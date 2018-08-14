import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.autograd import Variable
from model import AttnEncoder, AttnDecoder,Model
from dataset import Dataset
from torch import optim
from direction_correctness import direction_correctness
import config
import pickle

class Trainer:

    def __init__(self, time_step, split, lr):
        self.dataset = Dataset(T = time_step, split_ratio=split,binary_file=config.BINARY_DATASET)
        self.encoder = AttnEncoder(input_size=self.dataset.get_num_features(), hidden_size=config.ENCODER_HIDDEN_SIZE, time_step=time_step)
        self.decoder = AttnDecoder(code_hidden_size=config.ENCODER_HIDDEN_SIZE, hidden_size=config.DECODER_HIDDEN_SIZE, time_step=time_step)
        self.model = Model(self.encoder,self.decoder)
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.model = self.model.cuda()

        self.model_optim = optim.Adam(self.model.parameters(),lr)
        # self.encoder_optim = optim.Adam(self.encoder.parameters(), lr)
        # self.decoder_optim = optim.Adam(self.decoder.parameters(), lr)
        self.loss_func = nn.MSELoss()
        self.train_size, self.test_size = self.dataset.get_size()

    def train_minibatch(self, num_epochs, batch_size, interval):
        x_train, y_train, y_seq_train = self.dataset.get_train_set()
        for epoch in range(num_epochs):
            max_acc = 0
            i = 0
            loss_sum = 0
            while (i < self.train_size):
                self.model_optim.zero_grad()
                batch_end = i + batch_size
                if (batch_end >= self.train_size):
                    break
                var_x = self.to_variable(x_train[i: batch_end])
                var_y = self.to_variable(y_train[i: batch_end])
                var_y_seq = self.to_variable(y_seq_train[i: batch_end])
                if var_x.dim() == 2:
                    var_x = var_x.unsqueeze(2)
                y_res,y_var = self.model(var_x,var_y_seq)
                loss = self.loss_func(y_res, var_y)
                loss.backward()
                self.model_optim.step()
                print('[%d], loss is %f' % (epoch, 10000 * loss.data[0]))
                loss_sum += loss.data.item()
                i = batch_end
            print('epoch [%d] finished, the average loss is %f' % (epoch, loss_sum))

            x_dev ,y_dev,y_seq_dev = self.dataset.get_dev_set()
            y_pred_dev = self.predict(x_dev,y_dev,y_seq_dev,batch_size)
            acc = direction_correctness(y_pred_test=y_pred_dev,y_test=y_dev)
            if(acc > max_acc):
                max_acc = acc
            elif acc < max_acc*0.9:#prevent overfit
                break
            if (epoch + 1) % (interval) == 0 or epoch + 1 == num_epochs:
                torch.save(self.encoder.state_dict(), 'models/encoder' + str(epoch + 1) + '.model')
                torch.save(self.decoder.state_dict(), 'models/decoder' + str(epoch + 1) + '.model')


    def test(self, num_epochs, batch_size):
        x_test, y_test, y_seq_test = self.dataset.get_test_set()
        y_pred_test = self.predict(x_test, y_seq_test, batch_size)
        f = open('y_test','wb')
        pickle.dump(y_test,f)
        f.close()
        f = open('y_pred_test','wb')
        pickle.dump(y_pred_test,f)
        f.close()



        # plt.figure()
        # plt.ylim(0,1)
        # # plt.plot(range(1, 1 + self.train_size), y_train, label='train')
        # plt.plot(range(1 + self.train_size, 1 + self.train_size + self.test_size//50), y_test[:self.test_size//50], label='ground truth')
        # # plt.plot(range(1, 1 + self.train_size), y_pred_train, label.='predicted train')
        # plt.plot(range(1 + self.train_size, 1 + self.train_size + self.test_size//50), y_pred_test[:self.test_size//50], label='predicted test')
        # plt.savefig('res-' + str(num_epochs) + '.png')


    def predict(self, x, y_seq, batch_size):
        y_pred = np.zeros(x.shape[0])
        i = 0
        while (i < x.shape[0]):
            batch_end = i + batch_size
            if batch_end > x.shape[0]:
                break
                #batch_end = x.shape[0]
            var_x_input = self.to_variable(x[i: batch_end])
            var_y_input = self.to_variable(y_seq[i: batch_end])
            if var_x_input.dim() == 2:
                var_x_input = var_x_input.unsqueeze(2)
            # code = self.encoder(var_x_input)
            # y_res = self.decoder(code, var_y_input)
            y_res,_ = self.model(var_x_input,var_y_input)
            for j in range(i, batch_end):
                y_pred[j] = y_res[j - i]
            i = batch_end
        return y_pred

    def single_predict(self,x,y_seq):
        var_x_input = self.to_variable(x)
        var_y_input = self.to_variable(y_seq)
        if var_x_input.dim() == 2:
            var_x_input = var_x_input.unsqueeze(2)
        y_res, _ = self.model(var_x_input, var_y_input)
        return y_res

    def load_model(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))
        self.model = Model(self.encoder,self.decoder)
    def to_variable(self, x):
        if torch.cuda.is_available():
            return Variable(torch.from_numpy(x).float()).cuda()
        else:
            return Variable(torch.from_numpy(x).float())



def getArgParser():
    parser = argparse.ArgumentParser(description='Train the dual-stage attention-based model on stock')
    parser.add_argument(
        '-e', '--epoch', type=int, default =100,
        help='the number of epochs')
    # parser.add_argument(hg
    #     '-b', '--batch', type=int, default=1,
    #     help='the mini-batch size')
    parser.add_argument(
        '-s', '--split', type=float, default=0.8,
        help='the split ratio of validation set')
    parser.add_argument(
        '-i', '--interval', type=int, default=1,
        help='save models every interval epoch')
    parser.add_argument(
        '-l', '--lrate', type=float, default=0.0001,
        help='learning rate')
    parser.add_argument(
        '-t', '--test', action='store_true',
        help='train or test')
    return parser


if __name__ == '__main__':
    args = getArgParser().parse_args()
    num_epochs = args.epoch
    batch_size = config.BATCH_SIZE
    split = args.split
    interval = args.interval
    lr = args.lrate
    test = args.test
    trainer = Trainer(config.TIME_STEP, config.SPLIT_RATIO, lr)
    if not test:
        # trainer.load_model('models/encoder10.model', 'models/decoder10.model')
        trainer.train_minibatch(num_epochs, batch_size, interval)
    else:
        trainer.load_model('models/encoder1.model', 'models/decoder1.model')
        trainer.test(num_epochs, batch_size)
