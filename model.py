import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from weight_drop import WeightDrop
from replay_memory import Transition
import config
import numpy as np

class DQN(nn.Module):
    def __init__(self,encoder,decoder):
        super(DQN,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc1 = nn.Linear(in_features = decoder.get_hidden_size() + 2,out_features = 3)
        # self.fc2 = nn.Linear(in_features = decoder.get_hidden_size(),out_features = 2)
    def forward(self,state):
        driving_x = state[0]
        y_seq = state[1]
        position = state[2]
        code = self.encoder(driving_x)
        hidden = self.decoder(code,y_seq,pre_train = False)
        action_value = self.fc1((torch.cat((hidden,position),dim = 1)))
        action_value = F.softmax(action_value,dim = 1)
        return action_value

class Model(nn.Module):
    def __init__(self,encoder,decoder,dropout = True):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dropout = dropout
    def forward(self,driving_x,var_y_seq):
        batch_size = driving_x.size(0)
        if(self.dropout):
            y_drop = torch.zeros([config.B,batch_size,1])
            for i in range(config.B): #
                code = self.encoder(driving_x)
                y_drop[i] = self.decoder(code,var_y_seq)
            return y_drop.mean(dim=0).squeeze(1),y_drop.var(dim=0).squeeze(1)
        else:
            code = self.encoder(driving_x)
            return self.decoder(code,var_y_seq),0

class AttnEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, time_step):
        super(AttnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = time_step
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.wdrnn = WeightDrop(self.lstm, ['weight_hh_l0','weight_ih_l0'], dropout=config.DROP_OUT)
        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=self.T)
        self.attn2 = nn.Linear(in_features=input_size, out_features=input_size)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=self.T, out_features=1)


    def forward(self, driving_x):
        batch_size = driving_x.size(0)
        # batch_size * time_step * hidden_size
        code = self.init_variable(batch_size, self.T, self.hidden_size)
        # initialize hidden state
        h = self.init_variable(1, batch_size, self.hidden_size)
        # initialize cell state
        s = self.init_variable(1, batch_size, self.hidden_size)
        for t in range(self.T):
            x = torch.cat((self.embedding_hidden(h), self.embedding_hidden(s)), 2)
            z1 = self.attn1(x)
            z2 = self.attn2(driving_x).permute(0,2,1)
            x = z1 + z2
            z3 = self.attn3(self.tanh(x))
            if batch_size > 1:
                attn_w = F.softmax(z3.view(batch_size, self.input_size), dim=1)
            else:
                attn_w = self.init_variable(batch_size, self.input_size) + 1
            weighted_x = torch.mul(attn_w, driving_x[:, t, :])
            # _, states = self.lstm(weighted_x.unsqueeze(0), (h, s))
            _, states = self.wdrnn(weighted_x.unsqueeze(0), (h, s)) #drop out

            h = states[0]
            s = states[1]

            code[:, t, :] = h

        return code

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.repeat(self.input_size, 1, 1).permute(1, 0, 2)



class AttnDecoder(nn.Module):

    def __init__(self, code_hidden_size, hidden_size, time_step):
        super(AttnDecoder, self).__init__()
        self.code_hidden_size = code_hidden_size
        self.hidden_size = hidden_size
        self.T = time_step

        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=code_hidden_size)
        self.attn2 = nn.Linear(in_features=code_hidden_size, out_features=code_hidden_size)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=code_hidden_size, out_features=1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size)
        self.wdrnn = WeightDrop(self.lstm, ['weight_hh_l0', 'weight_ih_l0'], dropout=config.DROP_OUT)
        self.tilde = nn.Linear(in_features=self.code_hidden_size + 1, out_features=1)
        self.fc1 = nn.Linear(in_features=code_hidden_size + hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=1)
        # self.fc3 = nn.Linear(in_features=hidden_size, out_features=1)
    def forward(self, h, y_seq,pre_train = True):
        batch_size = h.size(0)
        d = self.init_variable(1, batch_size, self.hidden_size)
        s = self.init_variable(1, batch_size, self.hidden_size)
        ct = self.init_variable(batch_size, self.hidden_size)

        for t in range(self.T):
            x = torch.cat((self.embedding_hidden(d), self.embedding_hidden(s)), 2)
            z1 = self.attn1(x)
            z2 = self.attn2(h)
            x = z1 + z2
            z3 = self.attn3(self.tanh(x))
            if batch_size > 1:
                beta_t = F.softmax(z3.view(batch_size, -1), dim=1)
            else:
                beta_t = self.init_variable(batch_size, self.T) + 1
            ct = torch.bmm(beta_t.unsqueeze(1), h).squeeze(1)
            if t < self.T - 1:
                yc = torch.cat((y_seq[:, t].unsqueeze(1), ct), dim=1)
                y_tilde = self.tilde(yc)
                # _, states = self.lstm(y_tilde.unsqueeze(0), (d, s))
                _, states = self.wdrnn(y_tilde.unsqueeze(0), (d, s))
                d = states[0]
                s = states[1]
        state = self.fc1(torch.cat((d.squeeze(0), ct), dim=1))
        if(pre_train):
            y_res = self.fc2(state)
            return y_res
        else:
            return state

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.repeat(self.T, 1, 1).permute(1, 0, 2)

    def get_hidden_size(self):
        return self.hidden_size
