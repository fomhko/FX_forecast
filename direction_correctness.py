import numpy as np
import pickle
import matplotlib.pyplot as plt
import config
from dataset import Dataset
gt_filename = 'y_test'
pred_filename = 'y_pred_test'
pred_var_filename = 'y_pred_var'
cut =  10

def direction_correctness(y_pred_test,y_test):
    seq_len = len(y_test)
    gt_direction = (y_test[1:] - y_test[:seq_len -1])>0
    pred_direction = (y_pred_test[1:] - y_test[:seq_len - 1])>0
    correct = np.sum(gt_direction == pred_direction)/(seq_len - 1)
    print("movement direction correctness:",correct)
    return correct

if __name__ == '__main__':
    gt = open(gt_filename,'rb')
    pred = open(pred_filename,'rb')
    var = open(pred_var_filename,'rb')
    y_test = pickle.load(gt)
    y_pred_test = pickle.load(pred)
    direction_correctness(y_pred_test,y_test)


    plt.figure()


    train_size = int(y_test.shape[0]*config.SPLIT_RATIO)
    test_size = y_test.shape[0] - train_size
    x = range(1 + train_size, 1 + train_size + test_size // cut)
    #
    plt.plot(x, y_pred_test[:test_size // cut],
             label='predicted test')
    plt.plot(x, y_test[:test_size // cut],
             label='ground truth')

    plt.legend()
    plt.savefig('USDCAD_part' + '.png')
