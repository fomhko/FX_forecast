import numpy as np
import pandas as pd
import math
import config
import pickle
import os.path

class Dataset:
    def __init__(self, driving_csv=[], target_csv='', T=50, split_ratio=0.9,binary_file = '',shuffle = True):
        self.split_ratio = split_ratio

        if(os.path.isfile(binary_file)):
            self.load(binary_file)
        else:
            FXrate_frame_x = []
            self.gran = config.GRANULARITY
            for i in range(len(driving_csv)):
                col = pd.read_csv(driving_csv[i])['Close'].fillna(method='pad')
                col = (col - col.min())/(col.max() - col.min())
                FXrate_frame_x.append(col.values)
            FXrate_frame_x = np.array(FXrate_frame_x)
            y = pd.read_csv(target_csv)['Close'].fillna(method = 'pad')
            FXrate_frame_y = (y - y.min())/(y.max() - y.min())
            self.X, self.y, self.y_seq = self.time_series_gen(FXrate_frame_x,FXrate_frame_y,T,shuffle=shuffle)

    def get_size(self):
        return int(self.X.shape[0]*self.split_ratio), self.X.shape[0] - int(self.X.shape[0]*self.split_ratio)
    def get_train_set(self):
        train_size,_ = self.get_size()
        return self.X[:train_size], self.y[:train_size], self.y_seq[:train_size]
    def get_test_set(self):
        train_size,_ = self.get_size()
        return self.X[train_size:], self.y[train_size:], self.y_seq[train_size:]
    def get_dev_set(self):
        train_size,dev_size = self.get_size()
        dev_size = dev_size//10
        return self.X[train_size:train_size+dev_size],self.y[train_size:train_size+dev_size],self.y_seq[train_size:train_size+dev_size]
    def get_num_features(self):
        return self.X.shape[2]
    def time_series_gen(self,X,y,T,shuffle = False):
        ts_x, ts_y, ts_y_seq = [], [], []
        for i in range(T*max(self.gran),X.shape[1],min(self.gran)):
            # last = i + T
            col = []
            for g in self.gran:
                if g == min(self.gran):
                    col.append(X[:,i-T*g:i:g])
                else:
                    temp = np.zeros([X.shape[0], T])
                    for j in range(T):
                        temp[:,j] = X[:,i+(j-T)*g:i+(j-T+1)*g:min(self.gran)].mean(axis = 1)
                    col.append(temp)

            col = np.array(col)
            col = col.reshape(-1,col.shape[0]*col.shape[1])

            ts_x.append(col)
            ts_y.append(y[i])
            ts_y_seq.append(y[i-T:i])
        self.train_size = int(self.split_ratio * len(ts_x))
        self.test_size = len(ts_x) - self.train_size

        randomize = np.arange(len(ts_x))
        if shuffle:
            np.random.shuffle(randomize[:self.train_size])
        return np.array(ts_x)[randomize], np.array(ts_y)[randomize], np.array(ts_y_seq)[randomize]
    def save(self,binary_file):
        f = open(binary_file,'wb')
        pickle.dump(self,f)
        f.close()
    def load(self,binary_file):
        f = open(binary_file,'rb')
        ds = pickle.load(f)
        self.X,self.y,self.y_seq = ds.X,ds.y,ds.y_seq

        f.close()
if __name__ == '__main__':
    dir = config.DATA_DIR
    driving_csv = [
                   dir + 'AUDUSD_1 Min_Ask_2004.01.01_2017.12.08.csv',
                   dir + 'EURUSD_1 Min_Ask_2004.01.01_2017.12.08.csv',
                   dir + 'GBPUSD_1 Min_Ask_2004.01.01_2017.12.08.csv',
                   dir + 'USDCAD_1 Min_Ask_2004.01.01_2017.12.08.csv',
                   dir + 'USDJPY_1 Min_Ask_2004.01.01_2017.12.08.csv',
                   ] #driving series
    ds = Dataset(driving_csv,dir + 'USDCAD_1 Min_Ask_2004.01.01_2017.12.08.csv',T = config.TIME_STEP)#target series
    ds.save(config.binary_file)
