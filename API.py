import config
import numpy as np
from trainer import Trainer
import pickle
from dataset import Dataset
class API:
    def __init__(self,encoder,decoder):
        self.trainer = Trainer(config.TIME_STEP, config.SPLIT_RATIO, 0.0001)
        self.trainer.load_model(encoder, decoder)
    def process(self,x):
        x[0] = (x[1] - 0.60113) / (1.10802 - 0.60113)
        x[1] = (x[1] - 1.03436) / (1.60315 - 1.03436)
        x[2] = (x[2] - 1.19506) / (2.11585 - 1.19506)
        x[3] = (x[3] - 0.9062) / (1.469 - 0.9062)
        x[4] = (x[4] - 75.625) / (125.82 - 75.625)
        col = []
        T = config.TIME_STEP
        i = x.shape[1]
        for g in config.GRANULARITY:
            if g == min(config.GRANULARITY):
                col.append(x[:, -T:i])
            else:
                temp = np.zeros([x.shape[0], T])
                for j in range(T):
                    temp[:, j] = x[:, i + (j - T) * g:i + (j - T + 1) * g].mean(axis=1)
                col.append(temp)
        col = np.array(col)
        features = col.reshape(-1, col.shape[0] * col.shape[1])
        y_seq = x[3][i - T:i]
        return features,y_seq
    def predict(self,x):
        features,y_seq = self.process(x)
        y_res = self.trainer.single_predict(features,y_seq)
        print(y_res)
if __name__ == '__main__':
    api =  API('models/encoder10.model','models/decoder10.model')
    f = open('x','rb')
    x = pickle.load(f).transpose()
#input is a numpy array of shap(e(