#add DeepFrame project path to system path otherwise it may throw "no module named 'DeepFrame'"
import sys
sys.path.append('/Users/zhiyue/Desktop/DeepFrame')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DeepFrame.tensor import Tensor
from DeepFrame.module import Module, Parameter, save_model, load_model
from DeepFrame.optimizers import Adam
from DeepFrame.losses import MSE_loss, MAE_loss
from DeepFrame.layers import RNN

def load_data()->'df':
    path = 'data/GOOG.csv'
    df = pd.read_csv(path)
    data = df[['Date','Close']]
    data['Close'] = data['Close']-data['Close'].mean()
    time_window = 10
    for i in range(1,time_window):
        data['delay'+str(i)] = data['Close'].shift(-i).fillna(0)
    data['y'] = data['Close'].shift(-11).fillna(0)
    data.drop(columns=['Date'], inplace=True)
    train = data.iloc[:int(0.7*len(data))].values
    validate = data.iloc[int(0.7*len(data)):].values
    train_x = train[:,:-1].reshape(-1,10,1)
    train_y = train[:,-1].reshape(-1,1)
    validate_x = validate[:-12,:-1].reshape(-1,10,1)
    validate_y = validate[:-12,-1].reshape(-1,1)
    return Tensor(train_x), Tensor(train_y), Tensor(validate_x), Tensor(validate_y)

class stock_price_model(Module):
    def __init__(self, input_size):
        self.layer1 = RNN(input_size, hidden_size=20, num_layers=1, activation='linear', bias=True)
    def forward(self, input_data:'tensor')->'tensor':
        return self.layer1.forward(input_data)
    def fit(self, train_x:'tensor', train_y:'tensor'):
        epochs = 10000
        lr = 0.01
        optimizer = Adam(lr, self)
        for epoch in range(epochs):
            self.zero_grad()
            y_pred, _ = self.forward(train_x)
            loss = MSE_loss(y_pred, train_y)
            loss.backward()
            optimizer.update(self)
            print('epoch', epoch, '     loss', loss.data)
    def validate(self, train_x:'tensor', train_y:'tensor',validate_x:'tensor', validate_y:'tensor'):
        y_pred_train, _ = self.forward(train_x)
        y_pred_validate, _ = self.forward(validate_x)
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot([i for i in range(train_x.data.shape[0])], y_pred_train.data)
        plt.plot([i for i in range(train_x.data.shape[0])], train_y.data)
        plt.legend(['prediction(train)', 'ground_truth(train)'])
        plt.subplot(1,2,2)
        plt.plot([i for i in range(validate_x.data.shape[0])], y_pred_validate.data)
        plt.plot([i for i in range(validate_x.data.shape[0])], validate_y.data)
        plt.legend(['prediction(validation)', 'ground_truth(validation)'])
        plt.show()



if __name__=='__main__':
    train_x, train_y, validate_x, validate_y = load_data()
    model = stock_price_model(input_size=1)
    model.fit(train_x, train_y)
    model.validate(train_x, train_y, validate_x, validate_y)
    save_model(model, 'goog_price_prediction_model')
    #model = load_model('goog_price_prediction_model')
    #model.fit(train_x, train_y)
    #model.validate(train_x, train_y, validate_x, validate_y)
