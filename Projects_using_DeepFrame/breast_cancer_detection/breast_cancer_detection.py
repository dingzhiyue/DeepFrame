#add DeepFrame project path to system path otherwise it may throw "no module named 'DeepFrame'"
import sys
sys.path.append('/Users/zhiyue/Desktop/DeepFrame')


import pandas as pd
import numpy as np

from DeepFrame.tensor import Tensor
from DeepFrame.module import Module, save_model, load_model
from DeepFrame.optimizers import Adam
from DeepFrame.functions import sigmoid, tanh
from DeepFrame.losses import binary_CE_loss, focal_loss
from DeepFrame.metrics import accuracy
from DeepFrame.layers import Dense


def load_data()->'df':
    path = 'data/breast_cancer_dataset.csv'
    df = pd.read_csv(path)
    df['diagnosis'].loc[df['diagnosis']=='B'] = 0
    df['diagnosis'].loc[df['diagnosis']=='M'] = 1
    train, validate = np.split(df.sample(frac=1), [int(0.8*len(df))])
    df_train_y = train['diagnosis']
    df_train_x = train.drop(columns=['diagnosis','id','Unnamed: 32'])
    df_validate_y = validate['diagnosis']
    df_validate_x = validate.drop(columns=['diagnosis','id', 'Unnamed: 32'])
    return df_train_x, df_train_y, df_validate_x, df_validate_y

def data_prepare(df_train_x:'df', df_train_y, df_validate_x, df_validate_y)->'tensor-t.data.dtype=float':
    train_x = Tensor(df_train_x.values.astype(float))
    train_y = Tensor(df_train_y.values.astype(float).reshape(-1,1))
    validate_x = Tensor(df_validate_x.values.astype(float))
    validate_y = Tensor(df_validate_y.values.astype(float).reshape(-1,1))
    return train_x, train_y, validate_x, validate_y

class breast_cancer_model(Module):
    def __init__(self, input_shape):
        self.layer1 = Dense(input_shape, 10, 'sigmoid', 'layer1')
        self.layer2 = Dense(10, 1, 'sigmoid', 'layer2')
    def forward(self, train_x:'tensor')->'tensor':
        y1 = self.layer1.forward(train_x)
        y2 = self.layer2.forward(y1)
        return y2
    def fit(self, train_x:'tensor', train_y:'tensor'):
        epochs = 30000
        lr = 0.001
        optimizer = Adam(lr, self)
        for epoch in range(epochs):
            self.zero_grad()
            y_pred = self.forward(train_x)
            loss = binary_CE_loss(y_pred, train_y)
            loss.backward()
            optimizer.update(self)
            print('epoch', epoch, '     loss', loss.data)
        y_pred = self.forward(train_x)
        acc = accuracy(y_pred, train_y)
        print('train acc', acc)
    def validate(self, validate_x:'tensor', validate_y:'tensor'):
        y_pred = self.forward(validate_x)
        acc = accuracy(y_pred, validate_y)
        print('validate acc', acc)

if __name__=='__main__':
    train_x, train_y, validate_x, validate_y = load_data()
    train_x, train_y, validate_x, validate_y = data_prepare(train_x, train_y, validate_x, validate_y)
    model = breast_cancer_model(train_x.data.shape[1])
    model.fit(train_x, train_y)
    model.validate(validate_x, validate_y)
    save_model(model,'bc_model')
    #model.save_parameters('parameters')
    #model.load_parameters('parameters')
    #model = load_model('bc_model')
    #model.validate(validate_x, validate_y)
