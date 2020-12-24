import sys
sys.path.append('/Users/zhiyue/Desktop/DeepFrame')
import pandas as pd
import numpy as np

from DeepFrame.tensor import Tensor
from DeepFrame.model import Model, Parameter
from DeepFrame.optimizers import SGD, Adam
from DeepFrame.functions import sigmoid, tanh
from DeepFrame.losses import binary_CE_loss, focal_loss
from DeepFrame.metrics import accuracy


def load_data()->'df':
    path = '/Users/zhiyue/Desktop/DeepFrame/dataset/breast_cancer_dataset.csv'
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

class model(Model):
    def __init__(self, input_shape):
        self.w1 = Parameter([input_shape, 10])
        self.b1 = Parameter([1,10])
        self.w2 = Parameter([10, 1])
        self.b2 = Parameter([1,1])
    def forward(self, train_x:'tensor')->'tensor':
        y1 = train_x @ self.w1 + self.b1
        y1 = sigmoid(y1)
        y2 = y1 @ self.w2 + self.b2
        y_pred = sigmoid(y2)
        return y_pred
    def fit(self, train_x:'tensor', train_y:'tensor'):
        epochs = 30000
        lr = 0.001
        #optimizer = SGD(lr)
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

    model = model(train_x.data.shape[1])
    model.fit(train_x, train_y)
    model.validate(validate_x, validate_y)
