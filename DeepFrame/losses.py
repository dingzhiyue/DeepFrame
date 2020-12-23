import numpy as np
from DeepFrame.functions import abs, log

def MSE_loss(y_pred:'tensor', y_true:'tensor')->'tensor':
    return ((y_pred - y_true)*(y_pred - y_true)).sum()

def MAE_loss(y_pred:'tensor', y_true:'tensor')->'tensor':
    return abs(y_pred - y_true).sum()

def binary_CE_loss(y_pred:'tensor:sigmoid_output-[[0.9],[0.2],...]', y_true:'tensor-zero_one_encode-[[0],[1]...]')
    return (-y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred)).sum()

def focal_loss(y_pred:'tensor:sigmoid_output-[[0.9],[0.2],...]', y_true:'tensor-zero_one_encode-[[0],[1]...]', gamma=2.0, alpha=0.25)->'tensor':
    return (-alpha * y_true * (1 - y_pred)**gamma * log(y_pred) - (1 - alpha) * (1 - y_true) * y_pred**gamma * log(1 - y_pred)).sum()

def CE_loss(y_pred:'tensor:softmax_output-[[p1,p2..]..]', y_true:'tensor:onehot_encode-[[class1,class2..]..]')->'tensor':
    return (-y_true * log(y_pred)).sum()
