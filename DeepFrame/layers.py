import numpy as np
from DeepFrame.tensor import Tensor
from DeepFrame.module import Module, Parameter
from DeepFrame.functions import sigmoid, tanh, ReLU, softmax

class Dense(Module):
    '''
    fully connected layer
    '''
    def __init__(self, input_size:'int', output_size:'int', activation:'str'='linear', bias:'bool'=True):
        self.bias = bias
        self.w = Parameter([input_size, output_size])
        if self.bias:
            self.b = Parameter([1, output_size])
        else:
            self.b = Tensor(np.zeros([1, output_size]))
        self.activation = activation

    def forward(self, input_data:'tensor')->'tensor':
        y = input_data @ self.w + self.b
        if self.activation=='linear':
            return y
        if self.activation=='sigmoid':
            return sigmoid(y)
        elif self.activation=='tanh':
            return tanh(y)
        elif self.activation=='softmax':
            return softmax(y)


class RNN(Module):
    '''
    unidirectional base RNN layer -> single output
    '''
    def __init__(self, input_size: 'int', hidden_size: 'int', num_layers: 'int'=1, activation:'str' = 'linear', bias: 'bool' = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.bias = bias
        self.W_list, self.U_list, self.b_list = self.init_parameters()#list of parameters
        #output layer
        self.v = Parameter([self.hidden_size, 1])
        if self.bias:
            self.c = Parameter([1, 1])
        else:
            self.c = Tensor(np.zeros([1,1]))

    def init_parameters(self)->'lists:[Parameter,....]':
        W_list, U_list, b_list = [], [], []
        for i in range(self.num_layers):
            if i==0:
                U_list.append(Parameter([self.input_size, self.hidden_size]))
            else:
                U_list.append(Parameter([self.hidden_size, self.hidden_size]))
            W_list.append(Parameter([self.hidden_size, self.hidden_size]))
            if self.bias:
                b_list.append(Parameter([1, self.hidden_size]))
            else:
                b_list.append(Tensor(np.zeros([1, self.hidden_size])))
        return W_list, U_list, b_list

    def forward(self, input_data:'tensor--[batch_size, time, feature]', hidden0: 'list[tensor,...], optional'=None)->'tensor, list[tensor,...]':
        time = input_data.data.shape[1]
        batch_size = input_data.data.shape[0]
        #initialize hidden0
        if hidden0 is None:
            hidden0 = []
            for _ in range(self.num_layers):
                hidden0.append(Tensor(np.zeros((batch_size, self.hidden_size)), requires_grad=True))
        out, hidden_out = [], []
        self.zero_grad()
        # go through each layer
        for i in range(self.num_layers):
            H = []
            if i==0:
                x = [input_data[:,t,:] for t in range(time)]
            #go through each timestamp
            for t in range(time):
                if t==0:
                    a = hidden0[i] @ self.W_list[i] + x[t] @ self.U_list[i] + self.b_list[i]
                else:
                    a = H[-1] @ self.W_list[i] + x[t] @ self.U_list[i] + self.b_list[i]
                H.append(tanh(a))
            hidden_out.append(H[-1])
            x = H[:]
        #output layer
        for t in range(time):
            o = x[t] @ self.v + self.c
            if self.activation=='linear':
                out.append(o)
            elif self.activation=='sigmoid':
                out.append(sigmoid(o))
            elif self.activation=='softmax':
                out.append(softmax(o))
            elif self.activation=='tanh':
                out.append(tanh(o))
            elif self.activation=='ReLU':
                out.append(ReLU(o))
        return out[-1], hidden_out
