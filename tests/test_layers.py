#import sys
#sys.path.append('/Users/zhiyue/Desktop/DeepFrame')
import unittest
import pytest
import numpy as np
from DeepFrame.tensor import Tensor
from DeepFrame.layers import Dense, RNN
from DeepFrame.functions import sigmoid, tanh


class Testlayers(unittest.TestCase):
    def test_dense1(self):
        '''
        actiation='linear', bias = True
        '''
        x = [[1,2,3],[4,5,6]]
        w = [[1,2],[3,4],[5,6]]
        b = [[-1,-2]]
        #convert to Tensor
        X = Tensor(x)
        W = Tensor(w, requires_grad=True)
        B = Tensor(b, requires_grad=True)

        #manual calculation
        out = X @ W + B
        out.backward([1,1])
        #use Dense.forward
        dense = Dense(input_size=3, output_size=2, bias=True)
        dense.w.data = np.array(w)
        dense.b.data = np.array(b)
        out_dense = dense.forward(Tensor(x))
        out_dense.backward([1,1])

        #check value
        np.testing.assert_array_equal(out.data, out_dense.data)
        #check gradient
        np.testing.assert_array_equal(W.grad, dense.w.grad)
        np.testing.assert_array_equal(B.grad, dense.b.grad)

    def test_dense2(self):
        '''
        actiation='sigmoid', bias = False
        '''
        x = [[-11,7,-6],[1,9,4]]
        w = [[5,2],[-3,4],[5,16]]
        #convert to Tensor
        X = Tensor(x)
        W = Tensor(w, requires_grad=True)

        #manual calculation
        y = X @ W
        out = sigmoid(y)
        out.backward([1,1])
        #use Dense.forward
        dense = Dense(input_size=3, output_size=2, activation='sigmoid', bias=False)
        dense.w.data = np.array(w)
        out_dense = dense.forward(Tensor(x))
        out_dense.backward([1,1])

        #check value
        np.testing.assert_array_equal(out.data, out_dense.data)
        #check gradient
        np.testing.assert_array_equal(W.grad, dense.w.grad)
        np.testing.assert_array_equal(np.zeros_like(dense.b.grad), dense.b.grad)


    def test_rnn1(self):
        '''
        test single layer: num_layers=1, bias = True
        '''
        #generate test case
        x = [[[11,-2,31],[14,5,-6]]]
        x1 = [[11,-2,31]]
        x2 = [[14,5,-6]]
        w = [[6,-2],[12,7]]
        v = [[3],[5]]
        u = [[1,-1],[11,17],[-21,2]]
        b=[[1,-5]]
        c=[4]
        h0 = [[0,0]]
        #convert to tensor
        X = Tensor(x)
        X1 = Tensor(x1)
        X2 = Tensor(x2)
        W = Tensor(w, requires_grad=True)
        V = Tensor(v, requires_grad=True)
        U = Tensor(u, requires_grad=True)
        B = Tensor(b, requires_grad=True)
        C = Tensor(c, requires_grad=True)
        H0 = Tensor(h0, requires_grad=True)

        #manual calculation
        y = X1@U+H0@W+B
        H1 = tanh(y)
        y2 = X2@U+H1@W+B
        H2 = tanh(y2)
        y3 = H2@V + C
        out = sigmoid(y3)
        out.backward(1)

        #use RNN.forward()
        rnn = RNN(input_size=3, hidden_size=2, num_layers=1, activation='sigmoid', bias=True)
        rnn.W_list[0].data=np.array(w)
        rnn.v.data=np.array(v)
        rnn.U_list[0].data=np.array(u)
        rnn.b_list[0].data=np.array(b)
        rnn.c.data=np.array(c)
        out_rnn, _ = rnn.forward(X)
        #print(out_rnn)
        out_rnn.backward(1)

        #check value
        np.testing.assert_array_equal(out.data, out_rnn.data)
        #check gradient
        print(W.grad)
        print(rnn.W_list[0].grad)
        np.testing.assert_array_equal(W.grad, rnn.W_list[0].grad)
        np.testing.assert_array_equal(V.grad, rnn.v.grad)
        np.testing.assert_array_equal(U.grad, rnn.U_list[0].grad)
        np.testing.assert_array_equal(B.grad, rnn.b_list[0].grad)
        np.testing.assert_array_equal(C.grad, rnn.c.grad)
        #assert 1==2

    def test_rnn2(self):
        '''
        test single layer: num_layers=1, bias = False, activation = 'linear'
        '''
        #generate test case
        x = [[[1,2,3],[4,5,6]]]
        x1 = [[1,2,3]]
        x2 = [[4,5,6]]
        w = [[2,2],[2,2]]
        v = [[3],[3]]
        u = [[1,1],[1,1],[1,1]]
        h0 = [[0,0]]
        #convert to tensor
        X = Tensor(x)
        X1 = Tensor(x1)
        X2 = Tensor(x2)
        W = Tensor(w, requires_grad=True)
        V = Tensor(v, requires_grad=True)
        U = Tensor(u, requires_grad=True)
        H0 = Tensor(h0, requires_grad=True)

        #manual calculation
        y = X1@U+H0@W
        H1 = tanh(y)
        y2 = X2@U+H1@W
        H2 = tanh(y2)
        out = H2@V
        out.backward(1)

        #use RNN.forward()
        rnn = RNN(input_size=3, hidden_size=2, num_layers=1, activation='linear', bias=False)
        rnn.W_list[0].data=np.array(w)
        rnn.v.data=np.array(v)
        rnn.U_list[0].data=np.array(u)
        out_rnn, _ = rnn.forward(X)
        out_rnn.backward(1)

        #check value
        np.testing.assert_array_equal(out.data, out_rnn.data)
        #check gradient
        np.testing.assert_array_equal(W.grad, rnn.W_list[0].grad)
        np.testing.assert_array_equal(V.grad, rnn.v.grad)
        np.testing.assert_array_equal(U.grad, rnn.U_list[0].grad)
        np.testing.assert_array_equal(np.zeros_like(rnn.b_list[0].grad), rnn.b_list[0].grad)

    def test_rnn3(self):
        '''
        test multiple layer: num_layers=2, bias = True, activation='sigmoid'
        '''
        #generate test case
        x = [[[-1,12,3],[24,-5,-6]]]
        x1 = [[-1,12,3]]
        x2 = [[24,-5,-6]]
        w = [[12,-2],[21,1]]#first layer
        w1 = [[-1,1],[2,-2]]#hidden layers
        v = [[1],[3]]#output layer
        u1 = [[1,2], [3,4]]#hidden layers
        u = [[1,-1],[2,1],[1,3]]
        b=[[2,-1]]#first layer
        b1 = [[1,2]]#hidden layers
        c=[5]#output layer
        h0 = [[0,0]]#first layer
        h1 = [[0,0]]#2nd layer
        #convert to tensor
        X = Tensor(x)
        X1 = Tensor(x1)
        X2 = Tensor(x2)
        W = Tensor(w, requires_grad=True)
        W1 = Tensor(w1, requires_grad=True)
        V = Tensor(v, requires_grad=True)
        U1 = Tensor(u1, requires_grad=True)
        U = Tensor(u, requires_grad=True)
        B = Tensor(b, requires_grad=True)
        B1 = Tensor(b1, requires_grad=True)
        C = Tensor(c, requires_grad=True)
        H0 = Tensor(h0, requires_grad=True)
        H1 = Tensor(h1, requires_grad=True)

        #manual calculation
        y = X1@U+H0@W+B
        h0_right = tanh(y)
        h0_up = h0_right
        y2 = h0_up@U1+H1@W1 + B1
        h1_right = tanh(y2)
        y2 = X2@U+h0_right@W+B
        h0_right_2 = tanh(y2)
        h0_up_2 = h0_right_2
        y3 = h0_up_2@U1+h1_right@W1+B1
        h1_up_2 = tanh(y3)
        out = h1_up_2@V+C
        out = sigmoid(out)
        out.backward(1)

        #use RNN.forward()
        rnn = RNN(input_size=3, hidden_size=2, num_layers=2, activation='sigmoid', bias=True)
        rnn.W_list[0].data=np.array(w)
        rnn.U_list[0].data=np.array(u)
        rnn.b_list[0].data = np.array(b)
        rnn.W_list[1].data = np.array(w1)
        rnn.U_list[1].data = np.array(u1)
        rnn.b_list[1].data = np.array(b1)
        rnn.c.data = np.array(c)
        rnn.v.data = np.array(v)
        out_rnn, _ = rnn.forward(X)
        print(out_rnn)
        out_rnn.backward(1)

        #check value
        np.testing.assert_array_equal(out.data, out_rnn.data)
        #check gradient
        np.testing.assert_array_equal(W.grad, rnn.W_list[0].grad)
        np.testing.assert_array_equal(V.grad, rnn.v.grad)
        np.testing.assert_array_equal(U.grad, rnn.U_list[0].grad)
        np.testing.assert_array_equal(B.grad, rnn.b_list[0].grad)
        np.testing.assert_array_equal(C.grad, rnn.c.grad)
        np.testing.assert_array_equal(W1.grad, rnn.W_list[1].grad)
        np.testing.assert_array_equal(U1.grad, rnn.U_list[1].grad)
        np.testing.assert_array_equal(B1.grad, rnn.b_list[1].grad)
        #assert 1==2
