import unittest
import pytest
import numpy as np
from DeepFrame.tensor import Tensor
from DeepFrame.functions import sigmoid, ReLU, tanh, softmax, abs, log, dropout


class test_Tensor(unittest.TestCase):
    def test_sum(self):
        t1 = Tensor([[1,2,3],[4,5,6]], requires_grad=True)
        t2 = t1.sum()
        t2.backward(3)
        assert t1.grad.tolist() == [[3,3,3],[3,3,3]]
        assert t2.data ==21
        assert t1.data.tolist()==[[1,2,3],[4,5,6]]

    def test_neg(self):
        t = Tensor([[1,2,3],[4,5,6]],requires_grad=True)
        t2 = -t
        t2.backward([[1,1,1],[1,1,1]])
        assert t.grad.tolist()==[[-1,-1,-1],[-1,-1,-1]]

    def test_matmul(self):
        t1 = Tensor([[1,2,3],[4,5,6]], requires_grad=True)
        t2 = Tensor([[1,2],[3,4],[5,6]], requires_grad=True)
        t3 = t1 @ t2
        t3.backward([[1,1],[1,1]])
        assert t3.data.tolist()==[[22,28],[49,64]]
        assert t1.grad.tolist()==[[3,7,11],[3,7,11]]
        assert t2.grad.tolist()==[[5,5],[7,7],[9,9]]

    def test_add(self):
        t1 = Tensor([[1,2,3],[4,5,6]], requires_grad=True)
        t2 = Tensor([[-1,-2,-3],[-4,-5,-6]], requires_grad=True)
        t3 = t1+t2
        t3.backward([[2,4,6],[8,10,12]])
        assert t1.grad.tolist()==[[2,4,6],[8,10,12]]
        assert t2.grad.tolist()==[[2,4,6],[8,10,12]]
        assert t3.data.tolist() == [[0,0,0],[0,0,0]]

    def test_add_broadcast(self):
        t1 = Tensor([[1,2,3],[4,5,6]], requires_grad=True)
        t2 = Tensor([[-1,-2,-3]], requires_grad=True)
        t3 = t1+t2
        #t3.backward(Tensor([[2,4,6],[8,10,12]]))
        #assert t1.grad.data.tolist()==[[2,4,6],[8,10,12]]
        #assert t2.grad.data.tolist()==[[10,14,18]]
        t3.backward([[1,1,1],[1,1,1]])
        assert t1.grad.tolist()==[[1,1,1],[1,1,1]]
        assert t2.grad.tolist()==[[2,2,2]]
        assert t3.data.tolist() == [[0,0,0],[3,3,3]]

    def test_sub_broadcast(self):
        t1 = Tensor([[1,2,3],[4,5,6]], requires_grad=True)
        t2 = Tensor([[7,8,9]], requires_grad=True)
        t3 = t1-t2
        t3.backward([[11,12,13],[14,15,16]])
        assert t1.grad.tolist()==[[11,12,13],[14,15,16]]
        assert t2.grad.tolist()==[[-25,-27,-29]]
        assert t3.data.tolist()==[[-6,-6,-6],[-3,-3,-3]]

        t4 = Tensor([[1,2,3],[4,5,6]], requires_grad=True)
        t5 = Tensor([[7,8,9]], requires_grad=True)
        t6 = t5-t4
        t6.backward([[11,12,13],[14,15,16]])
        assert t4.grad.tolist()==[[-11,-12,-13],[-14,-15,-16]]
        assert t5.grad.tolist()==[[25,27,29]]
        assert t6.data.tolist()==[[6,6,6],[3,3,3]]

    def test_mul_broadcast(self):
        t1 = Tensor([[1,2,3],[4,5,6]], requires_grad=True)
        t2 = Tensor([[7,8,9]], requires_grad=True)
        t3 = t1*t2
        t3.backward([[1,1,1],[1,1,1]])
        assert t1.grad.tolist()==[[7,8,9],[7,8,9]]
        assert t2.grad.tolist()==[[5,7,9]]
        assert t3.data.tolist() == [[7,16,27],[28,40,54]]

        t4 = Tensor([[7,8,9]], requires_grad=True)
        t5 = 3*t4
        t5.backward([[1,1,1]])
        assert t4.grad.tolist()==[[3,3,3]]
        assert t5.data.tolist()==[[21,24,27]]

    #functions
    def test_sigmoid(self):
        A = Tensor([[1,2,3],[4,5,6]], requires_grad=True)
        T = sigmoid(A)
        grad = np.array([[2,4,6],[8,10,12]])
        T.backward(grad)
        sig = 1/(1+np.exp(-A.data))
        assert T.data.tolist()==sig.tolist()
        assert A.grad.tolist()==(grad * (T.data - T.data * T.data)).tolist()

    def test_ReLU(self):
        A = Tensor([[1,-2,3],[-4,5,6]], requires_grad=True)
        T = ReLU(A)
        T.backward([[2,4,6],[8,10,12]])
        assert T.data.tolist()==[[1,0,3],[0,5,6]]
        assert A.grad.tolist()==[[2,0,6],[0,10,12]]

    def test_tanh(self):
        A = Tensor([[1,2,3],[4,5,6]], requires_grad=True)
        T = tanh(A)
        grad = np.array([[2,4,6],[8,10,12]])
        T.backward(grad)
        assert T.data.tolist()==np.tanh(A.data).tolist()
        assert A.grad.tolist()==(grad * (1 - T.data * T.data)).tolist()

    def test_softmax(self):
        A = Tensor([[1,2,3]], requires_grad=True)
        T = softmax(A)
        temp = np.exp(A.data-np.max(A.data))
        softmax_true = temp/np.sum(temp)
        grad = np.array([3,1,2])
        T.backward(grad)

        np.testing.assert_almost_equal(T.data, softmax_true)
        temp_sum = np.sum(-grad*softmax_true)
        grad_true = (temp_sum+grad)*softmax_true
        np.testing.assert_almost_equal(A.grad, grad_true)

    def test_abs(self):
        A = Tensor([[1,-2,3],[-4,5,6]], requires_grad=True)
        T = abs(A)
        T.backward([[2,4,6],[8,10,12]])
        assert T.data.tolist()==[[1,2,3],[4,5,6]]
        assert A.grad.tolist()==[[2,-4,6],[-8,10,12]]

    def test_log(self):
        A = Tensor([[1,2,3],[4,5,6]], requires_grad=True)
        T = log(A)
        T.backward([[2,4,6],[8,10,12]])
        log_true = np.log(np.array([[1,2,3],[4,5,6]]))
        grad = np.array([[2,4,6],[8,10,12]])
        grad_true = grad / A.data
        assert T.data.tolist()==log_true.tolist()
        np.testing.assert_almost_equal(A.grad, grad_true)

    def test_dropout(self):
        A = Tensor([[1,-2,3],[-4,5,6]], requires_grad=True)
        T = dropout(A, 0.2)
        T.backward([[2,4,6],[8,10,12]])
        print(T.data)
        print(A.grad)
        #assert 1==2
