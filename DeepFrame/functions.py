import numpy as np
from DeepFrame.tensor import Tensor, Parents

def sigmoid(t:'tensor')->'tensor':
    data = 1 / (1+np.exp(-t.data))
    requires_grad = t.requires_grad
    parents = []
    if t.requires_grad:
        def grad_fn(grad):
            return grad * (data-data*data)
        parents.append(Parents(t, grad_fn))
    return Tensor(data, requires_grad, parents)

def ReLU(t:'tensor')->'tensor':
    data = t.data.copy()
    data[t.data<0] = 0.0
    requires_grad = t.requires_grad
    parents = []
    if t.requires_grad:
        def grad_fn(grad):
            temp = np.zeros_like(grad)
            temp[t.data>=0] = 1.0
            return grad * temp
        parents.append(Parents(t, grad_fn))
    return Tensor(data, requires_grad, parents)

def tanh(t: 'tensor')->'tensor':
    data = np.tanh(t.data)
    requires_grad = t.requires_grad
    parents = []
    if t.requires_grad:
        def grad_fn(grad:'ndarray')->'ndarray':
            return grad * (1 - data * data)
        parents.append(Parents(t, grad_fn))
    return Tensor(data, requires_grad, parents)

def softmax(t:'tensor-(1,n)')->'tensor':
    epsilon = 0.0000000001
    temp = np.exp(t.data-np.max(t.data))
    data = temp/(np.sum(temp)+epsilon)
    requires_grad = t.requires_grad
    parents = []
    if t.requires_grad:
        def grad_fn(grad):
            temp_sum = np.sum(-grad*data)
            return (temp_sum + grad) * data
        parents.append(Parents(t, grad_fn))
    return Tensor(data, requires_grad, parents)

def abs(t:'tensor')->'tensor':
    data  = np.abs(t.data)
    requires_grad = t.requires_grad
    parents = []
    if t.requires_grad:
        def grad_fn(grad):
            temp = np.ones_like(grad)
            temp[t.data<0] = -1.0
            return grad * temp
        parents.append(Parents(t, grad_fn))
    return Tensor(data, requires_grad, parents)

def log(t:'tensor')->'tensor':
    epsilon = 0.0000000001
    data = np.log(t.data)
    requires_grad = t.requires_grad
    parents = []
    if t.requires_grad:
        def grad_fn(grad):
            return grad / (t.data + epsilon)
        parents.append(Parents(t, grad_fn))
    return Tensor(data, requires_grad, parents)

def dropout(t:'tensor', p:'drop prob')->'tensor':
    if p<0 or p>1:
        raise Exception('probability should be between 0 and 1')
    epsilon = 0.0000000001
    mask = np.random.uniform(size=t.data.shape)
    mask[mask<=p]= 0
    mask[mask>p] = 1 / (1 - p + epsilon)
    data = t.data * mask
    requires_grad = t.requires_grad
    parents = []
    if t.requires_grad:
        def grad_fn(grad):
            return grad * mask
        parents.append(Parents(t, grad_fn))
    return Tensor(data, requires_grad, parents)
