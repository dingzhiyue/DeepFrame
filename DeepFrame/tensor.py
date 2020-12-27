import numpy as np

class Parents:
    def __init__(self, t:'tensor', grad_fn:'function_obj'):
        self.tensor = t
        self.grad_fn = grad_fn

class Tensor:
    def __init__(self, data:'ndarray', requires_grad:'bool'=False, parents:'list[(parent_tensor, grad_fn),...]'=[]):
        self.data = transfer_to_ndarray(data)
        self.requires_grad = requires_grad
        self.parents = parents
        self.grad:'ndarray'= []
        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def backward(self, grad:'ndarray'=None):
        '''
        backprop calculate all tensor gradient
        '''
        #if no grad passed in
        if grad is None:
            grad = transfer_to_ndarray(1.0)
        else:
            grad = transfer_to_ndarray(grad)
        self.grad = self.grad + grad
        for parent in self.parents:
            grad_pass_backward = parent.grad_fn(self.grad)
            parent.tensor.backward(grad_pass_backward)

    def sum(self)->'tensor':
        return _tensor_sum(self)
    #overload operators
    def __neg__(self)->'tensor':
        return _tensor_neg(self)
    def __matmul__(self, other_tensor:'tensor')->'tensor':
        return _tensor_matmul(self, other_tensor)
    def __add__(self, other:'int, list, ndarray, tensor')->'tensor':
        return _tensor_add(self, transfer_to_tensor(other))
    def __radd__(self, other):
        return _tensor_add(transfer_to_tensor(other), self)
    def __iadd__(self, other):
        self.data = self.data + transfer_to_tensor(other).data
        return self
    def __sub__(self, other):
        return _tensor_sub(self, transfer_to_tensor(other))
    def __rsub__(self, other):
        return _tensor_sub(transfer_to_tensor(other), self)
    def __isub__(self, other):
        self.data = self.data - transfer_to_tensor(other).data
        return self
    def __mul__(self, other):
        return _tensor_mul(self, transfer_to_tensor(other))
    def __rmul__(self, other):
        return _tensor_mul(transfer_to_tensor(other), self)
    def __imul__(self, other):
        self.data = self.data * transfer_to_tensor(other).data
        return self
    def __pow__(self, other):
        return _tensor_pow(self, transfer_to_tensor(other))
    def __getitem__(self, idx:'slice'):
        return _tensor_slice(self, idx)


#functions
def transfer_to_ndarray(data):
    '''
    convert int, float, list.. to ndarray
    '''
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)

def transfer_to_tensor(data):
    if isinstance(data, Tensor):
        return data
    else:
        return Tensor(data)

#operators
def _tensor_sum(t:'tensor')->'tensor':
    data = t.data.sum()
    requires_grad = t.requires_grad
    parents = []
    if t.requires_grad:
        def grad_fn(grad:'ndarray')->'ndarray':
            return grad * np.ones_like(t.data)
        parents.append(Parents(t, grad_fn))
    return Tensor(data, requires_grad, parents)

def _tensor_neg(t:'tensor')->'tensor':
    data = -t.data
    requires_grad = t.requires_grad
    parents = []
    if t.requires_grad:
        def grad_fn(grad):
            return -grad
        parents.append(Parents(t, grad_fn))
    return Tensor(data, requires_grad, parents)

def _tensor_matmul(t1:'tensor', t2:'tensor')->'tensor':
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2. requires_grad
    parents = []
    if t1.requires_grad:
        def grad_fn1(grad):
            return grad @ t2.data.T
        parents.append(Parents(t1, grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad):
            return t1.data.T @ grad
        parents.append(Parents(t2, grad_fn2))
    return Tensor(data, requires_grad, parents)

def boardcast_grad(t:'tensor', grad:'ndarray')->'ndarry':
    '''
    calculate the grad for boardcast operator: A=boardcast(t)->d/dt
    '''
    for _ in range(grad.ndim-t.data.ndim):#sum up extra dims, output grad has the same dim as t
        grad = grad.sum(axis=0)
    for i, dim in enumerate(t.data.shape):#handle when one dim=1, keepdims so that grad has the same dim as t
        if dim==1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

def _tensor_add(t1:'tensor', t2:'tensor')->'tensor':
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    parents = []
    if t1.requires_grad:
        def grad_fn1(grad):
            return boardcast_grad(t1, grad)
        parents.append(Parents(t1, grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad):
            return boardcast_grad(t2, grad)
        parents.append(Parents(t2, grad_fn2))
    return Tensor(data, requires_grad, parents)

def _tensor_sub(t1:'tensor', t2:'tensor')->'tensor':
    return _tensor_add(t1, _tensor_neg(t2))

def _tensor_mul(t1:'tensor', t2:'tensor')->'tensor':
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    parents = []
    if t1.requires_grad:
        def grad_fn1(grad):
            grad = grad * t2.data
            return boardcast_grad(t1, grad)
        parents.append(Parents(t1, grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad):
            grad = grad * t1.data
            return boardcast_grad(t2, grad)
        parents.append(Parents(t2, grad_fn2))
    return Tensor(data, requires_grad, parents)

def _tensor_pow(t1:'tensor', t2:'tensor')->'tensor':
    data = t1.data**t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    parents = []
    if t1.requires_grad:
        def grad_fn1(grad):
            grad = grad * t2.data * t1.data**(t2.data -1 )
            return boardcast_grad(t1, grad)
        parents.append(Parents(t1, grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad):
            grad = grad * (np.log(t1.data)) * data
            return boardcast_grad(t2, grad)
        parents.append(Parents(t2, grad_fn2))
    return Tensor(data, requires_grad, parents)

def _tensor_slice(t:'tensor', idx:'slice')->'tensor':
    data = t.data[idx]
    requires_grad = t.requires_grad
    parents = []
    if t.requires_grad:
        def grad_fn(grad):
            temp = np.zeros_like(t.data)
            temp[idx] = grad
            return temp
        parents.append(Parents(t, grad_fn))
    return Tensor(data, requires_grad, parents)
