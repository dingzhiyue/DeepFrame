import numpy as np
from DeepFrame.tensor import Tensor
import inspect

class Parameter(Tensor):
    '''
    parameter tensor with initialization
    '''
    def __init__(self, parameter_shape:'list'):
        data = np.random.randn(*parameter_shape)
        super().__init__(data, requires_grad=True)

class Model:
    def get_parameters(self):
        '''
        get all parameters of the model
        '''
        for name, obj in inspect.getmembers(self):
            if isinstance(obj, Parameter):
                yield obj
            elif isinstance(obj, Model):
                yield from obj.get_parameters()

    def zero_grad(self):
        '''
        set 0 for all the parameter grad
        '''
        for parameter in self.get_parameters():
            parameter.zero_grad()

    
