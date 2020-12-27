import numpy as np
from DeepFrame.tensor import Tensor
import inspect
import pickle

class Parameter(Tensor):
    '''
    parameter tensor with initialization
    '''
    def __init__(self, parameter_shape:'list'):
        data = np.random.randn(*parameter_shape)
        super().__init__(data, requires_grad=True)

class Module:
    def get_parameters(self):
        '''
        get all parameters of the module
        '''
        for name, obj in inspect.getmembers(self):
            if isinstance(obj, Parameter):
                yield obj
            elif isinstance(obj, Module):
                yield from obj.get_parameters()
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, Parameter):
                        yield item

    def zero_grad(self):
        '''
        set 0 for all the parameter grad
        '''
        for parameter in self.get_parameters():
            parameter.zero_grad()

    def save_parameters(self, save_path):
        '''
        save all the parameters to a numpy .npy file
        '''
        model_parameters = []
        for parameter in self.get_parameters():
            model_parameters.append(parameter.data)
        with open(save_path, 'wb') as file:
            np.save(file, model_parameters)
            
    def load_parameters(self, load_path):
        '''
        load from numpy .npy file and set to model parameters
        '''
        with open(load_path, 'rb') as file:
            model_parameters = np.load(load_path, allow_pickle=True)
        model_parameters = iter(model_parameters)
        for parameter in self.get_parameters():
            parameter.data = next(model_parameters)


#save, load model functions
def save_model(model, save_path):
    '''
    save model into a pickle
    '''
    with open(save_path, 'wb') as file:
        pickle.dump(model, file)

def load_model(load_path):
    '''
    load saved model
    '''
    with open(load_path, 'rb') as file:
        model = pickle.load(file)
    return model
