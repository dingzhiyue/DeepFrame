import numpy as np


class SGD:
    def __init__(self, lr=0.001):
        self.lr = lr
    def update(self, Module:'DeepFrame.Module'):
        for parameter in Module.get_parameters():
            parameter.data -= self.lr * parameter.grad

class Adam:
    def __init__(self, lr, Module:'DeepFrame.Module'):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.alpha = lr
        self.V, self.S = self.create_S_V(Module)
    def create_S_V(self, Module:'DeepFrame.Module')->'lists[ndarray,...]':
        V, S= [], []
        for parameter in Module.get_parameters():
            V.append(np.zeros_like(parameter.grad))
            S.append(np.zeros_like(parameter.grad))
        return V, S
    def update(self, Module):
        epsilon = 0.0000000001
        for i, parameter in enumerate(Module.get_parameters()):
            V_new = (self.beta1*self.V[i]+(1-self.beta1)*parameter.grad)#/(1-self.beta1**self.t) #ndarray
            S_new = (self.beta2*self.S[i]+(1-self.beta2)*parameter.grad*parameter.grad)#/(1-self.beta2**self.t) #ndarray
            parameter.data -= self.alpha * V_new / (S_new**0.5+epsilon)
            self.V[i] = V_new
            self.S[i] = S_new
