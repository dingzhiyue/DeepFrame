import numpy as np


class SGD:
    def __init__(self, lr=0.001):
        self.lr = lr
    def update(self, Model:'DeepFrame.Model'):
        for parameter in Model.get_parameters():
            parameter.data -= self.lr * parameter.grad

class Adam:
    def __init__(self, lr, Model:'DeepFrame.Model'):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.alpha = lr
        self.V, self.S = self.create_S_V(Model)

    def create_S_V(self, Model:'DeepFrame.Model')->'lists':
        V, S= [], []
        for parameter in Model.get_parameters():
            V.append(np.zeros_like(parameter.grad))
            S.append(np.zeros_like(parameter.grad))
        return V, S

    def update(self, Model):
        epsilon = 0.0000000001
        #i = 0
        for i, parameter in enumerate(Model.get_parameters()):
            V_new = (self.beta1*self.V[i]+(1-self.beta1)*parameter.grad)#/(1-self.beta1**self.t) #ndarray
            S_new = (self.beta2*self.S[i]+(1-self.beta2)*parameter.gradient*parameter.grad)#/(1-self.beta2**self.t) #ndarray
            parameter.data -= self.alpha * V_new / (S_new**0.5+epsilon)
            self.V[i] = V_new
            self.S[i] = S_new
            #i += 1
