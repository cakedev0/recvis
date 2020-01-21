from torch import nn
import torch
from abc import ABC, abstractmethod

class Regularizer(ABC):
    
    @abstractmethod
    def __call__(self, W):
        pass

class RegularizedParameter(nn.Parameter):
    
    def __init__(self, data=None, requires_grad=True, reg=None):
        self.regularizer = reg
    
    def __new__(cls, data=None, requires_grad=True, reg=None):
        return super(RegularizedParameter, cls).__new__(cls, data=data, requires_grad=requires_grad)
    
    def __repr__(self):
        return 'Regularized ' + super(RegularizedParameter, self).__repr__() + \
            '\n' + self._reg_repr()
    
    def _reg_repr(self):
        if(self.regularizer is None):
            return 'No regularizers'
        elif(type(self.regularizer) is dict):
            return 'Regularizers: ' + str(list(self.regularizer.keys()))
        else:
            return 'Regularizer: ' + self.regularizer.__code__.co_name

    def eval_regularizer(self):
        if(self.regularizer is None):
            return 0
        elif(type(self.regularizer) is dict):
            regs = {}
            for reg_name in self.regularizer:
                regs[reg_name] = self.regularizer[reg_name](self)
            return regs
        else:
            return self.regularizer(self)
    
class NAU_Regularizer(Regularizer):
    
    def __init__(self, squared=True):
        self.squared = squared
        
    def __call__(self, W):
        if(self.squared):
            return torch.mean(W**2 * (1 - torch.abs(W))**2) 
        else:
            W_abs = torch.abs(W)
            return torch.mean(torch.min(
                W_abs,
                torch.abs(1 - W_abs)
            ))
        
class NMU_Regularizer(Regularizer):
    
    def __init__(self, squared=True):
        self.squared = squared
        
    def __call__(self, W):
        if(self.squared):
            return torch.mean(W**2 * (1 - W)**2)
        else:
            return torch.mean(torch.min(
                torch.abs(W),
                torch.abs(1 - W)
            ))
        
class NumberActiveCoeffs(Regularizer):
    
    def __init__(self, n, squared=True, slack=0):
        self.n = n
        self.squared = squared
        self.slack = slack
    
    def __call__(self, W):
        if(self.squared):
            delta = (W.sum(axis=1) - self.n)**2 - self.slack**2
        else:
            delta = torch.abs(W.sum(axis=1) - self.n) - self.slack
        return torch.clamp_min(delta, 0).mean()

class MaximumActiveCoeffs(Regularizer):
    
    def __init__(self, max):
        self.max = max

    def __call__(self, W):
        return torch.clamp_min(W.sum(axis=1) - self.max, 0).mean()
    
def eval_regularizers(model, multipliers=None):
    regs = {}
    for param in model.parameters():
        if(type(param) is RegularizedParameter):
            params_regs = param.eval_regularizer()
            for reg_name in params_regs:
                regs.setdefault(reg_name, 0)
                regs[reg_name] += params_regs[reg_name]
    if(multipliers is None):
        return regs
    else:
        agg_reg = 0
        for reg_name in regs:
            agg_reg += regs[reg_name] * multipliers[reg_name]
        return agg_reg