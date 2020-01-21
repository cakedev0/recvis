import torch
from torch import nn
import numpy as np
import regularization as R

class NMU(nn.Module):
    
    def __init__(self, n_in, n_out, squared=True, extra_regs={}):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        regs = dict(sparsity=R.NMU_Regularizer(squared))
        regs = {**regs, **extra_regs}
        self.W = R.RegularizedParameter(torch.Tensor(n_out, n_in), reg=regs)
        self.reset_parameters()
        
    def reset_parameters(self):
        std = np.sqrt(0.25)
        r = min(0.25, np.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, 0.5 - r, 0.5 + r)
        
    def forward(self, x):
        W = torch.clamp(self.W, 0, 1).unsqueeze(1)
        return torch.prod(W * x + 1 - W, 2).t()
    
    def __repr__(self):
        return f"NMU({self.n_in}, {self.n_out})"
    
    def transform_terms(self, terms):
        W = np.round(self.W.data.cpu().numpy())
        new_terms = []
        for i in range(W.shape[0]):
            term = ""
            for j in range(W.shape[1]):
                if(W[i, j] == 1):
                    term += f'*{terms[j]}'
            if(len(term) > 0):
                term = term[1:]
            else:
                term = '1'
            new_terms.append(term)

        return new_terms
        
        
class NAU(nn.Module):
    
    def __init__(self, n_in, n_out, squared=True, extra_regs={}):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        regs = dict(sparsity=R.NAU_Regularizer(squared))
        regs = {**regs, **extra_regs}
        self.W = R.RegularizedParameter(torch.Tensor(n_out, n_in), reg=regs)
        self.reset_parameters()
        
    def reset_parameters(self):
        std = np.sqrt(2.0 / (self.n_in + self.n_out))
        r = min(0.5, np.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, -r, r)
        
    def forward(self, x):
        W = torch.clamp(self.W, -1.0, 1.0)
        return torch.nn.functional.linear(x, W)
    
    def __repr__(self):
        return f"NAU({self.n_in}, {self.n_out})"

    def transform_terms(self, terms):
        W = np.round(self.W.data.cpu().numpy())
        new_terms = []
        for i in range(W.shape[0]):
            term = ""
            for j in range(W.shape[1]):
                if(W[i, j] == 1):
                    term += f'+{terms[j]}'
                if(W[i, j] == -1):
                    term += f'-{terms[j]}'
            if(len(term) > 0):
                if(term[0] == '+'):
                    term = term[1:]
                if(len(term) > 1):
                    term = '(' + term + ')'
            else:
                term = '0'
            new_terms.append(term)
        return new_terms

from IPython.display import Markdown

class LeibnizModule(nn.Module):
    
    def __init__(self, input_size, n_hidden, squared=True, prod_sizes=None):
        super().__init__()
        extra_regs = {}
        if(prod_sizes is not None):
            extra_regs = {"n_coeffs": R.MaximumActiveCoeffs(prod_sizes + 0.5)}
        self.module = nn.Sequential(
            NMU(input_size, n_hidden, squared=squared, extra_regs=extra_regs),
            NAU(n_hidden, 1, squared=squared)
        )
    
    def forward(self, x):
        return self.module(x)
    
    def disp_equation(self, alphabet):
        terms = self.module[0].transform_terms(alphabet)
        terms = self.module[1].transform_terms(terms)
        equation = terms[0][1:-1]
        return Markdown("$" + equation.replace("*", "") + "$")