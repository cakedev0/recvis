import torch
import numpy as np

def one_mean_prod_sample(size=1):
    return np.exp(-1) + np.exp(-np.random.uniform(0, 1, size=size))

class FunctionDataset(torch.utils.data.Dataset):
    
    def __init__(self, batch_size=128, sample_range=[-1, 1], seed=0, use_cuda=False):
        self.sample_range = sample_range
        self.seed = seed
        self.batch_size = batch_size
        self.use_cuda = use_cuda
    
    def _fork(self, *args, sample_range=None, seed=None, batch_size=None, **kwargs):
        if(sample_range is None):
            sample_range = self.sample_range
        if(seed is None):
            seed = self.seed
        if(batch_size is None):
            batch_size = self.batch_size
        return self.__class__(*args, sample_range=sample_range, seed=seed, batch_size=batch_size, use_cuda=self.use_cuda, **kwargs)
    
    def __len__(self):
        return self.batch_size
    
    def dataloader(self):
        return torch.utils.data.DataLoader(self, batch_size=self.batch_size)
    
    def get_input_size(self):
        return self.d
    
    def variance(self, n=10000, sample_range=None):
        if(sample_range is None):
            sample_range = self.sample_range
        _, t = next(iter(self.fork(sample_range=sample_range, batch_size=n).dataloader()))
        return t.cpu().numpy().var()
        

class LambdaDataset(FunctionDataset):
    
    def __init__(self, lambda_func, batch_size=128, sample_range=[-1, 1], seed=0, use_cuda=False):
        super().__init__(sample_range=sample_range, seed=seed, batch_size=batch_size, use_cuda=use_cuda)
        self.lambda_func = lambda_func
        self.d = lambda_func.__code__.co_argcount
    
    def fork(self, sample_range=None, seed=None, batch_size=None):
        return self._fork(self.lambda_func, sample_range=sample_range, seed=seed, batch_size=batch_size)
    
    def __getitem__(self, idx):
        v = np.random.uniform(*self.sample_range, size=self.d)
        v[np.random.uniform(size=v.shape) < 0.1] = 0
        t = self.lambda_func(*v)
        v, t = torch.tensor(v, dtype=torch.float), torch.tensor([t], dtype=torch.float)
        if(self.use_cuda):
            v, t = v.cuda(), t.cuda()
        return v, t
    
class MatrixDeterminantDataset(FunctionDataset):
    
    def __init__(self, matrix_structure, batch_size=128, sample_range=[-1, 1], seed=0, use_cuda=False):
        super().__init__(sample_range=sample_range, seed=seed, batch_size=batch_size, use_cuda=use_cuda)
        self.matrix_structure = matrix_structure
        self.d = np.max(matrix_structure)
        
    def __getitem__(self, idx):
        v = np.random.uniform(*self.sample_range, size=self.d)
        fill_vals = np.concatenate(([0], v))
        matrix = np.zeros_like(self.matrix_structure, dtype=float)
        matrix = fill_vals[self.matrix_structure]
        t = np.linalg.det(matrix)
        v, t = torch.tensor(v, dtype=torch.float), torch.tensor([t], dtype=torch.float)
        if(self.use_cuda):
            v, t = v.cuda(), t.cuda()
        return v, t

    def fork(self, sample_range=None, seed=None, batch_size=None):
        return self._fork(self.matrix_structure, sample_range=sample_range, seed=seed, batch_size=batch_size)


from stable_nalu.abstract import ExtendedTorchModule
from stable_nalu.layer import GeneralizedLayer

class EquationNetwork(ExtendedTorchModule):
    UNIT_NAMES = GeneralizedLayer.UNIT_NAMES
    LAYER_NAMES = {
        "+": "ReRegualizedLinearNAC",
        "x": "ReRegualizedLinearMNAC"
    }
    BASE_ALPHABET = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, layers=["x", "+"], hidden_sizes=[2], input_size=100, writer=None, **kwags):
        super().__init__('network', writer=writer, **kwags)
        self.input_size = input_size
        self.layers = []
        self.layer_types = layers
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        
        for layer_type, input_size, output_size in zip(self.layer_types, [input_size] + hidden_sizes, hidden_sizes + [1]):
            layer_name = self.LAYER_NAMES[layer_type]
            layer = GeneralizedLayer(input_size, output_size, layer_name,
                                        writer=self.writer, name=layer_type, **kwags)
            self.layers.append(layer)
            setattr(self, f'layer_{len(self.layers)}', self.layers[-1])
        
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def regualizer(self):
        return super().regualizer()

    def forward(self, input):
        zs = [input]
        for layer in self.layers:
            zs.append(layer(zs[-1]))
        
        return zs[-1]
    
    def extra_repr(self):
        return f'input_size={self.input_size}, [{", ".join(self.layer_types)}]'
    
    def find_equation(self, alphabet=None):
        for layer, layer_type in zip(self.layers, self.layer_types):
            W = layer.layer.W.detach().cpu().numpy()
            W = np.round(W).astype(int)
            if(layer_type == "+"):
                alphabet = self._NAU_layer_equation(W, alphabet)
            else:
                alphabet = self._NMU_layer_equation(W, alphabet)
        eq = alphabet[0]
        return eq
    
    def _NAU_layer_equation(self, W, alphabet=None):
        if(alphabet is None):
            alphabet = self.BASE_ALPHABET
        terms = []
        for i in range(W.shape[0]):
            term = ""
            for j in range(W.shape[1]):
                if(W[i, j] == 1):
                    term += f'+{alphabet[j]}'
                if(W[i, j] == -1):
                    term += f'-{alphabet[j]}'
            if(len(term) > 0):
                if(term[0] == '+'):
                    term = term[1:]
                if(len(term) > 1):
                    term = '(' + term + ')'
            else:
                term = '0'
            terms.append(term)
        return terms

    def _NMU_layer_equation(self, W, alphabet=None):
        if(alphabet is None):
            alphabet = self.BASE_ALPHABET
        
        terms = []
        for i in range(W.shape[0]):
            term = ""
            for j in range(W.shape[1]):
                if(W[i, j] == 1):
                    term += f'*{alphabet[j]}'
            if(len(term) > 0):
                term = term[1:]
            else:
                term = '1'
            terms.append(term)

        return terms