import numpy as np
import torch

class BatchDataLoader:
    
    def __init__(self, dataset, batch_size, samplers, n_var=10**4, use_cuda=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samplers = samplers
        self.d = dataset.d
        self.use_cuda = use_cuda
        self.variances = []
        for sampler in samplers:
            v = sampler((n_var, self.d))
            _, dets = dataset.batch(v)
            self.variances.append(dets.var())
        
    def __iter__(self):
        while(True):
            sampler_idx = np.random.choice(len(self.samplers))
            sampler = self.samplers[sampler_idx]
            v = sampler((self.batch_size, self.d))
            v, t = self.dataset.batch(v)
            v, t = torch.tensor(v, dtype=torch.float), torch.tensor(t, dtype=torch.float)
            if(self.use_cuda):
                v, t = v.cuda(), t.cuda()
            yield self.variances[sampler_idx], v, t

class MatrixDeterminantDataset:
    
    def __init__(self, 
                 matrix_structure,
                 with_multiplicity=False,
                 matrix_form=False,
                 alphabet=None):
        assert not(with_multiplicity and matrix_form)
        
        self.matrix_structure = matrix_structure
        self.d = np.max(np.abs(matrix_structure))
        self.n_coefs = np.sum(matrix_structure != 0)
        
        self.with_multiplicity = with_multiplicity
        self.matrix_form = matrix_form
        
        self.alphabet = alphabet
        if(self.alphabet is None):
            self.alphabet = ["a_{%d}" % i for i in range(1, self.d + 1)]
        if(type(self.alphabet) is str):
            self.alphabet = list(self.alphabet)
        
    def get_input_size(self):
        if(self.matrix_form):
            return self.matrix_structure.shape
        elif(self.with_multiplicity):
            return self.n_coefs
        else:
            return self.d
        
    def batch(self, v):
        ms = self.matrix_structure
        n = v.shape[0]
        fill_vals = np.hstack((np.zeros((n, 1)), v))
        A = np.zeros((n, *ms.shape))
        for i in range(n):
            A[i, :, :] = fill_vals[i][np.maximum(0, ms)]
            A[i, :, :] -= fill_vals[i][np.maximum(0, -ms)]
        t = np.linalg.det(A)
        
        if(self.matrix_form):
            v = A
        elif(self.with_multiplicity):
            v = v[:, np.abs(ms[ms != 0]) - 1]
        
        t = t.reshape((-1, 1))
        
        return v, t

    def dataloader(self, batch_size, samplers, n_var=10**4, use_cuda=True):
        return BatchDataLoader(self, batch_size, samplers, n_var, use_cuda)
    
    def adapt_alphabet(self):
        if(self.with_multiplicity):
            adapted = []
            idxs = np.abs(self.matrix_structure[self.matrix_structure != 0]) - 1
            for i in idxs:
                adapted.append(self.alphabet[i])
            return adapted
        else:
            return self.alphabet
    
    def _repr_html_(self):
        name = "MatrixDeterminantDataset for matrices of form : <br>"
        symbols = ["0"] + self.alphabet
        matrix = [ [("" if self.matrix_structure[i, j] >= 0 else "-") +symbols[np.abs(self.matrix_structure[i, j])]\
                    for j in range(self.matrix_structure.shape[1])] \
                  for i in range(self.matrix_structure.shape[0])]
        matrix = [" & ".join(line) for line in matrix]
        matrix = " \\\\ ".join(matrix)
        return name + "$ \\begin{pmatrix} " + matrix + " \\end{pmatrix}$ "