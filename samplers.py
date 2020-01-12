import numpy as np

def uniform(a=0, b=1):
    return lambda size : np.random.uniform(a, b, size=size)

def mixin(*samplers):
    ns = len(samplers)
    def f(size):
        n, d = size
        samples = np.zeros((n, d))
        ps = n // ns
        for i, sampler in enumerate(samplers):
            b = (i+1)*ps if i < ns - 1 else n
            samples[i*ps:b, :] = sampler((b-i*ps, d))
        return samples
    return f

def batch_mixin(*samplers):
    ns = len(samplers)
    def f(size):
        i = np.random.choice(ns)
        return samplers[i](size)
    return f

def one_mean_prod_sample(size=1):
    return np.exp(-1) + np.exp(-np.random.uniform(0, 1, size=size))

def random_sign(sampler):
    def f(size=None):
        s = np.random.randint(0, 2, size=size)*2 - 1
        return s*sampler(size)
    return f