import numpy as np
from .functions import sigmoid


class Module:
    def __init__(self):
        self.params = []
        self.grads = []

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = np.random.randn(in_features, out_features)
        self.b = np.zeros(out_features)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

        self.params.extend([self.w, self.b])
        self.grads.extend([self.grad_w, self.grad_b])
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dout):
        self.grad_w[...] = np.dot(self.x.T, dout)
        self.grad_b[...] = np.sum(dout, axis=0)
        return np.dot(dout, self.w.T)


class Sigmoid(Module):
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)


class ReLU(Module):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask
    

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

        for layer in self.layers:
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
