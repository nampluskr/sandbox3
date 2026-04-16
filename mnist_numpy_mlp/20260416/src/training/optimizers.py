import numpy as np


class SGD:
    def __init__(self, model, lr):
        self.params = model.params
        self.grads = model.grads
        self.lr = lr

    def step(self):
        for param, grad in zip(self.params, self.grads):
            param -= self.lr * grad
            

class Adam:
    def __init__(self, model, lr, beta1=0.9, beta2=0.999):
        self.params = model.params
        self.grads = model.grads
        self.lr = lr

        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.ms = [np.zeros_like(param) for param in self.params]
        self.vs = [np.zeros_like(param) for param in self.params]

    def step(self):
        self.iter += 1
        for param, grad, m, v in zip(self.params, self.grads, self.ms, self.vs):
            m[...] = self.beta1 * m + (1 - self.beta1) * grad
            v[...] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            m_hat = m / (1.0 - self.beta1 ** self.iter)
            v_hat = v / (1.0 - self.beta2 ** self.iter)

            param[...] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
