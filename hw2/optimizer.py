import numpy as np

class AdaGrad():
    def __init__(self, learning_rate):
        self._lr = learning_rate
        self._grad_sq_sum = 0.0

    def get_eta(self, grad):
        square = np.power(np.linalg.norm(grad), 2)
        self._grad_sq_sum += square
        return self._lr / np.power(self._grad_sq_sum + 1e-8, 0.5)

class Adam():
    def __init__(self, learning_rate, beta_1=0.9, beta_2=0.99):
        self._lr = learning_rate
        self._m = 0.0
        self._v = 0.0
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._time = 0

    def get_eta(self, grad):
        self._time += 1
        self._m = self._beta_1 * self._m + (1-self._beta_1) * grad
        self._v = self._beta_2 * self._v + (1-self._beta_2) * grad * grad
        m = self._m / (1 - np.power(self._beta_1, self._time))
        v = self._v / (1 - np.power(self._beta_2, self._time))
        eta = self._lr * m / (np.power(v, 0.5) + 1e-8)
        return eta

