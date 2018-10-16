import numpy as np

class AdaGrad():
    def __init__(self, learning_rate):
        self._lr = learning_rate
        self._grad_sq_sum = 0.0

    def get_eta(self, grad):
        square = np.power(np.linalg.norm(grad), 2)
        self._grad_sq_sum += square
        return self._lr / np.power(self._grad_sq_sum + 1e-6, 0.5)
