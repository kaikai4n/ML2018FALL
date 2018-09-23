import numpy as np

class LinearRegression():
    def __init__(self, x):
        self.x = x
        self._init_parameters()
        self.new_epoch()
    
    def new_epoch(self):
        self.batcher = Batcher(self.x, 9, 1)

    def _init_parameters(self):
        A = np.random.normal(0.0, 1.0, 18*9)
        B = np.random.normal(0.0, 1.0, 1)
        self.params = [A, B]

    def get_batch(self):
        return self.batcher.get_batch()

    def forward(self, x):
        return np.sum(self.params[0] * x) + self.params[1]

    def backward(self, grad, x, grad_clip=False):
        if grad_clip:
            threshold = 0.001
            grad = threshold if grad > threshold else grad
        self.params[0] = self.params[0] - grad * np.array(x)
        self.params[1] -= grad
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.params)

class Batcher():
    def __init__(self, x, x_num, y_num):
        self.x = x
        self.total_epoches = x.shape[0] - 9
        self.permutation = np.random.permutation(self.total_epoches)
    def get_batch(self):
        return [(self.x[permu_num:permu_num+9].reshape(-1),
                self.x[permu_num+9][9])
                for permu_num in self.permutation]
