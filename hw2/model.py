import numpy as np

class LogisticRegression():
    def __init__(self, feature_num, optimizer=None, train=True):
        self.feature_num = feature_num
        if train and optimizer is None:
            raise Exception('Optimizer is required when training.')
        self._optimizer = optimizer
        self._init_parameters()

    def _init_parameters(self):
        self._weights = np.random.normal(0.0, 1.0, self.feature_num)

    def _sigmoid(self, x):
        # Prevent overflow
        x[x < -1e2] = -1e2
        x[x > 1e2] = 1e2
        return 1.0/(1.0+np.exp(-x))

    def forward(self, x):
        output = x.dot(self._weights)
        output = self._sigmoid(output)
        return output.reshape(-1, 1)

    def backward(self, x, y, pred):
        grad = -np.sum(np.multiply(y-pred, x), axis=0)
        eta = self._optimizer.get_eta(grad)
        self._weights -= eta * grad

    def count_loss(self, x, y):
        x = x.squeeze()
        x[x < 1e-8] = 1e-8
        x[x > 1 - 1e-8] = 1 - 1e-8
        y = y.squeeze()
        return -((y * np.log(x)) + \
                (np.ones(y.shape)-y) * (np.log(np.ones(x.shape)-x)))
    
    def count_accuracy(self, pred, y):
        pred[pred >= 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        return np.mean(pred.squeeze() == y.squeeze())

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self._weights)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self._weights = np.load(f)
