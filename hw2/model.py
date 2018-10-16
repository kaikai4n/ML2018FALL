import numpy as np

class LogisticRegression():
    def __init__(self, train_x, train_y, feature_num, optimizer):
        self.x = train_x
        self.y = train_y
        self.feature_num = feature_num
        self._optimizer = optimizer
        if self.x.shape[1] != feature_num:
            raise Exception('Feature number doesn\'t match input traning data.')
        self._init_parameters()

    def _init_parameters(self):
        self._weights = np.random.normal(0.0, 1.0, self.feature_num)

    def _sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

    def forward(self, x):
        output = x.dot(self._weights)
        output = self._sigmoid(output)
        return output

    def backward(self, x, y, pred):
        grad = -np.sum(y-pred) * x
        grad = np.sum(grad, 0)
        eta = self._optimizer.get_eta(grad)
        self._weights -= eta * grad

    def count_loss(self, x, y):
        x = x.squeeze()
        x[x < 1e-6] = 1e-6
        x[x > 1 - 1e-6] = 1 - 1e-6
        y = y.squeeze()
        return -((y * np.log(x)) + \
                (np.ones(y.shape)-y) * (np.log(np.ones(x.shape)-x)))
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.params)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.params = np.load(f)
