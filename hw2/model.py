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
        weight_y = y
        times = 1
        weight_y = weight_y * (times-1) + 1
        grad = -np.sum(np.multiply(np.multiply(y-pred, weight_y), x), axis=0)
        update_grad = self._optimizer.get_update_grad(grad)
        self._weights -= update_grad

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


class GaussianNaiveBayes():
    def __init__(self):
        # The probabilistic generative model
        # P(C1|X) = P(C1|X)P(C1) / (P(C1|X)P(C1) + P(C2|X)P(C2))
        self._have_parameters = False

    def fit(self, x, y, feature_num):
        # x = [data_num, feature_dim]
        # y = [data_num]
        if self._have_parameters:
            raise Exception('Check carefully \
                    if you try to fit a new model. Previous \
                    fitted model parameters will be replaced.')
        self._labels = np.sort(np.unique(y))
        self._dim = feature_num
        self._prior = []
        self._mus = []
        self._sigmas = []
        for one_label in self._labels:
            label_indexes = (y == one_label).squeeze()
            self._prior.append(np.sum(label_indexes)/y.shape[0])
            self._mus.append(np.mean(x[label_indexes], axis=0))
            self._sigmas.append(np.cov(x[label_indexes].T))
        for i, (prior, sigma) in enumerate(zip(self._prior, self._sigmas)):
            if i == 0:
                self._sigma = prior * sigma
            else:
                self._sigma += prior * sigma
        self._inv_sigma = np.linalg.pinv(self._sigma)
        self._have_parameters = True

    def predict(self, x):
        if self._have_parameters == False:
            raise Exception('Called predict before fit or load model.')
            exit()
        z = x.dot((self._mus[0] - self._mus[1]).T.dot(self._inv_sigma).T) - \
                (1/2) * self._mus[0].T.dot(self._inv_sigma).dot(self._mus[0]) +\
                (1/2) * self._mus[1].T.dot(self._inv_sigma).dot(self._mus[1]) +\
                np.log(self._prior[0]/self._prior[1])
        pred_prob = self._sigmoid(z)
        pred = self._prob_to_label(pred_prob)
        return pred

    def _sigmoid(self, x):
        # Prevent overflow
        x[x < -1e2] = -1e2
        x[x > 1e2] = 1e2
        return 1.0/(1.0+np.exp(-x))

    def _prob_to_label(self, prob):
        class1_indexes = (prob >= 0.5)
        class2_indexes = (prob < 0.5)
        label = prob
        label[class1_indexes] = self._labels[0]
        label[class2_indexes] = self._labels[1]
        return label

    def count_accuracy(self, pred, y):
        return np.sum(pred == y.squeeze()) / y.shape[0]
    
    def save_model(self, filename):
        weights = (self._labels, self._prior, \
                self._mus, self._sigma, self._inv_sigma)
        with open(filename, 'wb') as f:
            np.save(f, weights)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            weights = np.load(f)
        (self._labels, self._prior, self._mus, \
                self._sigma, self._inv_sigma) = weights 
        self._have_parameters = True
