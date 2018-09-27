import numpy as np

class LinearRegression():
    def __init__(self, 
            data=None, 
            train=True, 
            validation=False):
        if train and data is None:
            raise Exception('Training data not found error\n')
        self.data = data
        self.train = train
        self.validation = validation
        if train:
            self.total_attributes = self.data[0][0].shape[0]
            self._init_parameters(self.total_attributes)
            self._init_batcher()

    def _init_batcher(self):
        self.batcher = Batcher(self.data,
                validation=self.validation)

    def new_epoch(self):
        self.batcher.random_shuffle()

    def _init_parameters(self, total_attributes):
        A = np.random.normal(0.0, 1.0, total_attributes)
        B = np.random.normal(0.0, 1.0, 1)
        self.params = [A, B]

    def get_data(self):
        return self.batcher.get_data()

    def forward(self, x):
        return np.sum(self.params[0] * x) + self.params[1]

    def backward(self, grad, x, grad_clip=False):
        if grad_clip:
            threshold = 0.001
            grad = threshold if grad > threshold else grad
        self.params[0] = self.params[0] - grad * np.array(x)
        self.params[1] -= grad

    def get_validation_data(self):
        return self.batcher.get_validation_data()

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.params)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.params = np.load(f)

class Batcher():
    def __init__(self, data, validation=False):
        self.data = data
        self.total_data = self.data.shape[0]
        self._init_shuffle()
        if validation:
            self._cut_validation(validation_rate=0.9)
        else:
            self._cut_validation(validation_rate=1.0)

    def _cut_validation(self, validation_rate):
        train_num = int(self.total_data * validation_rate)
        self.permu_train = self.permutation[:train_num]
        self.permu_valid = self.permutation[train_num:]

    def _init_shuffle(self):
        self.permutation = np.random.permutation(self.total_data)

    def random_shuffle(self):
        # shuffle training data
        np.random.shuffle(self.permu_train)
    
    def get_data(self):
        data = [self.data[permu_num]
                for permu_num in self.permu_train]
        return data

    def get_validation_data(self):
        data = [self.data[permu_num]
                for permu_num in self.permu_valid]
        return data
