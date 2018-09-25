import numpy as np

class LinearRegression():
    def __init__(self, x=None, y=None, validation=False):
        self.x = x
        self.y = y
        self.total_attributes = x.shape[1]
        self._init_parameters(self.total_attributes)
        self.validation = validation
        if x is not None:
            self._init_epoch()
    
    def _init_epoch(self):
        self.batcher = Batcher(self.x, self.y,
                9, 1, validation=self.validation)

    def new_epoch(self):
        self.batcher.random_shuffle()

    def _init_parameters(self, total_attributes):
        A = np.random.normal(0.0, 1.0, total_attributes*9)
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

    def get_validation_batch(self):
        return self.batcher.get_validation_batch()

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.params)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.params = np.load(f)

class Batcher():
    def __init__(self, x, y, x_num, y_num, validation=False):
        self.x = x
        self.y = y
        self.total_epoches = int((x.shape[0]/12 - 9) * 12)
        self._init_shuffle()
        if validation:
            self._cut_validation(validation_rate=0.9)
        else:
            self._cut_validation(validation_rate=1.0)

    def _cut_validation(self, validation_rate):
        total_data = self.total_epoches
        train_num = int(total_data * validation_rate)
        #valid_num = total_data - train_num
        self.permu_train_x = self.permutation[:train_num]
        self.permu_valid_x = self.permutation[train_num:]

    def _init_shuffle(self):
        permutation = np.random.permutation(self.x.shape[0])
        self.permutation = np.array([permu_num 
            for permu_num in permutation
            if (permu_num+9) % 480 >= 9])

    def random_shuffle(self):
        # shuffle training data
        np.random.shuffle(self.permu_train_x)
    
    def get_batch(self):
        data = [(self.x[permu_num:permu_num+9].reshape(-1),
                self.y[permu_num+9])
                for permu_num in self.permu_train_x]
                # data augmentation from x/1 ~ x/20 -> 480 days
        return data

    def get_validation_batch(self):
        data = [(self.x[permu_num:permu_num+9].reshape(-1),
                self.y[permu_num+9])
                for permu_num in self.permu_valid_x]
                # data augmentation from x/1 ~ x/20 -> 480 days
        return data
