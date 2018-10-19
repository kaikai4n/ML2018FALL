import numpy as np

class DataProcessor():
    def __init__(self):
        pass
    
    def _add_bias(self, x):
        ones = np.ones((x.shape[0], 1))
        return np.concatenate([ones, x], 1)

    def _normalize(self, x):
        for i in range(15):
            mean, std = np.mean(x[i]), np.std(x[i])
            x[i] = (x[i] - mean)/std
        return x

    def _augment(self, x):
        for i in range(15):
            for exp_num in range (20):
                aug_data = np.power(x[:, i], exp_num)
                x = np.concatenate([x, aug_data.reshape(-1, 1)], axis=1)
        return x

    def augment_features(self, train_x):
        train_x = self._add_bias(train_x)
        #train_x = self._normalize(train_x)
        #train_x = self._augment(train_x)
        return train_x
