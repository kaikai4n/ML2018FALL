import numpy as np

class DataProcessor():
    def __init__(self):
        pass
    
    def _add_bias(self, x):
        ones = np.ones((x.shape[0], 1))
        return np.concatenate([ones, x], 1)

    def augment_features(self, train_x):
        train_x = self._add_bias(train_x)
        return train_x
