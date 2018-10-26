import numpy as np

class DataProcessor():
    def __init__(self):
        pass
    
    def _add_bias(self, x):
        ones = np.ones((x.shape[0], 1))
        return np.concatenate([ones, x], 1)

    def _normalize(self, x):
        normalize_list = [i for i in range(1, 15)]
        normalize_list += [i for i in range(92, x.shape[1])]
        for i in normalize_list:
            mean, std = np.mean(x[:,i]), np.std(x[:,i])
            x[:,i] = (x[:,i] - mean)/std
        return x

    def _augment(self, x):
        for i in range(1, 15):
            for exp_num in range (2, 7):
                aug_data = np.power(x[:, i], exp_num)
                x = np.concatenate([x, aug_data.reshape(-1, 1)], axis=1)
        return x

    def _take_log(self, x):
        take_log_dim = [dim for dim in range(1, 15) if dim != 2]
        e = np.exp(1)
        for dim in take_log_dim:
            x[:,dim][x[:,dim] > e] = \
                np.log(x[:,dim][x[:,dim] > e]) + e - 1
        return x

    def augment_features(self, train_x):
        train_x = self._add_bias(train_x)
        #train_x = self._take_log(train_x)
        #train_x = self._augment(train_x)
        train_x = self._normalize(train_x)
        return train_x

    def cut_validation(self, train_x, train_y, proportion=0.9):
        total = train_x.shape[0]
        train_num = int(total * proportion)
        return train_x[:train_num], train_y[:train_num], \
                train_x[train_num:], train_y[train_num:]
