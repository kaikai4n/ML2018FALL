from torch.utils.data import Dataset
import numpy as np

class FaceDataset(Dataset):
    def __init__(self, x, y, data_length):
        super(FaceDataset, self).__init__()
        self.x = x
        self.y = y
        self.data_length = data_length

    def __len__(self):
        return self.data_length

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class DataProcessor():
    def __init__(self, x, y, data_length):
        self._x = x
        self._y = y
        self._data_length = data_length
        self._validation = False

    def normalize(self):
        self._x = self._x / 255

    def get_data(self):
        if self._validation:
            return self._x[self._permut][:self._train_num],\
                    self._y[self._permut][:self._train_num],\
                    self._x[self._permut][self._train_num:],\
                    self._y[self._permut][self._train_num:]
        else:
            return self._x, self._y, None, None

    def cut_validation(self):
        self._validation = True
        self._train_num = int(self._data_length * 0.9)
        self._permut = np.random.permutation(self._data_length)
