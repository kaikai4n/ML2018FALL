from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class FaceDataset(Dataset):
    def __init__(self, x, y, data_length, transform=None):
        super(FaceDataset, self).__init__()
        self.x = x
        self.y = y
        self.data_length = data_length
        self.counter = [0 for _ in range(data_length)]
        self.previous_x = [[] for _ in range(data_length)]
        if transform is None:
            self.transform = transforms.Compose([
                    #transforms.RandomResizedCrop(48),
                    transforms.ColorJitter(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(
                            degrees=(-30, 30),
                            translate=(0.1, 0.1),
                            scale=(0.8, 1.2),
                            shear=(-10, 10),
                        ),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return self.data_length

    def __getitem__(self, i):
        if self.transform is not None:
            if self.counter[i] % 10 == 0:
                self.counter[i] += 1
                img = Image.fromarray(self.x[i].squeeze())
                img = self.transform(img)
                ret_x = img.reshape(1, 48, 48)
                self.previous_x[i] = ret_x
            else:
                ret_x = self.previous_x[i]
            return ret_x, self.y[i]
        else:
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
