from torch.utils.data import Dataset


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
