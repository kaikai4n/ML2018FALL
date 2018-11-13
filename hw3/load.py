import pandas as pd
import numpy as np


class DataLoader():
    def __init__(self):
        pass

    def load_training_data(self, filename):
        df_training = pd.read_csv(filename)
        y = df_training['label'].values
        x = df_training['feature'].tolist()
        x = [[int(value) for value in ele.split()] for ele in x]
        x = np.array(x, dtype=np.float32).reshape(-1, 1, 48, 48)
        return x, y

    def load_testing_data(self, filename):
        df_training = pd.read_csv(filename)
        x_id = df_training['id'].values
        x = df_training['feature'].tolist()
        x = [[int(value) for value in ele.split()] for ele in x]
        x = np.array(x, dtype=np.float32).reshape(-1, 1, 48, 48)
        return x


if __name__ == '__main__':
    #x, y = load_training_data('data/train.csv')
    x_id, test_x = load_testing_data('data/test.csv')
    print(x_id.shape, test_x.shape)
    print(x_id[:10])
