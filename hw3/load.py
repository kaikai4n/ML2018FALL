import pandas as pd
import numpy as np

def load_training_data(filename):
    df_training = pd.read_csv(filename)
    y = df_training['label'].values
    x = df_training['feature'].tolist()
    x = [[int(value) for value in ele.split()] for ele in x]
    x = np.array(x)
    return x, y

def load_testing_data(filename):
    df_training = pd.read_csv(filename)
    x_id = df_training['id'].values
    x = df_training['feature'].tolist()
    x = [[int(value) for value in ele.split()] for ele in x]
    x = np.array(x)
    return x_id, x


if __name__ == '__main__':
    #x, y = load_training_data('data/train.csv')
    x_id, test_x = load_testing_data('data/test.csv')
    print(x_id.shape, test_x.shape)
    print(x_id[:10])
