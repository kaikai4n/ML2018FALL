import pandas
import numpy as np

def read_csv(filename):
    return pandas.read_csv(filename, encoding='big5')

def parse_csv(data):
    for key_name in data.keys()[3:]:
        one_data = data[key_name]
        for i in range(len(one_data)):
            try:
                one_data[i] = float(one_data[i])
            except:
                one_data[i] = 0
    return data

def csv_to_np(data):
    processed_training_x = list()
    train_x = np.array([data[key_name] for key_name in data.keys()[3:]])
    train_x = train_x.transpose().reshape(-1, 18, train_x.shape[0])
    train_x = train_x.transpose(0,2,1).reshape(-1, 18)
    
    return train_x

