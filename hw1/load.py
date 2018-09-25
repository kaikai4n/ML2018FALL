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
                one_data[i] = 10e-1
    return data

def csv_to_np(data):
    processed_training_x = list()
    train_x = np.array([data[key_name] for key_name in data.keys()[3:]])
    train_x = train_x.transpose().reshape(-1, 18, 24)
    train_x = train_x.transpose(0,2,1).reshape(-1, 18)
    
    train_y = train_x[:,9]
    return train_x, train_y

def load_test_csv(filename):
    with open(filename, 'r') as f:
        data = list(filter(None, f.read().split('\n')))
    data = [[[ele for ele in line.split(',')[2:]]
        for line in data[i*18:i*18+18]] 
        for i in range(int(len(data)/18))]
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                try:
                    data[i][j][k] = float(data[i][j][k])
                except:
                    data[i][j][k] = 0.0
    return np.array(data).transpose(0,2,1).reshape(-1, 18*9)

