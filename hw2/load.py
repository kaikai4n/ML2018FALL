import pandas

class DataLoader():
    def __init__(self):
        pass
    
    def read_training_data(self, train_x_filename, train_y_filename):
        print(train_x_filename, train_y_filename)
        train_x = self.read_csv(train_x_filename)
        train_x = self.to_numpy(train_x)
        train_y = self.to_numpy(self.read_csv(train_y_filename))
        return train_x, train_y

    def read_csv(self, filename):
        content = pandas.read_csv(filename, dtype=float)
        return content

    def filter_feature(self, content):
        keys = content.keys()
        content = content[keys]
        return content

    def to_numpy(self, content):
        return content.values

if __name__ == '__main__':
    content = read_csv('data/train_x.csv')
    content = filter_feature(content)
    content = to_numpy(content)
    print(content.shape)
