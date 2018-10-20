import pandas
import numpy as np
import data

class DataLoader():
    def __init__(self):
        self.data_processor = data.DataProcessor()


    def read_data(self, train_x_filename, train_y_filename, test_filename):
        train_x = self.read_csv(train_x_filename)
        train_y = self.to_numpy(self.read_csv(train_y_filename))
        test_x = self.read_csv(test_filename)
        all_x = train_x.append(test_x, ignore_index=True)
        all_x = self.data_processing(all_x)
        all_x = self.to_numpy(all_x)
        all_x = self.data_processor.augment_features(all_x)
        train_x, test_x = all_x[:20000], all_x[20000:]
        return train_x, train_y, test_x

    def read_training_data(self, train_x_filename, train_y_filename, test_filename):
        train_x , train_y , _ = \
                self.read_data(train_x_filename, train_y_filename, test_filename)
        return train_x, train_y

    def read_testing_data(self, train_x_filename, train_y_filename, test_filename):
        _ , _ , test_x = self.read_data(train_x_filename, train_y_filename, test_filename)
        return test_x
    
    def data_processing(self, x):
        #keys = ['EDUCATION']
        x = self.filter_feature(x)
        x = self.to_one_hot(x)
        return x

    def read_csv(self, filename):
        content = pandas.read_csv(filename, dtype=float)
        return content

    def filter_feature(self, content, keys=None):
        if keys is None:
            # All keys are needed
            return content
        else:
            content = content[keys]
            return content

    def to_one_hot(self, content):
        keys = ['SEX', 'EDUCATION', 'MARRIAGE', \
                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        for one_key in keys:
            if one_key in content.keys():
                this_content = content[one_key]
                this_one_hot = pandas.get_dummies(this_content, prefix=one_key)
                content = content.drop(columns=one_key)
                content = pandas.concat([content, this_one_hot], axis=1)
        return content

    def to_numpy(self, content):
        return content.values

if __name__ == '__main__':
    content = read_csv('data/train_x.csv')
    content = filter_feature(content)
    content = to_numpy(content)
    print(content.shape)
