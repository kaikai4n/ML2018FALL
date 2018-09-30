# Load model parameters to test
import model
import load
import pandas
import numpy as np
import argparse
import train_multiple_models as train_mm
import os

def load_trainer(model_path):
    trainer = model.LinearRegression(train=False)
    trainer.load_model(model_path)
    return trainer

def filter_attributes(data, filename):
    with open(filename, 'rb') as f:
        booleans = np.load(f)
    return data[:,booleans]

def clean_data(data, attributes_filename, data_bounds_filename):
    def _check_bound(num, l_bound, r_bound):
        return num >= u_bound or num <= l_bound
    data_bounds = get_data_bounds(data_bounds_filename)
    total_testing_data = data.shape[0]
    data = data.reshape(-1, 18)
    data = filter_attributes(data, attributes_filename)
    for attr_index in range(data.shape[1]):
        l_bound, u_bound, middle_mean = data_bounds[attr_index]
        for i in range(data.shape[0]):
            if _check_bound(data[i][attr_index], l_bound, u_bound):
                if i != 0:
                    data[i][attr_index] = data[i-1][attr_index] 
                else:
                    data[i][attr_index] = middle_mean
    return data.reshape(total_testing_data, -1)

def get_data_bounds(filename):
    with open(filename, 'rb') as f:
        data_bounds = np.load(f)
    return data_bounds

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_main',
            required=True,
            help='The training model parameters path,\
                    this is the main one.')
    parser.add_argument('--model_minor',
            required=True,
            help='The training model parameters path,\
                    this is the minor one,\
                    a total of five small models.')
    parser.add_argument('-t','--testing_filename',
            default='data/test.csv',
            help='The testing.csv file path')
    parser.add_argument('-o','--output',
            default='ans.csv',
            help='The output testing prediction filename')
    parser.add_argument('--attributes_filename',
            default='models/attributes_PM2.5_PM10.npy',
            help='The attributes used boolean file')
    parser.add_argument('--data_bounds_filename',
            default='models/data_bounds.npy',
            help='The data bounds used in training,\
                    required loaded to filter out\
                    possible invalid data.')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main_model_path = args.model_main
    trainer = load_trainer(main_model_path)
    
    small_trainer = []
    for i in range(5):
        small_trainer.append(load_trainer(
            os.path.join(args.model_minor, \
                    'split_%d'%i, 'model_e4000.npy')))
    split_values = [2, 14, 22, 30, 40, 130]

    test_path = args.testing_filename
    testing_data = load.load_test_csv(test_path)
    testing_data = clean_data(testing_data,
            args.attributes_filename,
            args.data_bounds_filename)
    
    output_path = args.output
    outputs = [['id', 'value']]
    for i in range(testing_data.shape[0]):
        test_x = testing_data[i]
        prediction = trainer.forward(test_x)
        model_index = train_mm.get_split_index(prediction, split_values)
        final_prediction = small_trainer[model_index].forward(test_x)
        final_prediction = np.mean([prediction, final_prediction])
        outputs.append(['id_%d' % i, final_prediction])
    pandas.DataFrame(outputs).to_csv(output_path, 
            header=False, index=False)    
