# Load model parameters to test
import model
import load
import pandas
import numpy as np

def load_trainer(model_path):
    trainer = model.LinearRegression()
    trainer.load_model(model_path)
    return trainer

def filter_attributes(data):
    with open('models/filtered_boolean_attributes.npy',
            'rb') as f:
        booleans = np.load(f)
    return data[:,booleans]

def clean_data(data):
    def _check_bound(num, l_bound, r_bound):
        return num >= u_bound or num <= l_bound
    data_bounds = get_data_bounds()
    total_testing_data = data.shape[0]
    data = data.reshape(-1, 18)
    data = filter_attributes(data)
    for attr_index in range(data.shape[1]):
        l_bound, u_bound, middle_mean = data_bounds[attr_index]
        for i in range(data.shape[0]):
            if _check_bound(data[i][attr_index], l_bound, u_bound):
                if i != 0:
                    data[i][attr_index] = data[i-1][attr_index] 
                else:
                    data[i][attr_index] = middle_mean
    return data.reshape(total_testing_data, -1)

def get_data_bounds():
    with open('models/data_bounds.npy', 'rb') as f:
        data_bounds = np.load(f)
    return data_bounds

if __name__ == '__main__':
    model_path = 'models/model_e2000.npy'
    trainer = load_trainer(model_path)

    test_path = 'data/test.csv'
    testing_data = load.load_test_csv(test_path)
    testing_data = clean_data(testing_data)
    
    output_path = 'ans.csv'
    outputs = [['id', 'value']]
    for i in range(testing_data.shape[0]):
        test_x = testing_data[i]
        prediction = trainer.forward(test_x)
        outputs.append(['id_%d' % i, prediction[0]])
    pandas.DataFrame(outputs).to_csv(output_path, 
            header=False, index=False)    
