import load
import model
import numpy as np

def clean_data(data):
    def _find_bound(np_array, l_num, u_num):
        sorted_np_array = np.sort(np_array)
        return sorted_np_array[l_num],\
            sorted_np_array[u_num],\
            np.mean(sorted_np_array[l_num+1:u_num])
    def _check_bound(num, l_bound, r_bound):
        return num >= u_bound or num <= l_bound

    filtered_l_num = int(data.shape[0] * 0.01)
    filtered_u_num = data.shape[0] - filtered_l_num
    data_bounds = []
    for attr_index in range(data.shape[1]):
        l_bound, u_bound, middle_mean = _find_bound(
                data[:,attr_index], filtered_l_num, filtered_u_num)
        data_bounds.append((l_bound, u_bound, middle_mean))
        for i in range(data.shape[0]):
            if _check_bound(data[i][attr_index], l_bound, u_bound):
                if i != 0:
                    data[i][attr_index] = data[i-1][attr_index] 
                else:
                    data[i][attr_index] = middle_mean
    with open('models/data_bounds.npy', 'wb') as f:
        np.save(f, data_bounds)
    return data

def filter_attributes(data):
    with open('models/filtered_boolean_attributes.npy',
            'rb') as f:
        booleans = np.load(f)
    return data[:,booleans]

def preprocessing():
    data = load.read_csv('data/train.csv')
    data = load.parse_csv(data)
    train_x, train_y = load.csv_to_np(data)
    train_x = filter_attributes(train_x)
    train_x = clean_data(train_x)
    return train_x, train_y

def validate(trainer):
    total_loss = 0.0
    batches = trainer.get_validation_batch()
    if len(batches) == 0:
        return 0.0
    for x, y in batches:
        prediction = trainer.forward(x) 
        total_loss += np.power(prediction - y, 2)
    return total_loss/len(batches)
    

def train(train_x, train_y):
    trainer = model.LinearRegression(
            x=train_x, y=train_y, validation=False)
    
    total_epoches = 2000
    learning_rate = 1e-3
    adagrad_n = 0
    for epoch in range(total_epoches):
        trainer.new_epoch()
        total_loss = 0.0
        batches = trainer.get_batch()
        for step, (x, y) in enumerate(batches):
            prediction = trainer.forward(x) 
            total_loss += np.power(prediction - y, 2)
            adagrad_n += np.power(prediction - y, 2)
            grad = learning_rate * (prediction-y) / np.power(adagrad_n+1e-6, 0.5)
            trainer.backward(grad, x, grad_clip=False)
        total_loss /= len(batches)
        valid = validate(trainer)
        print('epoch:%d, total loss:%.3f, validation:%.3f' 
                % (epoch, total_loss, valid))
        if (epoch+1) % 100 == 0:
            trainer.save_model('models/model_e%d.npy' % (epoch+1))

if __name__ == '__main__':
    np.random.seed(7122)
    train_x, train_y = preprocessing()
    train(train_x, train_y)
