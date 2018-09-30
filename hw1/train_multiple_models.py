import train
import args
import numpy as np
import os

def get_split_index(y, split_values):
    for i, thres in enumerate(split_values):
        if y < thres:
            return i - 1
    return len(split_values) - 1

def split_data(data, split_values, splitted_data):
    for one_data in data:
        one_index = get_split_index(one_data[1], split_values)
        splitted_data[one_index].append(one_data)

if __name__ == '__main__':
    args = args.get_args()
    np.random.seed(7122)
    data = train.preprocessing(
            args.train_filename, 
            args.attributes_filename)
    split_values = [2, 14, 22, 30, 40, 130]
    splitted_data = [list() for _ in range(5)]
    split_data(data, split_values, splitted_data)
    
    log_path = os.path.join('logs', args.prefix)
    model_path = os.path.join('models', args.prefix)
    train.check_dir(log_path)
    train.check_dir(model_path)
    for split_index, one_split_data in enumerate(splitted_data):
        train.train(data=np.array(one_split_data),
            validation=args.validation, 
            prefix=os.path.join(args.prefix,\
                    'split_%d' % split_index), 
            total_epoches=args.epoches, 
            learning_rate=args.learning_rate,
            save_intervals=args.save_intervals,
            params_init_model=args.params_init_model)
