import pickle
import torch
import numpy as np
import os
    
def check_save_path(prefix, validation):
    # process saveing path
    # log file
    if os.path.isdir('logs') == False:
        os.mkdir('logs')
    log_save_path = os.path.join('logs', prefix + '.log')
    with open(log_save_path, 'w') as f_log:
        if validation:
            f_log.write('epoch,loss,validation\n')
        else:
            f_log.write('epoch,loss\n')
    # model path
    if os.path.isdir('models') == False:
        os.mkdir('models')
    model_path = os.path.join('models', prefix)
    if os.path.isdir(model_path) == False:
        os.mkdir(model_path)

    training_args_path = os.path.join(model_path, 'training_args.pkl')

    return log_save_path, model_path, training_args_path

def save_training_args(args, path):
    with open(path, 'wb') as f:
        pickle.dump(args, f)

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
