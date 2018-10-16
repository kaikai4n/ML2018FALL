import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation',
            action='store_true',
            default=False,
            help='To split validation or not.')
    parser.add_argument('--train_x_filename',
            default='data/train_x.csv',
            help='The csv file to train')
    parser.add_argument('--train_y_filename',
            default='data/train_y.csv',
            help='The csv file to train')
    parser.add_argument('--attributes_filename',
            default='models/attributes_PM2.5_PM10.npy',
            help='The filtered boolean numpy file,\
                    specified which attributes are used \
                    to train.')
    parser.add_argument('-e', '--epoches',
            type=int,
            default=2000)
    parser.add_argument('-lr', '--learning_rate',
            type=float,
            default=0.005)
    parser.add_argument('--save_intervals',
            default=100,
            type=int,
            help='The epoch intervals to save models')
    parser.add_argument('--prefix',
            required=True,
            help='The prefix of saving name')
    parser.add_argument('--params_init_model',
            default=None,
            type=str,
            help='The initialization parameters \
                    from a given model name.')
    parser.add_argument('--lambda_value',
            default=0.0,
            type=float,
            help='The regularization hyperparameter,\
                    default=0.0')
    parser.add_argument('--seed',
            default=7122,
            type=int,
            help='Random seed for numpy.')
    parser.add_argument('--batch_size',
            default=8,
            type=int,
            help='The batch size when training.')
    args = parser.parse_args()
    return args