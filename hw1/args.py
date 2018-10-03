import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation',
            action='store_true',
            default=False,
            help='To split validation or not.')
    parser.add_argument('--train_filename',
            default='data/train.csv',
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
            default=1e-3)
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
    args = parser.parse_args()
    return args
