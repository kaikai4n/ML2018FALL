import argparse
import model

def get_args(train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_filename',
            default='data/train.csv',
            help='The csv file to train')
    parser.add_argument('--test_filename',
            default='data/test.csv',
            help='The csv file to test.')
    parser.add_argument('--model',
            default='SimpleCNN',
            help='The model user would like to use.')
    parser.add_argument('--use_cuda',
            default=True,
            action='store_false',
            help='Use cuda or not.')
    parser.add_argument('--seed',
            default=7122,
            type=int,
            help='Random seed for numpy.')
    if train:
        parser.add_argument('--validation',
                action='store_true',
                default=False,
                help='To split validation or not.')
        parser.add_argument('-e', '--epoches',
                type=int,
                default=2000)
        parser.add_argument('-lr', '--learning_rate',
                type=float,
                default=0.001)
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
        parser.add_argument('-b', '--batch_size',
                default=128,
                type=int,
                help='The batch size when training.')
        parser.add_argument('--optimizer',
                default='Adam',
                help='The optimizer name in optimizer.py.')
    else:
        parser.add_argument('--output',
                default='ans.csv',
                help='When inferencing, the designated output\
                        filename.')
        parser.add_argument('--model_filename',
                required=True,
                help='When inferencing, the designated model.')
    args = parser.parse_args()
    try:
        model_object = getattr(model, args.model)
    except AttributeError:
        print('Model not found.')
        exit()

    return args
