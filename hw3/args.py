import argparse
import model

def check_model(model_name):
    try:
        model_object = getattr(model, model_name)
    except AttributeError:
        print('Model not found:', model_name)
        exit()

def get_args(train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_filename',
            default='data/train.csv',
            help='The csv file to train')
    parser.add_argument('--test_filename',
            default='data/test.csv',
            help='The csv file to test.')
    parser.add_argument('--model',
            default=None,
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
                default=3000)
        parser.add_argument('-lr', '--learning_rate',
                type=float,
                default=0.0005)
        parser.add_argument('--save_intervals',
                default=500,
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
        parser.add_argument('--ensemble',
                default=False,
                action='store_true',
                help='To ensemble models or not, if True, \
                        type in multiple model names and \
                        model filenames at \
                        models and model_filenames.')
        parser.add_argument('--models',
                nargs='*',
                help='When ensemble is True, this argument\
                        is required.')
        parser.add_argument('--model_filenames',
                nargs='*',
                help='When ensemble is True, this argument\
                        is required.')
        parser.add_argument('--output',
                default='ans.csv',
                help='When inferencing, the designated output\
                        filename.')
        parser.add_argument('--model_filename',
                default=None,
                type=str,
                help='When inferencing one model, \
                        the designated model.')
    args = parser.parse_args()
    
    if train:
        check_model(args.model)
    else:
        if args.ensemble:
            if args.models is None or len(args.models) <= 1 or \
                    args.model_filenames is None or len(args.model_filenames) <= 1:
                raise Exception("Ensemble set true, expect to have\
                        models and model_filenames arguments at least two")
            elif len(args.models) != len(args.model_filenames):
                raise Exception("Receive different length of models\
                        and corresponding model_filenames.")
            for model_name in args.models:
                check_model(model_name)
        else:
            if args.model is None or args.model_filename is None:
                raise Exception("Expect to have model and model_filename\
                        arguments, but not given.")
            check_model(args.model)
    return args


if __name__ == '__main__':
    args = get_args(False)
