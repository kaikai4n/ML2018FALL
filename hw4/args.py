import argparse
import model

def check_model(model_name):
    if model_name is None:
        raise Exception('Model name must be provided.')
    try:
        model_object = getattr(model, model_name)
    except AttributeError:
        raise Exception('Model not found:', model_name)
        

def get_args(train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_x_filename',
            default='data/train_x.csv',
            help='The csv file to train for x')
    parser.add_argument('--train_y_filename',
            default='data/train_y.csv',
            help='The csv file to train for y')
    parser.add_argument('--test_x_filename',
            default='data/test_x.csv',
            help='The csv file to test.')
    parser.add_argument('--model',
            default=None,
            help='The model user would like to use.')
    parser.add_argument('--no_cuda',
            default=True,
            action='store_false',
            help='Force not to use GPU.')
    parser.add_argument('--seed',
            default=7122,
            type=int,
            help='Random seed for numpy and torch.')
    parser.add_argument('-b', '--batch_size',
            default=512,
            type=int,
            help='The batch size for training.')
    parser.add_argument('--load_word_dict',
            default=False,
            action='store_true',
            help='To load preset word dictionary or not.')
    parser.add_argument('--word_dict_filename',
            default='word_dict.pkl',
            help='If "--load_word_dict" is true\
                    then this filename must given.')
    if train:
        parser.add_argument('--validation',
                action='store_true',
                default=False,
                help='To split validation or not.')
        parser.add_argument('-e', '--epoches',
                type=int,
                default=50)
        parser.add_argument('-lr', '--learning_rate',
                type=float,
                default=0.001)
        parser.add_argument('--save_intervals',
                default=5,
                type=int,
                help='The epoch intervals to save models')
        parser.add_argument('--prefix',
                required=True,
                help='The prefix of saving name')
        parser.add_argument('--load_model_filename',
                default=None,
                type=str,
                help='The initialization parameters \
                        from a given model name.')
        parser.add_argument('--hidden_size',
                default=256,
                type=int,
                help='The hidden size of RNN.')
        parser.add_argument('--rnn_layers',
                default=1,
                type=int,
                help='The rnn layers.')
        parser.add_argument('--embed_dim',
                default=128,
                type=int,
                help='The word embedding dimension.')
        parser.add_argument('--dropout_rate',
                default=0.0,
                type=float,
                help='The dropout rate for RNN')
        parser.add_argument('--no_bidirectional',
                default=True,
                action='store_false',
                help='To specify not to use bidirectional\
                        for RNN.')
                
    else:
        parser.add_argument('--args_filename',
                required=True,
                help='When initializing model, it is neccessary\
                        to give the training arguments from a file.')
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
        if args.model is None or args.model_filename is None:
            raise Exception("Expect to have model and model_filename\
                    arguments, but not given.")
        check_model(args.model)
    return args


if __name__ == '__main__':
    args = get_args(False)
