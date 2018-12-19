import model
from args import get_args

def main():
    args = get_args(train=True)
    training_args = {
            'load_model_filename': None,
            'vocabulary_size': 7768,
            'embed_dim': args.embed_dim,
            'hidden_size': args.hidden_size,
            'rnn_layers': args.rnn_layers,
            'dropout': args.dropout_rate,
            'bidirectional': args.no_bidirectional}
    for model_name in dir(model):
        if model_name == 'BaseModel':
            continue
        the_object = getattr(model, model_name)
        if type(the_object) is type:
            paras = sum([ele.numel() for ele in the_object(training_args, train=True).parameters()])
            print('%s: %d' % (model_name, paras))


if __name__ == '__main__':
    main()
