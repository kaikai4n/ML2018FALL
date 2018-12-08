from data import DataLoader
from data import DcardDataset
from data import customed_collate_fn
import model
from args import get_args
import torch
from utils import save_training_args
from utils import check_save_path
from utils import set_random_seed

def train(
        total_data,
        train_x,
        train_y,
        sentence_length,
        prefix, 
        validation,
        batch_size,
        collate_fn,
        model_name,
        vocabulary_size,
        embed_dim,
        hidden_size,
        dropout_rate,
        bidirectional,
        learning_rate,
        epoches,
        use_cuda=True):
    print('Training preprocessing...')
    # processing saving path
    log_save_path, model_path, save_args_path = \
            check_save_path(prefix, validation)

    # make dataset
    dcard_dataset = DcardDataset(
            total_data, train_x, train_y, sentence_length)
    train_loader = torch.utils.data.DataLoader(
                dataset=dcard_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )

    # Initialize model
    try:
        model_class_object = getattr(model, model_name)
    except AttributeError:
        raise Exception('Model %s not found in model.py' % model_name)
    
    # training arguments to send
    # load_model_filename: the filename of init parmeters
    # word_dict_len: the length of word dictionary
    # embed_dim: embedding dimension
    # hidden_size: hidden size of RNN
    # dropout: dropout rate of RNN 
    # bidirectional: RNN is bidirectional or not
    training_args = {
            'load_model_filename': None,
            'vocabulary_size': vocabulary_size,
            'embed_dim': embed_dim,
            'hidden_size': hidden_size,
            'dropout': dropout_rate,
            'bidirectional': bidirectional}
    save_training_args(training_args, save_args_path)
    my_model = model_class_object(training_args, train=True)
    my_model = my_model.cuda() if use_cuda else my_model

    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    
    print('Start training...')
    for epoch in range(epoches):
        for step, (x, y, length) in enumerate(train_loader):
            if use_cuda:
                x, y, length = x.cuda(), y.cuda(), length.cuda()
            optimizer.zero_grad()
            pred_y = my_model.forward(x, length)
            loss = loss_func(pred_y, y)
            loss.backward()
            optimizer.step()
            exit()

def main():
    args = get_args(train=True)
    set_random_seed(args.seed)
    if args.load_word_dict:
        dl = DataLoader(
                create_word_dict=False,
                word_dict_filename=args.word_dict_filename) 
    else:
        dl = DataLoader(
                create_word_dict=True, 
                filenames=[args.train_x_filename, args.test_x_filename],
                save_word_dict=True,
                word_dict_filename=args.word_dict_filename)
    train_x = dl.load_data_x(args.train_x_filename)
    train_y = dl.load_data_y(args.train_y_filename)
    sentence_length = dl.get_sentence_length()
    word_dict_len = dl.get_word_dict_len()
    train(
            total_data=len(train_x),
            train_x=train_x,
            train_y=train_y,
            sentence_length=sentence_length,
            prefix=args.prefix,
            validation=args.validation,
            batch_size=args.batch_size,
            collate_fn=customed_collate_fn,
            model_name=args.model,
            vocabulary_size=word_dict_len,
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            dropout_rate=args.dropout_rate,
            bidirectional=args.no_bidirectional,
            learning_rate=args.learning_rate,
            epoches=args.epoches,
            use_cuda=args.no_cuda)


if __name__ == '__main__':
    main()
