import model
from args import get_args
from data import DataLoader
from data import DcardDataset
from data import customed_collate_fn
import pickle
from utils import load_training_args
import torch

def infer(
        total_test, 
        test_id,
        test_x, 
        my_model, 
        batch_size,
        output_filename,
        vocabulary_size,
        use_cuda=True):
    if use_cuda:
        my_model = my_model.cuda()
    dcard_test_dataset = DcardDataset(
            total_test, test_x, test_id, vocabulary_size)
    test_loader = torch.utils.data.DataLoader(
                dataset=dcard_test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=customed_collate_fn
            )
    preds = None
    with torch.no_grad():
        for (x, x_id) in test_loader:
            if use_cuda:
                x, x_id = x.cuda(), x_id.cuda()
            pred = my_model.forward(x)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred = pred[torch.sort(x_id)[1]]
            if preds is None:
                preds = pred
            else:
                preds = torch.cat([preds, pred])
    with open(output_filename, 'w') as f:
        f.write('id,label\n')
        for i, ele in enumerate(preds):
            f.write('%d,%d\n' % (i, ele))

def get_test_data(filename, word_dict_filename):
    dl = DataLoader(
            create_word_dict=False,
            word_dict_filename=args.word_dict_filename) 
    test_x = dl.load_data_x(filename)
    vocabulary_size = dl.get_word_dict_len()
    return test_x, vocabulary_size

def load_model(model_name, model_filename, args_filename):
    try:
        model_class_object = getattr(model, model_name)
    except AttributeError:
        raise Exception('Model %s not found in model.py' % model_name)
    
    training_args = load_training_args(args_filename)
    training_args['load_model_filename'] = model_filename
    my_model = model_class_object(training_args, train=False)
    return my_model

if __name__ == '__main__':
    args = get_args(train=False)
    use_cuda = args.no_cuda
    test_x, vocabulary_size\
            = get_test_data(args.test_x_filename, args.word_dict_filename)
    total_test = len(test_x)
    test_id = [i for i in range(total_test)]
    my_model = load_model(args.model, args.model_filename, args.args_filename)
    my_model.eval()
    print('Start infering...')
    infer(
            total_test=total_test,
            test_id=test_id,
            test_x=test_x, 
            my_model=my_model, 
            batch_size=args.batch_size,
            output_filename=args.output,
            vocabulary_size=vocabulary_size,
            use_cuda=use_cuda)
