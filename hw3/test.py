import load
import model
from args import get_args
import torch
import os
        
def ensemble(
        test_x, 
        output_filename, 
        model_names, 
        model_filenames, 
        use_cuda, 
        batch_size):
    test_len = test_x.shape[0]
    votes = torch.zeros(test_len, 7)
    for model_name, model_filename in zip(model_names, model_filenames):
        print('Start inference model %s' % model_name)
        pred = test(test_x, output_filename, model_name, model_filename, 
                use_cuda, batch_size, write=False)
        votes[torch.arange(test_len),pred.type(torch.long)] += 1
    final_pred = torch.argmax(votes, dim=1)
    
    with open(output_filename, 'w') as f:
        f.write('id,label\n')
        for i, pred_value in enumerate(final_pred):
            f.write('%d,%d\n' % (i, pred_value))
    

def test(
        test_x, 
        output_filename, 
        model_name, 
        model_filename, 
        use_cuda, 
        batch_size,
        write=True):
    try:
        model_class_object = getattr(model, model_name)
    except AttributeError:
        raise Exception('Model not found in model.py')
    my_model = model_class_object()
    my_model.load(model_filename)
    my_model.eval()
    if use_cuda:
        my_model = my_model.cuda()
    if write:
        with open(output_filename, 'w') as f:
            f.write('id,label\n')
            with torch.no_grad():
                test_len = test_x.shape[0]
                batch_num = int(test_len / batch_size) + 1
                for i in range(batch_num):
                    pred_prob = my_model.forward(\
                            test_x[batch_size*i:min(batch_size*(i+1), test_len)])
                    pred = torch.argmax(pred_prob, dim=1)
                    for j, pred_value in enumerate(pred):
                        f.write('%d,%d\n' % (i*batch_size+j, pred_value))
    else:
        preds = None
        with torch.no_grad():
            test_len = test_x.shape[0]
            batch_num = int(test_len / batch_size) + 1
            for i in range(batch_num):
                pred_prob = my_model.forward(\
                        test_x[batch_size*i:min(batch_size*(i+1), test_len)])
                pred = torch.argmax(pred_prob, dim=1)
                if preds is None:
                    preds = pred
                else:
                    preds = torch.cat([preds, pred.squeeze()])
        return preds

if __name__ == '__main__':
    args = get_args(train=False)
    torch.manual_seed(args.seed)
    data_loader = load.DataLoader()
    print('Loading data ...')
    test_x = data_loader.load_testing_data(args.test_filename)
    print('Processing data ...')
    test_x = torch.tensor(test_x)
    if args.use_cuda:
        test_x = test_x.cuda()
    test_x /= 255
    print('Start testing ...')
    if args.ensemble == True:
        ensemble(
                test_x,
                args.output,
                args.models,
                args.model_filenames,
                use_cuda=args.use_cuda,
                batch_size=args.batch_size)
    else:
        test(
                test_x, 
                args.output, 
                args.model, 
                args.model_filename, 
                use_cuda=args.use_cuda,
                batch_size=args.batch_size)
