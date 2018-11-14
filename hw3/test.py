import load
import model
from args import get_args
import data
import torch
import os

def test(test_x, output_filename, model_name, model_filename, use_cuda):
    try:
        model_class_object = getattr(model, model_name)
    except AttributeError:
        raise Exception('Model not found in model.py')
    my_model = model_class_object()
    my_model.load(model_filename)
    my_model.eval()
    if use_cuda:
        my_model = my_model.cuda()
    with open(output_filename, 'w') as f:
        f.write('id,label\n')
        with torch.no_grad():
            test_len = test_x.shape[0]
            batch_num = int(test_len / 1000) + 1
            for i in range(batch_num):
                pred_prob = my_model.forward(\
                        test_x[1000*i:min(1000*(i+1), test_len)])
                pred = torch.argmax(pred_prob, dim=1)
                for j, pred_value in enumerate(pred):
                    f.write('%d,%d\n' % (i*1000+j, pred_value))

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
    #data_processor = data.DataProcessor(train_x, train_y, train_x.shape[0])
    #data_processor.normalize()
    #train_x, train_y = data_processor.get_data()
    print('Start testing ...')
    test(
            test_x, 
            args.output, 
            args.model, 
            args.model_filename, 
            use_cuda=args.use_cuda)
