import load
import model
from args import get_args
import data
import torch
import os
import numpy as np

def train(
        train_x, 
        train_y, 
        valid_x,
        valid_y,
        batch_size, 
        model_name, 
        optimizer_name,
        learning_rate,
        epoch_num,
        use_cuda,
        save_intervals,
        prefix,
        validation):
    print('Training preprocessing ...')
    if validation and (valid_x is None or valid_y is None):
        raise Exception('validation==True, expect valid_x, valid_y is not None.')
    # process saveing path
    # log file
    if os.path.isdir('logs') == False:
        os.mkdir('logs')
    log_save_path = os.path.join('logs', prefix + '.log')
    f_log = open(log_save_path, 'w+')
    if validation:
        f_log.write('epoch,loss,validation\n')
    else:
        f_log.write('epoch,loss\n')
    # model path
    if os.path.isdir('models') == False:
        os.mkdir('models')
    model_path = os.path.join('models', prefix)
    if os.path.isdir(model_path) == False:
        os.mkdir(model_path)

    # process validation
    if validation:
        valid_x, valid_y = torch.tensor(valid_x), torch.tensor(valid_y)
        if use_cuda:
            valid_x, valid_y = valid_x.cuda(), valid_y.cuda()

    # process dataset
    face_dataset = data.FaceDataset(train_x, train_y, train_x.shape[0])
    train_loader = torch.utils.data.DataLoader(
                dataset=face_dataset,
                batch_size=batch_size,
                shuffle=True
            )
    try:
        model_class_object = getattr(model, model_name)
    except AttributeError:
        raise Exception('Model not found in model.py')
    my_model = model_class_object()
    if use_cuda:
        my_model = my_model.cuda()
    try:
        optimizer_class_object = getattr(torch.optim, optimizer_name)
    except AttributeError:
        raise Exception('Optimizer not found.')
    optimizer = optimizer_class_object(my_model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    print('Start Training ...')
    for epoch in range(epoch_num):
        total_loss, total_steps = 0, 0
        for step, (x, y) in enumerate(train_loader):
            if use_cuda:
                x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            pred = my_model.forward(x)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss
            total_steps += 1

        total_loss /= total_steps
        if validation:
            with torch.no_grad():
                valid_loss = loss_func(my_model.forward(valid_x), valid_y)
            progress_msg = 'epoch:%3d, loss:%.3f, validation:%.3f' % \
                    (epoch, total_loss, valid_loss)
            log_msg = '%d,%.5f,%.5f\n' % \
                    (epoch, total_loss, valid_loss)
        else:
            progress_msg = 'epoch:%3d, loss:%.3f' % (epoch, total_loss)
            log_msg = '%d,%.5f\n' % (epoch, total_loss)
        print(progress_msg)
        f_log.write(log_msg)
        if (epoch + 1) % save_intervals == 0:
            model_save_path = os.path.join(model_path, 'models_e%d.pt' % (epoch+1))
            my_model.save(model_save_path)

    f_log.close()

if __name__ == '__main__':
    args = get_args()
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_loader = load.DataLoader()
    print('Loading data ...')
    train_x, train_y = data_loader.load_training_data(args.train_filename)
    print('Processing data ...')
    data_processor = data.DataProcessor(train_x, train_y, train_x.shape[0])
    data_processor.normalize()
    if args.validation:
        data_processor.cut_validation()
    train_x, train_y, valid_x, valid_y = data_processor.get_data()
    train(
        train_x=train_x, 
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        batch_size=args.batch_size, 
        model_name=args.model, 
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        epoch_num=args.epoches,
        use_cuda=args.use_cuda,
        save_intervals=args.save_intervals,
        prefix=args.prefix,
        validation=args.validation)
