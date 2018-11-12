import load
import model
from args import get_args
import data
import torch

def train(
        train_x, 
        train_y, 
        batch_size, 
        model_name, 
        optimizer_name,
        learning_rate,
        epoch_num,
        use_cuda):
    print('Training preprocessing ...')
    face_dataset = data.FaceDataset(train_x, train_y, train_x.shape[0])
    train_loader = torch.utils.data.DataLoader(
                dataset=face_dataset,
                batch_size=batch_size,
                shuffle=True
            )
    try:
        model_class_object = getattr(model, model_name)
    except AttributeError:
        print('Model not found in model.py')
        exit()
    my_model = model_class_object()
    if use_cuda:
        my_model = my_model.cuda()
    try:
        optimizer_class_object = getattr(torch.optim, optimizer_name)
    except AttributeError:
        print('Optimizer not found.')
        exit()
    optimizer = optimizer_class_object(my_model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    print('Start Training ...')
    for epoch in range(epoch_num):
        total_loss = 0
        for step, (x, y) in enumerate(train_loader):
            if use_cuda:
                x, y = x.cuda(), y.cuda()
            pred = my_model.forward(x)
            loss = loss_func(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        print(pred[:10])
        print(y[10:])
        print('epoch:%3d, loss:%.3f' %\
                (epoch, total_loss))


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    data_loader = load.DataLoader()
    print('Loading data ...')
    train_x, train_y = data_loader.load_training_data(args.train_filename)
    train(
        train_x=train_x, 
        train_y=train_y,
        batch_size=args.batch_size, 
        model_name=args.model, 
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        epoch_num=args.epoches,
        use_cuda=args.use_cuda)
