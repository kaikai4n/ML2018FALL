import load
import args
import data
import model
import batch
import optimizer
import numpy as np
import os

def train(
        train_x, 
        train_y, 
        batch_size, 
        total_epoch, 
        learning_rate, 
        save_intervals, 
        prefix,
        optim_name):
    if os.path.isdir('models') == False:
        os.mkdir('models')
    feature_num = train_x.shape[1]
    total_data = train_x.shape[0]
    try:
        optim_object = getattr(optimizer, optim_name)
    except AttributeError:
        print('Optimizer not found.')
    optim = optim_object(learning_rate)
    trainer = model.LogisticRegression(
            feature_num,
            optim,
            train=True)
    batcher = batch.Batcher(train_x, train_y, batch_size)
    for epoch in range(total_epoch):
        total_loss = 0.0
        batcher.new_epoch()
        total_accuracy = 0.0
        for step, (x, y) in enumerate(batcher):
            pred = trainer.forward(x)
            loss = trainer.count_loss(pred, y)
            trainer.backward(x, y, pred)
            total_loss += np.sum(loss)
            total_accuracy += trainer.count_accuracy(pred, y)
        total_loss /= total_data
        accuracy = total_accuracy * batch_size / total_data
        print('epoch:%d, loss:%.3f, Ein:%.3f' \
                % (epoch, total_loss, accuracy))
        if (epoch + 1) % save_intervals == 0:
            save_model_path = os.path.join('models', '%s_e%d.npy' % (prefix, epoch+1))
            trainer.save_model(save_model_path)

if __name__ == '__main__':
    args = args.get_args()
    np.random.seed(args.seed)
    data_loader = load.DataLoader()
    train_x, train_y = data_loader.read_training_data(
            args.train_x_filename, args.train_y_filename, args.test_x_filename)
    print(train_x.shape, train_y.shape)
    data_processor = data.DataProcessor()
    train_x = data_processor.augment_features(train_x)
    print(train_x.shape, train_y.shape)
    train(
            train_x, 
            train_y,
            args.batch_size,
            args.epoches,
            args.learning_rate,
            args.save_intervals,
            args.prefix,
            args.optimizer)
