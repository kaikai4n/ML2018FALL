import load
import args
import data
import model
import batch
import optimizer
import numpy as np

def train(train_x, train_y, batch_size, total_epoch, learning_rate):
    feature_num = train_x.shape[1]
    total_data = train_x.shape[0]
    optim = optimizer.AdaGrad(learning_rate)
    trainer = model.LogisticRegression(
            train_x, 
            train_y, 
            feature_num,
            optim)
    batcher = batch.Batcher(train_x, train_y, batch_size)
    for epoch in range(total_epoch):
        total_loss = 0.0
        batcher.new_epoch()
        total_correct = 0
        for step, (x, y) in enumerate(batcher):
            pred = trainer.forward(x)
            loss = trainer.count_loss(pred, y)
            trainer.backward(x, y, pred)
            total_loss += np.sum(loss)
            pred[pred >= 0.5] = 1.0
            pred[pred < 0.5] = 0.0
            total_correct += np.sum(pred.squeeze() == y.squeeze())
        total_loss /= total_data
        accuracy = total_correct / total_data
        print('epoch:%d, loss:%.3f, Ein:%.3f' \
                % (epoch, total_loss, accuracy))

if __name__ == '__main__':
    args = args.get_args()
    np.random.seed(args.seed)
    data_loader = load.DataLoader()
    train_x, train_y = data_loader.read_training_data(
            args.train_x_filename, args.train_y_filename)
    print(train_x.shape, train_y.shape)
    data_processor = data.DataProcessor()
    train_x = data_processor.augment_features(train_x)
    train(
            train_x, 
            train_y,
            args.batch_size,
            args.epoches,
            args.learning_rate)
