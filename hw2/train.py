import load
import args
import data
import model
import batch
import optimizer
import numpy as np
import os
    
def train_gaussian_naive_bayes(
            train_x, 
            train_y,
            prefix,
            validation):
    # Make sure the save pathes are valid
    # check model save path
    if os.path.isdir('models') == False:
        os.mkdir('models')
    save_model_prefix = os.path.join('models', prefix)
    if os.path.isdir(save_model_prefix) == False:
        os.mkdir(save_model_prefix)
    # check log save path
    if os.path.isdir('logs') == False:
        os.mkdir('logs')
    log_f = open(os.path.join('logs', prefix + '.log'), 'w')
    if validation:
        log_f.write('accuracy,validation\n')
    else:
        log_f.write('accuracy\n')

    feature_num = train_x.shape[1]
    if validation:
        data_processor = data.DataProcessor()
        train_x, train_y, valid_x, valid_y = \
                data_processor.cut_validation(train_x, train_y)
    total_data = train_x.shape[0]
    trainer = model.GaussianNaiveBayes()
    trainer.fit(train_x, train_y, feature_num)
    pred_y = trainer.predict(train_x)
    accuracy = trainer.count_accuracy(pred_y, train_y)
    if validation:
        valid_pred = trainer.predict(valid_x)
        valid_accuracy = trainer.count_accuracy(valid_pred, valid_y)
        message = 'accuracy:%.3f, validation:%.3f'\
                % (accuracy, valid_accuracy)
        log_f.write('%.3f,%.3f\n'\
                % (accuracy, valid_accuracy))
    else:
        message = 'accuracy:%.3f' % (accuracy)
        log_f.write('%.3f\n'% (accuracy))
    print(message)
    save_model_path = os.path.join(
            save_model_prefix, 'model.npy')
    trainer.save_model(save_model_path)

    log_f.close()


def train(
        train_x, 
        train_y, 
        batch_size, 
        total_epoch, 
        learning_rate, 
        save_intervals, 
        prefix,
        optim_name,
        validation):
    # Make sure the save pathes are valid
    # check model save path
    if os.path.isdir('models') == False:
        os.mkdir('models')
    save_model_prefix = os.path.join('models', prefix)
    if os.path.isdir(save_model_prefix) == False:
        os.mkdir(save_model_prefix)
    # check log save path
    if os.path.isdir('logs') == False:
        os.mkdir('logs')
    log_f = open(os.path.join('logs', prefix + '.log'), 'w')
    if validation:
        log_f.write('epoch,loss,accuracy,validation\n')
    else:
        log_f.write('epoch,loss,accuracy\n')

    feature_num = train_x.shape[1]
    if validation:
        data_processor = data.DataProcessor()
        train_x, train_y, valid_x, valid_y = \
                data_processor.cut_validation(train_x, train_y)
    total_data = train_x.shape[0]
    try:
        optim_object = getattr(optimizer, optim_name)
    except AttributeError:
        print('Optimizer not found.')
        exit()
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
        if validation:
            valid_pred = trainer.forward(valid_x)
            print(valid_pred)
            valid_accuracy = trainer.count_accuracy(valid_pred, valid_y)
            message = 'epoch:%3d, loss:%.3f, accuracy:%.3f, validation:%.3f'\
                    % (epoch, total_loss, accuracy, valid_accuracy)
            log_f.write('%d,%.3f,%.3f,%.3f\n'\
                    % (epoch, total_loss, accuracy, valid_accuracy))
        else:
            message = 'epoch:%3d, loss:%.3f, accuracy:%.3f' % \
                    (epoch, total_loss, accuracy)
            log_f.write('%d,%.3f,%.3f\n'\
                    % (epoch, total_loss, accuracy))
        print(message)
        if (epoch + 1) % save_intervals == 0:
            save_model_path = os.path.join(
                    save_model_prefix, 'model_e%d.npy' % (epoch+1))
            trainer.save_model(save_model_path)

    log_f.close()

if __name__ == '__main__':
    args = args.get_args()
    np.random.seed(args.seed)
    data_loader = load.DataLoader()
    train_x, train_y = data_loader.read_training_data(
            args.train_x_filename, args.train_y_filename, args.test_x_filename)
    if args.model == 'GaussianNaiveBayes':
        train_gaussian_naive_bayes(
                train_x, 
                train_y,
                args.prefix,
                args.validation)
    else:
        train(
                train_x, 
                train_y,
                args.batch_size,
                args.epoches,
                args.learning_rate,
                args.save_intervals,
                args.prefix,
                args.optimizer,
                args.validation)
