import load
import model
import numpy as np

def preprocessing():
    data = load.read_csv('data/train.csv')
    data = load.parse_csv(data)
    train_x = load.csv_to_np(data)
    return train_x

    

def train(train_x):
    train_x = preprocessing()
    trainer = model.LinearRegression(x = train_x)
    
    total_epoches = 1000
    learning_rate = 0.001
    adagrad_n = 0
    for epoch in range(total_epoches):
        trainer.new_epoch()
        total_loss = 0.0
        for step, (x, y) in enumerate(trainer.get_batch()):
            prediction = trainer.forward(x) 
            total_loss += np.power(prediction - y, 2)
            adagrad_n += np.power(prediction - y, 2)
            grad = learning_rate * (prediction-y) / np.power(adagrad_n+10e-6, 0.5)
            trainer.backward(grad, x, grad_clip=False)
        print('epoch:%d, total loss:%.3f' % (epoch, total_loss))
        if (epoch+1) % 100 == 0:
            trainer.save_model('models/model_e%d.npy' % epoch)

if __name__ == '__main__':
    np.random.seed(7122)
    train_x = preprocessing()
    train(train_x)
