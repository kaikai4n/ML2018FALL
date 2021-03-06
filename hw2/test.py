import model
import load
import data
import args

def test(train_x_filename, 
        train_y_filename,
        test_x_filename, 
        output_filename, 
        model_filename,
        model_name):
    data_loader = load.DataLoader()
    test_x = data_loader.read_testing_data(
            train_x_filename, 
            train_y_filename,
            test_x_filename)
    feature_num = test_x.shape[1]
    if model_name == 'LogisticRegression':
        trainer = model.LogisticRegression(
                feature_num,
                train=False)
        trainer.load_model(model_filename)
        pred_y = trainer.forward(test_x)
        pred_y[pred_y >= 0.5] = 1
        pred_y[pred_y < 0.5] = 0
    elif model_name == 'GaussianNaiveBayes':
        trainer = model.GaussianNaiveBayes()
        trainer.load_model(model_filename)
        pred_y = trainer.predict(test_x)
    with open(output_filename, 'w') as f:
        f.write('id,Value\n')
        for i, pred in enumerate(pred_y):
            f.write('id_%d,%d\n' % (i, int(pred)))

if __name__ == '__main__':
    args = args.get_args(train=False)
    test(args.train_x_filename,
            args.train_y_filename,
            args.test_x_filename, 
            args.output, 
            args.model_filename,
            args.model)
