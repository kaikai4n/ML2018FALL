import model
import load
import data
import args

def test(train_x_filename, 
        train_y_filename,
        test_x_filename, 
        output_filename, 
        model_filename):
    data_loader = load.DataLoader()
    test_x = data_loader.read_testing_data(
            train_x_filename, 
            train_y_filename,
            test_x_filename)
    print(test_x.shape)
    data_processor = data.DataProcessor()
    test_x = data_processor.augment_features(test_x)
    print(test_x.shape)
    feature_num = test_x.shape[1]
    trainer = model.LogisticRegression(
            feature_num,
            train=False)
    trainer.load_model(model_filename)
    pred_y = trainer.forward(test_x)
    pred_y[pred_y >= 0.5] = 1
    pred_y[pred_y < 0.5] = 0
    with open(output_filename, 'w') as f:
        f.write('id,Value\n')
        for i, pred in enumerate(pred_y):
            f.write('id_%d,%d\n' % (i, int(pred)))

if __name__ == '__main__':
    args = args.get_args()
    output_filename = 'ans.csv'
    model_filename = 'models/try_e20.npy'
    test(args.train_x_filename,
            args.train_y_filename,
            args.test_x_filename, 
            output_filename, 
            model_filename)
