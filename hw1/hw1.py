# Load model parameters to test
import model
import load
import pandas

def load_trainer(model_path):
    trainer = model.LinearRegression()
    trainer.load_model(model_path)
    return trainer

if __name__ == '__main__':
    model_path = 'models/model_e999.npy'
    trainer = load_trainer(model_path)

    test_path = 'data/test.csv'
    testing_data = load.load_test_csv(test_path)

    output_path = 'ans.csv'
    outputs = [['id', 'value']]
    for i in range(testing_data.shape[0]):
        test_x = testing_data[i]
        prediction = trainer.forward(test_x)
        outputs.append(['id_%d' % i, prediction[0]])
    pandas.DataFrame(outputs).to_csv(output_path, 
            header=False, index=False)    
