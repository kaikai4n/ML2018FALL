##########################################################
# To print confusion matrix, command:                    #
# python3 print_confusion_matrix.py --model=$model_name  #
# --model_filename=$model_filename                       # 
##########################################################
from args import get_args
import torch
import numpy as np
import load
import data
import model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def inference_validation(
        valid_x,
        batch_size,
        model_name,
        model_filename,
        use_cuda):
    try:
        model_class_object = getattr(model, model_name)
    except AttributeError:
        raise Exception('Model not found in model.py')

    my_model = model_class_object()
    my_model.load(model_filename)
    if use_cuda:
        my_model = my_model.cuda()

    valid_len = valid_x.shape[0]
    batch_num = int(valid_len/batch_size) + 1
    pred_y = None
    for batch_i in range(batch_num):
        start_i = batch_i * batch_size
        end_i = min((batch_i+1) * batch_size, valid_len)
        x = valid_x[start_i:end_i]
        x = torch.tensor(x).cuda() if use_cuda else torch.tensor(x)
        y = my_model.forward(x)
        y = torch.argmax(y, dim=1)
        if pred_y is None:
            pred_y = y
        else:
            pred_y = torch.cat([pred_y,y])
    return pred_y

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def print_training_label_percentage(data):
    unique, counts = np.unique(data, return_counts=True)
    total_counts = np.sum(counts)
    counts = counts / total_counts
    label_counts_dict = dict(zip(unique, counts))
    print(label_counts_dict)

if __name__ == '__main__':
    args = get_args(train=False)
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_loader = load.DataLoader()
    print('Loading data ...')
    train_x, train_y = data_loader.load_training_data(args.train_filename)
    print('Processing data ...')
    data_processor = data.DataProcessor(train_x, train_y, train_x.shape[0])
    data_processor.normalize()
    data_processor.cut_validation()
    train_x, train_y, valid_x, valid_y = data_processor.get_data()
    #print_training_label_percentage(train_y)
    pred_y = inference_validation(
            valid_x,
            args.batch_size,
            args.model,
            args.model_filename,
            args.use_cuda)
    #print(valid_x.shape, valid_y.shape, pred_y.shape)
    pred_y = pred_y.cpu().numpy() if args.use_cuda else pred_y.numpy()
    plt.figure()
    cnf_matrix = confusion_matrix(valid_y, pred_y)
    senti_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
    plot_confusion_matrix(cnf_matrix, senti_classes, normalize=False)
    plt.savefig('confusion_matrix.png')
