from data import DataLoader
import model
from args import get_args


def train(
        total_data,
        train_x,
        train_y,
        sentence_length,
        batch_size,
        epoches):
    dcard_dataset = data.DcardDataset(
            total_data, train_x, train_y, sentence_length)
    train_loader = torch.utils.data.DataLoader(
                dataset=dcard_dataset,
                batch_size=batch_size,
                shuffle=True
            )
    for epoch in range(epoches):
        for step, (x, y) in enumerate(train_loader):

def main():
    args = get_args(train=True)
    if args.load_word_dict:
        dl = DataLoader(
                create_word_dict=False,
                word_dict_filename=args.word_dict_filename) 
    else:
        dl = DataLoader(
                create_word_dict=True, 
                filenames=[args.train_x_filename, args.test_x_filename],
                save_word_dict=True,
                word_dict_filename=args.word_dict_filename)
    train_x = dl.load_data_x(args.train_x_filename)
    train_y = dl.load_data_y(args.train_y_filename)
    sentence_length = dl.get_sentence_length()
    train(
            total_data=len(train_x),
            train_x=train_x,
            train_y=train_y,
            sentence_length = setence_length,
            batch_size=args.batch_size,
            epoches=args.epoches)


if __name__ == '__main__':
    main()
