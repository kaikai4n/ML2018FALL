import model
from args import get_args
from data import DataLoader
from dataset import VideoCaptionDataset
from dataset import customed_collate_fn_for_testing as collate_fn
from utils import load_training_args
import torch

def infer(
            total_test,
            test_video,
            test_caption, 
            batch_size,
            collate_fn,
            ensemble,
            my_model,
            my_models,
            output_filename,
            use_cuda=True):
    print('Infering preprocessing...')
    test_dataset = VideoCaptionDataset(
            total_test, test_video, test_caption, train=False)
    test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
    print('Start infering...')
    if use_cuda:
        if ensemble:
            for my_model in my_models:
                my_model = my_model.cuda()
        else:
            my_model = my_model.cuda()

    preds = None
    with torch.no_grad():
        for step, (video, (caption, length, indices)) in enumerate(test_loader):
            if use_cuda:
                video = video.cuda()
                caption, length, indices = \
                        caption.cuda(), length.cuda(), indices.cuda()
            if ensemble:
                total_distances = torch.zeros(video.size(0), 5)
                if use_cuda:
                    total_distances = total_distances.cuda()
                for my_model in my_models:
                    pred_video, pred_caption  = my_model.infer(video, caption, length)
                    pred_caption = pred_caption[indices]
                    distances = my_model.count_distance(pred_video, pred_caption)
                    total_distances += standardize(distances)
                final_pred = torch.argmin(total_distances, dim=1)
            else:
                pred_video, pred_caption  = my_model.infer(video, caption, length)
                pred_caption = pred_caption[indices]
                final_pred = my_model.count_argmin_distance(
                        pred_video, pred_caption)
            if preds is None:
                preds = final_pred
            else:
                preds = torch.cat([preds, final_pred])
    output_preds(output_filename, preds)

def get_test_data(video_dir, caption_filename, word_dict_filename):
    dl = DataLoader(
            video_dir=video_dir,
            caption_filename=caption_filename,
            load_word_dict=True,
            word_dict_filename=word_dict_filename)
    test_video, test_caption = dl.load_data(train=False)
    return test_video, test_caption

def load_model(model_name, model_filename, args_filename):
    try:
        model_class_object = getattr(model, model_name)
    except AttributeError:
        raise Exception('Model %s not found in model.py' % model_name)
    
    training_args = load_training_args(args_filename)
    training_args['load_model_filename'] = model_filename
    my_model = model_class_object(training_args, train=False)
    return my_model

def load_ensemble_models(model_names, model_filenames, args_filenames):
    return [load_model(*x) for x in \
            zip(model_names, model_filenames, args_filenames)]

def output_preds(output_filename, preds):
    with open(output_filename, 'w') as f:
        f.write('id,Ans\n')
        for i, ele in enumerate(preds):
            f.write('%d,%d\n' % (i+1, ele))

def standardize(d):
    return (d-d.mean(dim=1).view(-1, 1))/d.std(dim=1).view(-1, 1)

def main():
    args = get_args(train=False)
    use_cuda = args.no_cuda
    test_video, test_caption = get_test_data(
            args.video_dir, 
            args.caption_filename, 
            args.word_dict_filename)
    total_test = len(test_video)

    my_model, my_models = None, None
    if args.ensemble:
        my_models = load_ensemble_models(
                args.models, args.model_filenames, args.args_filenames)
        for my_model in my_models:
            my_model.eval()
    else:
        my_model = load_model(args.model, args.model_filename, args.args_filename)
        my_model.eval()

    infer(
            total_test=total_test,
            test_video=test_video,
            test_caption=test_caption, 
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            ensemble=args.ensemble,
            my_model=my_model,
            my_models=my_models,
            output_filename=args.output,
            use_cuda=use_cuda)

if __name__ == '__main__':
    main()
