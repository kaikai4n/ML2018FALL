# $1 test filename
# $2 jieba dictionary filename
# $3 output filename
# Inference time: it takes about a half minute for me to inference on Tesla K80
# If TA finds it to out of memory, please change the batch size to be smaller
# What's more, one can even inference with cpu with '--no_cuda' argument
python3 inference.py --model=RNNWord2VecMeanPooling --model_filename=models/RNNWord2VecMeanPooling/models_e5.pt --args_filename=models/RNNWord2VecMeanPooling/training_args.pkl --batch_size=512 --word_dict_filename=word_dict.pkl --test_x_filename=$1 --output=$3
