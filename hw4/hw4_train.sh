# $1 train x filename
# $2 train y filename
# $3 test x filename
# $4 jieba userdict filename
prefix="TA_train" # the prefix when saving log file and model directory name
model_name="RNNWord2VecMeanPooling" # the model name in model.py
# add "--load_word_dict" to load word dictionary from given dictionary file,
# otherwise, create its own word dictionary
word_dict="word_dict.pkl" # Created or Loaded word dictionary filename
# add "--validation" to validate while training
python3 train.py --prefix=$prefix --model=$model_name --train_x_filename=$1 --train_y_filename=$2 --word_dict_filename=$word_dict
