# This bash file tells to how to train with arguments
# log files are saved at logs/prefix.log
# model files are saved at models/prefix/models_e[epoch_num].pt
# type '--validation' to cut validation
python3 train.py --prefix TA_training_Mobilenet --model Mobilenet --train_filename $1 --epoches 1500
python3 train.py --prefix TA_training_VGG --model VGG --train_filename $1 --epoches 2000
python3 train.py --prefix TA_training_Kai --model Kai --train_filename $1 --epoches 2500
python3 train.py --prefix TA_training_Kai2 --model Kai2 --train_filename $1 --epoches 500
python3 train.py --prefix TA_training_Kai3 --model Kai3 --train_filename $1 --epoches 3000
python3 train.py --prefix TA_training_Kai4 --model Kai4 --train_filename $1 --epoches 500

