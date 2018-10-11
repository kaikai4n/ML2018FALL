Machine Learning hw1: PM2.5
===

# Task description
- An observed wheather data of 大里, total 18 features per hour. Target is to predict the next PM2.5 value given the last 9 hours data.
- Homework designer regulates to use linear regression model only.

# How to train
## Download training data
- As TAs have regulated, I can't provide the training and testing file here.
- Please refer to competition on Kaggle to get the data.
## Train main model
- Example args: 
    ```=
        python3 train.py --attributes_filename=models/attributes_2_5_6_8_9.npy --epoches=6000 --prefix=2_5_6_8_9_both_PM_lr0.005 --learning_rate=0.005 --train_filename=$train_filename
    ```
    - ``--train_filename`` is the ``train.csv`` file.
    - ``--attributes_filename`` refers to a numpy parameter file storing True and False array specifying which attributes are used. In this case, I use the 2'th, 5'th, 6'th, 8'th and 9'th for training and testing.
    - ``--prefix`` is the prefix name for saving logs and models.
    - ``--validation`` if you want to cut validation (train:test=9:1).
    - ``--lambda_value`` is the regularization weight, better to set $1e-6$ at first if you are interested to use.
## Inference
- Example args:
    ```=
        python3 hw1.py --model_main=models/2_5_6_8_9_both_PM_lr0.005/model_e6000.npy --attributes_filename=models/attributes_2_5_6_8_9.npy --testing_filename=$test_filename --output=$output
    ```
    - ``--model_main`` is the model path trained above.
    - ``--testing_filename`` specifies the path of ``test.csv``.
    - ``--output`` is the submission csv file path.
    - ``--data_bounds_filename`` should exist at ``models/`` directory after training main model. Please do not alter the file.
## TAs inference guidance
- ```=
        bash hw1.sh $test_filename $output
    ```
