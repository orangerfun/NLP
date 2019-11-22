# -*- coding: utf-8 -*-

class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = './raw_data/train_and_test/train_y.txt'
    target_train = './raw_data/train_and_test/train_x.txt'
    source_test = './raw_data/train_and_test/test_y.txt'
    target_test = './raw_data/train_and_test/test_x.txt'
    
    # training
    batch_size = 32 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = './raw_data/logdir' # log directory
    
    # model
    maxlen = 35 # Maximum number of words in a sentence. alias = T.
    min_cnt = 5 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 100
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.编码方式选择
    
    
    
    
