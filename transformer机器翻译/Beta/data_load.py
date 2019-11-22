# -*- coding: utf-8 -*-

from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex


def load_de_vocab():
    ''' 给德语的每个词分配一个id并返回两个字典，一个是根据词找id，一个是根据id找词 '''
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(source_sents, target_sents): 
    """
    将每个句子用索引表示，一行一个句子，并将词数少于限定的最大单词长度的句子pad 0
    args:
        source_sents: 源语言，list, 元素是句子
        target_sents: 目标语言，list, 元素是句子
    return：
            X,Y:转换成数字的源目标语言句子(长度小于maxlen的已经pad 0); list[[],[]]
            Source, Target: 与X,Y对应的未转换成数字的源目标语言; list[[],[]]
    """
    
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()] # 1:UNK , </S>: End of Text
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()] 
        if max(len(x), len(y)) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    
    return X, Y, Sources, Targets


def load_train_data():
    """
    返回用索引表示的句子列表，每行表示一个句子
    """
    de_sents = [regex.sub(r"[、|。|，|‘|’\]\[.,!?\"':;-_)(]", "", line) for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line.strip()]
    en_sents = [regex.sub(r"[、|。|，|‘|’\]\[.,!?\"':;-_)(]", "", line) for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line.strip()]
    
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y
 

def load_test_data():
    def _refine(line):
        # line = regex.sub("<[^>]+>", "", line)
        line = regex.sub(r"[、|。|，|‘|’\]\[.,!?\"':;-_)(]", "", line)
        return line.strip()
    
    de_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if line]
    en_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if line]
        
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets # (1064, 150)


def get_batch_data():
    """
    获取一个batch的数据
    """
    X, Y = load_train_data()
    
    # 计算总的batch数量
    num_batch = len(X) // hp.batch_size
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)  # 将不同数据变成张量
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])   # 按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor
            
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch # (N, T), (N, T), ()

