# -*- coding: utf-8 -*-

from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import os
import regex
from collections import Counter

def make_vocab(fpath, fname):
    '''根据文本构建词汇表
       样式： 词汇     词频
    
    Args:
      fpath: 文本文件所在路径
      fname: 构建的词汇表文件名
    '''  
    text = codecs.open(fpath, 'r', 'utf-8').read()
    text = regex.sub(r"[、|。|，|‘|’\]\[.,!?\"':;-_)(]", "", text)   # 去除标点符号
    words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists('preprocessed'):
        os.mkdir('preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))

if __name__ == '__main__':
    make_vocab(hp.source_train, "de.vocab.tsv")
    make_vocab(hp.target_train, "en.vocab.tsv")
    print("Done")