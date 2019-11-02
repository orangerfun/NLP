# -*- coding:utf-8 -*-

"""
数据预处理
主要流程：
	1.按照词频创建词汇文件，格式：词：词频
	2.从词汇文件中读取词，并建立词汇到索引(索引到词汇)的映射
	3.根据映射将文本转换成向量形式(数字形式)
	4.生成一个batch的训练和测试数据
@author:orangerfun@gmail.com
"""

import codecs
import re
from collections import Counter
import numpy as np
import tensorflow as tf

np.set_printoptions(threshold = np.inf)

ch_train = "./data/chinese.txt"
ch_test = "./data/ch_test.txt"
en_train = "./data/english.txt"
en_test = "./data/en_test.txt"

N = 1000000000
maxlen = 40
embedding = 512
batchsize = 32

def make_vocab(path, outputfile):
	'''创建词汇文件，格式：词：词频'''
	text = codecs.open(path, "r", "utf-8").read()
	text = re.sub(r"[,.\[\]'?|\"-()，。‘’“”？：；;:<>《》]", "", text)
	words = text.split()
	fre = Counter(words)
	with codecs.open("./data/"+outputfile, "w", "utf-8") as fout:
		fout.write("{}\t1000000000\n{}\t1000000000\n{}\t100000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
		for word, freq in fre.most_common(len(fre)):
			fout.write("{}\t{}\n".format(word, freq))

def load_vocab(file):
	'''构建从词到索引和索引到词的映射'''
	words = [line.split()[0] for line in codecs.open(file, 'r', 'utf-8').readlines()]
	w2ind = {word:index for index, word in enumerate(words)}
	ind2w = {index:word for index, word in enumerate(words)}
	return w2ind, ind2w

def sentence2id(sourcefile, targetfile):
	"""将句子映射成向量"""
	ch2id, id2ch = load_vocab("./data/ch_words.txt")
	en2id, id2en = load_vocab("./data/en_words.txt")
	x_list , y_list = [], []
	for ch_sen, en_sen in zip(sourcefile,targetfile):
		x = [ch2id.get(word, 1) for word in ch_sen.strip().split()]
		if len(x) < maxlen:
			x_list.append(np.concatenate((np.array(x),np.zeros(maxlen-len(x)))))
		else:
			x_list.append(np.array(x)[:maxlen])

		y = [en2id.get(word, 1) for word in en_sen.strip().split()]
		if len(y) < maxlen:
			y_list.append(np.concatenate((np.array(y),np.zeros(maxlen-len(y)))))
		else:
			y_list.append(np.array(y)[:maxlen])
	x_list, y_list = np.array(x_list,dtype=np.int), np.array(y_list,dtype=np.int)
	return x_list,y_list

def load_train_data():
	"""生成训练数据"""
	ch_sents = [re.sub(r"[,.\[\]'?|\"-()，。‘’“”？：；;:<>《》]", "", line) for line in codecs.open(ch_train, "r", "utf-8").readlines() if line.strip()]
	en_sents = [re.sub(r"[,.\[\]'?|\"-()，。‘’“”？：；;:<>《》]", "", line) for line in codecs.open(en_train, "r", "utf-8").readlines() if line.strip()]
	x,y = sentence2id(ch_sents, en_sents)
	return x,y

def load_test_data():
	"""生成测试数据"""
	ch_sents = [re.sub(r"[,.\[\]'?|\"-()，。‘’“”？：；;:<>《》]", "", line) for line in codecs.open(ch_test, "r", "utf-8").readlines() if line.strip()]
	en_sents = [re.sub(r"[,.\[\]'?|\"-()，。‘’“”？：；;:<>《》]", "", line) for line in codecs.open(en_test, "r", "utf-8").readlines() if line.strip()]
	x, y = sentence2id(ch_sents, en_sents)
	return x,y

def get_batch_data():
	"""生成一个batch的数据"""
	x, y = load_train_data()
	num_batch = len(x)//batchsize
	x, y = tf.convert_to_tensor(x), tf.convert_to_tensor(y)
	input_queues = tf.train.slice_input_producer([x,y])
	x,y = tf.train.shffle_batch(input_queues, batch_size=batchsize, num_threads=8, capacity=batchsize*64, min_after_dequeue=batchsize*32)
	return x, y, num_batch

if __name__ == '__main__':
	make_vocab(ch_train, "ch_words.txt")
	make_vocab(en_train, "en_words.txt")