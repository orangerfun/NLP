import sys
from collections import Counter
import numpy as np
import pandas as pd
import tensorflow.contrib.keras as kr

if sys.version_info[0]>2:
	is_py3=True
else:
	reload(sys)
	sys.setdefaultencoding("utf-8")
	is_py3=False

def native_content(word,encoding="utf-8"):
	if not is_py3:
		return word.encode(encoding)
	else:
		return word

def open_file(filename,mode="r"):
	if is_py3:
		return open(filename,mode,encoding="utf-8",errors="ignore")
	else:
		return open(filename,mode)

def read_file(filename):
	contents,labels=[],[]
	with open_file(filename) as f:
		for line in f:
			try:
				label,content=line.strip().split("\t")
				if content:
					contents.append(list(content))
					labels.append(label)
			except:
				pass
	return contents,labels

def build_vocab(train_dir,vocab_dir,vocab_size=5000):
	'''
	制作词汇表
	'''
	data_train,_=read_file(train_dir)
	all_data=[]
	for content in data_train:
		all_data.extend(content)
	counter=Counter(all_data)		#返回的是字典
	count_pairs=count.most_common(vocab_size-1)		#返回的是元素是tuple的列表
	words,_=list(zip(*count_pairs))		#返回单词和词频
	words=["<PAD>"]+list(words)
	open_file(vocab_dir,mode="w").write("\n".join(words)+"\n")

def read_vocab(vocab_dir):
	with open_file(vocab_dir) as fp:
		words=[_.strip() for _ in fp.readlines()]
	word_to_id=dict(zip(words,range(len(words))))
	return words,word_to_id

def read_category():
	categories=['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
	cat_to_id=dict(zip(categories,range(len(categories))))
	return categories,cat_to_id

def id_to_words(content,words):
	return "".join(words[x] for x in content)

def process_file(filename,word_to_id,cat_to_id,max_length=600):
	'''将文件转换成数字表示'''
	contents,labels=read_file(filename)
	data_id,label_id=[],[]

	for i in range(len(contents)):
		data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
		label_id.append(cat_to_id[labels[i]])

	x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length)		# 将句子都变成600大小的句子，超过600的从后边开始数，去除前边的
	y_pad=kr.utils.to_categorical(label_id,num_classes=len(cat_to_id))		# 将标签转换为one-hot表示
	return x_pad,y_pad

def batch_iter(x,y,batch_size=64):
	data_len=len(x)
	num_batch=int((data_len-1)/batch_size)+1
	indices=np.random.permutation(np.arange(data_len))
	x_shuffle=x[indices]
	y_shuffle=y[indices]

	for i in range(num_batch):
		start_id=i*batch_size
		end_id=min((i+1)*batch_size,data_len)
		yield x_shuffle[start_id:end_id],y_shuffle[start_id:end_id]
		

