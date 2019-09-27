# encoding:utf-8

import tensorflow as tf
import numpy as np
import os
from os.path import isfile, join
import re
from random import randint

#载入word列表
wordsList = np.load('wordsList.npy')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
# 载入文本向量
wordVectors = np.load('wordVectors.npy')

print("总共词的个数为：", len(wordsList))
print("wordVectors.shape:", wordVectors.shape)

# os.path.isfile()函数判断某一路径是否为文件；os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
#获取文件名，并放入列表中
pos_files = ['pos/' + f for f in os.listdir('pos/') if isfile(join('pos/', f))]
neg_files = ['neg/' + f for f in os.listdir('neg/') if isfile(join('neg/', f))]
num_words = []

#读取每个文件并分词获取每个文件中的词的个数，并将其存入列表当中
for pf in pos_files:
    with open(pf, "r", encoding='utf-8') as f:
        line = f.readline()
        #计算每个样本的长度
        counter = len(line.split())
        num_words.append(counter)
print('正面评价完结')

for nf in neg_files:
    with open(nf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('负面评价完结')

num_files = len(num_words)   #正负样本一共有多少个
print('文件总数', num_files)
print('所有的词的数量', sum(num_words))
print('平均文件词的长度', sum(num_words) / len(num_words))

#数据可视化
def data_show():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams["font.sans-serif"]=["SimHei"]
    mpl.rcParams["font.family"]="sans-serif"
    plt.hist(num_words,50,facecolor="g",alpha=0.5)
    plt.xlabel("序列长度")
    plt.ylabel("频次")
    plt.axis([0,1200,0,8000])
    plt.show()

#找出非字符(去除标点符号等)
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
num_dimensions = 300  # Dimensions for each word vector

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

max_seq_num = 250     #每个序列规定长度为250个词

#建立索引矩阵
def build_ids():
    ids = np.zeros((num_files, max_seq_num), dtype='int32')
    file_count = 0
    for pf in pos_files:
      with open(pf, "r", encoding='utf-8') as f:
        indexCounter = 0
        line = f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
          try:
            ids[file_count][indexCounter] = wordsList.index(word)
          except ValueError:
            ids[file_count][indexCounter] = 399999  # 未知的词
          indexCounter = indexCounter + 1
          if indexCounter >= max_seq_num:
            break
        file_count = file_count + 1

    for nf in neg_files:
      with open(nf, "r",encoding='utf-8') as f:
        indexCounter = 0
        line = f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
          try:
            ids[file_count][indexCounter] = wordsList.index(word)
          except ValueError:
            ids[file_count][indexCounter] = 399999  # 未知的词语
          indexCounter = indexCounter + 1
          if indexCounter >= max_seq_num:
            break
        file_count = file_count + 1
    np.save('idsMatrix', ids)

batch_size = 24
lstm_units = 64
num_labels = 2
iterations = 50001
lr = 0.001
ids = np.load('idsMatrix.npy')

#取一个batch的训练样本，取的时候从正例中取一个，从负类中取一个...
def get_train_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels

def get_test_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels

#构建网络模型
class Model():
    def __init__(self):
        self.lstm_model()

    def lstm_model(self):
        tf.reset_default_graph()
        self.labels = tf.placeholder(tf.float32, [batch_size, num_labels])         #[24,2]
        self.input_data = tf.placeholder(tf.int32, [batch_size, max_seq_num])      #[24,250]
        self.keep_prob=tf.placeholder(tf.float32)
        data = tf.Variable(
            tf.zeros([batch_size, max_seq_num, num_dimensions]), dtype=tf.float32)  #[24,250,300]
        data = tf.nn.embedding_lookup(wordVectors, self.input_data)

        lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)

        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=self.keep_prob)
        value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)  #[batchsize,max_step,input]--->[24,250,64]

        weight = tf.Variable(tf.truncated_normal([lstm_units, num_labels]))  #[64,2]
        bias = tf.Variable(tf.constant(0.1, shape=[num_labels]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)   #取最后一个时序
        prediction = (tf.matmul(last, weight) + bias)   #[24 2]

        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

#训练数据
def train():
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        startepoch=0
        init = tf.global_variables_initializer()
        sess.run(init)
        if os.path.exists("models") and os.path.exists("models/checkpoint"):
            kpt=tf.train.latest_checkpoint("./models")
            saver.restore(sess,kpt)
            idx = kpt.find("-")
            startepoch=int(kpt[idx+1:])
        for step in range(iterations):
            if step<startepoch:
                continue
            next_batch,next_batch_labels = get_train_batch()
            sess.run(model.optimizer,feed_dict={model.input_data:next_batch, model.labels:next_batch_labels, model.keep_prob:0.6})
            if step % 500 == 0:
                print("step:", step, " 正确率:", (sess.run(model.accuracy, feed_dict={model.input_data: next_batch, model.labels: next_batch_labels, model.keep_prob:1})))
                if not os.path.exists("models"):
                    os.mkdir("models")
                save_path = saver.save(sess, "models/model.ckpt",global_step=step)
                print("Model saved in path: %s" % save_path)
#测试数据
def test():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        total_acc=0
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint("./models"))
        for step in range(10):
            next_batch,next_batch_labels=get_test_batch()
            acc=sess.run(model.accuracy,feed_dict={model.input_data: next_batch, model.labels: next_batch_labels,model.keep_prob:1.0})
            print("第%d次测试准确度："%(step+1),acc)
            total_acc+=acc/10
        print("平均正确率为：",total_acc)


if __name__ == "__main__":
    data_show()
    model=Model()
    option=input("请选择训练还是测试：")
    if option=="train":
        train()
    else:
        test()



