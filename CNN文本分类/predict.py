import os
import tensorflow as tf
import tensorflow.contrib.keras as kr
from model import TCNNConfig,TextCNN
from data_load import read_category,read_vocab

vocabdir="./data/cnews.vocab.txt"
save_path="./checkpoints/textcnn/best_validation"

class CnnModel:
	def __init__(self):
		self.config=TCNNConfig()
		self.categories,self.cat_to_id=read_category()
		self.words,self.word_to_id=read_vocab(vocabdir)
		self.config.vocab_size=len(self.words)
		self.model=TextCNN(self.config)

		self.session=tf.Session()
		self.session.run(tf.global_variables_initializer())
		saver=tf.train.Saver()
		saver.restore(sess=self.session,save_path=save_path)

	def predict(self,message):
		content=message
		data=[self.word_to_id[x] for x in content if x in self.word_to_id]
		feed_dict={self.model.input_x:kr.preprocessing.sequence.pad_sequences([data],self.config.seq_length),self.model.keep_prob:1.0}
		y_pred_cls=self.session.run(self.model.y_pred_cls,feed_dict=feed_dict)
		return self.categories[y_pred_cls[0]]

if __name__=="__main__":
	cnn_model=CnnModel()
	test_demo=['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',\
                 '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00',\
                 '“双宋离婚”霸占了有一个星期多的头条，目前“男宋”宋仲基和“女宋”宋慧乔都陆续开始营业了，从宋仲基于日前同剧组合影的照片流出来看，他并没有受离婚影响太多，露面的时候也面带笑容',\
                 '王者的英雄中却有几个非常招恨的英雄，非得杀几次解气才行的那种，说到这里你们会想到谁呢？今天就让我们盘点一下峡谷中最招恨的几个英雄吧']
	for i in test_demo:
		print(cnn_model.predict(i))






