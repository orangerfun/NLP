import tensorflow as tf

class TCNNConfig(object):
	embedding_dim=64	#词向量的长度
	seq_length=600   #将每个句子的长度都设置成一样的长度，序列长度
	num_classes=10
	num_filters=256
	kernel_size=5
	vocab_size=3000

	hidden_dim=128
	dropout_keep_prob=0.5
	learning_rate=1e-3

	batch_size=64
	num_epochs=10

	print_per_batch=100		# 每多少轮输出一次结果
	save_per_batch=10		# 每多少轮存入tensorboard

class TextCNN(object):
	def __init__(self,config):
		self.config=config
		self.input_x=tf.placeholder(tf.int32,[None,self.config.seq_length])		#[batchsize,seq_length][64,600]
		self.input_y=tf.placeholder(tf.int32,[None,self.config.num_classes])	
		self.keep_prob=tf.placeholder(tf.float32,name="keep_prob")
		self.cnn()

	def cnn(self):
		embedding=tf.get_variable("embedding",[self.config.vocab_size,self.config.embedding_dim])	#每个单词的词向量
		embedding_inputs=tf.nn.embedding_lookup(embedding,self.input_x)		#根据input_x的索引在embedding中查找对应词向量[64,600,64]

		with tf.name_scope("cnn"):
			conv=tf.layers.conv1d(embedding_inputs,self.config.num_filters,self.config.kernel_size,name="conv")
			gmp=tf.reduce_max(conv,reduction_indices=1,name="gmp")		#[64,256]

		#全连接层
		with tf.name_scope("score"):
			fc=tf.layers.dense(gmp,self.config.hidden_dim,name="fc1")	#[64,128]
			fc=tf.contrib.layers.dropout(fc,self.keep_prob)
			fc=tf.nn.relu(fc)

			self.logits=tf.layers.dense(fc,self.config.num_classes,name="fc2") #[64,10]
			self.y_pred_cls=tf.argmax(tf.nn.softmax(self.logits),1)

		with tf.name_scope("optimize"):
			cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
			self.loss=tf.reduce_mean(cross_entropy)
			self.optim=tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

		with tf.name_scope("accuracy"):
			correct_pred=tf.equal(self.y_pred_cls,tf.argmax(self.input_y,1))
			self.acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))













