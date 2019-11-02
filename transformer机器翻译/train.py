import tensorflow as tf
import preprocess as pre
from preprocess import get_batch_data, load_vocab
import os, codecs
from tqdm import tqdm
from model import*

class Graph():
	def __init__(self, is_training=True):
		self.graph = Graph()
		with self.graph as_default():
			if is_training:
				self.x, self.y = get_batch_data()
			else:
				self.x = tf.placeholder(tf.int32, shape=(None, pre.maxlen))
				self.y = tf.placeholder(tf.int32, shape=(None, pre.maxlen))

			self.decoder_input = tf.concat((tf.ones_like(self.y[:,:1])*2,self.y[:,:-1]),axis=-1)
			en2id, id2en = load_vocab("./data/en_words.txt")
			ch2id, id2ch = load_vocab("./data/ch_words.txt")

			with tf.variable_scope("encoder"):
				self.enc = embedding(self.x, vocab_size=len(en2id), num_units=pre.embedding)
				key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.enc),axis=-1)), -1)
				self.enc = self.enc + position_embedding(self.x, embedding_dim=pre.embedding)
				self.enc = self.enc*key_masks
				self.enc = tf.layers.dropout(self.enc, rate=0.1, training=tf.convert_to_tensor(is_training))

				for i in range(6):
					self.enc = multihead_attention(queries=self.enc,
												keys=self.enc,
												embedding_dim=pre.embedding,
												num_head=8,
												dropout_rate=0.1,
												is_training=is_training,
												future_blind=False)
					self.enc = feedforward(inputs=self.enc)

			with tf.variable_scope("decode"):
				self.dec = embedding(inputs=self.y, vocab_size=len(ch2id), embedding_dim=pre.embedding)
				key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.dec), axis=-1)),-1)
				self.dec = self.dec + position_embedding(self.y, embedding_dim=pre.embedding)
				self.dec = self.dec*key_masks
				self.dec = tf.layers.dropout(self.dec, rate=0.1, training=tf.convert_to_tensor(is_training))

				for i in range(6):
					self.dec = multihead_attention(queries=self.dec,
													keys=self.dec,
													embedding_dim=pre.embedding,
													num_head=8,
													dropout_rate=0.1,
													is_training=is_training,
													future_blind=True)
					self.dec = multihead_attention(queries=self.dec,
													keys=self.enc,
													embedding_dim=pre.embedding,
													num_head=8,
													dropout_rate=0.1,
													is_training=is_training,
													future_blind=False)
					self.dec = feedforward(self.dec)

			self.logits = tf.layers.dense(self.dec, len(ch2id))
			self.preds = tf.to_int32(tf.argmax(self.logits,axis=-1))
			self.istarget = tf.to_float(tf.not_equal(self.y, 0))
			self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds,self.y))*self.istarget)/(tf.reduce_sum(self.istarget))
			if is_training:
				self.y_smoothed = label_smoothing(tf.onehot(self.y, depth=len(ch2id)))
				self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
				self.mean_loss = tf.reduce_sum(self.loss*self.istarget)/tf.reduce_sum(self.istarget)
				self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001,
														beta1=0.9,
														beta2=0.98,
														epsilon=1e-8)
				self.opt = self.optimizer.minimize(self.mean_loss)

if __name__ == '__main__':
	





