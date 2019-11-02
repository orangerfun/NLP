# -*- encoding:utf-8 -*-

"""
构建模型
@author: orangerfun@gamil.com
"""

import tensorflow as tf
import numpy as np

def LN(inputs, epsilon=1e-8):
	"""
	Layer Normalization
	"""
	temp = inputs.get_shape()[-1]
	gamma = tf.Variable(tf.ones(temp))
	beta = tf.Variable(tf.zeros(temp))
	mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
	output = (inputs-mean)/tf.sqrt(variance+epsilon)
	output = gamma*output + beta
	return output


def embedding(inputs, vocab_size, embedding_dim, pad, scale):
	'''
	词嵌入
	'''
	lookup_table = tf.Variable(dtype=tf.float32, shape=[vocab_size, embedding_dim], initializer=tf.contrib.layers.xavier_initializer())
	lookup_table = tf.concat((tf.zeros(shape=[1,embedding_dim]), lookup_table[1:,:]), 0)
	output = tf.nn.embedding_lookup(lookup_table, inputs)
	output = output/tf.sqrt(embedding_dim)
	return output


def position_embedding(inputs, vocab_size, embedding_dim):
	"""
	构建位置向量
	"""
	batchsize, maxlen = inputs.get_shape().as_list()
	pos_ind = tf.tile(tf.expand_dim(tf.range(maxlen),0),[batchsize,1])
	pos_enc = [[pos/np.power(10000, 2.*i/num_units) for i in range(embedding_dim)] for pos in range(maxlen)]
	pos_enc[:,0::2] = np.sin(pos_enc[:,0::2])
	pos_enc[:,1::3] = np.cos(pos_enc[:,1::3])
	lookup_table = tf.convert_to_tensor(pos_enc)
	lookup_table = tf.concat((tf.zeros(shape=[1,embedding_dim]),lookup_table[1:,:]),0)
	output = tf.nn.embedding_lookup(lookup_table, pos_ind)
	output = output/tf.sqrt(embedding_dim)
	return output

def multihead_attention(queries, keys, embedding_dim=512, num_head=8, dropout_rate=0, is_training=True, future_blind=False):
	Q = tf.layers.dense(queries, embedding_dim, activation=tf.nn.relu)
	K = tf.layers.dense(keys, embedding_dim, activation=tf.nn.relu)
	V = tf.layers.dense(keys, embedding_dim, activation=tf.nn.relu)

	Q_ = tf.concat(tf.split(Q, num_head, axis=2), axis=0)
	K_ = tf.concat(tf.split(K, num_head, axis=2), axis=0)
	V_ = tf.concat(tf.split(V, num_head, axis=2), axis=0)

	output = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
	output = output/tf.sqrt((K_.get_shape().as_list()[-1]))

# key masking 此处mask来自网络，我也没懂
	key_mask = tf.sign(tf.reduce_sum(tf.abs(keys), axis=2))
	key_mask = tf.tile(key_mask, [num_head,1])
	key_mask = tf.tile(tf.expand_dim(key_mask,1), [1, tf.shape(queries)[1], 1])

	pad = tf.ones_like(output)*(-2**32+1)
	output = tf.where(tf.equal(key_mask, 0), pad, output)

	if future_blind:
		diag_vals = tf.ones_like(output[0, :, :])
		tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
		mask = tf.tile(tf.expand_dim(tril, 0), [tf.shape(output)[0],1,1])
		padding = tf.ones_like(mask)*(-2**32+1)
		output = tf.where(tf.equal(mask, 0), padding, output)
	output = tf.nn.softmax(output)

	#queries masking
	query_mask = tf.sign(tf.reduce_sum(tf.abs(queries), axis=1))
	query_mask = tf.tile(query_mask, [num_head, 1])
	query_mask = tf.tile(tf.expand_dim(query_mask, -1), [1, 1, tf.shape(keys)[1]])
	output = output*query_mask

	output = tf.layers.dropout(output, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
	output = tf.matmul(output, V_)
	output = tf.concat(tf.split(output, num_head, axis=0), axis=2)
	#残差连接
	output = output + queries
	#LN
	output = LN(output)
	return output


def feedforward(inputs, num_units=[2018,512]):
	'''
	前馈神经网络层
	'''
	output = tf.layers.dense(inputs, num_units[0])
	output = tf.layers.dense(output, num_units[1])
	output = output + inputs
	output = LN(output)

def label_smoothing(inputs, epsilon=0.1):
	"""
	标签平滑归一化LSR
	"""
	k = inputs.get_shape().as_list()[-1]
	return ((1-epsilon)*inputs) + (epsilon/k)


















