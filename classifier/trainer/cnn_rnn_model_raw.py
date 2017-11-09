from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib as tf_contrib

import utils
import base_model


def get_activation(inputs, act, **kwargs):
	if act == 'relu':
		return tf.nn.relu(inputs)
	elif act == 'tanh':
		return tf.nn.tanh(inputs)
	elif act == 'softmax':
		return tf.nn.softmax(inputs)
	else:
		return inputs


def conv_layer(training,
			   inputs,
			   input_shape,
			   filters,
			   kernel_size,
			   conv_num=1,
			   use_bias=False,
			   activation='relu',
			   pooling=None,
			   batch_norm=True,
			   padding='valid', dropout=0.0, stride=1):
	assert len(input_shape) == 4

	def compute_output_length(input_length, filter_size, padding, stride, dilation=1):
		if input_length is None:
			return None
		assert padding in {'same', 'valid', 'full', 'causal'}
		dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
		if padding == 'same':
			output_length = input_length
		elif padding == 'valid':
			output_length = input_length - dilated_filter_size + 1
		elif padding == 'causal':
			output_length = input_length
		elif padding == 'full':
			output_length = input_length + dilated_filter_size - 1
		return (output_length + stride - 1) // stride

	feed_input = inputs
	shape = list(input_shape)

	for i in range(conv_num):
		feed_input = tf.layers.conv2d(feed_input,
									  filters=filters,
									  kernel_size=[kernel_size] * 2,
									  strides=[stride] * 2, padding=padding, use_bias=use_bias,
									  kernel_initializer=tf.contrib.layers.xavier_initializer())

		if batch_norm:
			# conv = BatchNormalization()(conv)
			# model.add(BatchNormalization())
			feed_input = tf.layers.batch_normalization(feed_input, training=training)
		feed_input = get_activation(feed_input, activation)

		shape[1:-1] = [compute_output_length(input_length=shape[1],
												   filter_size=kernel_size,
												   padding=padding,
												   stride=stride, )] * 2
		shape[-1] = filters

	return feed_input, shape


def dense_layer(training, inputs, input_shape, units,
				activation=None,
				use_bias=True,
				kernel_initializer=tf_contrib.layers.xavier_initializer(),
				trainable=True,
				name=None,
				batch_norm=True,
				reuse=None):
	print('Dense layer: {}'.format(units))
	inputs = tf.layers.dense(inputs,
							 units=units,
							 use_bias=use_bias,
							 trainable=trainable,
							 kernel_initializer=kernel_initializer,
							 reuse=reuse, name=name)
	if batch_norm:
		inputs = tf.layers.batch_normalization(inputs, training=training)

	inputs = get_activation(inputs, activation)

	assert input_shape and len(input_shape) >= 2
	assert input_shape[-1]
	output_shape = list(input_shape)
	output_shape[-1] = units

	return inputs, output_shape


def cnn_block(training,
			  inputs,
			  input_shape,
			  activation='relu',
			  pooling=None,
			  use_bias=False,
			  batch_norm=True,
			  padding='valid',
			  print_fn=print,
			  ):
	conv_configs = [
		{'filters': 16, 'conv_num': 2, 'kernel_size': 4, 'stride': 1},
		{'filters': 16, 'conv_num': 2, 'kernel_size': 4, 'stride': 2},
		{'filters': 16, 'conv_num': 2, 'kernel_size': 4, 'stride': 1},
	]
	dense_units = 32

	for i, config in enumerate(conv_configs):
		inputs, input_shape = conv_layer(training=training,
										 inputs=inputs,
										 input_shape=input_shape,
										 filters=config['filters'],
										 conv_num=config['conv_num'],
										 kernel_size=config['kernel_size'],
										 stride=config['stride'],
										 use_bias=use_bias,
										 padding=config.get('padding', padding),
										 pooling=pooling,
										 batch_norm=batch_norm,
										 activation=activation, )
		print('conv {}, {}, {}'.format(i, config, input_shape))

	# flatten
	input_shape = [input_shape[0], np.prod(input_shape[1:])]
	print('_flatten {}'.format(input_shape))
	inputs = tf.reshape(inputs, input_shape)

	# dense
	inputs, input_shape = dense_layer(training, inputs, input_shape, dense_units, 'relu', use_bias,
									  batch_norm=batch_norm)

	return inputs, input_shape


def time_distributed(training,
					 inputs,
					 input_shape, applier, applier_config):
	shape = list(input_shape)

	time_length = shape[1]
	inside_shape = list([-1, ] + shape[2:])
	print('inside_shape {}'.format(inside_shape))
	inputs = tf.reshape(inputs, inside_shape)
	inputs, output_shape = applier(training=training, inputs=inputs, input_shape=inside_shape, **applier_config)

	inputs = tf.reshape(inputs, [-1, time_length] + output_shape[1:])
	output_shape = [-1, time_length] + output_shape[1:]

	return inputs, output_shape


def bi_lstm(training,
			inputs,
			input_shape,
			num_hidden,
			forget_bias=1.0,
			activation='tanh',
			print_fn=print, ):

	timesteps = input_shape[1]
	inputs = tf.unstack(inputs, timesteps, 1)

	with tf.variable_scope('forward_pass'):
		fw_cell = rnn.LSTMCell(num_hidden,
							   # activation=activation,
							   forget_bias=forget_bias,
							   initializer=tf_contrib.layers.xavier_initializer()
							   )

	with tf.variable_scope('backward_pass'):
		bw_cell = rnn.LSTMCell(num_hidden,
							   # activation=activation,
							   forget_bias=forget_bias,
							   initializer=tf_contrib.layers.xavier_initializer()
							   )


	# Get lstm cell output
	with tf.variable_scope('birnn') as bi_scope:
		# try:
		inputs, _, _ = rnn.static_bidirectional_rnn(fw_cell, bw_cell, inputs, scope=bi_scope,
													dtype=tf.float32)
		print('here success')
		# except Exception:  # Old TensorFlow version only returns outputs not states
		# 	inputs = rnn.static_bidirectional_rnn(fw_cell, bw_cell, inputs,scope=bi_scope,
		# 										  dtype=tf.float32)


	shape = [-1, 2 * num_hidden]

	# inputs, shape = dense_layer(training, inputs, shape)
	return inputs[-1], shape


def cnn_rnn_sequential(**kwargs):
	input_shape = [None, 15, 64, 64, 3]
	inputs = tf.placeholder(tf.float32, shape=input_shape, name='inputs')
	training = tf.placeholder(tf.bool, name='training')
	labels = tf.placeholder(tf.float32, shape=[None, 6])

	num_hidden = 32

	shape = list([-1] + input_shape[1:])

	feed_input = inputs

	feed_input = tf.divide(tf.subtract(feed_input, 175.5), 175.5)

	feed_input, shape = time_distributed(training=training, inputs=feed_input, input_shape=shape,
										 applier=cnn_block, applier_config={
		})

	feed_input, shape = bi_lstm(training, feed_input, shape, num_hidden=num_hidden,)

	# output
	print('{} -- {}'.format(feed_input, shape))
	feed_input, shape = dense_layer(training=training,
									inputs=feed_input,
									input_shape=shape,
									units=6,
									batch_norm=False)


	outputs = get_activation(feed_input, 'softmax')
	predictions = tf.argmax(outputs, axis=1)
	correct_predictions = tf.equal(predictions, tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

	softmax_logits = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=feed_input)
	loss = tf.reduce_mean(softmax_logits)
	optimizer = tf.train.AdamOptimizer(learning_rate=kwargs.get('learning_rate', 0.001))

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = optimizer.minimize(loss)

	return {
		'inputs': inputs,
		'training': training,
		'labels': labels,
		'outputs': outputs,
		'predictions': predictions,
		'accuracy': accuracy,
		'loss': loss,
		'optimizer': optimizer,
		'train_op': train_op
	}


class CNN_RNN_Sequential_raw(base_model.ClassiferTfModel):
	def __init__(self, config_file, job_dir, checkpoint_path, print_f=print, sequence_length=15, input_dim=64,
				 label_set=None, batch_norm=False):
		super().__init__(config_file, job_dir, checkpoint_path, print_f, sequence_length, input_dim, label_set,
						 batch_norm)
		self.model_ops = None
		self.session = None

	def initialize(self):
		assert self.model_ops is not None
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())

	def end_train(self):
		if self.session is not None:
			self.session.close()
			self.session = None

	def compile(self, **kwargs):
		super().compile(**kwargs)
		self.model_ops = cnn_rnn_sequential()

	def predict(self, data):
		# super().predict(data)
		assert self.model_ops is not None
		assert self.session is not None

		return self.session.run(self.model_ops['outputs'], feed_dict={
			self.model_ops['inputs']: data,
			self.model_ops['training']: False
		})

	def fit(self, train_files, test_files=None, batch_size=32, epochs=10, validation_split=0.1, callbacks=None,
			**kwargs):
		# super().fit(train_files, test_files, batch_size, epochs, validation_split, callbacks, **kwargs)

		self.process_training_data(train_files, split=validation_split)
		self.initialize()



if __name__ == '__main__':
	model_ops = cnn_rnn_sequential()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		size = 5
		d = np.random.randint(0, 255, size=[size, 15, 64, 64, 3])
		eye = np.eye(6)
		l = np.array([eye[i] for i in [np.random.randint(0, 6) for j in range(size)]])
		print('{} {}'.format(d.shape, l.shape))

		_, loss, acc = sess.run((model_ops['train_op'], model_ops['loss'], model_ops['accuracy']),
								feed_dict={
									model_ops['inputs']: d,
									model_ops['labels']: l,
									model_ops['training']: True
								})

		print('Loss: {}, acc: {}'.format(loss, acc))


