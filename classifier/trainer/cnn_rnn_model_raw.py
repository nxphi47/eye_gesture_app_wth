from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib as tf_contrib
import json
import pprint
import utils
import base_model


def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)

def neurons_histogram(var, name):
	tf.summary.histogram(name=name, values=var)

def neurons_scalar(var, name):
	tf.summary.scalar(name=name, tensor=var)
	tf.summary.histogram(name=name, values=var)

def get_activation(inputs, act, name=None, **kwargs):
	if act == 'relu':
		return tf.nn.relu(inputs, name=name)
	elif act == 'tanh':
		return tf.nn.tanh(inputs, name=name)
	elif act == 'softmax':
		return tf.nn.softmax(inputs, name=name)
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
			   padding='valid',
			   dropout=0.0,
			   stride=1,
			   name=None):
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
									  strides=[stride] * 2, padding=padding,
									  use_bias=use_bias,
									  kernel_initializer=tf.contrib.layers.xavier_initializer(),
									  name='{}_conv_{}'.format(name, i)
									  )

		if batch_norm:
			# conv = BatchNormalization()(conv)
			# model.add(BatchNormalization())
			feed_input = tf.layers.batch_normalization(feed_input, training=training, name='{}_bn_{}'.format(name, i))
		feed_input = get_activation(feed_input, activation, name='{}_act_{}'.format(name, i))

		neurons_histogram(feed_input, '{}_convbnact_{}'.format(name, i))

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
				reuse=None,
				):
	print('Dense layer: {}'.format(units))
	inputs = tf.layers.dense(inputs,
							 units=units,
							 use_bias=use_bias,
							 trainable=trainable,
							 kernel_initializer=kernel_initializer,
							 reuse=reuse,
							 name='{}_dense'.format(name))
	if batch_norm:
		inputs = tf.layers.batch_normalization(inputs, training=training, name='{}_bn'.format(name))

	inputs = get_activation(inputs, activation, name='{}_act'.format(name))
	neurons_histogram(inputs, '{}_densebnact'.format(name))

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
										 activation=activation,
										 name='conv_{}'.format(i)
										 )
		print('conv {}, {}, {}'.format(i, config, input_shape))

	# flatten
	input_shape = [input_shape[0], np.prod(input_shape[1:])]
	print('_flatten {}'.format(input_shape))
	inputs = tf.reshape(inputs, input_shape)

	# dense
	inputs, input_shape = dense_layer(training,
									  inputs,
									  input_shape, dense_units, 'relu', use_bias,
									  batch_norm=batch_norm, name='conv_fc')

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
							   initializer=tf_contrib.layers.xavier_initializer(),
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
	neurons_histogram(inputs[-1], 'birnn_final')

	# inputs, shape = dense_layer(training, inputs, shape)
	return inputs[-1], shape


def cnn_rnn_sequential(**kwargs):
	input_shape = [None, 15, 64, 64, 3]
	inputs = tf.placeholder(tf.float32, shape=input_shape, name='inputs')
	training = tf.placeholder(tf.bool, name='training')
	labels = tf.placeholder(tf.float32, shape=[None, 6], name='labels')

	num_hidden = 32

	shape = list([-1] + input_shape[1:])

	feed_input = inputs

	feed_input = tf.divide(tf.subtract(feed_input, 175.5), 175.5)

	feed_input, shape = time_distributed(training=training, inputs=feed_input, input_shape=shape,
										 applier=cnn_block, applier_config={
		})

	feed_input, shape = bi_lstm(training, feed_input, shape, num_hidden=num_hidden, )

	# output
	print('{} -- {}'.format(feed_input, shape))
	feed_input, shape = dense_layer(training=training,
									inputs=feed_input,
									input_shape=shape,
									units=6,
									batch_norm=False, name='fc_out')

	outputs = get_activation(feed_input, 'softmax', name='outputs')
	# neurons_scalar(outputs, 'outputs_scalar')

	predictions = tf.argmax(outputs, axis=1, name='predictions')

	correct_predictions = tf.equal(predictions, tf.argmax(labels, 1))

	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
	neurons_scalar(accuracy, 'accuracy_scalar')

	softmax_logits = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=feed_input)

	loss = tf.reduce_mean(softmax_logits)
	neurons_scalar(loss, 'loss_scalar')
	optimizer = tf.train.AdamOptimizer(learning_rate=kwargs.get('learning_rate', 0.001))

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = optimizer.minimize(loss)

	summaries = tf.summary.merge_all()
	# train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
	# 									 sess.graph)
	# test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

	return {
		'inputs': inputs,
		'training': training,
		'labels': labels,
		'outputs': outputs,
		'predictions': predictions,
		'accuracy': accuracy,
		'loss': loss,
		'optimizer': optimizer,
		'train_op': train_op,
		'summaries': summaries
	}


class CNN_RNN_Sequential_raw(base_model.ClassiferTfModel):
	def __init__(self, config_file=None, job_dir=None, checkpoint_path=None, print_f=print, sequence_length=15, input_dim=64,
				 label_set=None, batch_norm=False):
		super().__init__(config_file, job_dir, checkpoint_path, print_f, sequence_length, input_dim, label_set,
						 batch_norm)
		self.model_ops = None
		self.session = None
		self.tfboard_train_writer = None
		self.tfboard_test_writer = None

	def load_model_from_savedmodel(self, export_path):
		self.compile()
		self.initialize(init_weights=False, init_tfboard=False)
		utils.session_from_savedmodel(session=self.session, export_dir=export_path)

	def load_config(self):
		if self.config_file is not None:
			self.config = json.load(open(self.config_file, 'r'))
			pprint.pprint(self.config)
		if self.config is not None and "cnn_block" in self.config:
			self.batch_norm = self.config['cnn_block'].get('batch_norm', True)
		else:
			self.batch_norm = True

	def init_tensorboard(self):
		assert self.model_ops is not None
		assert self.session is not None
		self.tfboard_train_writer = tf.summary.FileWriter(os.path.join(self.job_dir, 'train'), self.session.graph)
		self.tfboard_test_writer = tf.summary.FileWriter(os.path.join(self.job_dir, 'test'), self.session.graph)

	def initialize(self, init_weights=True, init_tfboard=True):
		assert self.model_ops is not None
		self.session = tf.Session()
		if init_weights:
			self.session.run(tf.global_variables_initializer())
		if init_tfboard:
			self.init_tensorboard()

	def end_train(self):
		if self.session is not None:
			self.session.close()
			self.session = None

	def compile(self, **kwargs):
		super().compile(**kwargs)
		self.load_config()
		self.model_ops = cnn_rnn_sequential()

	def predict(self, data):
		# super().predict(data)
		assert self.model_ops is not None
		assert self.session is not None

		return self.session.run(self.model_ops['outputs'], feed_dict={
			self.model_ops['inputs']: data,
			self.model_ops['training']: False
		})

	def eval(self, data, labels):
		assert self.model_ops is not None
		assert self.session is not None

		return self.session.run([self.model_ops['outputs'], self.model_ops['summaries']], feed_dict={
			self.model_ops['inputs']: data,
			self.model_ops['labels']: labels,
			self.model_ops['training']: False
		})

	def test_on_trained(self, test_files):
		if test_files is not None:
			self.print_f('-- Perform Testing --')
			if isinstance(test_files, (list, tuple)):
				test_files = test_files[0]
			X, y = utils.load_npz(test_files)
			print('Test Shape x: {}'.format(self.X.shape))
			print('Test Shape y: {}'.format(self.y.shape))
			assert self.session is not None
			pred_val = np.argmax(self.predict(X), axis=1)
			true_val = np.argmax(y, axis=1)

			utils.report(true_val, pred_val, self.label_set, print_fn=self.print_f)



		# model_name = 'eye_final_model.hdf5'
		# self.model.save(os.path.join(self.job_dir, model_name))
		#
		# # Convert the Keras model to TensorFlow SavedModel
		self.print_f('Save model to {}'.format(self.job_dir))
		# utils.to_savedmsesodel(self.model, os.path.join(self.job_dir, 'export'))
		utils.session_to_savedmodel(self.session, self.model_ops['inputs'], self.model_ops['outputs'], os.path.join(self.job_dir, 'export'))

	def on_epoch_end(self, epoch, **kwargs):
		assert self.session is not None
		if epoch > 0 and (epoch % kwargs.get('eval_freq', 4) == 0 or epoch == kwargs.get('epochs', 15)):
			preds, summaries = self.eval(self.X_val)
			pred_val = np.argmax(preds, axis=1)

			if self.true_val is None:
				self.true_val = np.array(np.argmax(self.y_val, axis=1))
			utils.report(self.true_val, pred_val, self.label_set, print_fn=self.print_f)
			self.tfboard_test_writer.add_summary(summaries, epoch)

	def mid_eval(self, epoch, step, **kwargs):
		assert self.session is not None
		sum_step = kwargs['steps'] * epoch + step
		self.tfboard_train_writer.add_summary(kwargs['train_summaries'], sum_step)

		# if epoch > 0 and (epoch % kwargs.get('eval_freq', 4) == 0 or epoch == kwargs.get('epochs', 15)):
		preds, summaries = self.eval(self.X_val, self.y_val)
		pred_val = np.argmax(preds, axis=1)

		if self.true_val is None:
			self.true_val = np.array(np.argmax(self.y_val, axis=1))
		utils.report(self.true_val, pred_val, self.label_set, print_fn=self.print_f)
		self.tfboard_test_writer.add_summary(summaries, epoch)

	def fit(self, train_files,
			test_files=None,
			batch_size=32,
			epochs=10,
			validation_split=0.1,
			callbacks=None,
			**kwargs):
		# super().fit(train_files, test_files, batch_size, epochs, validation_split, callbacks, **kwargs)
		eval_per_epoch = 30

		self.process_training_data(train_files, split=validation_split)
		self.initialize()

		train_idx = np.arange(0, len(self.y))
		steps = len(self.y) // batch_size - 1
		assert self.session is not None
		for e in range(epochs):
			self.print_f('Epoch {}'.format(e))
			np.random.shuffle(train_idx)

			for s in range(steps):
				inputs = self.X[train_idx[s * batch_size: (s + 1) * batch_size]]
				labels = self.y[train_idx[s * batch_size: (s + 1) * batch_size]]

				_, loss, acc, summaries = self.session.run(
					[self.model_ops['train_op'], self.model_ops['loss'], self.model_ops['accuracy'], self.model_ops['summaries']],
					feed_dict={
						self.model_ops['inputs']: inputs,
						self.model_ops['labels']: labels,
						self.model_ops['training']: True
					})
				if s % eval_per_epoch == 0:
					self.print_f('--E({}) -- step ({}) -- loss ({}) -- acc ({})'.format(e, s, loss, acc))
					# self.tfboard_train_writer.add_summary(summaries, steps * e + s)
					self.mid_eval(e, s, steps=steps, train_loss=loss, train_acc=acc, train_summaries=summaries)


			# self.on_epoch_end(e, eval_freq=kwargs.get('eval_freq', 4), epochs=epochs)

		self.test_on_trained(test_files=test_files)
		self.end_train()


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
