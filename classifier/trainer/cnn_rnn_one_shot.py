#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from sklearn.model_selection import train_test_split
# from keras.layers.merge import Concatenate, Add, Dot, Multiply
import glob
import os
from PIL import Image
import numpy as np
from keras import backend as K
from keras.layers import Input, Activation, Conv2D, Dense, Dropout, Concatenate, Reshape, \
	LSTM, Bidirectional, TimeDistributed, MaxPooling2D, BatchNormalization, AveragePooling2D, Flatten
from keras.models import Model, Sequential
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from keras import backend as K
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import utils
import cnn_rnn_model

# all assume 2 dim
def fc_layer(inputs, units=128, activation='relu'):
	flatten = Flatten()(inputs)
	return Dense(units=units, activation=activation)(flatten)

def hoz_temporal_conv(inputs, filters=8, kernel_size=4, pooling=2):
	out_shape = list(inputs._keras_shape)[1:] + [1]
	print (out_shape)
	inputs = Reshape(out_shape)(inputs)
	conv = Conv2D(filters=filters, kernel_size=(2, kernel_size))(inputs)
	pooling = MaxPooling2D(pool_size=(1, pooling))(conv)
	flatten = Flatten()(pooling)
	return flatten

def one_shot_model(print_f=print,
				 sequence_length=15,
				 input_dim=64,
				 label_set=None
				 ):

	if label_set is None:
		label_set = ['left', 'right', 'up', 'down', 'center', 'double_blink']

	cnn_rnn_sequential_model = cnn_rnn_model.CNN_RNN_Sequential_model(print_f, sequence_length, input_dim, label_set)

	# TODO: building the one shot model
	inputs_train = Input(shape=(sequence_length, input_dim, input_dim, 3))
	inputs_target = Input(shape=(sequence_length, input_dim, input_dim, 3))
	inputs_train_reshape = Reshape((1, sequence_length, input_dim, input_dim, 3))(inputs_train)
	inputs_target_reshape = Reshape((1, sequence_length, input_dim, input_dim, 3))(inputs_target)
	inputs = Concatenate(axis=1)([inputs_train_reshape, inputs_target_reshape])

	print ('inputs shape: {}'.format(inputs._keras_shape))

	cnn_rnn_fc_layer = cnn_rnn_sequential_model.layers[-2]

	cnn_rnn_fc_model = Model(inputs=cnn_rnn_sequential_model.inputs, outputs=cnn_rnn_fc_layer.output)

	# print ('cnn_rnn_last: {} {}'.format(cnn_rnn_fc_layer, cnn_rnn_fc_layer.get_shape()))
	cnn_rnn_time_distributed = TimeDistributed(cnn_rnn_fc_model)(inputs)

	print ('distributed: {}'.format(cnn_rnn_time_distributed._keras_shape))
	# TODO: output is 2 fc layer
	layer_list = [
		fc_layer(cnn_rnn_time_distributed, units=64, activation='relu'),
		hoz_temporal_conv(cnn_rnn_time_distributed, filters=4)
	]

	concat = Concatenate(axis=1)(layer_list)
	print ('concat shape: {}'.format(concat._keras_shape))

	dropout = Dropout(0.5)(concat)

	output_layer = Dense(1, activation='sigmoid')(dropout)

	one_shot = Model(inputs=[inputs_train, inputs_target], outputs=output_layer)

	return cnn_rnn_sequential_model, one_shot


class CNN_RNN_ONE_SHOT():
	def __init__(self, print_f=print,
				 sequence_length=15,
				 input_dim=64,
				 label_set=None
				 ):
		self.print_f = print_f
		self.sequence_length = sequence_length
		self.input_dim = input_dim
		self.label_set = label_set

		self.model = None
		self.one_shot_model = None
		self.feed_dict = None
		self.feed_dict_train = None
		self.feed_dict_test = None
		self.X = None
		self.y = None
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		self.eye = np.eye(len(self.label_set))

	def compile(self, **kwargs):
		self.model, self.one_shot_model = one_shot_model(self.print_f, self.sequence_length, self.input_dim, self.label_set)
		self.compile_classifier()
		self.compile_one_shot()
		self.one_shot_model.summary(print_fn=self.print_f)

	def compile_classifier(self):
		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mae'], )

	def compile_one_shot(self):
		self.one_shot_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], )


	def label_list(self, length, i):
		# np.vstack([np.array([np.array(self.eye[i])] * len(self.feed_dict[k])) for i, k in enumerate(self.feed_dict)])
		return np.array([np.array(self.eye[i])] * length)

	def process_data(self, train_dirs, split):

		X, y, y_raws, label_set = utils.load_dataset(train_dirs, self.label_set, self.sequence_length)
		# self.feed_dict = feed_dict
		# self.feed_dict_train = {}
		# self.feed_dict_test = {}
		# X = np.vstack([feed_dict[k] for k in feed_dict])
		# y = np.vstack([np.array([np.array(self.eye[i])] * len(feed_dict[k])) for i, k in enumerate(feed_dict)])

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=split)

		# for i, k in feed_dict:
		# 	# X_train, X_test, y_train, y_test = train_test_split(feed_dict[k], self.label_list(len(feed_dict[k]), i), test_size=split)
		# 	X_train, X_test, y_train, y_test = train_test_split(feed_dict[k], [0] * len(feed_dict[k]), test_size=split)
		# 	self.feed_dict_train[k] = np.array(X_train)
		# 	self.feed_dict_test[k] = np.array(X_test)
		#
		# self.X_train = np.vstack([self.feed_dict_train[k] for k in self.feed_dict_train])
		# self.X_test = np.vstack([self.feed_dict_test[k] for k in self.feed_dict_test])
		# self.y_train = np.vstack([np.array([np.array(self.eye[i])] * len(self.feed_dict_train[k])) for i, k in enumerate(self.feed_dict_train)])
		# self.y_test = np.vstack([np.array([np.array(self.eye[i])] * len(self.feed_dict_test[k])) for i, k in enumerate(self.feed_dict_test)])

		print ('Train Shape x: {}'.format(self.X_train.shape))
		print ('Train Shape y: {}'.format(self.y_train.shape))
		print ('Eval Shape x: {}'.format(self.X_test.shape))
		print ('Eval Shape y: {}'.format(self.y_test.shape))

	def fit(self, train_dirs, batch_size=32, epochs=10, validation_split=0.1, callbacks=None, one_shot_freq=2, **kwargs):
		self.process_data(train_dirs, split=validation_split)

		train_size = len(self.y_train)
		test_size = len(self.y_test)
		steps = train_size // batch_size
		train_idxs = range(train_size)
		one_shot_idxs = range(test_size)

		for callback in callbacks:
			callback.on_train_begin(logs={})

		for e in range(epochs):
			self.print_f('Epoch: {}'.format(e))
			# get batch
			for callback in callbacks:
				callback.on_epoch_begin(e)

			np.random.shuffle(train_idxs)
			for s in range(steps):
				if batch_size * s + batch_size >= train_size:
					break

				# x_train_batch = self.X_train[batch_size * s:batch_size * s + batch_size]
				# y_train_batch = self.y_train[batch_size * s:batch_size * s + batch_size]
				x_train_batch = self.X_train[train_idxs[batch_size * s:batch_size * s + batch_size]]
				y_train_batch = self.y_train[train_idxs[batch_size * s:batch_size * s + batch_size]]

				self.compile_classifier()
				# K.set_learning_phase(1)
				loss = self.model.train_on_batch(x_train_batch, y_train_batch)

				self.print_f('-Step {}/{}: loss={}'.format(s, steps, loss))

				self.compile_one_shot()
				# K.set_learning_phase(1)
				for o in range(one_shot_freq):
					np.random.shuffle(one_shot_idxs)
					x_test_batch = self.X_test[one_shot_idxs[0:batch_size]]
					y_test_batch = self.y_test[one_shot_idxs[0:batch_size]]
					y_compare = np.equal(np.argmax(y_train_batch, 1), np.argmax(y_test_batch, 1)).astype(int)

					loss_one_shot = self.one_shot_model.train_on_batch([x_train_batch, x_test_batch], y_compare)
					# self.print_f('--One shot {}: {}'.format(o, loss_one_shot))

			for callback in callbacks:
				callback.on_epoch_end(e)

		for callback in callbacks:
			callback.on_train_end()