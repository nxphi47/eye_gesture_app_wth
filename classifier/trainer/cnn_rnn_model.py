#!/usr/bin/env python
from __future__ import print_function

# from keras.layers.merge import Concatenate, Add, Dot, Multiply
import glob
import os
from PIL import Image
import numpy as np
from keras import backend as K
from keras.layers import Input, Activation, Conv2D, Dense, Dropout, \
	LSTM, Bidirectional, TimeDistributed, MaxPooling2D, BatchNormalization, AveragePooling2D, Flatten
from keras.models import Model, Sequential
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import utils


"""Deprecated"""
def load_dataset(base_dir, label_set, sequence_length=15):
	globs = {}
	for l in label_set:
		# source_dir / dimension / labels / batches / images...
		globs[l] = glob.glob('{src_dir}/{label}/*/*.jpg'.format(src_dir=base_dir, label=l))

	# datasets
	X = []
	y = []
	y_raws = []
	eye = np.eye(len(label_set))
	for i, l in enumerate(label_set):
		data = []
		for j, img_url in enumerate(globs[l]):
			img = Image.open(img_url)
			img_array = utils.normalize_image(np.array(img))
			if j % sequence_length == 0 and j != 0:
				# package into sequence
				X.append(np.array(data))
				y.append(np.array(eye[i]))
				y_raws.append(l)
				data = []
			# else:
			data.append(img_array)

	X = np.array(X)
	y = np.array(y)
	return X, y, y_raws, label_set



# Convolutional blocks
def add_conv_layer(model, filters, kernel_size, use_bias, activation='relu', pooling='max_pool', batch_norm=False,
				   input_shape=False):
	if input_shape:
		model.add(Conv2D(input_shape=input_shape, filters=filters, kernel_size=(kernel_size, kernel_size),
						 use_bias=use_bias))
	else:
		model.add(Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
						 use_bias=use_bias))

	if batch_norm:
		# conv = BatchNormalization()(conv)
		model.add(BatchNormalization())
	if activation and activation != '':
		# conv = Activation(activation)(conv)
		model.add(Activation(activation))
	if pooling == 'max_pool':
		# conv = MaxPooling2D()(conv)
		model.add(MaxPooling2D())
	elif pooling == 'avg_pool':
		# conv = AveragePooling2D()(conv)
		model.add(AveragePooling2D())
	else:
		raise Exception('Pooling invalid: {}'.format(pooling))

	return model


def CNN_block(input_shape, print_fn=print):
	use_bias = False
	batch_norm = False
	pooling = 'max_pool'

	model = Sequential()

	model = add_conv_layer(model, filters=8, kernel_size=4, use_bias=use_bias, pooling=pooling, batch_norm=batch_norm,
						   input_shape=input_shape)

	model = add_conv_layer(model, filters=16, kernel_size=3, use_bias=use_bias, pooling=pooling, batch_norm=batch_norm)

	model = add_conv_layer(model, filters=32, kernel_size=3, use_bias=use_bias, pooling=pooling, batch_norm=batch_norm)

	model = add_conv_layer(model, filters=64, kernel_size=3, use_bias=use_bias, pooling=pooling, batch_norm=batch_norm)

	# flatten = Flatten()(conv)
	model.add(Flatten())

	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))

	# model = Model(outputs=fc)

	# CNN model
	print_fn('-----------  CNN model ------------------')
	model.summary(print_fn=print_fn)
	print_fn('----------- <CNN model> -----------------')

	return model


def CNN_RNN_Sequential_model(print_f=print,
							 sequence_length=15,
							 input_dim=64,
							 label_set=None
							 ):
	if label_set is None:
		label_set = ['left', 'right', 'up', 'down', 'center', 'double_blink']

	# inputs = Input(shape=(config.SEQUENCE_LENGTH, config.INPUT_DIM, config.INPUT_DIM, config.CHANNEL))
	inputs = Input(shape=(sequence_length, input_dim, input_dim, 3))

	cnn_input_shape = (input_dim, input_dim, 3)

	timedistributed = TimeDistributed(CNN_block(cnn_input_shape, print_fn=print_f))(inputs)

	lstm = Bidirectional(LSTM(units=128))(timedistributed)
	out_layer = Dense(len(label_set), activation='softmax')(lstm)

	model = Model(inputs=inputs, outputs=out_layer)

	# CNN model
	print_f('-----------  CNN sequential model -----------------')
	model.summary(print_fn=print_f)
	print_f('----------- <CNN sequential model> -----------------')

	return model


class CNN_RNN_Sequential():
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
		self.feed_dict = None
		self.X = None
		self.y = None
		self.eye = np.eye(len(self.label_set))

	def compile(self, **kwargs):
		self.model = CNN_RNN_Sequential_model(self.print_f, self.sequence_length, self.input_dim, self.label_set)
		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mae'], )

	def process_feed_dict(self, feed_dict):
		self.feed_dict = feed_dict
		self.X = np.vstack([self.feed_dict[k] for k in self.feed_dict])
		self.y = np.vstack([np.array([np.array(self.eye[i])] * len(self.feed_dict[k])) for i, k in enumerate(self.feed_dict)])
		print ('Shape x: {}'.format(self.X.shape))
		print ('Shape y: {}'.format(self.y.shape))

	def process_data(self, train_dirs, split=0.2):

		self.X, self.y, y_raws, label_set = utils.load_dataset(train_dirs, self.label_set, self.sequence_length)

		print ('Train Shape x: {}'.format(self.X.shape))
		print ('Train Shape y: {}'.format(self.y.shape))
		# print ('Eval Shape x: {}'.format(self.X_test.shape))
		# print ('Eval Shape y: {}'.format(self.y_test.shape))

	# TODO: feed_dict is data keyed by label
	def fit(self, train_dirs, batch_size=32, epochs=10, validation_split=0.1, callbacks=None, **kwargs):
		self.process_data(train_dirs)
		self.model.fit(self.X, self.y, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
					  callbacks=callbacks
					  )
