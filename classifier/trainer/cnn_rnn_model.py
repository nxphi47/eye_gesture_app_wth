#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
from keras.layers.merge import Concatenate, Add, Dot, Multiply
import glob
import os
from PIL import Image
import numpy as np
from keras import backend as K
from keras.layers import Input, Activation, Conv2D, Dense, Dropout, Lambda, \
	LSTM, Bidirectional, TimeDistributed, MaxPooling2D, BatchNormalization, AveragePooling2D, Flatten
from keras.models import Model, Sequential

from sklearn.model_selection import train_test_split
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import utils
import base_model


# Convolutional blocks
def add_conv_layer(model, filters, kernel_size,
				   conv_num=1, use_bias=False,
				   activation='relu', pooling=None, batch_norm=False,
				   input_shape=False, padding='valid', dropout=0.0, stride=1):

	for i in range(conv_num):

		if input_shape and i == 0:
			model.add(Conv2D(input_shape=input_shape, filters=filters, kernel_size=(kernel_size, kernel_size),
							 padding=padding,strides=[stride, stride],
							 use_bias=use_bias))
		else:
			model.add(Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
							 padding=padding, strides=[stride, stride],
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
			# raise Exception('Pooling invalid: {}'.format(pooling))
			print('no pooling')

	if 0.0 < dropout < 1.0:
		model.add(Dropout(dropout))

	return model


def CNN_block(input_shape, print_fn=print):
	use_bias = False
	batch_norm = True
	# pooling = 'max_pool'
	pooling = None
	padding = 'valid'
	conv_dropout = -1

	model = Sequential()

	model = add_conv_layer(model,
						   filters=16,
						   conv_num=2,
						   kernel_size=4,
						   stride=1,
						   use_bias=use_bias,
						   padding=padding,
						   pooling=pooling, batch_norm=batch_norm,
						   input_shape=input_shape, dropout=conv_dropout)

	# model = add_conv_layer(model, filters=8, kernel_size=3,
	# 					   use_bias=use_bias, pooling=pooling,
	# 					   batch_norm=batch_norm, dropout=conv_dropout)

	model = add_conv_layer(model, filters=16, kernel_size=4, conv_num=2,
						   padding=padding,
						   stride=2,
						   use_bias=use_bias, pooling=pooling,
						   batch_norm=batch_norm, dropout=conv_dropout)

	# model = add_conv_layer(model, filters=16, kernel_size=3,
	# 					   use_bias=use_bias, pooling=pooling,
	# 					   batch_norm=batch_norm, dropout=conv_dropout)

	model = add_conv_layer(model, filters=16, kernel_size=4, conv_num=2,
						   padding=padding,
						   stride=1,
						   use_bias=use_bias, pooling=pooling,
						   batch_norm=batch_norm, dropout=conv_dropout)

	# model = add_conv_layer(model, filters=32, kernel_size=3, conv_num=2,
	# 					   padding=padding,
	# 					   stride=1,
	# 					   use_bias=use_bias, pooling=pooling,
	# 					   batch_norm=batch_norm, dropout=conv_dropout)

	# 4x4x64

	# flatten = Flatten()(conv)
	model.add(Flatten())

	model.add(Dense(32,))
	if batch_norm:
		model.add(BatchNormalization())
	model.add(Activation(activation='relu'))

	# model.add(Dropout(0.5))

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

	batch_norm = True

	# inputs = Input(shape=(config.SEQUENCE_LENGTH, config.INPUT_DIM, config.INPUT_DIM, config.CHANNEL))

	inputs = Input(shape=(sequence_length, input_dim, input_dim, 3))
	preprocess = Lambda(lambda x: (x - 127.5) / 127.5)(inputs)
	# preprocess = inputs

	cnn_input_shape = (input_dim, input_dim, 3)

	timedistributed = TimeDistributed(CNN_block(cnn_input_shape, print_fn=print_f))(preprocess)

	feed_input = Bidirectional(LSTM(units=32))(timedistributed)

	# dropout = Dropout(0.4)(lstm)
	# feed_input = Dense(32)(feed_input)
	#
	# if batch_norm:
	# 	feed_input = BatchNormalization()(feed_input)
	# feed_input = Activation('relu')(feed_input)
	# dropout = Dropout(0.2)(dense)

	out_layer = Dense(len(label_set), activation='softmax')(feed_input)

	model = Model(inputs=inputs, outputs=out_layer)

	# CNN model
	print_f('-----------  CNN sequential model -----------------')
	model.summary(print_fn=print_f)
	print_f('----------- <CNN sequential model> -----------------')

	return model


class CNN_RNN_Sequential(base_model.ClassiferKerasModel):
	def __init__(self,
				 config_file,
				 job_dir,
				 checkpoint_path,
				 print_f=print,
				 sequence_length=15,
				 input_dim=64,
				 label_set=None,
				 batch_norm=False
				 ):
		super().__init__(config_file, job_dir, checkpoint_path, print_f, sequence_length, input_dim, label_set, batch_norm)


	def compile(self, **kwargs):
		self.load_config()
		K.set_learning_phase(1)
		self.model = CNN_RNN_Sequential_model(self.print_f, self.sequence_length, self.input_dim, self.label_set)
		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mae'], )

	# def process_feed_dict(self, feed_dict):
	# 	self.feed_dict = feed_dict
	# 	self.X = np.vstack([self.feed_dict[k] for k in self.feed_dict])
	# 	self.y = np.vstack([np.array([np.array(self.eye[i])] * len(self.feed_dict[k])) for i, k in enumerate(self.feed_dict)])
	# 	print ('Shape x: {}'.format(self.X.shape))
	# 	print ('Shape y: {}'.format(self.y.shape))
	#
	# def process_data(self, train_dirs, split=0.2):
	#
	# 	X, y, y_raws, label_set = utils.load_dataset(train_dirs, self.label_set, self.sequence_length)
	#
	# 	self.X, self.X_val, self.y, self.y_val = train_test_split(X, y, test_size=split, random_state=42, stratify=y)
	#
	# 	print ('Train Shape x: {}'.format(self.X.shape))
	# 	print ('Train Shape y: {}'.format(self.y.shape))
	# 	# print ('Eval Shape x: {}'.format(self.X_test.shape))
	# 	# print ('Eval Shape y: {}'.format(self.y_test.shape))
	#
	# # TODO: feed_dict is data keyed by label
	# def fit(self, train_dirs, batch_size=32, epochs=10, validation_split=0.1, callbacks=None, **kwargs):
	# 	self.process_data(train_dirs, validation_split)
	# 	self.model.fit(self.X, self.y,
	# 				   validation_data=[self.X_val, self.y_val],
	# 				   batch_size=batch_size, epochs=epochs,
	# 				   # validation_split=validation_split,
	# 				  callbacks=callbacks
	# 				  )
