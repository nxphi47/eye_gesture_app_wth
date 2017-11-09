"""
This model is  GAN which take a synthetic image from eye app and train on real image
to refine synthetic image to pseudo image for classification model
"""
from __future__ import print_function

import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
from keras.models import Model
import os
from keras.layers import Input, Conv2D, Dense, BatchNormalization, Activation, Add, MaxPooling2D,\
	AveragePooling2D, Lambda, Flatten, TimeDistributed, Bidirectional, LSTM, Dropout, Multiply, Subtract, multiply, subtract, division
from keras.initializers import VarianceScaling

from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import json
import pprint
import utils
import base_model

INPUT_DIM = 64
CHANNEL = 3


def residual_module(inputs, conv_filters, kernel_sizes=None, paddings=None, activation='relu', batch_norm=False):
	feed_input = inputs
	# size = 3
	# conv_filters : [64, 64, 64]
	# kernel: [1, 3,
	# paddings: ['valid', 'same', 'valid']
	if kernel_sizes is None:
		kernel_sizes = [1, 3, 1]
	if not isinstance(conv_filters, (list, tuple)):
		conv_filters = [conv_filters] * len(kernel_sizes)

	input_branch = inputs
	# print('{} {}'.format(input_branch.get_shape(), feed_input.get_shape()[-1]))
	if input_branch.get_shape()[-1] != conv_filters[-1]:
		# new_filters = feed_input.get_shape()[-1]
		# print('new filter: {}'.format(new_filters))
		input_branch = Conv2D(filters=conv_filters[-1],
							  kernel_size=[1, 1],
							  padding='valid',
							  # kernel_initializer=VarianceScaling(mode='fan_avg', distribution='uniform'),
							  )(input_branch)
		if batch_norm:
			input_branch = BatchNormalization(epsilon=0.001, )(input_branch)

	if paddings is None:
		paddings = []
		for k in kernel_sizes:
			if k == 1:
				paddings.append('valid')
			else:
				paddings.append('same')

	activations = [activation] * (len(kernel_sizes) - 1) + ['linear']

	for _filter, _kernel, _padding, _activation in zip(conv_filters, kernel_sizes, paddings, activations):
		feed_input = Conv2D(filters=_filter, kernel_size=[_kernel, _kernel], padding=_padding,
							# kernel_initializer=VarianceScaling(mode='fan_avg', distribution='uniform')
							)(feed_input)
		if batch_norm:
			feed_input = BatchNormalization(epsilon=0.001, )(feed_input)
		if _activation != 'linear':
			feed_input = Activation(activation=_activation)(feed_input)



	feed_input = Add()([feed_input, input_branch])
	feed_input = Activation(activation)(feed_input)

	return feed_input



def discriminator():
	inputs = Input(shape=[INPUT_DIM, INPUT_DIM, CHANNEL])
	normalized_inputs = Lambda(lambda x: tf.divide(tf.subtract(x, 175.5), 175.5))(inputs)
	conv1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[2, 2], padding='same', kernel_initializer=VarianceScaling(mode='fan_avg'))(normalized_inputs)
	bn1 = BatchNormalization()(conv1)
	act1 = Activation('relu')(bn1)
	maxpool1 = MaxPooling2D(pool_size=[2, 2])(act1)

	feed_input = maxpool1
	res_modules = [
		{'conv_filters': 32},
		{'conv_filters': 32},
		{'conv_filters': [32, 32, 64]},
		{'conv_filters': [32, 32, 64]},
		# {'conv_filters': [32, 32, 128]},
		# {'conv_filters': [32, 32, 128]},
	]

	for res in res_modules:
		feed_input = residual_module(feed_input, **res)

	feed_input = Conv2D(3, kernel_size=[3, 3], padding='same')(feed_input)
	feed_input = BatchNormalization()(feed_input)
	feed_input = Activation('tanh')(feed_input)

	post_process = Lambda(lambda x: tf.add(tf.multiply(x, 175.5), 175.5))(feed_input)


	model = Model(inputs=inputs, outputs=post_process)
	return model

# More like the discriminator but output the fully connected cnn blocks
def resnet_block(input_shape, config, print_fn=print, batch_norm=True, dropout=0.0):
	inputs = Input(shape=input_shape)
	normalized_inputs = Lambda(lambda x: (x - 127.5) / 127.5)(inputs)

	batch_norm = config['batch_norm']
	dropout = config['dropout']

	feed_input = Conv2D(**config['first_conv']
				   # kernel_initializer=VarianceScaling(mode='fan_avg')
						)(normalized_inputs)
	if batch_norm:
		feed_input = BatchNormalization()(feed_input)

	feed_input = Activation(**config['first_activation'])(feed_input)
	feed_input = MaxPooling2D(**config['first_pool'])(feed_input)

	# feed_input = maxpool1
	# res_modules = [
	# 	{'conv_filters': 32},
	# 	{'conv_filters': 32},
	# 	{'conv_filters': [32, 32, 64]},
	# 	{'conv_filters': [32, 32, 64]},
	# 	# {'conv_filters': [32, 32, 128]},
	# 	# {'conv_filters': [32, 32, 128]},
	# ]
	res_modules = config['res_modules']

	for res in res_modules:
		feed_input = residual_module(feed_input, **res)

	feed_input = Conv2D(**config['second_conv'])(feed_input)
	if batch_norm:
		feed_input = BatchNormalization(epsilon=0.001)(feed_input)
	feed_input = Activation(**config['second_activation'])(feed_input)
	feed_input = AveragePooling2D()(feed_input)

	flatten = Flatten()(feed_input)
	fc = Dense(**config['second_dense'])(flatten)
	if dropout > 0.0:
		fc = Dropout(dropout)(fc)

	block = Model(inputs=inputs, outputs=fc)

	print_fn('--------- CNN restnet Block ----------')
	block.summary(print_fn=print_fn)
	print_fn('######### CNN restnet Block ##########')

	return block



def report(true_val, pred_val, label_set, epoch=0, print_fn=print, digits=4):
	report = classification_report(true_val, pred_val, target_names=label_set, digits=digits)
	matrix = confusion_matrix(true_val, pred_val)
	print_fn("----- Epoch:{} -----".format(epoch))
	print_fn(report)
	print_fn(matrix)


def resnet_sequential_model(config, print_fn=print, sequence_length=15, input_dim=15, label_set=None, dropout=0.0, batch_norm=False):
	if label_set is None:
		label_set = ['left', 'right', 'up', 'down', 'center', 'double_blink']

	# inputs = Input(shape=(config.SEQUENCE_LENGTH, config.INPUT_DIM, config.INPUT_DIM, config.CHANNEL))
	inputs = Input(shape=(sequence_length, input_dim, input_dim, 3))

	cnn_input_shape = (input_dim, input_dim, 3)

	timedistributed = TimeDistributed(resnet_block(config=config['cnn_block'],
												   input_shape=cnn_input_shape, print_fn=print_fn, batch_norm=batch_norm, dropout=dropout / 10))(inputs)

	feed_input = Bidirectional(LSTM(**config['LSTM']))(timedistributed)
	# dropout = Dropout(0.4)(lstm)
	# dense = Dense(32, activation='relu')(lstm)
	# dropout = Dropout(0.2)(dense)
	if config.get('dropout') and config['dropout']['rate'] > 0.0:
		feed_input = Dropout(**config['dropout'])(feed_input)

	out_layer = Dense(len(label_set), activation='softmax')(feed_input)
	sequential_model = Model(inputs=inputs, outputs=out_layer)

	print_fn('-----------  CNN sequential model -----------------')
	sequential_model.summary(print_fn=print_fn)
	print_fn('######### <CNN sequential model> #########')

	return sequential_model


class ResNet_RNN_classifier(base_model.ClassiferKerasModel):
	def __init__(self, config_file, job_dir, checkpoint_path, print_f=print, sequence_length=15, input_dim=64,
				 label_set=None, batch_norm=False):
		super().__init__(config_file, job_dir, checkpoint_path, print_f, sequence_length, input_dim, label_set,
						 batch_norm)

	def load_config(self):
		self.config = json.load(open(self.config_file, 'r'))
		pprint.pprint(self.config)
		self.batch_norm = self.config['cnn_block']['batch_norm']

	def compile(self, **kwargs):
		self.load_config()
		if self.batch_norm:
			K.set_learning_phase(1)
		self.model = resnet_sequential_model(config=self.config,
											 print_fn=self.print_f,
											 sequence_length=self.sequence_length,
											 input_dim=self.input_dim,
											 label_set=self.label_set,
											 dropout=0.5, batch_norm=self.batch_norm)
		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mae'])


if __name__ == '__main__':
	d = discriminator()
	d.summary()