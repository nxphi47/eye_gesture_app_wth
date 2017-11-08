"""
This is model implement densenet
"""
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
from keras.models import Model
import os
from keras.layers import Input, Conv2D, Dense, BatchNormalization, Activation, Add, MaxPooling2D, Concatenate, \
	AveragePooling2D, Lambda, Flatten, TimeDistributed, Bidirectional, LSTM, Dropout, Multiply, \
	Subtract, multiply, subtract, division, GlobalAveragePooling2D
from keras.initializers import VarianceScaling
from keras.regularizers import l2
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import json
import pprint
import utils
import base_model

INPUT_DIM = 64
CHANNEL = 3


def conv_factory(x, nb_filter, dropout_rate=None, weight_decay=1E-4):
	"""Apply BatchNorm, Relu 3x3Conv2D, optional dropout
	:param x: Input keras network
	:param nb_filter: int -- number of filters
	:param dropout_rate: int -- dropout rate
	:param weight_decay: int -- weight decay factor
	:returns: keras network with b_norm, relu and Conv2D added
	:rtype: keras network
	"""

	x = BatchNormalization(axis=1,
						   gamma_regularizer=l2(weight_decay),
						   beta_regularizer=l2(weight_decay))(x)
	x = Activation('relu')(x)
	x = Conv2D(nb_filter, (3, 3),
			   kernel_initializer="he_uniform",
			   padding="same",
			   use_bias=False,
			   kernel_regularizer=l2(weight_decay))(x)
	if dropout_rate:
		x = Dropout(dropout_rate)(x)

	return x


def transition(x, nb_filter, dropout_rate=None, weight_decay=1E-4):
	"""Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D
	:param x: keras model
	:param nb_filter: int -- number of filters
	:param dropout_rate: int -- dropout rate
	:param weight_decay: int -- weight decay factor
	:returns: model
	:rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
	"""

	x = BatchNormalization(axis=1,
						   gamma_regularizer=l2(weight_decay),
						   beta_regularizer=l2(weight_decay))(x)
	x = Activation('relu')(x)
	x = Conv2D(nb_filter, (1, 1),
			   kernel_initializer="he_uniform",
			   padding="same",
			   use_bias=False,
			   kernel_regularizer=l2(weight_decay))(x)
	if dropout_rate:
		x = Dropout(dropout_rate)(x)
	x = AveragePooling2D((2, 2), strides=(2, 2))(x)

	return x


def denseblock(x, nb_layers, nb_filter, growth_rate,
			   dropout_rate=None, weight_decay=1E-4):
	"""Build a denseblock where the output of each
	   conv_factory is fed to subsequent ones
	:param x: keras model
	:param nb_layers: int -- the number of layers of conv_
					  factory to append to the model.
	:param nb_filter: int -- number of filters
	:param dropout_rate: int -- dropout rate
	:param weight_decay: int -- weight decay factor
	:returns: keras model with nb_layers of conv_factory appended
	:rtype: keras model
	"""

	list_feat = [x]

	if K.image_dim_ordering() == "th":
		concat_axis = 1
	elif K.image_dim_ordering() == "tf":
		concat_axis = -1

	for i in range(nb_layers):
		x = conv_factory(x, growth_rate, dropout_rate, weight_decay)
		list_feat.append(x)
		x = Concatenate(axis=concat_axis)(list_feat)
		nb_filter += growth_rate

	return x, nb_filter


def denseblock_altern(x, nb_layers, nb_filter, growth_rate,
					  dropout_rate=None, weight_decay=1E-4):
	"""Build a denseblock where the output of each conv_factory
	   is fed to subsequent ones. (Alternative of a above)
	:param x: keras model
	:param nb_layers: int -- the number of layers of conv_
					  factory to append to the model.
	:param nb_filter: int -- number of filters
	:param dropout_rate: int -- dropout rate
	:param weight_decay: int -- weight decay factor
	:returns: keras model with nb_layers of conv_factory appended
	:rtype: keras model
	* The main difference between this implementation and the implementation
	above is that the one above
	"""

	if K.image_dim_ordering() == "th":
		concat_axis = 1
	elif K.image_dim_ordering() == "tf":
		concat_axis = -1

	for i in range(nb_layers):
		merge_tensor = conv_factory(x, growth_rate, dropout_rate, weight_decay)
		x = Concatenate(axis=concat_axis)([merge_tensor, x])
		nb_filter += growth_rate

	return x, nb_filter


def DenseNet(nb_fc, img_dim, depth, nb_dense_block, growth_rate,
			 nb_filter, dropout_rate=None, weight_decay=1E-4, batch_norm=False, print_fn=print):
	""" Build the DenseNet model
	:param nb_fc: int -- number of output
	:param img_dim: tuple -- (channels, rows, columns)
	:param depth: int -- how many layers
	:param nb_dense_block: int -- number of dense blocks to add to end
	:param growth_rate: int -- number of filters to add
	:param nb_filter: int -- number of filters
	:param dropout_rate: float -- dropout rate
	:param weight_decay: float -- weight decay
	:returns: keras model with nb_layers of conv_factory appended
	:rtype: keras model
	"""

	model_input = Input(shape=img_dim)

	assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

	# layers in each dense block
	nb_layers = int((depth - 4) / 3)

	# Initial convolution
	x = Conv2D(nb_filter, (3, 3),
			   kernel_initializer="he_uniform",
			   padding="same",
			   name="initial_conv2D",
			   use_bias=False,
			   kernel_regularizer=l2(weight_decay))(model_input)

	# Add dense blocks
	for block_idx in range(nb_dense_block - 1):
		x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate,
								  dropout_rate=dropout_rate,
								  weight_decay=weight_decay)
		# add transition
		x = transition(x, nb_filter, dropout_rate=dropout_rate,
					   weight_decay=weight_decay)

	# The last denseblock does not have a transition
	x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate,
							  dropout_rate=dropout_rate,
							  weight_decay=weight_decay)

	x = BatchNormalization(axis=1,
						   gamma_regularizer=l2(weight_decay),
						   beta_regularizer=l2(weight_decay))(x)
	x = Activation('relu')(x)
	x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
	x = Dense(nb_fc,
			  activation='relu',
			  kernel_regularizer=l2(weight_decay),
			  bias_regularizer=l2(weight_decay))(x)

	densenet = Model(inputs=[model_input], outputs=[x], name="DenseNet")

	densenet.summary(print_fn=print_fn)
	return densenet


def densecnn_module(inputs, k_step, conv_filters, kernel_sizes=None, paddings=None, activation='relu',
					batch_norm=False):
	feed_input = inputs

	return feed_input


# More like the discriminator but output the fully connected cnn blocks


def densenet_sequential_model(config, print_fn=print, sequence_length=15, input_dim=15, label_set=None, dropout=0.0,
							  batch_norm=False):
	if label_set is None:
		label_set = ['left', 'right', 'up', 'down', 'center', 'double_blink']

	# inputs = Input(shape=(config.SEQUENCE_LENGTH, config.INPUT_DIM, config.INPUT_DIM, config.CHANNEL))
	inputs = Input(shape=(sequence_length, input_dim, input_dim, 3))

	cnn_input_shape = (input_dim, input_dim, 3)

	timedistributed = TimeDistributed(DenseNet(img_dim=cnn_input_shape, print_fn=print_fn, **config['cnn_block']))(inputs)

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


class DenseNet_RNN_classifier(base_model.ClassiferModel):
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


	# def load_config(self):
	# 	self.config = json.load(open(self.config_file, 'r'))
	# 	pprint.pprint(self.config)
	# 	self.batch_norm = self.config['cnn_block']['batch_norm']

	def compile(self, **kwargs):
		self.load_config()
		if self.batch_norm:
			K.set_learning_phase(1)
		self.model = densenet_sequential_model(config=self.config,
											   print_fn=self.print_f,
											   sequence_length=self.sequence_length,
											   input_dim=self.input_dim,
											   label_set=self.label_set,
											   dropout=0.5, batch_norm=self.batch_norm)
		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mae'])

	# def predict(self, data):
	# 	if self.batch_norm:
	# 		K.set_learning_phase(0)
	# 	result = self.model.predict(data)
	# 	if self.batch_norm:
	# 		K.set_learning_phase(1)
	# 	return result

	# it will only accept tfrecords files
	# def process_training_data(self, train_files, split=0.15):
	# 	# FIXME: only accept 1 single file at this momment
	# 	if isinstance(train_files, (list, tuple)):
	# 		train_files = train_files[0]
	#
	# 	X, y = utils.load_npz(train_files)
	# 	self.X, self.X_val, self.y, self.y_val = train_test_split(X, y, test_size=split, random_state=42, stratify=y)
	#
	# 	print('Train Shape x: {}'.format(self.X.shape))
	# 	print('Train Shape y: {}'.format(self.y.shape))
	#
	# def test_on_trained(self, test_files):
	# 	if test_files is not None:
	# 		self.print_f('-- Perform Testing --')
	# 		if isinstance(test_files, (list, tuple)):
	# 			test_files = test_files[0]
	# 		X, y = utils.load_npz(test_files)
	# 		assert self.model is not None
	# 		pred_val = np.argmax(self.predict(X), axis=1)
	# 		true_val = np.argmax(y, axis=1)
	#
	# 		utils.report(true_val, pred_val, self.label_set, print_fn=self.print_f)
	#
	# 	# save the model
	# 	# if job_dir.startswith("gs://"):
	# 	# 	model.save(model_name)
	# 	# 	copy_file_to_gcs(job_dir, model_name)
	# 	# else:
	# 	# 	model.save(os.path.join(job_dir, model_name))
	# 	model_name = 'eye_final_model.hdf5'
	# 	self.model.save(os.path.join(self.job_dir, model_name))
	#
	# 	# Convert the Keras model to TensorFlow SavedModel
	# 	self.print_f('Save model to {}'.format(self.job_dir))
	# 	utils.to_savedmodel(self.model, os.path.join(self.job_dir, 'export'))

	# def fit(self, train_files, test_files=None, batch_size=32, epochs=10, validation_split=0.1, callbacks=None,
	# 		**kwargs):
	# 	self.process_training_data(train_files, split=validation_split)
	#
	# 	if callbacks is None:
	# 		callbacks = [
	# 			keras.callbacks.ModelCheckpoint(
	# 				filepath=self.checkpoint_path,
	# 				monitor='val_loss',
	# 				verbose=1,
	# 				period=kwargs.get('checkpoint_epochs', 2),
	# 				mode='max'),
	# 			TerminateOnNaN(),
	# 			EarlyStopping(patience=10),
	# 			ReduceLROnPlateau(patience=4),
	# 			utils.EvalCheckPoint(self.model,
	# 								 self.checkpoint_path,
	# 								 self.X_val,
	# 								 self.y_val,
	# 								 self.label_set,
	# 								 self.sequence_length,
	# 								 eval_freq=kwargs.get('eval_freq', 1),
	# 								 print_func=self.print_f,
	# 								 epochs=epochs,
	# 								 batch_norm=self.batch_norm
	# 								 )
	#
	# 		]
	#
	# 	self.model.fit(self.X, self.y, batch_size=batch_size, epochs=epochs,
	# 				   validation_data=[self.X_val, self.y_val], callbacks=callbacks)
	#
	# 	self.print_f('--Training Done--')
	# 	self.test_on_trained(test_files=test_files)


if __name__ == '__main__':
	pass
