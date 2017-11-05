"""
This model is  GAN which take a synthetic image from eye app and train on real image
to refine synthetic image to pseudo image for classification model
"""

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
	normalized_inputs = Lambda(lambda x: tf.divide(tf.subtract(x, 175.5), 175.5))(inputs)
	# normalized_inputs = subtract(inputs, 175.5)
	# normalized_inputs = multiply(normalized_inputs, 1 / 175.5)
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


class ResNet_RNN_classifier():
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
		self.config_file = config_file
		self.config = None
		self.job_dir = job_dir
		self.checkpoint_path = checkpoint_path
		self.print_f = print_f
		self.sequence_length = sequence_length
		self.input_dim = input_dim
		self.label_set = label_set
		self.batch_norm = batch_norm

		self.model = None
		self.feed_dict = None
		self.X = None
		self.X_val = None
		self.y = None
		self.y_val = None
		self.eye = np.eye(len(self.label_set))

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

	def predict(self, data):
		if self.batch_norm:
			K.set_learning_phase(0)
		result = self.model.predict(data)
		if self.batch_norm:
			K.set_learning_phase(1)
		return result

	# it will only accept tfrecords files
	def process_training_data(self, train_files, split=0.15):
		# FIXME: only accept 1 single file at this momment
		if isinstance(train_files, (list, tuple)):
			train_files = train_files[0]

		X, y = utils.load_npz(train_files)
		self.X, self.X_val, self.y, self.y_val = train_test_split(X, y, test_size=split, random_state=42, stratify=y)

		print ('Train Shape x: {}'.format(self.X.shape))
		print ('Train Shape y: {}'.format(self.y.shape))

	def test_on_trained(self, test_files):
		if test_files is not None:
			self.print_f('-- Perform Testing --')
			if isinstance(test_files, (list, tuple)):
				test_files = test_files[0]
			X, y = utils.load_npz(test_files)
			assert self.model is not None
			pred_val = np.argmax(self.predict(X), axis=1)
			true_val = np.argmax(y, axis=1)

			report(true_val, pred_val, self.label_set, print_fn=self.print_f)

		# save the model
		# if job_dir.startswith("gs://"):
		# 	model.save(model_name)
		# 	copy_file_to_gcs(job_dir, model_name)
		# else:
		# 	model.save(os.path.join(job_dir, model_name))
		model_name = 'eye_final_model.hdf5'
		self.model.save(os.path.join(self.job_dir, model_name))

		# Convert the Keras model to TensorFlow SavedModel
		self.print_f('Save model to {}'.format(self.job_dir))
		utils.to_savedmodel(self.model, os.path.join(self.job_dir, 'export'))


	def fit(self, train_files, test_files=None, batch_size=32, epochs=10, validation_split=0.1, callbacks=None, **kwargs):
		self.process_training_data(train_files, split=validation_split)

		if callbacks is None:
			callbacks = [
				keras.callbacks.ModelCheckpoint(
					filepath=self.checkpoint_path,
					monitor='val_loss',
					verbose=1,
					period=kwargs.get('checkpoint_epochs', 2),
					mode='max'),
				TerminateOnNaN(),
				EarlyStopping(patience=10),
				ReduceLROnPlateau(patience=4),
				utils.EvalCheckPoint(self.model,
							   self.checkpoint_path,
							   self.X_val,
							   self.y_val,
							   self.label_set,
							   self.sequence_length,
							   eval_freq=kwargs.get('eval_freq', 1),
							   print_func=self.print_f,
							   epochs=epochs,
							   batch_norm=self.batch_norm
							   )

			]

		self.model.fit(self.X, self.y, batch_size=batch_size, epochs=epochs,
					   validation_data=[self.X_val, self.y_val], callbacks=callbacks)

		self.print_f('--Training Done--')
		self.test_on_trained(test_files=test_files)



if __name__ == '__main__':
	d = discriminator()
	d.summary()