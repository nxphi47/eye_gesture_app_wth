#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import random
import pprint

import argparse

import keras
from keras.models import Model, Sequential
from keras.layers import Layer, InputLayer, Input, Activation, Conv2D, Dense, Dropout, \
	LSTM, Bidirectional, TimeDistributed, MaxPooling2D, BatchNormalization, AveragePooling2D, Flatten
# from keras.layers.merge import Concatenate, Add, Dot, Multiply
from keras.callbacks import BaseLogger, ProgbarLogger, ReduceLROnPlateau, TensorBoard

import glob
import os
import numpy as np
from PIL import Image
from PIL import ImageOps

LABEL_SET = ['left', 'right', 'up', 'down', 'center', 'double_blink']
# DATASETS_SRC_DIR = './datasets/{dim}/'.format(dim=config.INPUT_DIM)
DATASETS_SRC_DIR = './datasets/'
SEQUENCE_LENGTH = 15
INPUT_DIM = 64
EPOCH = 10
BATCH = 32
SPLIT = 0.2


def normalize_image(img):
	# return (img - 127.5) / 127.5
	return (img.astype(np.float32) - 127.5) / 127.5

def denormalize_image(img):
	result = img * 127.5 + 127.5
	return result.astype(np.uint8)


def load_datasets():
	globs = {}
	for l in LABEL_SET:
		# source_dir / dimension / labels / batches / images...
		globs[l] = glob.glob('{src_dir}{label}/*/*.jpg'.format(src_dir=DATASETS_SRC_DIR, label=l))

	# datasets
	X = []
	y = []
	y_raws = []
	eye = np.eye(len(LABEL_SET))

	for i, l in enumerate(LABEL_SET):
		data = []
		for j, img_url in enumerate(globs[l]):
			img = Image.open(img_url)

			imgArray = normalize_image(np.array(img))
			if j % SEQUENCE_LENGTH == 0 and j != 0:
				# package into sequence
				X.append(np.array(data))
				y.append(np.array(eye[i]))
				y_raws.append(l)
				data = []
			# else:
			data.append(imgArray)

	X = np.array(X)
	y = np.array(y)
	return X, y, y_raws, LABEL_SET


# Convolutional blocks
def add_conv_layer(model, filters, kernel_size, use_bias, activation='relu', pooling='max_pool', batch_norm=False, input_shape=False):
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

def CNN_block(input_shape):
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
	print '-----------  CNN model ------------------'
	model.summary()
	print '----------- <CNN model> -----------------'

	return model


def CNN_RNN_Sequential_model():
	# inputs = Input(shape=(config.SEQUENCE_LENGTH, config.INPUT_DIM, config.INPUT_DIM, config.CHANNEL))
	inputs = Input(shape=(SEQUENCE_LENGTH, INPUT_DIM, INPUT_DIM, 3))

	cnn_input_shape = ( INPUT_DIM, INPUT_DIM, 3)

	timedistributed = TimeDistributed(CNN_block(cnn_input_shape))(inputs)

	lstm = Bidirectional(LSTM(units=128))(timedistributed)
	out_layer = Dense(len(LABEL_SET), activation='softmax')(lstm)

	model = Model(inputs=inputs, outputs=out_layer)

	# CNN model
	print '-----------  CNN sequential model -----------------'
	model.summary()
	print '----------- <CNN sequential model> -----------------'

	return model

def after_train(model, X_train, X_test, y_train, y_test, label_set, model_name='cnn_'):
	true_val = np.array(np.argmax(y_test, axis=1))
	pred_val = np.argmax(model.predict(X_test), axis=1)
	report = classification_report(true_val, pred_val, target_names=label_set)
	matrix = confusion_matrix(true_val, pred_val)
	print report
	print matrix

	save_dir = 'models/'
	# index = 0
	# model_name = 'cnn_sequential_6label_{}'.format(index)
	model.save(os.path.join(save_dir, "{}.h5".format(model_name)))
	# save_tf(model, os.path.join(save_dir, "{}".format(model_name)))

	print 'model save {}'.format(model_name)


def train(train_config={}):
	epoch = train_config.get('epoch', 10)
	batch_size = train_config.get('batch', 32)
	split = train_config.get('split', 0.2)

	model = CNN_RNN_Sequential_model()
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mae'],)

	X, y, y_raws, label_set = load_datasets()


	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state = 42)

	model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1,
			  validation_data=[X_test, y_test],
			  callbacks=[ProgbarLogger(), ReduceLROnPlateau(),
						 TensorBoard(log_dir='./tensorboard', histogram_freq=1, batch_size=100)])

	after_train(model, X_train, X_test, y_train, y_test, label_set, train_config.get('model', 'cnn_noname'))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--epoch', help='Epoch ({})'.format(EPOCH), type=int, default=EPOCH)
	parser.add_argument('--batch', help='Batch size ({})'.format(BATCH), type=int, default=BATCH)
	parser.add_argument('--split', help='Split ({})'.format(SPLIT), type=float, default=SPLIT)
	parser.add_argument('--model', help='Filename', type=str, required=True)
	parser.add_argument('--bw', help='Black and white (1 channel)', action="store_true")

	args = parser.parse_args()

	train_config = {
		'epoch': args.epoch,
		'batch_size': args.batch,
		'split': args.split,
		'model': args.model,
	}
	print 'Configuration'
	pprint.pprint(train_config)
	if os.path.exists(os.path.join('./models/', '{}.h5'.format(train_config['model']))):
		raise ValueError('Model {} already exists', train_config['model'])

	print '---------------------------------------------------------------------------'
	train(train_config)


if __name__ == '__main__':
	main()