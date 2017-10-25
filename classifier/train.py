#!/usr/bin/env python
from __future__ import print_function
import os
import shutil
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pprint
import argparse

from keras.models import Model, Sequential
from keras.layers import Input, Activation, Conv2D, Dense, Dropout, \
	LSTM, Bidirectional, TimeDistributed, MaxPooling2D, BatchNormalization, AveragePooling2D, Flatten
# from keras.layers.merge import Concatenate, Add, Dot, Multiply
from keras.callbacks import ProgbarLogger, ReduceLROnPlateau, TensorBoard, TerminateOnNaN, CSVLogger

import glob
from PIL import Image

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

def compare_url(a, b):
	ia = int(a.split('/')[-1].replace('img_', '').split('.')[0])
	prefix_a = '/'.join(a.split('/')[:-1])
	ib = int(b.split('/')[-1].replace('img_', '').split('.')[0])
	prefix_b = '/'.join(b.split('/')[:-1])
	if prefix_a == prefix_b:
		return ia - ib
	elif prefix_a > prefix_b:
		return 1
	else:
		return 0


def load_dataset(base_dir):
	globs = {}
	for l in LABEL_SET:
		# source_dir / dimension / labels / batches / images...
		globs[l] = glob.glob('{src_dir}/{label}/*/*.jpg'.format(src_dir=base_dir, label=l))
		globs[l].sort(compare_url)

	# datasets
	X = []
	y = []
	y_raws = []
	eye = np.eye(len(LABEL_SET))
	for i, l in enumerate(LABEL_SET):
		data = []
		for j, img_url in enumerate(globs[l]):
			img = Image.open(img_url)
			img_array = normalize_image(np.array(img))
			if j % SEQUENCE_LENGTH == 0 and j != 0:
				# package into sequence
				X.append(np.array(data))
				y.append(np.array(eye[i]))
				y_raws.append(l)
				data = []
			# else:
			data.append(img_array)

	X = np.array(X)
	y = np.array(y)
	return X, y, y_raws, LABEL_SET


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


def CNN_RNN_Sequential_model(print_f=print):
	# inputs = Input(shape=(config.SEQUENCE_LENGTH, config.INPUT_DIM, config.INPUT_DIM, config.CHANNEL))
	inputs = Input(shape=(SEQUENCE_LENGTH, INPUT_DIM, INPUT_DIM, 3))

	cnn_input_shape = (INPUT_DIM, INPUT_DIM, 3)

	timedistributed = TimeDistributed(CNN_block(cnn_input_shape, print_fn=print_f))(inputs)

	lstm = Bidirectional(LSTM(units=128))(timedistributed)
	out_layer = Dense(len(LABEL_SET), activation='softmax')(lstm)

	model = Model(inputs=inputs, outputs=out_layer)

	# CNN model
	print_f('-----------  CNN sequential model -----------------')
	model.summary(print_fn=print_f)
	print_f('----------- <CNN sequential model> -----------------')

	return model


def after_train(model, model_file, model_dir, train_config, label_set, model_name='cnn_', print_fn=print):
	X, y, y_raws, label_set = load_dataset(train_config['test'])

	true_val = np.array(np.argmax(y, axis=1))
	pred_val = np.argmax(model.predict(X), axis=1)
	report = classification_report(true_val, pred_val, target_names=label_set)
	matrix = confusion_matrix(true_val, pred_val)
	print_fn(report)
	print_fn(matrix)

	# save_dir = 'models/'
	# index = 0
	# model_name = 'cnn_sequential_6label_{}'.format(index)
	model.save(os.path.join(model_dir, "{}.h5".format(model_name)))
	# save_tf(model, os.path.join(save_dir, "{}".format(model_name)))

	print_fn('model save {}'.format(model_name))


def train(train_config):
	epoch = train_config.get('epoch', 10)
	batch_size = train_config.get('batch', 32)
	split = train_config.get('split', 0.2)
	model_name = train_config.get('model', 'noname')

	train_dir = train_config['train']
	test_dir = train_config['test']
	model_dir = os.path.join("./models", "{}".format(model_name))
	model_file = open(os.path.join(model_dir, "model_log.txt"), 'w')


	def print_f(val):
		model_file.write("{}\n".format(val))
		print(val)


	model = CNN_RNN_Sequential_model(print_f)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mae'], )

	# load training set
	print ('Loading data')
	X, y, y_raws, label_set = load_dataset(train_dir)

	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split, random_state=42)

	model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1,
			  validation_data=[X_val, y_val],
			  callbacks=[ProgbarLogger(),
						 ReduceLROnPlateau(),
						 TensorBoard(log_dir=model_dir, histogram_freq=1, batch_size=100),
						 CSVLogger(filename=os.path.join(model_dir, "logs.log"))
						 ])
	after_train(model, model_file, model_dir, train_config, label_set, model_name)

	model_file.close()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='Filename', type=str, required=True)
	parser.add_argument('--train', help='train folder', type=str, required=True)
	parser.add_argument('--test', help='test folder', type=str, required=True)

	parser.add_argument('--epoch', help='Epoch ({})'.format(EPOCH), type=int, default=EPOCH)
	parser.add_argument('--batch', help='Batch size ({})'.format(BATCH), type=int, default=BATCH)
	parser.add_argument('--split', help='Split ({})'.format(SPLIT), type=float, default=SPLIT)
	parser.add_argument('--bw', help='Black and white (1 channel)', action="store_true")

	args = parser.parse_args()

	# testing
	# print os.path.abspath(args.train)

	train_config = {
		'epoch': args.epoch,
		'batch_size': args.batch,
		'split': args.split,
		'model': args.model,
		'train': os.path.abspath(args.train),
		'test': os.path.abspath(args.test),
	}
	print('\n\n\nConfiguration')
	pprint.pprint(train_config)

	model_dir = os.path.join('./models/', '{}'.format(train_config['model']))
	if os.path.exists(model_dir):
		res = raw_input('Model {} exists, do you want to overwrite? (y/n): ')
		if res.lower() == 'y' or res.lower() == 'yes':
			shutil.rmtree(os.path.abspath(model_dir))
			os.mkdir(os.path.abspath(model_dir))
			print ('Overwrite folder {}'.format(model_dir))
		else:
			print ('No Overwrite folder, exit')
			exit()
		# raise ValueError('Model {} already exists', train_config['model'])
	else:
		os.mkdir(model_dir)

	print('---------------------------------------------------------------------------')
	train(train_config)


if __name__ == '__main__':
	main()
