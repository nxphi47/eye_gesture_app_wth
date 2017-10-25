#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import glob
import json
import numpy as np
import shutil

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pprint
from PIL import Image
import tensorflow as tf
import keras
import zipfile

from keras.models import load_model, Model

import utils
import cnn_rnn_model
import cnn_rnn_one_shot

LABEL_SET = ['left', 'right', 'up', 'down', 'center', 'double_blink']
DATASETS_SRC_DIR = './datasets/'
SEQUENCE_LENGTH = 15
INPUT_DIM = 64
EPOCH = 10
BATCH = 32
SPLIT = 0.2

FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
CHUNK_SIZE = 5000
MODEL_NAME = 'eye.hdf5'


# URL can be file



class EvalCheckPoint(keras.callbacks.Callback):
	def __init__(self, ml_model, job_dir, test_files, label_set, sequence_lenth, eval_freq=4, print_func=print,
				 epochs=10):
		self.job_dir = job_dir
		self.test_files = test_files
		self.label_set = label_set
		self.sequence_length = sequence_lenth
		self.X_test = None
		self.y_test = None
		self.epochs = epochs
		self.eval_freq = eval_freq
		self.model = None
		self.print_func = print_func
		self.set_model(ml_model)
		self.true_val = None
		self.pred_val = None

		self.load_test_data()

	def load_test_data(self):
		self.X_test, self.y_test, y_raws, label_set = utils.load_dataset(self.test_files, self.label_set,
																   self.sequence_length)
		# print (self.y_test[0])
		# print (self.y_test[1])
		self.true_val = np.array(np.argmax(self.y_test, axis=1))

	# self.pred_val = np.argmax(model.predict(X), axis=1)

	def on_epoch_begin(self, epoch, logs={}):
		if epoch > 0 and (epoch % self.eval_freq == 0 or epoch == self.epochs):
			if self.model is not None:
				pred_val = np.argmax(self.model.predict(self.X_test), axis=1)
				report = classification_report(self.true_val, pred_val, target_names=self.label_set, digits=4)
				matrix = confusion_matrix(self.true_val, pred_val)
				self.print_func("----- Epoch:{} -----".format(epoch))
				self.print_func(report)
				self.print_func(matrix)


"""Dispatch the training process"""


def dispatch(train_dirs,
			 eval_dirs,
			 job_dir,
			 epoch,
			 batch,
			 split,
			 eval_frequency,
			 get_zip,
			 one_shot_freq,

			 train_steps,
			 eval_steps,
			 train_batch_size,
			 eval_batch_size,
			 learning_rate,
			 first_layer_size,
			 num_layers,
			 scale_factor,
			 eval_num_epochs,
			 num_epochs,
			 checkpoint_epochs):
	try:
		if os.path.exists(job_dir):
			shutil.rmtree(job_dir)
		os.makedirs(job_dir)
	except:
		pass

	# very inefficient way
	custom_logs = []
	if "gs://" not in job_dir:
		model_file = open(os.path.join(job_dir, "custom_logs.txt"), 'w')

	def print_f(val):
		if "gs://" not in job_dir:
			model_file.write("{}\n".format(val))
		print(val)

	# training_feed_dict = load_data_dict(train_dirs, LABEL_SET, SEQUENCE_LENGTH, get_zip=get_zip)

	# eye_model = cnn_rnn_model.CNN_RNN_Sequential_model(print_f, SEQUENCE_LENGTH, INPUT_DIM, LABEL_SET)
	# eye_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mae'], )
	eye_model = cnn_rnn_model.CNN_RNN_Sequential(print_f, SEQUENCE_LENGTH, INPUT_DIM, LABEL_SET)
	# eye_model = cnn_rnn_one_shot.CNN_RNN_ONE_SHOT(print_f, SEQUENCE_LENGTH, INPUT_DIM, LABEL_SET)
	eye_model.compile(learning_phase=1)

	# exit()

	# Unhappy hack to work around h5py not being able to write to GCS.
	# Force snapshots and saves to local filesystem, then copy them over to GCS.
	checkpoint_path = FILE_PATH
	if not job_dir.startswith("gs://"):
		checkpoint_path = os.path.join(job_dir, checkpoint_path)

	# Model checkpoint callback
	checkpoint = keras.callbacks.ModelCheckpoint(
		checkpoint_path,
		monitor='val_loss',
		verbose=1,
		period=checkpoint_epochs,
		mode='max')

	# TODO: load data
	print('Loading data')

	# X, y, y_raws, label_set = load_dataset(train_dirs, LABEL_SET, SEQUENCE_LENGTH, get_zip=get_zip)

	# exit()
	evaluation = EvalCheckPoint(eye_model, job_dir, eval_dirs, LABEL_SET, SEQUENCE_LENGTH, eval_freq=eval_frequency,
								print_func=print_f, epochs=epoch)

	# Tensorboard logs callback
	tblog = keras.callbacks.TensorBoard(
		log_dir=os.path.join(job_dir, 'logs'),
		histogram_freq=4,
		write_graph=True,
		embeddings_freq=0)

	callbacks = [checkpoint,
				 evaluation,
				 # tblog
				 ]

	# FIXME: unused part copied from census
	# eye_model.fit_generator(
	# 	model.generator_input(train_files, chunk_size=CHUNK_SIZE),
	# 	steps_per_epoch=train_steps,
	# 	epochs=num_epochs,
	# 	callbacks=callbacks)

	# TODO: fit
	# eye_model.fit(X, y, batch_size=batch, epochs=epoch, validation_split=split,
	# 			  callbacks=callbacks
	# 			  )
	eye_model.fit(train_dirs=train_dirs,
				  batch_size=batch,
				  epochs=epoch,
				  one_shot_freq=one_shot_freq,
				  validation_split=split,
				  callbacks=callbacks)

	utils.after_train(eye_model.model, MODEL_NAME, job_dir, print_fn=print_f)

	if model_file is not None:
		model_file.close()


# with utils.write_file(job_dir, "custom_logs.txt") as f:
# 	f.write("\n".join(custom_logs))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_dirs',
						required=True,
						type=str,
						help='Training files local or GCS', nargs='+')
	# help='Training files local or GCS', )
	parser.add_argument('--eval_dirs',
						required=True,
						type=str,

						help='Evaluation files local or GCS', nargs="+")
	parser.add_argument('--job-dir',
						required=True,
						type=str,
						help='GCS or local dir to write checkpoints and export model')
	parser.add_argument('--get_zip', help='Get zip file instead of folder', action='store_true')
	parser.set_defaults(get_zip=False)

	parser.add_argument('--epoch', type=int, default=10, help="\nNumber of epochs")
	parser.add_argument('--batch', type=int, default=32, help="\nNumber in a batch")
	parser.add_argument('--split', type=float, default=0.1, help="\nSplit in training")
	parser.add_argument('--eval-frequency',
						type=int,
						default=10,
						help='Perform one evaluation per n epochs')
	parser.add_argument('--one_shot_freq', type=int, default=3)

	# unneccessary arguments
	parser.add_argument('--train-steps',
						type=int,
						default=100,
						help="""\
                       Maximum number of training steps to perform
                       Training steps are in the units of training-batch-size.
                       So if train-steps is 500 and train-batch-size if 100 then
                       at most 500 * 100 training instances will be used to train.
                      """)

	parser.add_argument('--eval-steps',
						help='Number of steps to run evalution for at each checkpoint',
						default=100,
						type=int)
	parser.add_argument('--train-batch-size',
						type=int,
						default=40,
						help='Batch size for training steps')
	parser.add_argument('--eval-batch-size',
						type=int,
						default=40,
						help='Batch size for evaluation steps')
	parser.add_argument('--learning-rate',
						type=float,
						default=0.003,
						help='Learning rate for SGD')
	parser.add_argument('--first-layer-size',
						type=int,
						default=256,
						help='Number of nodes in the first layer of DNN')
	parser.add_argument('--num-layers',
						type=int,
						default=2,
						help='Number of layers in DNN')
	parser.add_argument('--scale-factor',
						type=float,
						default=0.25,
						help="""\
                      Rate of decay size of layer for Deep Neural Net.
                      max(2, int(first_layer_size * scale_factor**i)) \
                      """)
	parser.add_argument('--eval-num-epochs',
						type=int,
						default=1,
						help='Number of epochs during evaluation')
	parser.add_argument('--num-epochs',
						type=int,
						default=20,
						help='Maximum number of epochs on which to train')
	parser.add_argument('--checkpoint-epochs',
						type=int,
						default=5,
						help='Checkpoint per n training epochs')
	parse_args, unknown = parser.parse_known_args()

	dispatch(**parse_args.__dict__)
