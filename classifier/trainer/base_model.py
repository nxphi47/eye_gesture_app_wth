"""
This is general model
"""
from __future__ import division
from __future__ import print_function

import json
import os
import pprint

import keras
import keras.backend as K
import numpy as np
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split

import utils

INPUT_DIM = 64
CHANNEL = 3

class ClassiferTfModel():
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
		self.true_val = None
		self.X_val = None
		self.y = None
		self.y_val = None
		if self.label_set is None:
			self.label_set = utils.LABEL_SET
		self.eye = np.eye(len(self.label_set))

	def load_config(self):
		pass

	def compile(self, **kwargs):
		pass

	def predict(self, data):
		return None

	def process_training_data(self, train_files, split=0.15):
		# FIXME: only accept 1 single file at this momment
		if isinstance(train_files, (list, tuple)):
			train_files = train_files[0]
		X, y = utils.load_npz(train_files)
		self.X, self.X_val, self.y, self.y_val = train_test_split(X, y, test_size=split, random_state=42, stratify=y)

		print('Train Shape x: {}'.format(self.X.shape))
		print('Train Shape y: {}'.format(self.y.shape))
		print('Eval Shape X: {}'.format(self.X_val.shape))
		print('Eval Shape y: {}'.format(self.y_val.shape))

	def test_on_trained(self, test_files):
		# if test_files is not None:
		# 	self.print_f('-- Perform Testing --')
		# 	if isinstance(test_files, (list, tuple)):
		# 		test_files = test_files[0]
		# 	X, y = utils.load_npz(test_files)
		# 	print('Test Shape x: {}'.format(self.X.shape))
		# 	print('Test Shape y: {}'.format(self.y.shape))
		# 	assert self.model is not None
		# 	pred_val = np.argmax(self.predict(X), axis=1)
		# 	true_val = np.argmax(y, axis=1)
		#
		# 	utils.report(true_val, pred_val, self.label_set, print_fn=self.print_f)
		#
		# model_name = 'eye_final_model.hdf5'
		# self.model.save(os.path.join(self.job_dir, model_name))
		#
		# # Convert the Keras model to TensorFlow SavedModel
		# self.print_f('Save model to {}'.format(self.job_dir))
		# utils.to_savedmodel(self.model, os.path.join(self.job_dir, 'export'))

		pass

	def fit(self, train_files, test_files=None, batch_size=32, epochs=10, validation_split=0.1, callbacks=None,
			**kwargs):
		pass



class ClassiferKerasModel(ClassiferTfModel):
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
		# self.config_file = config_file
		# self.config = None
		# self.job_dir = job_dir
		# self.checkpoint_path = checkpoint_path
		# self.print_f = print_f
		# self.sequence_length = sequence_length
		# self.input_dim = input_dim
		# self.label_set = label_set
		# self.batch_norm = batch_norm
		#
		# self.model = None
		# self.feed_dict = None
		# self.X = None
		# self.X_val = None
		# self.y = None
		# self.y_val = None
		# self.eye = np.eye(len(self.label_set))

	def load_config(self):
		self.config = json.load(open(self.config_file, 'r'))
		pprint.pprint(self.config)
		if "cnn_block" in self.config:
			self.batch_norm = self.config['cnn_block'].get('batch_norm', True)
		else:
			self.batch_norm = True


	def compile(self, **kwargs):
		self.load_config()
		pass
		# if self.batch_norm:
		# 	K.set_learning_phase(1)
		# self.model = densenet_sequential_model(config=self.config,
		# 									   print_fn=self.print_f,
		# 									   sequence_length=self.sequence_length,
		# 									   input_dim=self.input_dim,
		# 									   label_set=self.label_set,
		# 									   dropout=0.5, batch_norm=self.batch_norm)
		# self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mae'])

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

		print('Train Shape x: {}'.format(self.X.shape))
		print('Train Shape y: {}'.format(self.y.shape))

	def test_on_trained(self, test_files):
		if test_files is not None:
			self.print_f('-- Perform Testing --')
			if isinstance(test_files, (list, tuple)):
				test_files = test_files[0]
			X, y = utils.load_npz(test_files)
			print('Test Shape x: {}'.format(self.X.shape))
			print('Test Shape y: {}'.format(self.y.shape))
			assert self.model is not None
			pred_val = np.argmax(self.predict(X), axis=1)
			true_val = np.argmax(y, axis=1)

			utils.report(true_val, pred_val, self.label_set, print_fn=self.print_f)

		model_name = 'eye_final_model.hdf5'
		self.model.save(os.path.join(self.job_dir, model_name))

		# Convert the Keras model to TensorFlow SavedModel
		self.print_f('Save model to {}'.format(self.job_dir))
		utils.to_savedmodel(self.model, os.path.join(self.job_dir, 'export'))

	def fit(self, train_files,
			test_files=None,
			batch_size=32, epochs=10, validation_split=0.1, callbacks=None,
			**kwargs):
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
				TensorBoard(log_dir=self.job_dir,
							histogram_freq=1,
							batch_size=batch_size,
							),
				utils.EvalCheckPoint(self.model,
									 self.job_dir,
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