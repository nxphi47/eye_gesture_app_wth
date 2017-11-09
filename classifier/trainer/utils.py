#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

# from keras.layers.merge import Concatenate, Add, Dot, Multiply
import glob
import os
import zipfile

import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

LABEL_SET = ['left', 'right', 'up', 'down', 'center', 'double_blink']

def normalize_image(img):
	# return (img - 127.5) / 127.5
	return (img.astype(np.float32) - 127.5) / 127.5

def denormalize_image(img):
	result = img * 127.5 + 127.5
	return result.astype(np.uint8)

NORMALIZE = False

def to_savedmodel(model, export_path):
	"""Convert the Keras HDF5 model into TensorFlow SavedModel."""
	builder = saved_model_builder.SavedModelBuilder(export_path)

	signature = predict_signature_def(inputs={'input': model.inputs[0]},
									  outputs={'income': model.outputs[0]})

	with K.get_session() as sess:
		builder.add_meta_graph_and_variables(
			sess=sess,
			tags=[tag_constants.SERVING],
			signature_def_map={
				signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
		)
		builder.save()

def session_to_savedmodel(session, inputs, outputs, export_path):
	"""Convert the Keras HDF5 model into TensorFlow SavedModel."""
	builder = saved_model_builder.SavedModelBuilder(export_path)

	signature = predict_signature_def(inputs={'inputs': inputs},
									  outputs={'outputs': outputs})

	builder.add_meta_graph_and_variables(
		sess=session,
		tags=[tag_constants.SERVING],
		signature_def_map={
			signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
	)
	builder.save()

def session_from_savedmodel(session, export_dir):
	tf.saved_model.loader.load(session, [tag_constants.SERVING], export_dir)


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

# key compare urls
OFFSET_BATCH = 1000000
def key_compare_url(a):
	ia = int(a.split('/')[-1].replace('img_', '').split('.')[0])
	batch_num = int(a.split('/')[-2].replace('batch_', ''))
	# prefix_a = '/'.join(a.split('/')[:-1])
	return batch_num * OFFSET_BATCH + ia

def load_npz(url):
	files = np.load(url)
	return files['X'], files['y']


# big problem with sorting data!!!!!!
def load_dataset(base_urls, label_set, sequence_length=15, get_zip=True):
	globs = {}
	print(base_urls)
	zips = {}
	zip_dirs = {}
	if os.path.isdir(base_urls[0]):
		get_zip = False

	if not get_zip:
		for l in label_set:
			globs[l] = []
			for d in base_urls:
				# print ('Open folder label {} in {}'.format(l, d))
				path = os.path.join(d, l)
				# print (path)
				globs[l] += glob.glob('{dir}/*/*.jpg'.format(dir=path))
			globs[l].sort(compare_url)
	else:
		for d in base_urls:
			zips[d] = zipfile.ZipFile(retrieve_file(d), 'r')
			# zips[d] = GzipFile(d, 'r+b')
			zip_dirs[d] = {}
			z_namelist = [n for n in zips[d].namelist() if n.split(".")[-1].lower() == 'jpg']
			for l in label_set:
				zip_dirs[d][l] = [n for n in z_namelist if l in n]
				# zip_dirs[d][l].sort(compare_url)
				zip_dirs[d][l].sort(key=key_compare_url)
				# for u in zip_dirs[d][l]:
				# 	print(u)

	# datasets
	X = []
	y = []
	y_raws = []
	eye = np.eye(len(label_set))

	for i, l in enumerate(label_set):
		print('Label: {}'.format(l))
		if get_zip:
			for d in base_urls:
				data = []
				print('---Read Zip file: {}'.format(d))
				for j, img_url in enumerate(zip_dirs[d][l]):
					with Image.open(zips[d].open(img_url, 'r')) as img:
						# img = Image.open(zips[d].open(img_url, 'r'))
						if NORMALIZE:
							img_array = normalize_image(np.array(img))
						else:
							img_array = np.array(img)

						if j % sequence_length == 0 and j != 0:
							# package into sequence
							X.append(np.array(data))
							y.append(np.array(eye[i]))
							y_raws.append(l)

							data = []
						# else:
						data.append(img_array)


		else:
			data = []
			for j, img_url in enumerate(globs[l]):
				# if j >= 61:
				# 	break

				with Image.open(img_url) as img:
					# img = Image.open(img_url)
					if NORMALIZE:
						img_array = normalize_image(np.array(img))
					else:
						img_array = np.array(img)
					# img_array = normalize_image(np.array(img))
					if j % sequence_length == 0 and j != 0:
						# package into sequence
						X.append(np.array(data))
						y.append(np.array(eye[i]))
						y_raws.append(l)
						data = []
					# else:
					data.append(img_array)

	if get_zip:
		for d in base_urls:
			zips[d].close()

	X = np.array(X)
	y = np.array(y)
	print(X.shape)
	print(y.shape)
	return X, y, y_raws, label_set


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
	with file_io.FileIO(file_path, mode='r') as input_f:
		with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
			output_f.write(input_f.read())

def write_file(job_dir, file_path):
	if "gs://" in file_path:
		print ('Write file to: {}/{}'.format(job_dir, file_path))
		# with  as f:
		return copy_file_to_gcs(job_dir, file_path)
	else:
		return open(file_path, 'r')

# read file handle opening of gsc
def retrieve_file(file_path):
	if "gs://" in file_path:
		print ('readata from gcs: {}'.format(file_path))
		# with  as f:
		return file_io.FileIO(file_path, 'r')
	else:
		return open(file_path, 'r+b')

def after_train(model, model_name, job_dir, print_fn=print):
# def after_train(model, model_file, model_dir, train_config, label_set, model_name='cnn_', print_fn=print):
	# Unhappy hack to work around h5py not being able to write to GCS.
	# Force snapshots and saves to local filesystem, then copy them over to GCS.
	if job_dir.startswith("gs://"):
		model.save(model_name)
		copy_file_to_gcs(job_dir, model_name)
	else:
		model.save(os.path.join(job_dir, model_name))

	# Convert the Keras model to TensorFlow SavedModel
	print_fn('Save model to {}'.format(job_dir))
	to_savedmodel(model, os.path.join(job_dir, 'export'))


def report(true_val, pred_val, label_set, epoch=0, print_fn=print, digits=4):
	report = classification_report(true_val, pred_val, target_names=label_set, digits=digits)
	matrix = confusion_matrix(true_val, pred_val)
	print_fn("----- Epoch:{} -----".format(epoch))
	print_fn(report)
	print_fn(matrix)


class EvalCheckPoint(keras.callbacks.Callback):
	def __init__(self, ml_model,
				 job_dir,
				 X, y,
				 label_set,
				 sequence_lenth,
				 eval_freq=4,
				 print_func=print,
				 epochs=10,
				 batch_norm=False
				 ):
		self.job_dir = job_dir
		self.label_set = label_set
		self.sequence_length = sequence_lenth

		self.X_test = X
		self.y_test = y
		self.batch_norm = batch_norm

		self.epochs = epochs
		self.eval_freq = eval_freq
		self.model = None
		self.print_func = print_func
		self.set_model(ml_model)
		self.true_val = None
		self.pred_val = None

		self.true_val = np.array(np.argmax(self.y_test, axis=1))

	def on_epoch_begin(self, epoch, logs={}):
		if epoch > 0 and (epoch % self.eval_freq == 0 or epoch == self.epochs):
			if self.model is not None:
				# if self.batch_norm:
				K.set_learning_phase(0)
				pred_val = np.argmax(self.model.predict(self.X_test), axis=1)
				K.set_learning_phase(1)

				report(self.true_val, pred_val, self.label_set, print_fn=self.print_func)
