# import tensorflow as tf
from __future__ import absolute_import
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from tempfile import TemporaryFile
import os
from PIL import Image

import utils
import time

import cv2

LABEL_SET = ['left', 'right', 'up', 'down', 'center', 'double_blink']
DATASETS_SRC_DIR = './datasets/'
SEQUENCE_LENGTH = 15

INPUT_DIM = 64


# def _bytes_feature(value):
# 	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
# def _int64_feature(value):
# 	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# main function to transform zip file into tfrecords file

def augment_brightness_camera_images(image, bright_random):
	image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	# print(random_bright)
	image1[:, :, 2] = image1[:, :, 2] * bright_random
	image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
	return image1


def transform_image(img, ang_rot, pt1, pt2, tr_x, tr_y, bright_random):
	'''
	This function transforms images to generate new images.
	The function takes in following arguments,
	1- Image
	2- ang_range: Range of angles for rotation
	3- shear_range: Range of values to apply affine transform to
	4- trans_range: Range of values to apply translations over.

	A Random uniform distribution is used to generate different parameters for transformation

	'''
	# Rotation

	# ang_rot = np.random.uniform(ang_range) - ang_range / 2
	rows, cols, ch = img.shape
	Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

	# Translation
	# tr_x = trans_range * np.random.uniform() - trans_range / 2
	# tr_y = trans_range * np.random.uniform() - trans_range / 2
	Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

	# Shear
	pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

	# pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
	# pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

	# Brightness

	pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

	shear_M = cv2.getAffineTransform(pts1, pts2)

	img = cv2.warpAffine(img, Rot_M, (cols, rows))
	img = cv2.warpAffine(img, Trans_M, (cols, rows))
	img = cv2.warpAffine(img, shear_M, (cols, rows))

	img = augment_brightness_camera_images(img, bright_random)

	# testing only
	# path = '../augment'
	# if not os.path.exists(path):
	# 	os.mkdir(path)
	# Image.fromarray(img).save(os.path.join(path, '{}.jpg'.format(time.time())))

	return img

def transform_video(video):
	# rotation_random = np.random.uniform()
	ang_range = 5
	shear_range = 2
	trans_range = 5

	ang_rot = np.random.uniform(ang_range) - ang_range / 2

	tr_x = trans_range * np.random.uniform() - trans_range / 2
	tr_y = trans_range * np.random.uniform() - trans_range / 2

	pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
	pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

	random_bright = .25 + np.random.uniform()

	new_video=None
	for frame in video:
		new_img = transform_image(frame, ang_rot, pt1, pt2, tr_x, tr_y, random_bright)
		if new_video is None:
			new_video = np.array([new_img])
		else:
			new_video = np.concatenate((new_video, [new_img]))
	return new_video


def dispatch(files, out, split, augment_count=0):
	X, y, _, __ = utils.load_dataset(files, LABEL_SET, SEQUENCE_LENGTH)

	new_x=None
	new_y=None
	print('size {}'.format(X.shape))

	if augment_count > 0:
		# augment data
		for i in range(len(X)):
			video = X[i]
			label = y[i]
			if new_x is None:
				new_x = np.array([video])
				new_y = np.array([label])
			else:
				new_x = np.concatenate((new_x, [video]))
				new_y = np.concatenate((new_y, [label]))

			for j in range(augment_count):
				new_vid = transform_video(video)
				new_x = np.concatenate((new_x, [new_vid]))
				new_y = np.concatenate((new_y, [label]))

			if i % 100 == 0:
				print('augmenting {}'.format(i))

		X = new_x
		y = new_y
		print('size {}'.format(X.shape))

	out_root_folder = '/'.join(out.split('/')[:-1])
	if not os.path.exists(out_root_folder):
		os.makedirs(out_root_folder)

	if split <= 0.0:
		file_name = '{}.npz'.format(out.split('/')[-1])
		path = os.path.join(out_root_folder, file_name)
		# writer = tf.python_io.TFRecordWriter(path)
		f = open(path, 'wb')
		np.savez(f, X=X, y=y)
		print('Done {}'.format(file_name))
	else:
		X, X_val, y, y_val = train_test_split(X, y, test_size=split, random_state=42, stratify=y)
		file_name_train = '{}_train.npz'.format(out.split('/')[-1])
		path = os.path.join(out_root_folder, file_name_train)
		with open(path, 'wb') as f:
			np.savez(f, X=X, y=y)
			print('Done {} {}'.format(file_name_train, X.shape))

		file_name_test = '{}_test.npz'.format(out.split('/')[-1])
		path = os.path.join(out_root_folder, file_name_test)
		with open(path, 'wb') as f:
			np.savez(f, X=X_val, y=y_val)
			print('Done {} {}'.format(file_name_test, X_val.shape))

		# writer = tf.python_io.TFRecordWriter(path)
		print('Done')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--files',
						required=True,
						type=str,
						help='Training zipfile', nargs='+')
	parser.add_argument('--out', required=True, type=str)
	parser.add_argument('--split', default=0.0, type=float)
	parser.add_argument('--augment_count', default=0, type=int)

	parse_args, unknown = parser.parse_known_args()

	dispatch(**parse_args.__dict__)
