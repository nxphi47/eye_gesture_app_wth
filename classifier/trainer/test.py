import tensorflow as tf
import keras
import keras.backend as K
import numpy as np
from keras.models import load_model
import argparse
import utils

LABEL_SET = ['left', 'right', 'up', 'down', 'center', 'double_blink']


def dispatch(eval_files, model_file):

	K.set_learning_phase(1)
	eye_model = load_model(model_file, custom_objects={"tf": tf})
	print('finish loading model')
	if isinstance(eval_files, (list, tuple)):
		eval_files = eval_files[0]
	X, y = utils.load_npz(eval_files)
	print('finish loading data')
	# print(X.shape)
	# print(y.shape)
	K.set_learning_phase(0)
	idx = np.arange(0, len(y))
	print('single predict')
	print(eye_model.predict(np.array([X[10]])))

	preds = eye_model.predict(X[idx])
	pred_val = np.argmax(preds, axis=1)
	true_val = np.argmax(y[idx], axis=1)
	utils.report(true_val, pred_val, LABEL_SET)

	batch = 64
	print('dividing to batch {}'.format(batch))
	np.random.shuffle(idx)

	accuracies = []
	for i in range(0, len(y) // batch):
		xx = X[idx[i * batch:i * batch + batch]]
		yy = y[idx[i * batch:i * batch + batch]]
		preds = eye_model.predict(xx)
		# print(preds.shape)
		pred_val = np.argmax(preds, axis=1)
		# K.set_learning_phase(1)
		true_val = np.argmax(yy, axis=1)
		# utils.report(true_val, pred_val, LABEL_SET)
		acc = np.mean(np.equal(pred_val, true_val))
		print(acc)
		accuracies.append(acc)
	print(accuracies)
	print(np.mean(np.array(accuracies)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('--train_dirs',
	# 					required=True,
	# 					type=str,
	# 					help='Training files local or GCS', nargs='+')
	# help='Training files local or GCS', )
	parser.add_argument('--eval_files',
						required=True,
						type=str,)
	parser.add_argument('--model_file', required=True, type=str)
	parse_args, unknown = parser.parse_known_args()

	dispatch(**parse_args.__dict__)