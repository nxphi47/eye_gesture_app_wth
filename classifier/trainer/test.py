# from __future__ import print_function
# from __future__ import division

import tensorflow as tf
import keras
import keras.backend as K
import numpy as np
from keras.models import load_model
import argparse
import utils
import time
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

import cnn_rnn_model_raw
LABEL_SET = ['left', 'right', 'up', 'down', 'center', 'double_blink']


def dispatch(eval_files, model_file):

	# FIXME: for testing only
	# size = 1
	# X = np.random.randint(0, 255, [size, 15, 64, 64, 3])
	# eye = np.eye(6)
	# y = np.random.randint(0, 6, [size, ])
	# y = np.array([eye[i] for i in y])
	# print('{}'.format(y.shape))


	# K.set_learning_phase(1)
	# eye_model = load_model(model_file, compile=False)

	# eye_model = cnn_rnn_model_raw.CNN_RNN_Sequential_raw()
	# eye_model.load_model_from_savedmodel(model_file)
	with tf.Session(graph=tf.Graph()) as sess:
		tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_file)
		# ops = sess.graph.get_operations()
		# ops_name = [o.name for o in ops]
		# outputs = [n for n in ops_name if 'outputs' in n]
		# inputs = [n for n in ops_name if 'inputs' in n]
		# training = [n for n in ops_name if 'training' in n]
		# predictions = [n for n in ops_name if 'predictions' in n]
		# print(outputs)
		# print(inputs)
		# print(training)
		# print(predictions)
		inputs = sess.graph.get_tensor_by_name('inputs:0')
		outputs = sess.graph.get_tensor_by_name('outputs:0')
		training = sess.graph.get_tensor_by_name('training:0')
		predictions = sess.graph.get_tensor_by_name('predictions:0')
		# print()
		# print(outputs)
		# print(inputs)
		# print(training)
		# print(predictions)
		# x = np.random.randint(0, 255, size=[2, 15, 64, 64, 3])
		# eye = np.eye(6)
		# y = np.array([eye[i] for i in np.random.randint(0, 255, size=[2,])])


		print('finish loading model')
		# exit()

		if isinstance(eval_files, (list, tuple)):
			eval_files = eval_files[0]

		X, y = utils.load_npz(eval_files)
		# eye_model = load_model(model_file, custom_objects={"tf": tensorflow})

		print('finish loading data')
		# print(X.shape)
		# print(y.shape)
		# K.set_learning_phase(0)
		# idx = np.arange(0, len(y))
		# print('single predict')
		#
		# print(eye_model.predict(np.array([X[0]])))
		#

		# preds = eye_model.predict(X)

		# pred_val = sess.run(predictions, feed_dict={inputs: X, training: False})
		# true_val = np.argmax(y, axis=1)
		# utils.report(true_val, pred_val, LABEL_SET)

		s = time.time()
		pred_val = sess.run(predictions, feed_dict={inputs: [X[0]], training: False})
		print('Single Time: {}'.format(time.time() - s))



	# eye_model.end_train()

	# batch = 64
	# print('dividing to batch {}'.format(batch))
	# np.random.shuffle(idx)
	#
	# accuracies = []
	# for i in range(0, len(y) // batch):
	# 	xx = X[idx[i * batch:i * batch + batch]]
	# 	yy = y[idx[i * batch:i * batch + batch]]
	# 	preds = eye_model.predict(xx)
	# 	# print(preds.shape)
	# 	pred_val = np.argmax(preds, axis=1)
	# 	# K.set_learning_phase(1)
	# 	true_val = np.argmax(yy, axis=1)
	# 	# utils.report(true_val, pred_val, LABEL_SET)
	# 	acc = np.mean(np.equal(pred_val, true_val))
	# 	print(acc)
	# 	accuracies.append(acc)
	# print(accuracies)
	# print(np.mean(np.array(accuracies)))


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


"""
densenet big
# test on phi.test // may contains training data
----- Epoch:0 -----
              precision    recall  f1-score   support

        left     1.0000    0.9583    0.9787        24
       right     1.0000    1.0000    1.0000        34
          up     1.0000    1.0000    1.0000        34
        down     0.9714    1.0000    0.9855        34
      center     1.0000    0.9706    0.9851        34
double_blink     0.9714    1.0000    0.9855        34

 avg / total     0.9900    0.9897    0.9897       194

[[23  0  0  1  0  0]
 [ 0 34  0  0  0  0]
 [ 0  0 34  0  0  0]
 [ 0  0  0 34  0  0]
 [ 0  0  0  0 33  1]
 [ 0  0  0  0  0 34]]
Time: 63.73308181762695



"""