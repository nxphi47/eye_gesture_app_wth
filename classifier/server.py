#!/usr/bin/env python
from __future__ import print_function
import argparse

import keras.backend as K
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
from keras.models import load_model

import camera

CHANNEL = 3
RESIZE = True
PORT = 80
NORMALIZE = False # normalize to be handle by the model
LABEL_SET = ['left', 'right', 'up', 'down', 'center', 'double_blink']
# DATASETS_SRC_DIR = './datasets/{dim}/'.format(dim=config.INPUT_DIM)
DATASETS_SRC_DIR = './datasets/'
SEQUENCE_LENGTH = 15
INPUT_DIM = 64
EPOCH = 10
BATCH = 32
SPLIT = 0.2
FRAMERATE = 15
model = None
query = None
result = None
pi_camera = None

app = Flask(__name__)
cors = CORS(app)


def normalize_image(img):
	# return (img - 127.5) / 127.5
	return (img.astype(np.float32) - 127.5) / 127.5


def denormalize_image(img):
	result = img * 127.5 + 127.5
	return result.astype(np.uint8)


@app.route('/ping')
def ip_ping():
	return jsonify({'result': True})


@app.route('/inference')
def inference():
	global query, model, result, pi_camera
	if pi_camera is not None:
		data = pi_camera.capture_sequence_to_array(SEQUENCE_LENGTH)

		# data = np.random.uniform(-1, 1, (15, 64, 64, 3))
		print("data retrieve from camera")

		if model is not None:
			try:
				print("start prediction")
				predicted = model.predict(np.array([data]))
				print("end of prediction")
				# label = np.argmax(label, axis=1)
				prop = predicted[0]
				label = LABEL_SET[np.argmax(prop)]
				prop = prop.tolist()
				json_obj = {
					'prob': prop,
					'label': label,
					'label_set': LABEL_SET
				}
				return jsonify(json_obj)
			except Exception as e:
				print(e)
				return jsonify({})

			# return jsonify(json_obj)
		else:
			return jsonify({})

	else:
		print('ERROR')
		raise Exception('pi_camera not found!')


# FIXME: deprecated
# @app.route('/predict', methods=['POST'])
# def predict():
# 	global query, model, result
# 	if request.method == 'POST':
# 		# predict
# 		# data = request.get_json(silent=True)
# 		# data to be 3d array from js
# 		# img = np.array([np.array([np.array([np.array(channel) for channel in col]) for col in row]) for row in data])
#
# 		f = request.files['file']
# 		path = 'static/' + secure_filename(str(random.randint(0, 1000000)) + f.filename)
# 		# path = '/var/www/uploads/' + secure_filename('testfile.jpg')
# 		# f.save(path)
#
# 		# print("image shape", img.shape
#
# 		img = Image.open(f)
# 		# img = img.resize((INPUT_DIM, INPUT_DIM), Image.ANTIALIAS)
# 		img_array = np.array([np.array(img)])
# 		img.save(path)
# 		print("Shape"
# 		print(img_array.shape
#
# 		if NORMALIZE:
# 			img_array = (img_array.astype(np.float32) - 127.5) / 127.5
#
# 		if model is not None:
#
# 			# if True:
# 			prediction = model.predict(img_array)
# 			prediction = prediction[0]
# 			index_pred = np.argmax(prediction)
# 			response = {
# 				"prob": prediction.tolist(),
# 				"pred": index_pred,
# 				"label": LABEL_SET[index_pred],
# 				"image": "{}".format(path)
# 			}
# 			pprint.pprint(response)
# 			return jsonify(response)
#
# 		else:
# 			print("no model"
# 			return jsonify({})
#
# 	else:
# 		# return null
# 		print('ERROR: method wrong'
# 		return jsonify({})


def load_cnn_model(model_name):
	global model, pi_camera
	model_dir = './models/'
	# root_dir_files = [rdf for rdf in os.walk(model_dir)]
	# files = root_dir_files[0][2]
	# model_name = 'dense_connected_cnn_{}.h5'.format(len(files) - 1)
	# files.sort()

	# model_name = MODEL_NAME
	# model_name = files[-1]

	print("open filename: {}".format(model_name))
	K.set_learning_phase(1)
	model = load_model('{}{}.h5'.format(model_dir, model_name))
	K.set_learning_phase(0)


	print("Finish load model with learning phase 0")

	camera_config = {
		'duration': 15,
		'framerate': FRAMERATE
		# 'framerate': 30
	}

	pi_camera = camera.Camera(camera_config)
	pi_camera.initialize_pi_camera()
	pi_camera.start_preview(alpha=50)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='Filename', type=str, required=True)
	args = parser.parse_args()


	# thread.start_new_thread(prediction_loop, ())
	print('run server')
	try:
		load_cnn_model(args.model)
		app.run(host='0.0.0.0', port=PORT, debug=False)
	except:
		if pi_camera is not None:
			pi_camera.stop_preview()
			pi_camera.destroy()
		exit()
	# app.run(host='0.0.0.0', port=3000, debug=False)


if __name__ == '__main__':
	main()
