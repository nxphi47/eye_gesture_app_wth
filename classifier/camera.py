import time
import numpy as np
try :
	from picamera import PiCamera
except ImportError:
	print 'PiCamera is not present, ignore this message if you are not in Raspberry Pi'

from PIL import Image, ImageOps


def normalize_image(img):
	# return (img - 127.5) / 127.5
	return (img.astype(np.float32) - 127.5) / 127.5

def denormalize_image(img):
	result = img * 127.5 + 127.5
	return result.astype(np.uint8)


NORMALIZE = True

class Camera():
	def __init__(self, camera_config):
		self.CHANNEL = camera_config.get('channel', 3)
		self.INPUT_DIM = camera_config.get('input_dim', 64)
		self.FRAMERATE = camera_config.get('framerate', 15)
		self.RESIZE = False
		self.SEQUENCE_LENGTH = camera_config.get('sequence_length', 15)
		self.TARGET_DIR = camera_config.get('target_dir', './datasets/')
		self.camera_config = camera_config
		self.pi_camera = None

	def initialize_pi_camera(self):
		self.pi_camera = PiCamera()
		self.pi_camera.resolution = [self.INPUT_DIM] * 2
		self.pi_camera.framerate = self.FRAMERATE
		self.pi_camera.rotation = -90
		self.pi_camera.crop = (0.2, 0.05, 0.8, 0.8)

	def capture_sequence(self, file_list):
		self.initialize_pi_camera()

		self.pi_camera.start_preview()
		time.sleep(2.0)
		self.pi_camera.capture_sequence(file_list, use_video_port=True)
		self.pi_camera.stop_preview()

		self.pi_camera = None

	def capture_sequence_to_array(self, sequence_length):
		interval = 1.0 / self.FRAMERATE

		data = np.empty((sequence_length, self.CHANNEL * self.INPUT_DIM ** 2), dtype=np.uint8)

		self.pi_camera.capture_sequence(data, 'rgb', use_video_port=True)

		data = np.array(data).reshape((sequence_length, self.INPUT_DIM, self.INPUT_DIM, self.CHANNEL))
		if NORMALIZE:
			return normalize_image(data)
		else:
			return data

	def capture(self, params={}):
		pass

	def start_preview(self, **params):
		if self.pi_camera is not None:
			self.pi_camera.start_preview(**params)
		else:
			raise Exception('You have to init camera first')

	def stop_preview(self):
		if self.pi_camera is not None:
			self.pi_camera.stop_preview()

	def destroy(self):
		self.pi_camera = None