import glob
import os
import numpy as np
from PIL import Image
from PIL import ImageOps
import random
import pprint
import shutil

import argparse
import camera

TRAIN_TARGET_DIR = './datasets'
FRAMERATE = 15
LABEL_SET = ['left', 'right', 'up', 'down', 'center', 'double_blink']
CHANNEL = 3
INPUT_DIM = 64

def record(camera_config):
	# target_dir = config.TRAIN_TARGET_DIR
	target_dir = "{}/{}/{}/".format(TRAIN_TARGET_DIR, camera_config['person'],  camera_config['label'])



	if not os.path.exists(target_dir):
		os.makedirs(target_dir)
	# os.makedirs(target_dir)
	folders = os.listdir(target_dir)
	target_dir = "{}{}".format(target_dir, 'batch_{}/'.format(len(folders)))
	if not os.path.exists(target_dir):
		os.makedirs(target_dir)
	# os.makedirs(target_dir, exist_ok=True)

	target_paths = ["{}img_{}.jpg".format(target_dir, i) for i in range(camera_config.get('duration') * camera_config.get('framerate', FRAMERATE))]

	# exit()
	pi_camera = camera.Camera(camera_config)
	pi_camera.capture_sequence(target_paths)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--label', help='Label ({})'.format(LABEL_SET), type=str, choices=LABEL_SET, required=True)
	parser.add_argument('-p', '--person', help='person', type=str, required=True)
	parser.add_argument('-d', '--duration', help='Duration of recording, in seconds <30', type=int, default=10)
	parser.add_argument('-f', '--framerate', help='Frame rate of recording < 30', type=int, default=FRAMERATE)
	parser.add_argument('-c', '--channel', help='Channel 1 or 3', type=int, default=CHANNEL, choices=[3, 1])
	parser.add_argument('-s', '--size', help='image size', type=int, default=INPUT_DIM)

	# parser.add_argument('--batch', help='Batch size ({})'.format(config.BATCH_SIZE), type=int, default=config.BATCH_SIZE)
	# parser.add_argument('--split', help='Split ({})'.format(config.SPLIT), type=float, default=config.SPLIT)
	# parser.add_argument('--bw', help='Black and white (1 channel)', action="store_true")


	args = parser.parse_args()
	if args.label not in LABEL_SET:
		raise Exception('label {} is wrong'.format(args.label))
	camera_config = {
		'person': args.person,
		'label': args.label,
		'duration': args.duration,
		'framerate': args.framerate
	}

	pprint.pprint(camera_config)

	record(camera_config)

if __name__ == '__main__':
	main()
