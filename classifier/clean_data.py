from __future__ import print_function
import glob
import os
import numpy as np
from PIL import Image
# import tensorflow as tf
import argparse
import random
import traceback


def preprocess_data(src_dirs, target_dir):
	# source_dir = '/home/nxphi47/Downloads/DIP_DATA/'
	# # labels = ['left', 'right', 'up', 'down', 'center']
	# labels = ['right', 'up', 'down', 'center']
	# src_dirs = {}
	# # src_dirs['left'] = ['{}{}/'.format(source_dir, folder) for folder in  ['left', 'left2', 'left3']]
	# src_dirs['right'] = ['{}{}/'.format(source_dir, folder) for folder in  ['right', 'right2', 'right3']]
	# src_dirs['up'] = ['{}{}/'.format(source_dir, folder) for folder in  ['up', 'up2', 'up3', 'dip_up']]
	# src_dirs['down'] = ['{}{}/'.format(source_dir, folder) for folder in  ['down', 'down2', 'down3']]
	# src_dirs['center'] = ['{}{}/'.format(source_dir, folder) for folder in  ['center', 'center2', 'center3']]
	resolution = 64
	# target_dir = '../../datasets/{}/'.format(resolution)

	# ["dir/person/test(train)/left(...)/batch/*.jpg]
	count = {}
	print(src_dirs)
	# exit()

	for src_d in src_dirs:
		print('Preparing Src: {}'.format(src_d))
		# FIXME input dirs must have "/" at the end
		person_name = src_d.split("/")[-2]
		count = 0
		#img_list = glob.glob("{src}*/*/*/*.jpg".format(src=src_d))
		img_list = glob.glob("{src}*/*.JPG".format(src=src_d))
		#print (img_list)
		for img_url in img_list:
			target_url = "{target}{name}/{child}".format(target=target_dir, name=person_name, child=img_url.split(src_d)[-1])
			target_dir_url = "/".join(target_url.split("/")[:-1])
			# print (target_url)
			# print (target_dir_url)
			if not os.path.exists(target_dir_url):
				os.makedirs(target_dir_url)

			count += 1
			if count % 1000:
				print ("--Save image: {}".format(img_url))

			# saving
			try:
				#image = Image.open(img_url).crop((84, 0, 384, 384)).resize((resolution, resolution), Image.ANTIALIAS)
				image = Image.open(img_url).resize((resolution, resolution), Image.ANTIALIAS)
				image.save(target_url)
				# image.callback(target_url)
			except KeyboardInterrupt:
				print ('Keyboard interrupt')
				exit()
			except Exception as e:
				print('Cannot save {} at {}'.format(img_url, target_url))
				traceback.print_exc(e)
				exit()

	# for l in labels:
	# 	print('Saving label {}'.format(l))
	# 	count[l] = 0
	#
	# 	for src_dir in src_dirs[l]:
	# 		# print 'looking at dir {}'.format(src_dir + '*.JPG')
	# 		img_list = glob.glob('{}*.*'.format(src_dir))
	# 		# print img_list
	# 		for img in img_list:
	#
	# 			try:
	# 				image = Image.open(img).crop(64, 0, 384, 384).resize((resolution, resolution), Image.ANTIALIAS)
	# 				image.callback('{target_dir}{label}/{count}.jpg'.format(target_dir=target_dir, label=l, count=count[l]))
	# 				count[l] += 1
	#
	# 				# if count[l] % 300:
	# 				print('Save {} at index {}'.format(l, count[l]))
	# 			except Exception as e:
	# 				print('Cannot save {} at index {}'.format(l, count[l]))
	# 				traceback.print_stack(e)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--src_dirs', type=str, required=True, nargs="+")
	parser.add_argument('--target_dir', type=str, required=True)

	parse_args, unknown = parser.parse_known_args()
	preprocess_data(**parse_args.__dict__)

if __name__ == '__main__':
	main()
