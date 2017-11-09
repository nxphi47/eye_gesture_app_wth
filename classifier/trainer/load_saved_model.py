#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
# from keras.layers.merge import Concatenate, Add, Dot, Multiply
import glob
import os
from PIL import Image
import numpy as np
import keras
from keras import backend as K
from keras.layers import Input, Activation, Conv2D, Dense, Dropout, \
	LSTM, Bidirectional, TimeDistributed, MaxPooling2D, BatchNormalization, AveragePooling2D, Flatten
from keras.models import Model, Sequential
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
import zipfile
from gzip import GzipFile
from tensorflow.python.lib.io import file_io

if __name__ == '__main__':

	pass