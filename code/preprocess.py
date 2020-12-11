from __future__ import division
from __future__ import print_function

import random
import glob
import numpy as np
import cv2
import tensorflow as tf


def preprocess(img, imgSize):
	"put img into target img of size imgSize, and resizes it, normalizing values"

	# there are damaged files in dataset - just use black image instead
	if img is None:
		img = np.zeros([imgSize[1], imgSize[0]])

	# resize the image
	(wt, ht) = imgSize
	(h, w) = img.shape
	fx = w / wt
	fy = h / ht
	f = max(fx, fy)
	newSize = (max(min(ht, int(h / fy)), 1), max(min(wt, int(w / fx)), 1)) # scale according to f (result at least 1 and at most wt or ht)
	img = cv2.resize(img, newSize)
	target = np.ones([wt, ht]) * 255
	target[0:newSize[1], 0:newSize[0]] = img

	# normalize
	(m, s) = cv2.meanStdDev(img)
	m = m[0][0]
	s = s[0][0]
	img = img - m
	img = img / s if s>0 else img
	return img

class DataLoader:
	"loads data and provides an easy way to get batches of data from the given dataset"

	def __init__(self, filePath, batchSize, imgSize):
		"loader for dataset at given location, preprocess images and text according to parameters"

		assert filePath[-1]=='/'

		self.currIdx = 0
		self.batchSize = batchSize
		self.imgSize = imgSize
		self.samples = np.array(glob.glob(filePath+'/**/*.png', recursive=True))
		shuffle = tf.random.shuffle(tf.range(len(self.samples)))
		self.samples = np.take(np.array(self.samples), shuffle)


	def getNext(self):
		"iterator"
		batchRange = range(self.currIdx, self.currIdx + self.batchSize)
		imgs = [preprocess(cv2.imread(self.samples[i], cv2.IMREAD_GRAYSCALE), self.imgSize) for i in batchRange]
		self.currIdx += self.batchSize
		return np.float32(np.array(imgs))
