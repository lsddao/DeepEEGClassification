import model
import os
import numpy as np
from random import shuffle
from imageFilesTools import getImageData
from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

class CNNModel(model.Model):
	def __init__(self, config):
		super().__init__(config)
	
	def createModel(self):
		print("Creating model...")
		imageSize = self.config.imageSize
		nbClasses = self.config.nbClasses

		net = input_data(shape=[None, imageSize, imageSize, 1], name='input')

		net = conv_2d(net, imageSize/2, 2, activation='elu', weights_init="Xavier")
		net = max_pool_2d(net, 2)

		net = conv_2d(net, imageSize, 2, activation='elu', weights_init="Xavier")
		net = max_pool_2d(net, 2)

		net = conv_2d(net, imageSize*2, 2, activation='elu', weights_init="Xavier")
		net = max_pool_2d(net, 2)

		net = conv_2d(net, imageSize*4, 2, activation='elu', weights_init="Xavier")
		net = max_pool_2d(net, 2)

		net = fully_connected(net, imageSize*8, activation='elu')
		net = dropout(net, 0.5)

		net = fully_connected(net, nbClasses, activation='softmax')
		net = regression(net, optimizer='rmsprop', loss='categorical_crossentropy')

		self.model = DNN(net)
		print("Model created!")

	def datasetName(self):
		return 'CNNModel'

	def X_shape(self):
		imageSize = self.config.imageSize
		return [-1, imageSize, imageSize, 1]
		
	# Creates dataset from configured folder with PNG images
	# Subfolder == image class
	def getData(self):
		nbPerClass = self.config.nbPerClass
		classes = self.config.classes 
		imagesPath = self.config.imagesPath
		imageSize = self.config.imageSize

		data = []
		for image_class in classes:
			print("-> Adding {}...".format(image_class))
			#Get slices in class subfolder
			filenames = os.listdir(imagesPath+image_class)
			filenames = [filename for filename in filenames if filename.endswith('.png')]
			#Randomize file selection for this class
			shuffle(filenames)
			filenames = filenames[:nbPerClass]

			#Add data (X,y)
			for filename in filenames:
				imgData = getImageData(imagesPath+image_class+"/"+filename, imageSize)
				label = [1. if image_class == g else 0. for g in classes]
				data.append((imgData,label))

		return data