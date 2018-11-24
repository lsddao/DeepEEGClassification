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
		
	# Creates dataset from configured folder with PNG images
	# Subfolder == image class
	def createDataset(self):
		nbPerClass = self.config.nbPerClass
		classes = self.config.classes 
		sliceSize = self.config.imageSize
		validationRatio = self.config.validationRatio
		testRatio = self.config.testRatio
		imagesPath = self.config.imagesPath

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
				imgData = getImageData(imagesPath+image_class+"/"+filename, sliceSize)
				label = [1. if image_class == g else 0. for g in classes]
				data.append((imgData,label))

		#Shuffle data
		shuffle(data)

		#Extract X and y
		X,y = zip(*data)

		#Split data
		validationNb = int(len(X)*validationRatio)
		testNb = int(len(X)*testRatio)
		trainNb = len(X)-(validationNb + testNb)

		#Prepare for Tflearn at the same time
		self.train_X = np.array(X[:trainNb]).reshape([-1, sliceSize, sliceSize, 1])
		self.train_y = np.array(y[:trainNb])
		self.validation_X = np.array(X[trainNb:trainNb+validationNb]).reshape([-1, sliceSize, sliceSize, 1])
		self.validation_y = np.array(y[trainNb:trainNb+validationNb])
		self.test_X = np.array(X[-testNb:]).reshape([-1, sliceSize, sliceSize, 1])
		self.test_y = np.array(y[-testNb:])
		print("Dataset created!")