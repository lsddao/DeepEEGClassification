import pickle
from random import shuffle
import numpy as np

class Model:
	def __init__(self, config):
		self.config = config
		self.model = None
		self.dataProvider = None
	
	def createModel(self):
		raise NotImplementedError

	def loadModel(self):
		raise NotImplementedError

	def datasetName(self):
		raise NotImplementedError

	def loadDataset(self):
		datasetName = self.datasetName()
		datasetPath = self.config.datasetPath
		print("Loading datasets... ")
		self.train_X = pickle.load(open("{}train_X_{}.p".format(datasetPath,datasetName), "rb" ))
		self.train_y = pickle.load(open("{}train_y_{}.p".format(datasetPath,datasetName), "rb" ))
		self.validation_X = pickle.load(open("{}validation_X_{}.p".format(datasetPath,datasetName), "rb" ))
		self.validation_y = pickle.load(open("{}validation_y_{}.p".format(datasetPath,datasetName), "rb" ))
		print("Datasets loaded!")

	def saveDataset(self):
		print("Saving dataset... ")
		datasetName = self.datasetName()
		datasetPath = self.config.datasetPath
		pickle.dump(self.train_X, open("{}train_X_{}.p".format(datasetPath,datasetName), "wb" ))
		pickle.dump(self.train_y, open("{}train_y_{}.p".format(datasetPath,datasetName), "wb" ))
		pickle.dump(self.validation_X, open("{}validation_X_{}.p".format(datasetPath,datasetName), "wb" ))
		pickle.dump(self.validation_y, open("{}validation_y_{}.p".format(datasetPath,datasetName), "wb" ))
		print("Dataset saved!")

	def trainModel(self):
		raise NotImplementedError

	def saveModel(self):
		raise NotImplementedError

	def correctedAcc(self, acc):
		rnd_acc = 1.0/self.config.nbClasses
		return (acc - rnd_acc) / (1.0 - rnd_acc)
	
	def trainAccuracy(self):
		raise NotImplementedError

	def validationAccuracy(self):
		raise NotImplementedError

	def createDataset(self):
		print("Creating dataset...")	
		data = self.dataProvider.getData()

		if self.dataProvider.shuffleAllowed():
			shuffle(data)

		#Extract X and y
		X,y = zip(*data)

		#Split data
		validationNb = int(len(X)*self.config.validationRatio)
		trainNb = len(X)-validationNb
		
		#Prepare test arrays
		x_shape = self.dataProvider.X_shape()

		self.train_X = np.array(X[:trainNb]).reshape(x_shape)
		self.train_y = np.array(y[:trainNb])
		self.validation_X = np.array(X[trainNb:]).reshape(x_shape)
		self.validation_y = np.array(y[trainNb:])

		print("Dataset created!")