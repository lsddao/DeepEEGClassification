import pickle
import datetime as dt

class Model:
	def __init__(self, config):
		self.config = config
		self.model = None
	
	def createModel(self):
		raise NotImplementedError

	def loadModel(self):
		self.model.load('eegDNN.tflearn')

	def createDataset(self):
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
		self.test_X = pickle.load(open("{}test_X_{}.p".format(datasetPath,datasetName), "rb" ))
		self.test_y = pickle.load(open("{}test_y_{}.p".format(datasetPath,datasetName), "rb" ))
		print("Datasets loaded!")

	def saveDataset(self):
		print("Saving dataset... ")
		datasetName = self.datasetName()
		datasetPath = self.config.datasetPath
		pickle.dump(self.train_X, open("{}train_X_{}.p".format(datasetPath,datasetName), "wb" ))
		pickle.dump(self.train_y, open("{}train_y_{}.p".format(datasetPath,datasetName), "wb" ))
		pickle.dump(self.validation_X, open("{}validation_X_{}.p".format(datasetPath,datasetName), "wb" ))
		pickle.dump(self.validation_y, open("{}validation_y_{}.p".format(datasetPath,datasetName), "wb" ))
		pickle.dump(self.test_X, open("{}test_X_{}.p".format(datasetPath,datasetName), "wb" ))
		pickle.dump(self.test_y, open("{}test_y_{}.p".format(datasetPath,datasetName), "wb" ))
		print("Dataset saved!")

	def trainModel(self):
		run_id = 'EEG_{}'.format(dt.datetime.now().strftime("%Y%m%d%H%M%S"))
		self.model.fit(self.train_X, self.train_y, n_epoch=self.config.nbEpoch, batch_size=self.config.batchSize, 
			shuffle=True, validation_set=(self.validation_X, self.validation_y), snapshot_step=100, show_metric=True, run_id=run_id)

	def saveModel(self):
		self.model.save('eegDNN.tflearn')

	def correctedAcc(self, acc):
		rnd_acc = 1.0/self.config.nbClasses
		return (acc - rnd_acc) / (1.0 - acc)

	def testAccuracy(self):
		res = self.model.evaluate(self.test_X, self.test_y)
		return res[0]