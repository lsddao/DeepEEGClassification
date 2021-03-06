from model import Model

import datetime as dt

class BaseTFLearnModel(Model):
	def __init__(self, config):
		super().__init__(config)

	def getRunID(self):
		return 'EEG_{}_'.format(dt.datetime.now().strftime("%Y%m%d%H%M%S"))

	def loadModel(self):
		self.model.load('eegDNN.tflearn')

	def trainModel(self):
		self.model.fit(self.train_X, self.train_y, n_epoch=self.config.nbEpoch, batch_size=self.config.batchSize, 
			shuffle=True, validation_set=(self.validation_X, self.validation_y), snapshot_step=None, show_metric=True, run_id=self.getRunID())

	def saveModel(self):
		self.model.save('eegDNN.tflearn')

	def trainAccuracy(self):
		return self.model.evaluate(self.train_X, self.train_y, batch_size=self.config.batchSize)[0]

	def validationAccuracy(self):
		return self.model.evaluate(self.validation_X, self.validation_y, batch_size=self.config.batchSize)[0]