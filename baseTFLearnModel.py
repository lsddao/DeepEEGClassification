from model import Model

import datetime as dt

class BaseTFLearnModel(Model):
	def __init__(self, config):
		super().__init__(config)

	def loadModel(self):
		self.model.load('eegDNN.tflearn')

	def trainModel(self):
		run_id = 'EEG_{}'.format(dt.datetime.now().strftime("%Y%m%d%H%M%S"))
		self.model.fit(self.train_X, self.train_y, n_epoch=self.config.nbEpoch, batch_size=self.config.batchSize, 
			shuffle=True, validation_set=(self.validation_X, self.validation_y), snapshot_step=100, show_metric=True, run_id=run_id)

	def saveModel(self):
		self.model.save('eegDNN.tflearn')

	def testAccuracy(self):
		res = self.model.evaluate(self.test_X, self.test_y)
		return res[0]