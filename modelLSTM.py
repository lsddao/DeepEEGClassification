from baseTFLearnModel import BaseTFLearnModel
from tflearn import DNN, input_data, fully_connected, regression, lstm, dropout
from labelProvider import BaseLabelProvider

class LSTMModel(BaseTFLearnModel):
	def __init__(self, config):
		super().__init__(config)

	def trainModel(self):
		self.model.fit(self.train_X, self.train_y, n_epoch=self.config.nbEpoch, batch_size=self.config.batchSize, 
			shuffle=False, validation_set=(self.validation_X, self.validation_y), show_metric=True, run_id=self.getRunID())
	
	def createModel(self):
		print("Creating model...")
		sequenceLength = self.config.sequenceLength
		nbClasses = self.config.nbClasses
		nFeatures = self.config.nFeatures

		net = input_data(shape=[None, sequenceLength, nFeatures])
		net = lstm(net, n_units=nFeatures, dropout=0.8)
		net = fully_connected(net, nbClasses, activation='softmax')
		net = regression(net)

		self.model = DNN(net)

		print("Model created!")

	def datasetName(self):
		return 'LSTMModel'