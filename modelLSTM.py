from baseTFLearnModel import BaseTFLearnModel
from tflearn import DNN, input_data, fully_connected, regression, lstm, dropout
from labelProvider import BaseLabelProvider

class LSTMModel(BaseTFLearnModel):
	def __init__(self, config):
		super().__init__(config)
	
	def createModel(self):
		print("Creating model...")
		sequenceLength = self.config.sequenceLength
		nbClasses = self.config.nbClasses
		nChannels = self.config.nChannels

		net = input_data(shape=[None, sequenceLength, nChannels])
		net = lstm(net, n_units=nChannels*2, activation='relu', return_seq=True)
		net = dropout(net, 0.2)
		net = lstm(net, n_units=nChannels, activation='sigmoid')
		net = fully_connected(net, nbClasses, activation='sigmoid')
		net = regression(net, optimizer='rmsprop')

		self.model = DNN(net)

		print("Model created!")

	def datasetName(self):
		return 'LSTMModel'