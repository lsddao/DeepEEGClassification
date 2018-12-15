from baseTFLearnModel import BaseTFLearnModel
from tflearn import DNN, input_data, fully_connected, regression, lstm
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
		net = lstm(net, n_units=nChannels, activation='relu', dropout=0.2, return_seq=True)
		net = lstm(net, n_units=nChannels/2, activation='sigmoid')
		net = fully_connected(net, nbClasses, activation='softmax')
		net = regression(net)

		self.model = DNN(net, tensorboard_verbose=0)
		print("Model created!")

	def datasetName(self):
		return 'LSTMModel'