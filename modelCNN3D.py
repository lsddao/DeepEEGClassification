from baseTFLearnModel import BaseTFLearnModel

from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

class CNNModel3D(BaseTFLearnModel):
	def __init__(self, config):
		super().__init__(config)
	
	def createModel(self):
		print("Creating model...")
		imageSize = self.config.sequenceLength
		nbClasses = self.config.nbClasses
		nChannels = self.config.nChannels

		net = input_data(shape=[None, imageSize, imageSize, nChannels], name='input')

		net = conv_2d(net, 8, 2, activation='elu', weights_init="Xavier", name='conv1')
		net = max_pool_2d(net, 2)

		net = conv_2d(net, 4, 4, activation='elu', weights_init="Xavier", name='conv2')
		net = max_pool_2d(net, 2)

		net = conv_2d(net, 2, 8, activation='elu', weights_init="Xavier", name='conv3')
		net = max_pool_2d(net, 2)

		net = conv_2d(net, 1, 16, activation='elu', weights_init="Xavier", name='conv4')
		net = max_pool_2d(net, 2)

		net = fully_connected(net, imageSize*8, activation='elu')
		net = dropout(net, 0.5)

		net = fully_connected(net, nbClasses, activation='softmax')
		net = regression(net, optimizer='rmsprop', loss='categorical_crossentropy')

		self.model = DNN(net)
		print("Model created!")

	def datasetName(self):
		return 'CNNModel'