from baseTFLearnModel import BaseTFLearnModel

from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

class CNNModel(BaseTFLearnModel):
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