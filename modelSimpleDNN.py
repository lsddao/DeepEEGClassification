from baseTFLearnModel import BaseTFLearnModel
from tflearn import DNN, input_data, fully_connected, regression, DataPreprocessing
from labelProvider import BaseLabelProvider

class SimpleDNNModel(BaseTFLearnModel):
	def __init__(self, config):
		super().__init__(config)
	
	def createModel(self):
		print("Creating model...")
		nFeatures = self.config.nFeatures
		nbClasses = self.config.nbClasses

		preprocess = DataPreprocessing()
		preprocess.add_featurewise_zero_center()
		preprocess.add_featurewise_stdnorm()

		net = input_data(shape=[None, nFeatures], data_preprocessing=preprocess)
		net = fully_connected(net, 32, activation='elu', weights_init="Xavier")
		net = fully_connected(net, nbClasses, activation='softmax')
		net = regression(net)

		self.model = DNN(net, tensorboard_verbose=3)
		print("Model created!")

	def datasetName(self):
		return 'simpleDNNModel'

class SimpleDNNLabelProvider(BaseLabelProvider):
	def getLabel(self, value):
		to_label = {
			-2 : [1., 0., 0.],
			-1 : [1., 0., 0.],
			0 : [0., 1., 0.],
			1 : [0., 0., 1.],
			2 : [0., 0., 1.]
		}
		return to_label[value]
        
	def getClassName(self, value):
		to_class = {
 		   -2 : "bad",
			-1 : "bad",
			0 : "ok",
			1 : "good",
			2 : "good"
		}
		return to_class[value]

	def getClasses(self):
		return ['bad', 'ok', 'good']