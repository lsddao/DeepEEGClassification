import train
import config
from modelCNN import CNNModel
from FFT2DDataProvider import FFT2DDataProvider
from modelSimpleDNN import SimpleDNNModel, SimpleDNNLabelProvider

from visualize import display_convolutions
from eegToData import dump_png
import numpy as np

class CNNConfig(config.GenericConfig):
	def __init__(self):
		super().__init__()
		self.nbPerClass = 37857
		self.batchSize = 300
		self.nbEpoch = 8
		self.nbClasses = 3
		self.sequenceLength = 64
		self.nFeatures = 64
		self.sample_rate = 256
		self.window_step = 32

def try_CNN():
	cfg = CNNConfig()
	train.train_and_test(load_existing_dataset=False, load_existing_model=False, train_model=True, modelType=CNNModel, 
		dataProviderType=FFT2DDataProvider, labelProviderType=SimpleDNNLabelProvider, config=cfg)

def visualize_weights(layer):
	cfg = CNNConfig()
	model = CNNModel(cfg)
	labelProvider = SimpleDNNLabelProvider()
	model.dataProvider = FFT2DDataProvider(cfg, labelProvider)
	model.createModel()
	model.loadModel()
	weigths = display_convolutions(model.model, layer)
	weigths *= 255
	dump_png(weigths, '{}.png'.format(layer))

def datasetToImg():
	cfg = CNNConfig()
	model = CNNModel(cfg)
	labelProvider = SimpleDNNLabelProvider()
	model.dataProvider = FFT2DDataProvider(cfg, labelProvider)
	model.loadDataset()
	min = np.min(model.train_X)
	max = np.max(model.train_X)
	diff = max - min
	for idx in range(300):
		arr = model.train_X[idx]
		arr -= min
		arr /= diff
		arr *= 255
		image_class = labelProvider.classFromLabel(model.train_y[idx])
		dump_png(arr, 'Data/Slices/{}/{}.png'.format(image_class, idx))

#visualize_weights('conv1')
#visualize_weights('conv2')
#visualize_weights('conv3')
#visualize_weights('conv4')

try_CNN()