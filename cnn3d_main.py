import train
import config
from modelCNN3D import CNNModel3D
from FFT3DDataProvider import FFT3DDataProvider
from modelSimpleDNN import SimpleDNNModel, SimpleDNNLabelProvider

class CNNConfig(config.GenericConfig):
	def __init__(self):
		super().__init__()
		self.nbPerClass = 10000
		self.batchSize = 300
		self.nbEpoch = 8
		self.nbClasses = 3
		self.sequenceLength = 64
		self.nFeatures = 64
		self.sample_rate = 256
		self.window_step = 32
		self.nChannels = 4

def try_CNN():
	cfg = CNNConfig()
	train.train_and_test(load_existing_dataset=False, load_existing_model=False, train_model=True, check_accuracy=True, modelType=CNNModel3D, 
		dataProviderType=FFT3DDataProvider, labelProviderType=SimpleDNNLabelProvider, config=cfg)

try_CNN()