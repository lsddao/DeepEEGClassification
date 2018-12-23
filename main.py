import train
import config
from modelSimpleDNN import SimpleDNNModel, SimpleDNNLabelProvider
from simpleSVMmodel import SimpleSVMModel, SimpleSVMLabelProvider
from FFTDataProvider import FFTDataProvider
from modelLSTM import LSTMModel
from rawEEGDataProvider import RawEEGDataProvider
from FFT2DDataProvider import FFT2DDataProvider

class SimpleDNNConfig(config.GenericConfig):
	def __init__(self):
		super().__init__()
		self.nbPerClass = 30000
		self.batchSize = 30000
		self.nbEpoch = 128
		self.sample_rate = 256
		self.window_step = 32
		self.nFeatures = 64
		self.nbClasses = 3

def try_MLP():
	cfg = SimpleDNNConfig()
	train.train_and_test(load_existing_dataset=True, load_existing_model=False, train_model=True, 
		modelType=SimpleDNNModel, dataProviderType=FFTDataProvider, labelProviderType=SimpleDNNLabelProvider, config=cfg)

class SimpleSVMConfig(config.GenericConfig):
	def __init__(self):
		super().__init__()
		self.nbPerClass = 48500
		self.sample_rate = 256
		self.fft_window = 90
		self.nFeatures = 6*4
		self.nbClasses = 3

def try_SVM():
	cfg = SimpleSVMConfig()
	train.train_and_test(load_existing_dataset=False, load_existing_model=False, train_model=True, 
		modelType=SimpleSVMModel, dataProviderType=FFTDataProvider, labelProviderType=SimpleSVMLabelProvider, config=cfg)

class LSTMConfig(config.GenericConfig):
	def __init__(self):
		super().__init__()
		self.nbPerClass = 37902
		self.sample_rate = 256
		self.window_step = 32
		self.nChannels = 4
		self.nbClasses = 3
		self.sequenceLength = 8*2
		self.batchSize = 4*8192
		self.nbEpoch = 1024
		self.nFeatures = self.nChannels*3

def try_LSTM():
	cfg = LSTMConfig()
	train.train_and_test(load_existing_dataset=True, load_existing_model=True, train_model=True, 
		modelType=LSTMModel, dataProviderType=FFT2DDataProvider, labelProviderType=SimpleDNNLabelProvider, config=cfg)