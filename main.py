import train
import os
import config
from modelCNN import CNNModel
from modelSimpleDNN import SimpleDNNModel, SimpleDNNLabelProvider
from simpleSVMmodel import SimpleSVMModel, SimpleSVMLabelProvider
from FFTDataProvider import FFTDataProvider
from modelLSTM import LSTMModel
from rawEEGDataProvider import RawEEGDataProvider
from FFT2DDataProvider import FFT2DDataProvider

from eegToData import generate_slices_all


class CNNConfig(config.GenericConfig):
	def __init__(self):
		super().__init__()
		self.nbPerClass = 486
		self.batchSize = 300
		self.nbEpoch = 16
		self.nbClasses = 3
		self.fft_window = 90

def try_CNN():
	cfg = CNNConfig()
	train.train_and_test(load_existing_dataset=False, load_existing_model=False, train_model=True, modelType=CNNModel, 
		dataProviderType=FFT2DDataProvider, labelProviderType=SimpleDNNLabelProvider, config=cfg)

class SimpleDNNConfig(config.GenericConfig):
	def __init__(self):
		super().__init__()
		self.nbPerClass = 30000
		self.batchSize = 30000
		self.nbEpoch = 16
		self.sample_rate = 256
		self.fft_window = 90
		self.nFeatures = 6*4
		self.nbClasses = 3

def try_MLP():
	cfg = SimpleDNNConfig()
	train.train_and_test(load_existing_dataset=False, load_existing_model=False, 
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
	train.train_and_test(load_existing_dataset=False, load_existing_model=False, 
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

try_CNN()