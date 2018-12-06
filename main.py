import train
import os
import config
from modelCNN import CNNModel
from modelSimpleDNN import SimpleDNNModel
from FFTDataProvider import FFTDataProvider

from eegToData import generate_slices_all

imagesPath = "Data/Slices/"
imageSize = 64

class CNNConfig(config.GenericConfig):
	def __init__(self):
		super().__init__()
		self.imagesPath = imagesPath
		self.imageSize = imageSize
		self.nbPerClass = 486
		self.batchSize = 300
		self.nbEpoch = 16
		self.classes = os.listdir(self.imagesPath)
		self.classes = [filename for filename in self.classes if os.path.isdir(self.imagesPath + filename)]
		self.nbClasses = len(self.classes)

def train_and_test_CNN():
	cfg = CNNConfig()
	train.train_and_test(load_existing_dataset=False, load_existing_model=False, 
		modelType=CNNModel, config=cfg)

class SimpleDNNConfig(config.GenericConfig):
	def __init__(self):
		super().__init__()
		self.nbPerClass = 48500
		self.batchSize = 150000
		self.nbEpoch = 16
		self.sample_rate = 256
		self.fft_window = 90
		self.nFeatures = 45*4
		self.classes = ['bad', 'ok', 'good']
		self.nbClasses = len(self.classes)

cfg = SimpleDNNConfig()
train.train_and_test(load_existing_dataset=True, load_existing_model=True, modelType=SimpleDNNModel, dataProviderType=FFTDataProvider, config=cfg)