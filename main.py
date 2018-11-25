import train
import os
import config
from modelCNN import CNNModel
from modelSimpleDNN import SimpleDNNModel

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

#train_and_test_CNN()
#generate_slices_all(imagesPath=imagesPath, channel=1, max_image_len=imageSize, window=90)

class SimpleDNNConfig(config.GenericConfig):
	def __init__(self):
		super().__init__()
		self.nbPerClass = 30000
		self.batchSize = 300
		self.nbEpoch = 32
		self.sample_rate = 256
		self.channel = 1
		self.fft_window = 90
		self.nFeatures = 5
		self.nbClasses = 3

cfg = SimpleDNNConfig()
train.train_and_test(load_existing_dataset=False, load_existing_model=False, 
		modelType=SimpleDNNModel, config=cfg)

#model = SimpleDNNModel(cfg)
#model.createDataset()
#data = model.getData()
#print(data)