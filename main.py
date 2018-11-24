import train
import os
import config
from modelCNN import CNNModel

#from eegToData import generate_slices_all

#generate_slices_all(channel=c.channel, max_image_len=c.imageSize, window=c.fft_window)

class CNNConfig(config.GenericConfig):
	def __init__(self):
		super().__init__()
		self.imagesPath = "Data/Slices/"
		self.imageSize = 64
		self.nbPerClass = 486
		self.classes = os.listdir(self.imagesPath)
		self.classes = [filename for filename in self.classes if os.path.isdir(self.imagesPath + filename)]
		self.nbClasses = len(self.classes)

def train_and_test_CNN():
	cfg = CNNConfig()
	train.train_and_test(load_existing_dataset=False, load_existing_model=False, 
		modelType=CNNModel, config=cfg)

train_and_test_CNN()