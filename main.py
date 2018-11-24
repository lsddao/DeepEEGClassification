# -*- coding: utf-8 -*-
import os

from model import createModel
from datasetTools import getDataset
from config import *

from eegToData import generate_slices_all

def getClasses():
	classes = os.listdir(imagesPath)
	classes = [filename for filename in classes if os.path.isdir(imagesPath+filename)]
	return classes

def eeg_slice():
	generate_slices_all(channel=channel, max_image_len=image_size, window=fft_window)

def eeg_train(load_existing=True):
	classes = getClasses()

	#Create or load new dataset
	train_X, train_y, validation_X, validation_y = getDataset(filesPerClass, classes, image_size, validationRatio, testRatio, mode="train")

	model = createModel(len(classes), image_size)

	#Load weights
	if load_existing:
		print("[+] Loading weights...")
		model.load('eegDNN.tflearn')
		print("    Weights loaded! ✅")
	else:
		print("[+] Training model from scratch")

	#Train the model
	print("[+] Training the model...")
	model.fit(train_X, train_y, n_epoch=nbEpoch, batch_size=batchSize, shuffle=True, validation_set=(validation_X, validation_y), snapshot_step=100, show_metric=True)
	print("    Model trained! ✅")

	#Save trained model
	print("[+] Saving the weights...")
	model.save('eegDNN.tflearn')
	print("[+] Weights saved! ✅💾")

def eeg_test():
	classes = getClasses()

	#Create or load new dataset
	test_X, test_y = getDataset(filesPerClass, classes, image_size, validationRatio, testRatio, mode="test")

	model = createModel(len(classes), image_size)

	#Load weights
	print("[+] Loading weights...")
	model.load('eegDNN.tflearn')
	print("    Weights loaded! ✅")

	testAccuracy = model.evaluate(test_X, test_y)[0]
	print("[+] Test accuracy: {} ".format(testAccuracy))

#eeg_slice()
eeg_train(False)
#eeg_test()