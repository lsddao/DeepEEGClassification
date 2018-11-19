# -*- coding: utf-8 -*-
import random
import string
import os
import sys
import numpy as np

from model import createModel
from datasetTools import getDataset
from config import *

from eegToData import generate_slices

def eeg_slice(session_id, channel):
	generate_slices(session_id, channel)

#List genres
classes = os.listdir(slicesPath)
classes = [filename for filename in classes if os.path.isdir(slicesPath+filename)]
nbClasses = len(classes)

def eeg_train():
	model = createModel(nbClasses, sliceSize)

	#Create or load new dataset
	train_X, train_y, validation_X, validation_y = getDataset(filesPerClass, classes, sliceSize, validationRatio, testRatio, mode="train")

	#Define run id for graphs
	run_id = "EEGClasses - "+str(batchSize)+" "+''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))

	#Train the model
	print("[+] Training the model...")
	model.fit(train_X, train_y, n_epoch=nbEpoch, batch_size=batchSize, shuffle=True, validation_set=(validation_X, validation_y), snapshot_step=100, show_metric=True, run_id=run_id)
	print("    Model trained! ✅")

	#Save trained model
	print("[+] Saving the weights...")
	model.save('eegDNN.tflearn')
	print("[+] Weights saved! ✅💾")

def eeg_test():
	model = createModel(nbClasses, sliceSize)

	#Create or load new dataset
	test_X, test_y = getDataset(filesPerClass, classes, sliceSize, validationRatio, testRatio, mode="test")

	#Load weights
	print("[+] Loading weights...")
	model.load('eegDNN.tflearn')
	print("    Weights loaded! ✅")

	testAccuracy = model.evaluate(test_X, test_y)[0]
	print("[+] Test accuracy: {} ".format(testAccuracy))


#eeg_slice(session_id, channel)
#eeg_train()
eeg_test()