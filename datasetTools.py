# -*- coding: utf-8 -*-

import os
from PIL import Image
from random import shuffle
import numpy as np
import pickle

from imageFilesTools import getImageData
from config import datasetPath
from config import slicesPath

#Creates name of dataset from parameters
def getDatasetName(nbPerClass, sliceSize):
    name = "{}".format(nbPerClass)
    name += "_{}".format(sliceSize)
    return name

#Creates or loads dataset if it exists
#Mode = "train" or "test"
def getDataset(nbPerClass, classes, sliceSize, validationRatio, testRatio, mode):
    print("[+] Dataset name: {}".format(getDatasetName(nbPerClass,sliceSize)))
    if not os.path.isfile(datasetPath+"train_X_"+getDatasetName(nbPerClass, sliceSize)+".p"):
        print("[+] Creating dataset with {} slices of size {} per genre... ⌛️".format(nbPerClass,sliceSize))
        createDatasetFromSlices(nbPerClass, classes, sliceSize, validationRatio, testRatio) 
    else:
        print("[+] Using existing dataset")
    
    return loadDataset(nbPerClass, classes, sliceSize, mode)
        
#Loads dataset
#Mode = "train" or "test"
def loadDataset(nbPerClass, classes, sliceSize, mode):
    #Load existing
    datasetName = getDatasetName(nbPerClass, sliceSize)
    if mode == "train":
        print("[+] Loading training and validation datasets... ")
        train_X = pickle.load(open("{}train_X_{}.p".format(datasetPath,datasetName), "rb" ))
        train_y = pickle.load(open("{}train_y_{}.p".format(datasetPath,datasetName), "rb" ))
        validation_X = pickle.load(open("{}validation_X_{}.p".format(datasetPath,datasetName), "rb" ))
        validation_y = pickle.load(open("{}validation_y_{}.p".format(datasetPath,datasetName), "rb" ))
        print("    Training and validation datasets loaded! ✅")
        return train_X, train_y, validation_X, validation_y

    else:
        print("[+] Loading testing dataset... ")
        test_X = pickle.load(open("{}test_X_{}.p".format(datasetPath,datasetName), "rb" ))
        test_y = pickle.load(open("{}test_y_{}.p".format(datasetPath,datasetName), "rb" ))
        print("    Testing dataset loaded! ✅")
        return test_X, test_y

#Saves dataset
def saveDataset(train_X, train_y, validation_X, validation_y, test_X, test_y, nbPerClass, classes, sliceSize):
     #Create path for dataset if not existing
    if not os.path.exists(os.path.dirname(datasetPath)):
        #try:
        os.makedirs(os.path.dirname(datasetPath))
        #except OSError as exc: # Guard against race condition
        #    if exc.errno != errno.EEXIST:
        #        raise

    #SaveDataset
    print("[+] Saving dataset... ")
    datasetName = getDatasetName(nbPerClass, sliceSize)
    pickle.dump(train_X, open("{}train_X_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(train_y, open("{}train_y_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(validation_X, open("{}validation_X_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(validation_y, open("{}validation_y_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(test_X, open("{}test_X_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(test_y, open("{}test_y_{}.p".format(datasetPath,datasetName), "wb" ))
    print("    Dataset saved! ✅💾")

#Creates and save dataset from slices
def createDatasetFromSlices(nbPerClass, classes, sliceSize, validationRatio, testRatio):
    data = []
    for eeg_class in classes:
        print("-> Adding {}...".format(eeg_class))
        #Get slices in genre subfolder
        filenames = os.listdir(slicesPath+eeg_class)
        filenames = [filename for filename in filenames if filename.endswith('.png')]
        filenames = filenames[:nbPerClass]
        #Randomize file selection for this genre
        shuffle(filenames)

        #Add data (X,y)
        for filename in filenames:
            imgData = getImageData(slicesPath+eeg_class+"/"+filename, sliceSize)
            label = [1. if eeg_class == g else 0. for g in classes]
            data.append((imgData,label))

    #Shuffle data
    shuffle(data)

    #Extract X and y
    X,y = zip(*data)

    #Split data
    validationNb = int(len(X)*validationRatio)
    testNb = int(len(X)*testRatio)
    trainNb = len(X)-(validationNb + testNb)

    #Prepare for Tflearn at the same time
    train_X = np.array(X[:trainNb]).reshape([-1, sliceSize, sliceSize, 1])
    train_y = np.array(y[:trainNb])
    validation_X = np.array(X[trainNb:trainNb+validationNb]).reshape([-1, sliceSize, sliceSize, 1])
    validation_y = np.array(y[trainNb:trainNb+validationNb])
    test_X = np.array(X[-testNb:]).reshape([-1, sliceSize, sliceSize, 1])
    test_y = np.array(y[-testNb:])
    print("    Dataset created! ✅")
        
    #Save
    saveDataset(train_X, train_y, validation_X, validation_y, test_X, test_y, nbPerClass, classes, sliceSize)

    return train_X, train_y, validation_X, validation_y, test_X, test_y
