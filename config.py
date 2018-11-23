#Define paths for files
datasetPath = "Data/Dataset/"
slicesPath = "Data/Slices/"

#Slice parameters
sliceSize = 128
fft_window = 90

#Dataset parameters
filesPerClass = 185
validationRatio = 0.3
testRatio = 0.1

#Model parameters
batchSize = 24
nbEpoch = 16

channel = 1

dbconnection = "mongodb://localhost:27017/"