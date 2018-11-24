#Define paths for files
datasetPath = "Data/Dataset/"
imagesPath = "Data/Slices/"

#Slice parameters
image_size = 28
fft_window = 90

#Dataset parameters
filesPerClass = 486
validationRatio = 0.3
testRatio = 0.1

#Model parameters
batchSize = 3*12
nbEpoch = 16

channel = 1

dbconnection = "mongodb://192.168.178.30:27017/"