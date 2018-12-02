# main training procedure
def train_and_test(load_existing_dataset, load_existing_model, modelType, config):
	model = modelType(config)
	if load_existing_dataset:
		model.loadDataset()
	else:
		model.createDataset()
		model.saveDataset()

	model.createModel()

	if load_existing_model:
		print("Loading weights...")
		model.loadModel()
		print("Weights loaded!")
	else:
		print("Training model from scratch")

	#Train the model
	print("Training the model...")
	model.trainModel()
	print("Model trained!")

	#Save trained model
	print("Saving the weights...")
	model.saveModel()
	print("Weights saved!")

	acc = model.testAccuracy()
	print("Test accuracy: {} ".format(acc))
	print('Corrected accuracy (100%): {}'.format(100*model.correctedAcc(acc)))
	return acc