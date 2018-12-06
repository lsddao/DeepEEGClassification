# main training procedure
def train_and_test(load_existing_dataset, load_existing_model, modelType, dataProviderType, labelProviderType, config):
	model = modelType(config)
	labelProvider = labelProviderType()
	model.dataProvider = dataProviderType(config, labelProvider)
	
	if load_existing_dataset:
		model.loadDataset()
	else:
		model.createDataset()
		model.saveDataset()

	model.createModel()

	if load_existing_model:
		print("Loading model...")
		model.loadModel()
		print("Model loaded!")
	else:
		#Train the model
		print("Training the model...")
		model.trainModel()
		print("Model trained!")
		#Save trained model
		print("Saving the model...")
		model.saveModel()
		print("Model saved!")

	acc = model.validationAccuracy()
	print("Validation accuracy: {} ".format(acc))
	print('Corrected accuracy (100%): {}'.format(100*model.correctedAcc(acc)))

	acc = model.testAccuracy()
	print("Test accuracy: {} ".format(acc))
	print('Corrected accuracy (100%): {}'.format(100*model.correctedAcc(acc)))

	return acc