# main training procedure
def train_and_test(load_existing_dataset, load_existing_model, train_model, check_accuracy, modelType, dataProviderType, labelProviderType, config):
	model = modelType(config)
	labelProvider = labelProviderType()
	model.dataProvider = dataProviderType(config, labelProvider)
	
	if load_existing_dataset:
		model.loadDataset()
	else:
		model.createDataset()
		model.saveDataset()

	if load_existing_model or train_model or check_accuracy:
		model.createModel()

	if load_existing_model:
		print("Loading model...")
		model.loadModel()
		print("Model loaded!")
	
	if train_model:
		#Train the model
		print("Training the model...")
		model.trainModel()
		print("Model trained!")
		#Save trained model
		print("Saving the model...")
		model.saveModel()
		print("Model saved!")

	if check_accuracy:
		try:
			acc = model.trainAccuracy()
			print("Train accuracy: {} ".format(acc))
			print('Corrected accuracy (100%): {}'.format(100*model.correctedAcc(acc)))
		except NotImplementedError:
			print('Model does not implement trainAccuracy()')

		try:
			acc = model.validationAccuracy()
			print("Validation accuracy: {} ".format(acc))
			print('Corrected accuracy (100%): {}'.format(100*model.correctedAcc(acc)))
		except NotImplementedError:
			print('Model does not implement validationAccuracy()')