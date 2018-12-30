class BaseDataProvider():
	def __init__(self, config, labelProvider):
		self.config = config
		self.labelProvider = labelProvider
	
	def X_shape(self):
		raise NotImplementedError
		
	def getData(self):
		raise NotImplementedError

	def shuffleAllowed(self):
		return True