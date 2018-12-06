class BaseDataProvider():
	def __init__(self, config):
		self.config = config
	
	def X_shape(self):
		raise NotImplementedError
		
	def getData(self):
		raise NotImplementedError