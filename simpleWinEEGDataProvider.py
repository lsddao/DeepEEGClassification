from winDataProvider import WindowBasedDataProvider

class SimpleWindowBasedDataProvider(WindowBasedDataProvider):
	def __init__(self, config, labelProvider):
		super().__init__(config, labelProvider)

	def consumeWindow(self):
		if not self.isCurrentClassFull():
			self.appendData(self.getFeaturesFromWindow())

	def getFeaturesFromWindow(self):
		raise NotImplementedError