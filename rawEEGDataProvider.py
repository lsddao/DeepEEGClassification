from winDataProvider import WindowBasedDataProvider
import numpy as np

class RawEEGDataProvider(WindowBasedDataProvider):
	def __init__(self, config, labelProvider):
		super().__init__(config, labelProvider)

	def getFeaturesFromWindow(self):
		return np.array(self.samples)