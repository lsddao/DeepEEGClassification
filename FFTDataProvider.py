from simpleWinEEGDataProvider import SimpleWindowBasedDataProvider
from eegToData import fft_elements, fft_log
import numpy as np

class FFTDataProvider(SimpleWindowBasedDataProvider):
	def __init__(self, config, labelProvider):
		super().__init__(config, labelProvider)

	def X_shape(self):
		return [-1, self.config.nFeatures]

	def getFeaturesFromWindow(self):
		features = fft_elements(self.samples[0]) - fft_elements(self.samples[3])
		return features