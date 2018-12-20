from simpleWinEEGDataProvider import SimpleWindowBasedDataProvider
from eegToData import fft_elements

class FFTDataProvider(SimpleWindowBasedDataProvider):
	def __init__(self, config, labelProvider):
		super().__init__(config, labelProvider)

	def X_shape(self):
		return [-1, self.config.nFeatures]

	def getFeaturesFromWindow(self):
		features = []
		for channel_idx in range(4):
			features.extend(fft_elements(self.samples[channel_idx]))
		return features