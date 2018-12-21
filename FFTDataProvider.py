from simpleWinEEGDataProvider import SimpleWindowBasedDataProvider
from eegToData import fft_elements, fft_log

class FFTDataProvider(SimpleWindowBasedDataProvider):
	def __init__(self, config, labelProvider):
		super().__init__(config, labelProvider)

	def X_shape(self):
		return [-1, self.config.nFeatures]

	def getFeaturesFromWindow(self):
		features = []
		f = []
		for channel_idx in range(4):
			f.append( fft_log(self.samples[channel_idx])[:64] )
		features.extend(f[0] - f[3])
		return features