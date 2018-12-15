from winDataProvider import WindowBasedDataProvider
import numpy as np

class RawEEGDataProvider(WindowBasedDataProvider):
	def __init__(self, config, labelProvider):
		super().__init__(config, labelProvider)

	def X_shape(self):
		return [-1, self.config.sequenceLength, self.config.nChannels]

	def getFeaturesFromWindow(self):
		timesteps = []
		for i in range(self.config.sequenceLength):
			channel_data = []
			for ch in range(self.config.nChannels):
				channel_data.append(self.samples[ch][i])
			timesteps.append(channel_data)
		return np.array(timesteps)