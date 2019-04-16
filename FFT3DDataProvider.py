from winDataProvider import WindowBasedDataProvider
from eegToData import fft_elements, fft_log
from collections import deque
import numpy as np

class FFT3DDataProvider(WindowBasedDataProvider):
	def __init__(self, config, labelProvider):
		super().__init__(config, labelProvider)

		self.timesteps = deque(maxlen=self.config.sequenceLength)

	def X_shape(self):
		return [-1, self.config.sequenceLength, self.config.nFeatures, self.config.nChannels]

	def consumeWindow(self):
		f = []
		for channel_idx in range(self.config.nChannels):
			f.append( fft_log(self.samples[channel_idx])[:self.config.nFeatures] )
		f = np.array(f)
		f = np.rot90(f, k=3)
		self.timesteps.append(f)
		if len(self.timesteps) == self.timesteps.maxlen:
			if not self.isCurrentClassFull():
				self.appendData(np.array(self.timesteps))