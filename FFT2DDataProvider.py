from winDataProvider import WindowBasedDataProvider
from eegToData import fft_elements, fft_log
from collections import deque
import numpy as np

class FFT2DDataProvider(WindowBasedDataProvider):
	def __init__(self, config, labelProvider):
		super().__init__(config, labelProvider)

		self.timesteps = deque(maxlen=self.config.sequenceLength)

	def X_shape(self):
		return [-1, self.config.sequenceLength, self.config.nFeatures, 1]

	def consumeWindow(self):
		features = []
		f = []
		for channel_idx in range(4):
			f.append( fft_log(self.samples[channel_idx])[:self.config.nFeatures] )
		features.extend(f[0] - f[3])
		#features.extend(f[1] - f[2])
		self.timesteps.append(features)
		if len(self.timesteps) == self.timesteps.maxlen:
			if not self.isCurrentClassFull():
				self.appendData(np.array(self.timesteps))