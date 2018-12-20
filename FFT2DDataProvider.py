from winDataProvider import WindowBasedDataProvider
from eegToData import fft_elements
from collections import deque
import numpy as np

class FFT2DDataProvider(WindowBasedDataProvider):
	def __init__(self, config, labelProvider):
		super().__init__(config, labelProvider)

		self.timesteps = deque(maxlen=self.config.sequenceLength)
		self.fft_increment = 2
		self.fft_shift = 0

	def X_shape(self):
		return [-1, self.config.sequenceLength, self.config.nFeatures]

	def consumeWindow(self):
		if len(self.timesteps) == self.timesteps.maxlen:
			self.fft_shift += 1
		features = []
		f = []
		for channel_idx in range(4):
			f.append(np.array(fft_elements(self.samples[channel_idx])))
		features.extend(f[0] - f[3])
		features.extend(f[1] - f[2])
		self.timesteps.append(features)
		if self.fft_shift == self.fft_increment:
			self.fft_shift = 0
		if len(self.timesteps) == self.timesteps.maxlen:
			if not self.isCurrentClassFull():
				self.appendData(np.array(self.timesteps))