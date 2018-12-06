from base1DdataProvider import Base1DDataProvider

import collections
from eegToData import fft_elements

class FFTDataProvider(Base1DDataProvider):
	def __init__(self, config, labelProvider):
		super().__init__(config, labelProvider)

		sample_rate = self.config.sample_rate
		self.samples = [collections.deque(maxlen=sample_rate) for channel in range(4)]
		window = self.config.fft_window
		self.increment = int(sample_rate*(1-window/100))
		self.shift = 0

	def consumeEEG(self, channel_data):
		if len(self.samples[0]) == self.samples[0].maxlen:
			self.shift += 1
		for channel_idx in range(4):
			self.samples[channel_idx].append(channel_data[channel_idx])
		if self.shift == self.increment:
			self.shift = 0
			if not self.isCurrentClassFull():
				features = []
				for channel_idx in range(4):
					features.extend(fft_elements(self.samples[channel_idx]))
				self.appendData(features)