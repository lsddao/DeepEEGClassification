from seqDataProvider import SequentialDataProvider
from collections import deque

class WindowBasedDataProvider(SequentialDataProvider):
	def __init__(self, config, labelProvider):
		super().__init__(config, labelProvider)

		sample_rate = self.config.sample_rate
		self.samples = [deque(maxlen=sample_rate) for channel in range(4)]
		self.increment = self.config.window_step
		self.shift = 0
		self.y_samples = deque(maxlen=sample_rate)

	def shuffleAllowed(self):
		# Because of data augmentation through sliding window, 
		# portions of training data may leak into the validation set.
		# This is why shuffle is restricted for all window-based data providers.
		return False

	def consumeWindow(self):
		raise NotImplementedError

	def getEnjoy(self):
		return max(set(self.y_samples), key = self.y_samples.count)

	def consumeEEG(self, channel_data):
		if len(self.samples[0]) == self.samples[0].maxlen:
			self.shift += 1
		for channel_idx in range(4):
			self.samples[channel_idx].append(channel_data[channel_idx])
		self.y_samples.append(self.enjoy)
		if self.shift == self.increment:
			self.shift = 0
			self.consumeWindow()