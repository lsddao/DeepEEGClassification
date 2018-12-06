from baseDataProvider import BaseDataProvider
import trackdata

import collections
from eegToData import fft_elements, enjoy_to_class

class Base1DDataProvider(BaseDataProvider):
	def __init__(self, config):
		super().__init__(config)

	def X_shape(self):
		return [-1, self.config.nFeatures]

	def getData(self):
		nbPerClass = self.config.nbPerClass
		sample_rate = self.config.sample_rate
		channels = self.config.channels
		window = self.config.fft_window
		increment = int(sample_rate*(1-window/100))
		enjoy_to_label = {
			-2 : [1., 0., 0.],
			-1 : [1., 0., 0.],
			0 : [0., 1., 0.],
			1 : [0., 0., 1.],
			2 : [0., 0., 1.]
		}

		dataPerClass = {
			"bad" : 0,
			"ok" : 0,
			"good" : 0
		}

		data = []
		dbconn = trackdata.DBConnection()
		shift = 0
		for session_id in dbconn.all_sessions():
			samples = [collections.deque(maxlen=sample_rate) for channel in channels]
			enjoy = 0
			doc = dbconn.session_data(session_id, 'eeg')
			print("Creating data for session {}...".format(session_id))
			for x in doc:
				if dataPerClass['bad'] == nbPerClass and dataPerClass['ok'] == nbPerClass and dataPerClass['good'] == nbPerClass:
					print(dataPerClass)
					return data
				if "channel_data" in x:
					if len(samples[0]) == samples[0].maxlen:
						shift += 1
					U = x["channel_data"]
					for channel_idx in range(len(channels)):
						samples[channel_idx].append(U[channels[channel_idx]])
					if shift == increment:
						shift = 0
						lbl = enjoy_to_class[enjoy]
						if dataPerClass[lbl] < nbPerClass:
							label = enjoy_to_label[enjoy]
							f = []
							for channel_idx in range(len(channels)):
								f.extend(fft_elements(samples[channel_idx]))
							data.append((f,label))
							dataPerClass[lbl] += 1
				elif "event_name" in x:
					if x["event_name"] == "enjoy_changed":
						enjoy = x["value"]
		
		print(dataPerClass)
		return data