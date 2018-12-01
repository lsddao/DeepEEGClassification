import model
import trackdata

import numpy as np
from random import shuffle
import collections

from tflearn import DNN, input_data, fully_connected, regression, init_graph
from eegToData import fft_elements, enjoy_to_class

class SimpleDNNModel(model.Model):
	def __init__(self, config):
		super().__init__(config)
	
	def createModel(self):
		print("Creating model...")
		nFeatures = self.config.nFeatures
		nbClasses = self.config.nbClasses

		init_graph(num_cores=4)

		net = input_data(shape=[None, nFeatures])
		net = fully_connected(net, 4*nFeatures, activation='elu', weights_init="Xavier")
		net = fully_connected(net, 4*nFeatures, activation='elu', weights_init="Xavier")
		net = fully_connected(net, nbClasses, activation='softmax')
		net = regression(net)

		self.model = DNN(net)
		print("Model created!")

	def datasetName(self):
		return 'simpleDNNModel'

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
		
	def createDataset(self):
		print("Creating dataset...")
		validationRatio = self.config.validationRatio
		testRatio = self.config.testRatio
		numFeatures = self.config.nFeatures
	
		data = self.getData()

		#Shuffle data
		shuffle(data)

		#Extract X and y
		X,y = zip(*data)

		#Split data
		validationNb = int(len(X)*validationRatio)
		testNb = int(len(X)*testRatio)
		trainNb = len(X)-(validationNb + testNb)

		
		#Prepare for Tflearn at the same time
		self.train_X = np.array(X[:trainNb]).reshape([-1, numFeatures])
		self.train_y = np.array(y[:trainNb])
		self.validation_X = np.array(X[trainNb:trainNb+validationNb]).reshape([-1, numFeatures])
		self.validation_y = np.array(y[trainNb:trainNb+validationNb])
		self.test_X = np.array(X[-testNb:]).reshape([-1, numFeatures])
		self.test_y = np.array(y[-testNb:])

		print("Dataset created!")