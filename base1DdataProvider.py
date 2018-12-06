from baseDataProvider import BaseDataProvider
import trackdata

from eegToData import enjoy_to_class

class NotEnoughData(BaseException):
	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)

class Base1DDataProvider(BaseDataProvider):
	def __init__(self, config):
		super().__init__(config)
		self.dataPerClass = dict()
		for key in self.config.classes:
			self.dataPerClass[key] = 0

	def X_shape(self):
		return [-1, self.config.nFeatures]

	def getLabel(self):
		raise NotImplementedError

	def getClassName(self):
		return enjoy_to_class[self.enjoy]

	def consumeEEG(self, channel_data):
		raise NotImplementedError

	def appendData(self, data):
		label = self.getLabel()
		self.data.append((data,label))
		self.dataPerClass[self.getClassName()] += 1

	def isClassFull(self, className):
		return self.dataPerClass[className] >= self.config.nbPerClass

	def isCurrentClassFull(self):
		return self.isClassFull(self.getClassName())

	def isFull(self):
		for key in self.dataPerClass.keys():
			if not self.isClassFull(key):
				return False
		return True

	def getData(self):
		self.data = []
		dbconn = trackdata.DBConnection()
		
		for session_id in dbconn.all_sessions():
			self.enjoy = 0
			doc = dbconn.session_data(session_id, 'eeg')
			print("Creating data for session {}...".format(session_id))
			for x in doc:
				if self.isFull():
					return self.data
				if "channel_data" in x:
					U = x["channel_data"]
					self.consumeEEG(U)
				elif "event_name" in x:
					if x["event_name"] == "enjoy_changed":
						self.enjoy = x["value"]
		
		if not self.isFull():
			print(self.dataPerClass)
			raise NotEnoughData("Not enough data per class")

		return self.data