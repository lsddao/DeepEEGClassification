from baseDataProvider import BaseDataProvider
import trackdata

class NotEnoughData(BaseException):
	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)

class SequentialDataProvider(BaseDataProvider):
	def __init__(self, config, labelProvider):
		super().__init__(config, labelProvider)
		self.dataPerClass = dict()
		for key in self.labelProvider.getClasses():
			self.dataPerClass[key] = 0

	def getLabel(self):
		return self.labelProvider.getLabel(self.enjoy)

	def getClassName(self):
		return self.labelProvider.getClassName(self.enjoy)

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