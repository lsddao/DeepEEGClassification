from model import Model
from labelProvider import BaseLabelProvider
from sklearn import svm

class SimpleSVMModel(Model):
	def __init__(self, config):
		super().__init__(config)

	def loadModel(self):
		raise NotImplementedError

	def trainModel(self):
		self.model.fit(self.train_X, self.train_y)

	def saveModel(self):
		raise NotImplementedError

	def testAccuracy(self):
		size = len(self.test_X)
		accurate = 0
		for idx in range(size):
			predicted_label = self.model.predict(self.test_X[idx])
			actual_label = self.test_y[idx]
			if predicted_label == actual_label:
				accurate += 1
		return accurate / size

	def createModel(self):
		print("Creating model...")
		self.model = svm.SVC(gamma='scale', decision_function_shape='ovo', cache_size=1000)
		print("Model created!")

	def datasetName(self):
		return 'SimpleSVMModel'

class SimpleSVMLabelProvider(BaseLabelProvider):
	def getLabel(self, value):
		return self.getClassName(value)
        
	def getClassName(self, value):
		to_class = {
 		   -2 : "bad",
			-1 : "bad",
			0 : "ok",
			1 : "good",
			2 : "good"
		}
		return to_class[value]

	def getClasses(self):
		return ['bad', 'ok', 'good']