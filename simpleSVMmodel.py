from model import Model
from labelProvider import BaseLabelProvider
from sklearn import svm
import pickle

class SimpleSVMModel(Model):
	def __init__(self, config):
		super().__init__(config)

	def loadModel(self):
		self.model = pickle.load(open("{}.mdl".format(self.datasetName()), "rb" ))

	def trainModel(self):
		self.model.fit(self.train_X, self.train_y)

	def saveModel(self):
		pickle.dump(self.model, open("{}.mdl".format(self.datasetName()), "wb" ))

	def trainAccuracy(self):
		return self.dataSetAccuracy(self.train_X, self.train_y)

	def testAccuracy(self):
		return self.dataSetAccuracy(self.test_X, self.test_y)

	def validationAccuracy(self):
		return self.dataSetAccuracy(self.validation_X, self.validation_y)

	def dataSetAccuracy(self, vector_x, vector_y):
		size = len(vector_x)
		accurate = 0
		for idx in range(size):
			x = vector_x[idx].reshape(1, -1)
			predicted_label = self.model.predict(x)
			actual_label = vector_y[idx]
			if predicted_label == actual_label:
				accurate += 1
		return accurate / size

	def createModel(self):
		print("Creating model...")
		self.model = svm.SVC(gamma='scale', decision_function_shape='ovo', cache_size=2000)
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