class GenericConfig:
    def __init__(self):
        self.datasetPath = "Data/Dataset/"
        self.validationRatio = 0.3
        self.testRatio = 0.1
        self.batchSize = 36
        self.nbEpoch = 16
        self.dbconnection = "mongodb://192.168.178.30:27017/"

generic = GenericConfig()