class GenericConfig:
    def __init__(self):
        self.datasetPath = "Data/Dataset/"
        self.validationRatio = 0.3
        self.testRatio = 0.1
        self.dbconnection = "mongodb://localhost:27017/" #"mongodb://192.168.178.30:27017/"

generic = GenericConfig()