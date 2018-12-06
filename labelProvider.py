class BaseLabelProvider:
    def getLabel(self, value):
        raise NotImplementedError
        
    def getClassName(self, value):
        raise NotImplementedError

    def getClasses(self):
        raise NotImplementedError