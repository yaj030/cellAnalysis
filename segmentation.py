from os.path import abspath, dirname

class SegmentationMethods:
    def segment(self):
        pass
    def methodPreparation(self):
        pass
    def getPath(self):
        self.currentPath = dirname(abspath(__file__))
    def importFiles(self):
        # import required files, like weights, config
        pass
