from os import path
import imageFileClass
import experimentclass
import microscope
import pickle
import json
from tifffile import imwrite as tifwrite
import gc
import time
class IMAGE_ANALYSIS: 
    def __init__(self, experimentPath, fileNamePrefix, scope,fileType, experimentObj):
        self._experimentPath = experimentPath
        self._fileNamePrefix = fileNamePrefix
        self.scope = scope
        self.colors = []
        self.totalFrames = 0
        self.totalColors = 0
        self.totalZstacks = 1
        self.totalPositions = 0
        self.positions = []
        self.fovHeight = 0
        self.fovWidth = 0
        self.segmentation_options = {'equalize': True}
        self.motherCellAnalysis = []

        self.fileType = fileType
        if self.fileType=='tif_noZstack':
            self.fileClass = imageFileClass.TifFileNoZstackSplitColor(self)
        elif self.fileType == 'ND':
            self.fileClass = imageFileClass.NDfile(self)
        else:
            raise("Please specifiy file type")
        self.experimentObj = experimentObj
        self.experimentObj.setAttr(self)
    @property
    def experimentPath(self):
        return self._experimentPath
    @experimentPath.setter
    def experimentPath(self, pathStr):
        self._experimentPath = pathStr
        self.fileClass.updateFileInfo()

    @property
    def fileNamePrefix(self):
        return self._fileNamePrefix
    @fileNamePrefix.setter
    def fileNamePrefix(self,nameStr):
        self._fileNamePrefix = nameStr
        self.fileClass.updateFileInfo()
        
    def imRegistrationAllPosition(self):
        self.experimentObj.imageRegistration(positions=[])
    
    def loadRegistration(self):
        registrationPath = path.join(self.experimentPath, 'registration.pkl')
        if path.exists(registrationPath):
            self.registration = pickle.load(open(registrationPath,'rb'))
            print('loaded registration information, pickle')
            return True
        else:
            print('images not registered yet')
            return False
        
    def segmentPositionTimeZ(self,positions,frames=None,Zs=None,saveAsRuntime=True):
        # when saving mask as tifs, has to to be careful about the saveAsRuntime
        # the saving result will be different if there's z-stack
        # save in run time means each z has its only tif file
        # save at the end of run will save a hypertack with z,t
        if frames == None:
            frames = range(1,self.totalFrames+1)
        if saveAsRuntime:
            for position in positions:
                for frame in frames:
                    start_time = time.time()
                    print('segment frame '+str(frame))
                    if Zs==None:
                        _, masks = self.segmentOneSlice(position,frame,z=None)
                        self.experimentObj.append2MaskTif(masks,position)
                    else:
                        for z in Zs:
                            _, masks = self.segmentOneSlice(position,frame,z)
                            self.experimentObj.append2MaskTif(masks,position,z)
                    print(time.time() - start_time)
                collected = gc.collect() # or gc.collect(2) 
                print("postion, Garbage collector: collected {} objects.".format(collected)) 
        else:
            #allFrameResult = []
            allFrameMask = []
            for position in positions:
                allFrameMask.clear()
                allZmask = []
                #allZresult = []
                #allFrameResult.clear()
                for frame in frames:
                    start_time = time.time()
                    print('segment frame '+str(frame))
                    allZmask.clear()
                    #allZresult.clear()
                    if Zs==None:
                        Zs=[1]
                    for z in Zs:
                        _, masks = self.segmentOneSlice(position,frame,z)
                        #allZresult.append(result)
                        allZmask.append(masks)
                    print(time.time() - start_time)

                    #allFrameResult.append(allZresult)
                    allFrameMask.append(allZmask)
                #self.experimentObj.pickleSegResults(allFrameResult,position)
                self.experimentObj.saveMaskAsTif(allFrameMask,position)
                collected = gc.collect() # or gc.collect(2) 
                print("postion, Garbage collector: collected {} objects.".format(collected)) 


    def segmentOneSlice(self,position,frame,z):
        # although it say one slice, but it has all colors
        slice2segment = self.fileClass.getOneSlice(position,frame,z)
        segmentResult, masks = self.experimentObj.segmentMethodObj.segment(slice2segment, \
              position, frame)
        #collected = gc.collect() # or gc.collect(2) 
        #print("frame Garbage collector: collected {} objects.".format(collected)) 
        return segmentResult, masks

                        


