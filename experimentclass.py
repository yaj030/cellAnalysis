from os import path, mkdir
import importlib
from tifffile import imwrite as tifwrite
from tifffile import TiffWriter
import numpy as np
import pickle
import json
from skimage.io import imread, imsave
from skimage.feature import register_translation
from skimage import exposure, img_as_ubyte 
import time
#from PIL import Image

class EXPERIMENT_TYPE:
    def setAttr(self, imAnalysis):
        self.imAnalysis = imAnalysis
    def setSegmentMethod(self, segmentMethodName=''):
        self.segmentationMethodName = segmentMethodName
        self.startSegmentMethod
    def startSegmentMethod(self):
        self.segmentClass = getattr(importlib.import_module(self.segmentationMethodName), self.segmentationMethodName)
        self.segmentMethodObj = self.segmentClass(self.imAnalysis)
        self.segmentMethodObj.methodPreparation()
    def saveMaskAsTif(self):
        # it is called in the imageAnalysis class
        pass
    def append2MaskTif(self):
        pass
    def pickleSegResults(self):
        # it is called in the imageAnalysis class
        pass
    def getRegArea(self, image):
        # need to be override
        # crop the image for image shift calculation
        # might be different for differnt kind of experiment
        pass
    def imageRegistration(self, positions=[]):
        if len(positions) == 0:
            print('calculate image shift for all positions')
            positions2reg = self.imAnalysis.positions
        else:
            # sometimes you just want to test one position
            positions2reg = [x-1 for x in positions]
        start_time = time.time()
        shiftsAllPositions = []
        shifts = []
        for position in positions2reg:
            shifts.clear()
            shifts.append(np.zeros([2,]).tolist())
            zeroFrame = self.imAnalysis.fileClass.getOneSliceColor(position,1,1)
            for frame in range(2, self.imAnalysis.totalFrames+1):
                secondFrame = self.imAnalysis.fileClass.getOneSliceColor(position,frame,1)
                shift, _, _ = register_translation(self.getRegArea(zeroFrame), self.getRegArea(secondFrame),10)
                #shifts.append(shift)
                shifts.append(shift.tolist())
            shiftsAllPositions.append(shifts)
            print(time.time() - start_time)
        # shift is save as a dict
        self.imAnalysis.registration = dict(zip(positions2reg,shiftsAllPositions))
        saveFilePath = path.join(self.imAnalysis.experimentPath,'registration.pkl')
        pickle.dump(self.imAnalysis.registration,open(saveFilePath,'wb'))
        saveFilePath = path.join(self.imAnalysis.experimentPath,'registration.json')
        json.dump(self.imAnalysis.registration,open(saveFilePath,'w'))
        print('finished registration')

class agingExperiment(EXPERIMENT_TYPE):
    def __init__(self):
        self.segmentationMethodName = 'Adarsh2' #default method

    def getRegArea(self, image):
        # it is tested for Julie scope, should be fine with Yang's too
        return(image[:,:128])

    def pickleSegResults(self, allFrameResult, position):
        # only works for savingAsRuntime = False
        positionFolder = path.join(self.imAnalysis.experimentPath,'xy'+str(position)) 
        if not path.exists(positionFolder):
            try:
                mkdir(positionFolder)
            except OSError:
                print("can't create dir")
        saveFilePath = path.join(positionFolder, 'segmentationResults.pkl')
        with open(saveFilePath,'wb') as f:
            pickle.dump(allFrameResult,f)
        print('pickled segmentation result')

    def saveMaskAsTif(self,allFrameMask, position):
        # this will hyper stack, but can only save after segmentation of all frames are finished
        savePath = path.join(self.imAnalysis.experimentPath, 'xy'+str(position))
        if not path.exists(savePath):
            try:
                mkdir(savePath)
            except OSError:
                print("can't create dir")
        #mask tif
        numFrames = len(allFrameMask)
        numZ = len(allFrameMask[0])
        numChannels = len(allFrameMask[0][0])
        # first subscript frame, second is z, third is channel number
        # mask for channels is a dictionary, keys start at 1
        height = allFrameMask[0][0][1].shape[0]
        width = allFrameMask[0][0][1].shape[1]
        for channel in range(1,numChannels+1): #start at 1
            saveFilePath = path.join(savePath,'ch'+ str(channel)+'mask.tif')
            image2write = np.ndarray((numFrames,numZ,1,height,width), 'uint8')
            for fr in range(0,numFrames):
                for zi in range(0,numZ):
                    image2write[fr,zi,0,:,:] = allFrameMask[fr][zi][channel][:,:]
            tifwrite(saveFilePath,image2write,photometric='minisblack',imagej=True)
        print('masks saved as tiff successfully')

    def append2MaskTif(self,masks,position, z=None):
        # each z has its own file, but save as runtime
        savePath = path.join(self.imAnalysis.experimentPath, 'xy'+str(position))
        if not path.exists(savePath):
            try:
                mkdir(savePath)
            except OSError:
                print("can't create dir")
        #mask tif
        numChannels = len(masks)
        if z==None:
            assert(numChannels==1,'numChannel is one but z is not none')
            prefix = ''
        else:
            prefix = 'z'+str(z)
        for channel in range(1,numChannels+1): #start at 1
            print('saving channel '+str(channel))
            saveFilePath = path.join(savePath,prefix+'ch'+ str(channel)+'mask.tif')
            image2write = masks[channel][:,:]
            tifwrite(saveFilePath,image2write,photometric='minisblack',append=True)

    def getChLocAllPositions(self):
        phase_numb = self.imAnalysis.colors.index('phase')
        xs = []
        ys = []
        positions = self.imAnalysis.positions
        for position in positions:
            image = self.imAnalysis.fileClass.getOneSlice(position,1)
            x, y = self.imAnalysis.scope.getChannelLocations(image[phase_numb,:,:])
            xs.append(x)
            ys.append(y)
            print('channels searched at position '+ str(position))
        self.ChLocations_x = dict(zip(positions, xs))
        self.ChLocations_y = dict(zip(positions, ys))
        saveFilePath = path.join(self.imAnalysis.experimentPath,'channelLocation.pkl')
        pickle.dump([self.ChLocations_x,self.ChLocations_y],open(saveFilePath,'wb'))
        saveFilePath = path.join(self.imAnalysis.experimentPath,'channelLocation.json')
        json.dump([self.ChLocations_x,self.ChLocations_y],open(saveFilePath,'w'))

    def loadChLocations(self):
        ChLocPath = path.join(self.imAnalysis.experimentPath, 'channelLocation.pkl')
        if path.exists(ChLocPath):
            dataload = pickle.load(open(ChLocPath,'rb'))
            self.ChLocations_x = dataload[0]
            self.ChLocations_y = dataload[1]
            print('channel locations loaded, pickle')
            return True
        else:
            print('channel locations not searched yet')
            return False

    def getChannelImage(self,image,Ch, position, frame, color=None):
        shifts = self.imAnalysis.registration[position][frame-1]
        ChLocations_y = self.imAnalysis.experimentObj.ChLocations_y[position]
        ChLocations_x = self.imAnalysis.experimentObj.ChLocations_x[position]
        separatedImage = self.imAnalysis.scope.separateChannels(image,shifts, ChLocations_x, ChLocations_y)
        if color == None:
            return separatedImage[Ch-1]
        else:
            return separatedImage[Ch-1][color-1,:,:]

    def output4Training(self, folderName='', prefix='', positions=[], frames=[], channels=[], color=1, z=None,equalize=True):
        # only work with one color and one z
        if positions==[]:
            positions = self.imAnalysis.positions
        if frames==[]:
            frames = range(1,self.imAnalysis.totalFrames+1)
        if channels==[]:
            channels=range(1,self.imAnalysis.scope.numChannels+1)
        if not path.exists(path.join(folderName, 'tif')):
            mkdir(path.join(folderName,'tif'))
        if not path.exists(path.join(folderName, 'jpg')):
            mkdir(path.join(folderName,'jpg'))
        for position in positions:
            ChLocations_y = self.imAnalysis.experimentObj.ChLocations_y[position]
            ChLocations_x = self.imAnalysis.experimentObj.ChLocations_x[position]
            for frame in frames:
                shifts = self.imAnalysis.registration[position][frame-1]
                image = self.imAnalysis.fileClass.getOneSliceColor(position, frame, color, z)
                if equalize:
                    image = exposure.equalize_adapthist(image)
                    image = img_as_ubyte(image)
                    separatedImage = self.imAnalysis.scope.separateChannels(image,shifts, ChLocations_x, ChLocations_y)
                    for ch in channels:
                        subImage = separatedImage[ch-1]
                        fileName = prefix + '_xy' + str(position).zfill(2) + 'c' + str(color)\
                            + '_' + str(frame).zfill(3) + '_im_' + str(ch).zfill(2)
                        tifwrite(path.join(folderName, 'tif', fileName+'.tif'), subImage)
                        imsave(path.join(folderName,'jpg', fileName+'.jpg'), subImage,check_contrast=False, quality=90)
                        #cv2.imwrite(path.join(folderName,'jpg', fileName+'.jpg'), subImage, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        saveFilePath = path.join(folderName,'tif',fileName+'.tif')
                        try:
                            imread(saveFilePath)
                        except:
                            print("error reading")
                            print(saveFilePath)
                else:
                    separatedImage = self.imAnalysis.scope.separateChannels(image,shifts, ChLocations_x, ChLocations_y)
                    for ch in channels:
                        subImage = separatedImage[ch-1]
                        fileName = prefix + '_xy' + str(position).zfill(2) + 'c' + str(color)\
                            + '_' + str(frame).zfill(3) + '_im_' + str(ch).zfill(2)
                        # output original as tif
                        tifwrite(path.join(folderName, 'tif', fileName+'.tif'), subImage)
                        # save jpg after adjust contrast
                        subImage = exposure.equalize_adapthist(subImage)
                        imsave(path.join(folderName,'jpg', fileName+'.jpg'), subImage, quality=90)
                        #cv2.imwrite(path.join(folderName,'jpg', fileName+'.jpg'), subImage, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        saveFilePath = path.join(folderName,'tif',fileName+'.tif')
                        try:
                            imread(saveFilePath)
                        except:
                            print("error reading")
                            print(saveFilePath)
        

    def output4Separated(self, positions=[], color=1,outputTif=True):
        # only work with ONE color, not multicolor images
        # output all frames and all channels
        frames = range(1,self.imAnalysis.totalFrames+1)
        channels=range(1,self.imAnalysis.scope.numChannels+1)
        z=None
        # only work with one color and one z
        width = self.imAnalysis.scope.separateWidth*2
        height = self.imAnalysis.scope.separateHeight
        if positions==[]:
            positions = self.imAnalysis.positions
        for position in positions:
            print('outputing position ' + str(position))
            ChLocations_y = self.imAnalysis.experimentObj.ChLocations_y[position]
            ChLocations_x = self.imAnalysis.experimentObj.ChLocations_x[position]
            list_of_channelImage = [np.ndarray([len(frames),1,1,height, width],dtype=np.uint16) for _ in range(len(channels))]
            for frame in frames:
                shifts = self.imAnalysis.registration[position][frame-1]
                image = self.imAnalysis.fileClass.getOneSliceColor(position, frame, color, z)
                separatedImage = self.imAnalysis.scope.separateChannels(image,shifts, ChLocations_x, ChLocations_y)
                for ch in channels:
                    subImage = separatedImage[ch-1]
                    list_of_channelImage[ch-1][frame-1,:,:] = subImage
            
            if outputTif:
                savePath = path.join(self.imAnalysis.experimentPath, 'xy'+str(position))
                print(savePath)
                if not path.exists(savePath):
                    mkdir(savePath)
                for ch in channels:
                    saveFilePath = path.join(savePath,'xy'+str(position).zfill(2)+'c'+ str(color)+'ch'+ str(ch)+'.tif')
                    tifwrite(saveFilePath, list_of_channelImage[ch-1], imagej=True)
            else:
                # only work with one position
                image = [np.moveaxis(np.squeeze(list_of_channelImage[ch-1]),0,-1) for ch in channels]
                return image


