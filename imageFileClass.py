from yanfeiND2 import yanfeiND2reader
from os import path
import pickle
from os import mkdir
import numpy as np
from tifffile import imwrite as tifwrite
from tifffile import imread as tifread
from skimage.transform import rotate
from glob import glob
from PIL import Image
class EXPERIMENT_FILE_CLASS:
    def __init__(self, imAnalysis):
        self.imAnalysis = imAnalysis
        self.angle = 0
    def updateFileInfo(self):
        pass
    def getShapes(self):
        pass
    def getOneSlice(self,position,frame,z):
        # need to be override
        return 0
    def getOneSliceColor(self,position,frame,color,z=None):
        pass
    def getAngle(self):
        angletxt = path.join(self.imAnalysis.experimentPath, 'angle.txt')
        if path.exists(angletxt):
            try:
                f = open(angletxt,'r')
                self.angle = float(f.readline())
                print(self.angle)
                f.close()
            except:
                Warning('something wrong with the angle.txt')

class NDfile(EXPERIMENT_FILE_CLASS):
    def __init__(self, imAnalysis):
        super().__init__(imAnalysis)
        self.updateFileInfo()
    def updateFileInfo(self):
        fileName = self.imAnalysis.fileNamePrefix + '.nd2'
        fileFullPath = path.join(self.imAnalysis.experimentPath, fileName)
        try:
            #self.NDimages = ND2Reader(self.NDfileFullPath) # orignal reader, does not work well with truncated ND2
            self.NDimages = yanfeiND2reader(fileFullPath)
        except IOError:
            print('Can not open ND2 file')
        self.getShapes()
        colors = self.NDimages.metadata['channels']
        self.imAnalysis.colors = self.imAnalysis.scope.changeColorNaming(colors)
        self.getAngle()

    def getShapes(self):
        # this function does not get much use
        self.imAnalysis.totalFrames = self.NDimages.sizes['t']
        self.imAnalysis.totalColors = self.NDimages.sizes['c']
        if 'z' in self.NDimages.sizes:
            self.imAnalysis.totalZstacks = self.NDimages.sizes['z']
        else:
            self.imAnalysis.totalZstacks = 1
        self.imAnalysis.totalPositions = self.NDimages.sizes['v']
        self.imAnalysis.positions = range(self.imAnalysis.totalPositions)
        self.imAnalysis.fovHeight = self.NDimages.sizes['x']
        self.imAnalysis.fovWidth = self.NDimages.sizes['y']

        #self.NDimages.bundle_axes = 'yxt'
        # it seems 't' can not be in bundle axes, it gives some difficulty, 
        # I just 
        # self.NDimages.bundle_axes = 'yx'
        # two ways to get a frame of image
        # either iterate with vt, use position*size['t']+frame
        # like
            #secondFrame = self.NDimages[position*self.NDimages.sizes['t']+frame+1]
        # or iterate with v, set default_coords['t'] = frame
        # the first strategy will be a little messy if there is z stacks
        # here we use 'v' approach, it has been tested for saveTiffFiles function
        # but it seems both gives the same frame shift for whole time series
        # important thing is avoid bundle_axex = 'tyx' or 'zyx'
        # bundle 'cyx' or 'vyx' seems fine, not thoroughly tested though

        
    def getOneSliceColor(self,position,frame,color,z=None):
        # color is a number input, don't use like GFP
        self.NDimages.bundle_axes = 'yx'
        self.NDimages.iter_axes = 'v'
        self.NDimages.default_coords['t'] = frame-1
        self.NDimages.default_coords['c'] = color-1
        if 'z' in self.NDimages.sizes.keys():
            self.NDimages.default_coords['z'] = z-1
        elif z is not None:
            print('Warning, this ND2 does not have z stack')
        return self.NDimages[position-1]

    def getOneSlice(self,position,frame,z=None):
        # this function returns all color channels
        # it is called by the imageAnalysis, it is a real public, call with cautious
        # because frame, z and position are all from 1 instead of zero
        self.NDimages.bundle_axes = 'cyx'
        self.NDimages.iter_axes = 'v'
        self.NDimages.default_coords['t'] = frame-1
        if 'z' in self.NDimages.sizes.keys():
            self.NDimages.default_coords['z'] = z-1
        elif z is not None:
            print('Warning, this ND2 does not have z stack')
        return self.NDimages[position-1]

    def saveTiffFiles(self,positions):
        # it save the tif files for each position
        # special for ND files
        if len(positions) == 0:
            positions2save = [x+1 for x in self.NDimages.metadata['fields_of_view']]
        else:
            # sometimes you just want to test one position
            positions2save = positions

        self.NDimages._iter_axes = 'v'
        self.NDimages._bundle_axes = 'cyx'
        for position in positions2save:
            positionFolder = path.join(self.imAnalysis.experimentPath,str(position)) 
            if not path.exists(positionFolder):
                try:
                    mkdir(positionFolder)
                except OSError:
                    print("can't create dir")
            saveFilePath = path.join(positionFolder,'position' + str(position)+'.tif')

            sizes = self.NDimages.sizes
            if 'z' in self.NDimages.sizes.keys():
                image2write = np.ndarray([sizes['t'], self.NDimages.sizes['z'], sizes['c'], sizes['y'],sizes['x']],'uint16')
                for frame in range(0,sizes['t']):
                    for z in range(self.NDimages.sizes['z']):
                        self.NDimages.default_coords['t'] = frame
                        self.NDimages.default_coords['z'] = z
                        image2write[frame,z,:,:,:] = np.uint16(self.NDimages[position-1])
                tifwrite(saveFilePath,image2write,imagej=True)
            else:
                image2write = np.ndarray([sizes['t'], 1, sizes['c'], sizes['y'],sizes['x']],'uint16')
                for frame in range(0,sizes['t']):
                    self.NDimages.default_coords['t'] = frame
                    image2write[frame,0,:,:,:] = np.uint16(self.NDimages[position-1])
                tifwrite(saveFilePath,image2write,imagej=True)

class TifFileNoZstackSplitColor(EXPERIMENT_FILE_CLASS):
    def __init__(self, imAnalysis):
        super().__init__(imAnalysis)
        self.updateFileInfo()

    def updateFileInfo(self):
        allTifFiles = glob(path.join(self.imAnalysis.experimentPath,\
            self.imAnalysis.fileNamePrefix + 'xy*c?.tif'))
        self.allTifFiles = [path.basename(x) for x in allTifFiles]
        numbFiles = len(self.allTifFiles)
        channelNumbs = []
        positionNumbs = []
        for file in self.allTifFiles:
            channelNumbs.append(int(file[-5]))
            positionNumbs.append(int(file[2:-6]))
        channelNumbs = set(channelNumbs)
        positionNumbs = set(positionNumbs)
        if numbFiles == len(channelNumbs)*len(positionNumbs):
            self.imAnalysis.totalPositions = len(positionNumbs) # number of them, but 17-20 means 4
            self.imAnalysis.totalColors = len(channelNumbs)
            self.imAnalysis.positions = positionNumbs
        else:
            raise('the number or the names of files have something wrong')
        self.getShapes()
        self.getAngle()

    def setColors(self, colors=[]):
        # special for tif files
        if len(colors) == self.imAnalysis.totalColors:
            if colors[0] != 'phase':
                print('Warning, the first channel is not phase')
            self.imAnalysis.colors = colors
        else:
            raise('input number of colors does not match what is in the folder')

    def getShapes(self):
        # this function does not get much use
        firstFilePath = path.join(self.imAnalysis.experimentPath, self.allTifFiles[0])
        image = Image.open(firstFilePath)
        self.imAnalysis.totalFrames = image.n_frames
        self.imAnalysis.fovHeight = image.size[1]
        self.imAnalysis.fovWidth = image.size[0]
        image.close()

    def getFileName(self, position, color=1, z=None):
        # special function for tif files
        fileName = self.imAnalysis.fileNamePrefix + 'xy' + str(position).zfill(2) + 'c' + str(color) + '.tif'
        return fileName

    def getOneSlice(self,position, frame, z=None):
        image = np.ndarray([self.imAnalysis.totalColors, self.imAnalysis.fovHeight, self.imAnalysis.fovWidth],dtype=np.uint16)
        for color in range(self.imAnalysis.totalColors):
            # read multi colors and feed to segmentation might be redundant, it is
            # just for later convinence someone write a method using other than phase
            fileName = self.getFileName(position, color+1, z)
            print(fileName)
            fileFullPath = path.join(self.imAnalysis.experimentPath, fileName)
            if self.angle==0:
                image[color,:,:] = tifread(fileFullPath, key=frame-1)
            else:
                image[color,:,:] = rotate(tifread(fileFullPath, key=frame-1),self.angle,preserve_range=True)
        return image
            
    def getOneSliceColor(self,position, frame, color, z=None):
        fileName = self.getFileName(position, color, z)
        fileFullPath = path.join(self.imAnalysis.experimentPath, fileName)
        if self.angle==0:
            image = np.uint16(tifread(fileFullPath, key=frame-1))
        else:
            image = rotate(np.uint16(tifread(fileFullPath, key=frame-1)),self.angle,preserve_range=True)
        return image
