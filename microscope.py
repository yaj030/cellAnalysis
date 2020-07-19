
from os.path import abspath, dirname, join
from skimage.feature import match_template
from skimage.io import imread
import numpy as np
from replaceOutliers import replaceOutliers
class MICROSCOPE:
    def __init__(self): 
        # defaults are  all for Julie microscope
        self.numChannels = 13
        self.templateFileName = 'JulieTemplate.tif'
        self.separateWidth = 30#40 this gives 60 pixel width
        self.separateHeight = 160
        self.separateOffset = 0 # 0 means the bottom is the channel bottom
        self.currentPath = dirname(abspath(__file__))
        self.channel_distance = 113

    def separateChannels(self, image, shifts, ChLocations_x, ChLocations_y):
        # output separatedImage is a list, not dict, index start from 0
    # shifts[0] is y shift, positive is shifting up
    # shifts[1] is x shift, positive is shifting left
    # np.roll, positive is shifting right and down
        separatedImage = []
        if image.ndim==3:
            image_width = image.shape[2]
            print(shifts)
            image = np.roll(image, [int(shifts[0]), int(shifts[1])],axis=(1,2))
        else:
            assert(image.ndim==2, 'image dimension is neither 3 or 2')
            image_width = image.shape[1]
            image = np.roll(image, [int(shifts[0]), int(shifts[1])],axis=(0,1))
        for ii in range(self.numChannels):
            y = int(ChLocations_y[ii]) - self.separateOffset
            x = int(ChLocations_x[ii])
            if x<self.separateWidth:
                rollshift = self.separateWidth - x
                x = self.separateWidth
                if image.ndim==3:
                    appendImage = image[:,\
                        y-self.separateHeight:y, \
                        x-self.separateWidth:x+self.separateWidth]
                    appendImage = np.roll(appendImage, rollshift, axis=2)
                    separatedImage.append(appendImage)
                elif image.ndim==2:
                    appendImage = image[y-self.separateHeight:y, \
                        x-self.separateWidth:x+self.separateWidth]
                    appendImage = np.roll(appendImage, rollshift, axis=1)
                    separatedImage.append(appendImage)
            elif image_width<x+self.separateWidth:
                rollshift = self.separateWidth + x - image_width
                x = x - rollshift
                if image.ndim==3:
                    appendImage = image[:,\
                        y-self.separateHeight:y, \
                        x-self.separateWidth:x+self.separateWidth]
                    appendImage = np.roll(appendImage, -rollshift, axis=2)
                    separatedImage.append(appendImage)
                elif image.ndim==2:
                    appendImage = image[y-self.separateHeight:y, \
                        x-self.separateWidth:x+self.separateWidth]
                    appendImage = np.roll(appendImage, -rollshift, axis=1)
                    separatedImage.append(appendImage)
            else:
                if image.ndim==3:
                    appendImage = image[:,\
                        y-self.separateHeight:y, \
                        x-self.separateWidth:x+self.separateWidth]
                    separatedImage.append(appendImage)
                elif image.ndim==2:
                    appendImage = image[y-self.separateHeight:y, \
                        x-self.separateWidth:x+self.separateWidth]
                    separatedImage.append(appendImage)
        # a list not dict, ind start from 0
        return separatedImage

    def getChannelLocations(self,image):
        # get the left lower corner image
        template = imread(join(self.currentPath, self.templateFileName))
        result = match_template(image, template)
        ChLocations_x = []
        ChLocations_y = []
        for ii in range(self.numChannels):
            subResult = result[:,ii*self.channel_distance:(ii+1)*self.channel_distance]
            y, x = np.unravel_index(np.argmax(subResult),subResult.shape)
            y = y + template.shape[0]
            x = x + self.channel_distance*ii + template.shape[1]//2
            # y, x is the bottom center position of the cell trap
            ChLocations_x.append(x)
            ChLocations_y.append(y)
        x2 = np.int16(replaceOutliers(range(0,self.numChannels), ChLocations_x,15))
        y2 = np.int16(replaceOutliers(range(0,self.numChannels), ChLocations_y,15))
        return x2.tolist(), y2.tolist()

    def changeColorNaming(self, colors):
        res = [sub.replace('BF', 'phase') for sub in colors] 
        return res
class JulieMicroscope(MICROSCOPE):
    pass

class YangMicroscope(MICROSCOPE):
    def __init__(self): 
        # default is all for Julie microscope
        self.numChannels = 6
        self.templateFileName = 'YangTemplate.tif'
        self.separateWidth = 20 #this gives 60 pixel width
        self.separateHeight = 120
        self.separateOffset = 0 # 0 means the bottom is the channel bottom
        self.currentPath = dirname(abspath(__file__))
        self.channel_distance = 82 # distance between two traps