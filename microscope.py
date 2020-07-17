
from os.path import abspath, dirname, join
from skimage.feature import match_template
from skimage.io import imread
import numpy as np
from replaceOutliers import replaceOutliers
class MICROSCOPE:
    def changeColorNaming(self):
        pass
    def separateChannels(self):
        pass
    def getChannelLocations(self):
        pass
class JulieMicroscope(MICROSCOPE):
    numChannels = 13
    templateFileName = 'JulieTemplate.tif'
    separateWidth = 30#40 this gives 60 pixel width
    separateHeight = 160
    separateOffset = 0 # 0 means the bottom is the channel bottom
    currentPath = dirname(abspath(__file__))
    def changeColorNaming(self, colors):
        res = [sub.replace('BF', 'phase') for sub in colors] 
        return res

    def separateChannels(self, image, shifts, ChLocations_x, ChLocations_y):
        # output separatedImage is a list, not dict, index start from 0
    # shifts[0] is y shift, positive is shifting up
    # shifts[1] is x shift, positive is shifting left
    # np.roll, positive is shifting right and down
        separatedImage = []
        if image.ndim==3:
            image_width = image.shape[2]
            print(shifts)
            #image2 = np.roll(image, int(shifts[0]),axis=1)
            #image = np.roll(image2, int(shifts[1]),axis=2)
            image = np.roll(image, [int(shifts[0]), int(shifts[1])],axis=(1,2))
        else:
            assert(image.ndim==2, 'image dimension is neither 3 or 2')
            image_width = image.shape[1]
            image = np.roll(image, [int(shifts[0]), int(shifts[1])],axis=(0,1))
            #image2 = np.roll(image, shifts[0],axis=0)
            #image = np.roll(image2, shifts[1],axis=1)
        for ii in range(JulieMicroscope.numChannels):
            #y = int(ChLocations_y[ii] - shifts[0]) - JulieMicroscope.separateOffset
            #x = int(ChLocations_x[ii] - shifts[1])
            y = int(ChLocations_y[ii]) - JulieMicroscope.separateOffset
            x = int(ChLocations_x[ii])
            if x<JulieMicroscope.separateWidth:
                rollshift = JulieMicroscope.separateWidth - x
                x = JulieMicroscope.separateWidth
                if image.ndim==3:
                    appendImage = image[:,\
                        y-JulieMicroscope.separateHeight:y, \
                        x-JulieMicroscope.separateWidth:x+JulieMicroscope.separateWidth]
                    appendImage = np.roll(appendImage, rollshift, axis=2)
                    separatedImage.append(appendImage)
                elif image.ndim==2:
                    appendImage = image[y-JulieMicroscope.separateHeight:y, \
                        x-JulieMicroscope.separateWidth:x+JulieMicroscope.separateWidth]
                    appendImage = np.roll(appendImage, rollshift, axis=1)
                    separatedImage.append(appendImage)
            elif image_width<x+JulieMicroscope.separateWidth:
                rollshift = JulieMicroscope.separateWidth + x - image_width
                x = x - rollshift
                if image.ndim==3:
                    appendImage = image[:,\
                        y-JulieMicroscope.separateHeight:y, \
                        x-JulieMicroscope.separateWidth:x+JulieMicroscope.separateWidth]
                    appendImage = np.roll(appendImage, -rollshift, axis=2)
                    separatedImage.append(appendImage)
                elif image.ndim==2:
                    appendImage = image[y-JulieMicroscope.separateHeight:y, \
                        x-JulieMicroscope.separateWidth:x+JulieMicroscope.separateWidth]
                    appendImage = np.roll(appendImage, -rollshift, axis=1)
                    separatedImage.append(appendImage)
            else:
                if image.ndim==3:
                    appendImage = image[:,\
                        y-JulieMicroscope.separateHeight:y, \
                        x-JulieMicroscope.separateWidth:x+JulieMicroscope.separateWidth]
                    separatedImage.append(appendImage)
                elif image.ndim==2:
                    appendImage = image[y-JulieMicroscope.separateHeight:y, \
                        x-JulieMicroscope.separateWidth:x+JulieMicroscope.separateWidth]
                    separatedImage.append(appendImage)
        # a list not dict, ind start from 0
        return separatedImage

    def getChannelLocations(self,image):
        # get the left lower corner image
        template = imread(join(JulieMicroscope.currentPath, 'template.tif'))
        result = match_template(image, template)
        #
        ChLocations_x = []
        ChLocations_y = []
        for ii in range(JulieMicroscope.numChannels):
            subResult = result[:,ii*113:(ii+1)*113]
            y, x = np.unravel_index(np.argmax(subResult),subResult.shape)
            y = y + template.shape[0]
            x = x + 113*ii + template.shape[1]//2
            # y, x is the bottom center position of the cell trap
            ChLocations_x.append(x)
            ChLocations_y.append(y)
        x2 = np.uint16(replaceOutliers(range(0,13), ChLocations_x,15))
        y2 = np.uint16(replaceOutliers(range(0,13), ChLocations_y,15))
        return x2, y2


class YangMicroscope:
    numChannels = 6