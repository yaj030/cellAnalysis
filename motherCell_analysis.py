import imageAnalysis
from os import path,rename
import pims
import json
import numpy as np
import pandas as pd
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage import exposure
from skimage.measure import regionprops
from skimage.filters import gaussian

class MEASUREMENT:
    def __init__(self,imAnalysis,positions=[]):
        self.imAnalysis = imAnalysis
        self.chNumb = 0
        self.outputs = []
        self.positions = positions
        self.folderName = imAnalysis.experimentPath
        self.numbChannels = self.imAnalysis.scope.numChannels
        self.name = ''
    def measure(self):
        for pos in self.positions:
            self.position = pos
            self.currentFolderName = path.join(self.folderName,'xy'+str(pos))
            self.process()
    def getImage(self):
        filePath = path.join(self.folderName,'xy'+str(self.position).zfill(2)+'c'+str(self.chNumb)+'.tif')
        self.imageStacks = pims.open(filePath)
        print('reading image'+filePath)
    def getSeparatedChannels(self,image,fr):
        shifts = self.imAnalysis.registration[self.position][fr]
        ChLocations_y = self.imAnalysis.experimentObj.ChLocations_y[self.position]
        ChLocations_x = self.imAnalysis.experimentObj.ChLocations_x[self.position]
        separatedImage = self.imAnalysis.scope.separateChannels(image,shifts, ChLocations_x, ChLocations_y)
        return separatedImage
    def getMasks(self):
        # this only get the phase contrast mask
        self.masks = []
        for ch in range(self.numbChannels):
            if self.tifSaved[ch]:
                maskFile = path.join(self.currentFolderName, 'ch'+str(ch+1)+'mask_tracked.tif')
                self.masks.append(pims.open(maskFile))
            else:
                # this is purely for keep the list has the same number elements as numb_channels
                maskFile = path.join(self.currentFolderName, 'ch'+str(ch+1)+'mask.tif')
                self.masks.append(pims.open(maskFile))
    def getData(self):
        jsonFileName = path.join(self.currentFolderName,'data.json')
        with open(jsonFileName,'r') as datafile:
            self.data2save = (json.load(datafile))['data2save'] # list of all channels
        self.numb_frames = self.data2save[0]['numb_frames']
        self.tifSaved = []
        self.lastMotherFrame = []
        for ch in range(self.numbChannels):
            self.tifSaved.append(self.data2save[ch]['saved'])
            self.lastMotherFrame.append(self.data2save[ch]['mother_last_frame'])

    def refreshOutputLists(self):
        self.all_output_data = [] 
        numb_col = len(self.outputs)
        for ch in range(self.numbChannels):
            if self.tifSaved[ch]:
                numb_row = self.lastMotherFrame[ch]
                self.all_output_data.append(pd.DataFrame(np.zeros((numb_row,numb_col)),
                    index=range(numb_row),columns=self.outputs))
            else: 
                self.all_output_data.append(pd.DataFrame(np.zeros((1,numb_col)),
                    index=[1],columns=self.outputs))
    def saveData(self):
        xlsxPath = path.join(self.currentFolderName, self.name+'data.xlsx')
        writer = pd.ExcelWriter(xlsxPath, engine='xlsxwriter')
        for ch in range(self.numbChannels):
            sheetName = 'channel '+str(ch+1)
            self.all_output_data[ch].to_excel(writer, sheet_name=sheetName)
        writer.save()
        writer.close()
    def measureSingleFrame(self,fr):
        image = self.imageStacks[fr]
        background = self.getBackground(image)
        image = image - background
        self.separateImages = self.getSeparatedChannels(image,fr)
    def getBackground(self,image):
        return 0
    def process(self):
        self.getData()
        self.getImage()
        self.getMasks()
        self.refreshOutputLists()
        for fr in range(self.numb_frames):
            self.measureSingleFrame(fr)
        self.saveData()

class measureFluorInt(MEASUREMENT):
    def __init__(self,master,positions=[],color='',**options):
        super().__init__(master,positions)
        self.color = color
        self.name = color
        self.chNumb = self.imAnalysis.colors.index(self.color)+1
        outputs = ['Mean','Sum','NumbPixels','Median','Std',
        'Top5pixelMean', 'Top10pixelMean','Top20pixelMean',
        'Top5perMean','Top10perMean','Top20perMean']
        self.outputs = []
        for key in outputs:
            self.outputs.append(color + '_' + key)
        self.options = {}
        self.options.update(options)
    def measureSingleFrame(self,fr):
        super().measureSingleFrame(fr)
        for ch in range(self.numbChannels):
            if self.tifSaved[ch] and self.lastMotherFrame[ch]>fr:
                mask = self.masks[ch][fr]
                motherMask = (mask==12)
                chImage = self.separateImages[ch]
                allPixels = np.sort(chImage[motherMask])
                allPixels = np.flip(allPixels)
                totalPixels = len(allPixels)
                self.all_output_data[ch].at[fr,self.outputs[0]]=np.mean(allPixels)
                self.all_output_data[ch].at[fr,self.outputs[1]]=np.sum(allPixels)
                self.all_output_data[ch].at[fr,self.outputs[2]]=totalPixels
                self.all_output_data[ch].at[fr,self.outputs[3]]=np.median(allPixels)
                self.all_output_data[ch].at[fr,self.outputs[4]]=np.std(allPixels)
                self.all_output_data[ch].at[fr,self.outputs[5]]=np.mean(allPixels[0:5])
                self.all_output_data[ch].at[fr,self.outputs[6]]=np.mean(allPixels[0:10])
                self.all_output_data[ch].at[fr,self.outputs[7]]=np.mean(allPixels[0:20])
                numbPixels = int(np.ceil(totalPixels*5/100))
                self.all_output_data[ch].at[fr,self.outputs[8]]=np.mean(allPixels[0:numbPixels])
                numbPixels = int(np.ceil(totalPixels*10/100))
                self.all_output_data[ch].at[fr,self.outputs[9]]=np.mean(allPixels[0:numbPixels])
                numbPixels = int(np.ceil(totalPixels*20/100))
                self.all_output_data[ch].at[fr,self.outputs[10]]=np.mean(allPixels[0:numbPixels])

#%%
class measureFoci(MEASUREMENT):
    def __init__(self,master,positions=[],color='',**options):
        super().__init__(master,positions)
        self.color = color
        self.name = color+'foci'
        self.chNumb = self.imAnalysis.colors.index(color)+1
        self.outputs = ['Numb of Foci'] + [('foci mean'+str(s)) for s in range(1,8)]
        self.options = {'cutoff':2,'lb':11,'ub':-3}
        self.options.update(options)

    def measureSingleFrame(self,fr):
        super().measureSingleFrame(fr)
        for ch in range(self.numbChannels):
            if self.tifSaved[ch] and self.lastMotherFrame[ch]>fr:
                mask = self.masks[ch][fr]
                mask[mask!=12] = 0
                chImage = self.separateImages[ch]
                #chImage = gaussian(chImage,1)
                image_orig = regionprops(mask,chImage)[0].intensity_image
                low, high = np.percentile(image_orig, (5, 95))
                image = exposure.rescale_intensity(image_orig, in_range=(low,high))
                #image = exposure.equalize_hist(image_orig)
                segments = slic(image, n_segments = 24, sigma = 1,compactness=0.3)
                regions = regionprops(segments+1,intensity_image=image_orig)
                # all_intensities: the 4 dimmest corner is excluded
                all_intensities = np.sort([props.mean_intensity for props in regions])
                mean_int = np.mean(all_intensities[self.options['lb']:self.options['ub']])
                #std_int = np.std(all_intensities[11:-3])
                foci_mean = []
                numb_foci = 0
                for intensity in all_intensities:
                    #if intensity>mean_int+self.options['cutoff']*std_int or intensity > 50:
                    if intensity>mean_int*self.options['cutoff']:
                        foci_mean.append(intensity)
                        numb_foci = numb_foci+1
                self.all_output_data[ch].at[fr,self.outputs[0]]=numb_foci
                ii = 1
                for fc in foci_mean:
                    if ii>4:
                        print(ii)
                    self.all_output_data[ch].at[fr,self.outputs[ii]]=fc
                    ii = ii+1




