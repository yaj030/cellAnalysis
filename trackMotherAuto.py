# colors are all inherited from first frame
#%%
import numpy as np
from tifffile import imread as tifread
from tifffile import imwrite as tifwrite
from skimage.measure import regionprops
#from skimage.morphology import dilation,disk
import utils
from PIL import Image
from os import path
import pims
from os import path, rename
import json
from datetime import datetime
#%%
class TRACKMOTHER_MANAGER_AUTO:
    def __init__(self, experimentPath='',numb_channels=13,positions=[]):
        self.experimentPath = experimentPath
        self.numb_channels = numb_channels
        self.positions = positions
    def updateFolder(self, folderName):
        self.folderName = folderName
        self.filesPath = []
        self.all_motherTrackers = []
        self.all_chs_images = []
        self.json_path = path.join(self.folderName, 'data.json')

        for ch in range(self.numb_channels):
            self.filesPath.append(path.join(self.folderName, 'ch'+str(ch+1)+'mask.tif'))
            self.all_motherTrackers.append(TRACKMOTHER(self, ch))
            self.all_chs_images.append(self.all_motherTrackers[ch].all_images)
        self.fileNames = [path.basename(files) for files in self.filesPath]

        self.json_exist = path.exists(self.json_path)
        if self.json_exist:
            with open(self.json_path,'r') as datafile:
                data2save = (json.load(datafile))['data2save']
            for ch in range(self.numb_channels):
                self.all_motherTrackers[ch].data = data2save[ch]
        else:
            for ch in range(self.numb_channels):
                self.all_motherTrackers[ch].data = {"ch":ch+1}
            
    def saveResults(self):
        if self.json_exist:
            date = datetime.now().strftime("-%m%d%y-%H%M%S")
            json_bak_path = path.join(self.folderName,'data'+date+'.json')
            rename(self.json_path,path.join(self.folderName,json_bak_path))
        data2save = []
        for ch in range(self.numb_channels):
            #self.all_motherTrackers[ch].saveResults()
            data2save.append(self.all_motherTrackers[ch].data)
        with open(self.json_path,'w') as datafile:
            json.dump({"data2save":data2save},datafile, indent=2)

    def trackAllPositions(self):
        for pos in self.positions:
            folderName = path.join(self.experimentPath, 'xy'+str(pos))
            self.updateFolder(folderName)
            self.trackMotherCell()
            print('finished' + str(pos))


    def trackMotherCell(self):
        for ch in range(self.numb_channels):    
            self.all_motherTrackers[ch].linkMultiFrames()
            self.all_motherTrackers[ch].saveResults()
        self.saveResults()

class TRACKMOTHER:
    def __init__(self, motherTrackerManager, ch):
        self.motherTrackerManager = motherTrackerManager
        self.filePath = self.motherTrackerManager.filesPath[ch]
        self.ch = ch
        self.removeOverlap = False
        self.readImages()

        self.motherLabelNumb = 12
        self.completed = False
        self.saveOrNot = False
        self.tracking_status = 'NotTracked'
        self.motherLastFrame = self.numb_frames

    def saveResults(self):
        folderName = self.motherTrackerManager.folderName
        filePath = path.join(folderName, 'ch'+str(self.ch+1)+'mask_tracked.tif')
        if self.saveOrNot:
            shape = self.all_images.shape
            image2 = np.zeros((self.motherLastFrame,1,1,shape[0],shape[1],1), dtype = np.uint8)
            image2[:,0,0,:,:,0] = np.moveaxis(self.all_images,-1,0)[0:self.motherLastFrame,:,:]
            tifwrite(filePath, image2,imagej=True)
            
        if not self.motherTrackerManager.json_exist:
            self.contructDataDict() #save anyway no matter wanto to save or not
        elif self.saveOrNot:
            self.contructDataDict() #update only want to save
            
    def contructDataDict(self):
        self.data = {}
        self.data["ch"] = self.ch+1
        self.data['numb_frames'] = self.numb_frames
        self.data["saved"] = self.saveOrNot
        self.data["tracking_status"] = self.tracking_status
        self.data["mother_last_frame"] = self.motherLastFrame
        self.data["original_mask"] = self.filePath
        self.data["folder_name"] = self.motherTrackerManager.folderName
        self.data["tracked_mask"] = path.join(self.motherTrackerManager.folderName,
         'ch'+str(self.ch+1)+'mask_tracked.tif')

    def readImages(self):
        img = Image.open(self.filePath)
        self.numb_frames = img.n_frames
        self.height = img.height
        self.width = img.width
        img.close()

        self.all_images = np.zeros([self.height, self.width,self.numb_frames], dtype=np.uint8)
        # pims.open use TiffStack_tifffile as default, which will have problem with
        # tiff files saved in a attaching mode
        # self.all_images_orig = pims.open(self.filePath)
        self.all_images_orig = pims.TiffStack_pil(self.filePath)

    def linkMultiFrames(self):  
        for fr in range(self.numb_frames-1):
            if fr==0:
                image1 = np.copy(self.all_images_orig[0])
                regions1 = regionprops(image1)
                if not regions1:# the first frame is empty
                    image1 = self.all_images_orig[1]
                    regions1 = regionprops(image1)
                if self.removeOverlap:
                    image1 = utils.removeOverlapSmallObj(image1)
                    regions1 = regionprops(image1)
                if not regions1:# the first and the second frame are empty
                    self.tracking_status = 'No cells'
                    self.motherLastFrame = 1
                    break
                motherLabel = image1[105,30]
                image1[image1==motherLabel] = self.motherLabelNumb
                if (image1==self.motherLabelNumb).any():
                    motherUndertrack = True
                else:
                    motherUndertrack = False
                    self.tracking_status = 'No mother cell'
                    self.motherLastFrame = 1
                    break
                self.all_images[:,:,fr] = image1

            regions1 = regionprops(image1)
            image2 = self.all_images_orig[fr+1]
            if self.removeOverlap:
                image2 = utils.removeOverlapSmallObj(image2)
            regions2 = regionprops(image2)

            image2,_, _,_,motherFind = utils.link_twoframes(image1,image2,regions1,regions2,self.motherLabelNumb)
            if (not motherFind) and motherUndertrack:
                image2[:,:] = self.all_images_orig[fr]
                if self.removeOverlap:
                    image2 = utils.removeOverlapSmallObj(image2)
                regions2 = regionprops(image2)
                image2,_, _,_,motherFind = utils.link_twoframes(image1,image2,regions1,regions2,self.motherLabelNumb)
            if motherFind:
                motherUndertrack = motherFind
            else:
                self.motherLastFrame = fr+1
                self.saveOrNot = True
                self.tracking_status = 'Mother lost'
                break

            self.all_images[:,:,fr+1]=image2[:,:]
            image1[:,:] = image2[:,:] # don't do image1=image2!

            if fr == self.numb_frames-2:
                if motherFind:
                    self.motherLastFrame = fr+2
                    self.saveOrNot = True
                    self.completed = True
                    self.tracking_status = 'tracked to last frame'
                

#%%

#tm = TRACKMOTHER_MANAGER_AUTO('/home/nas2/yang/Julie/20180827/data',13,range(17,33))
#tm = TRACKMOTHER_MANAGER_AUTO('/home/yanfei/Julie_Aging/20191007/',13,[2])
#tm.trackAllPositions()




# %%
