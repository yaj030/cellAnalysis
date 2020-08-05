# this is a more standard way to do it, compared to gui_test3.py
#%%
import numpy as np
from tifffile import imread as tifread
from tifffile import imwrite as tifwrite
from skimage.measure import regionprops
from skimage.exposure import equalize_hist
from skimage.color import gray2rgb
from skimage import img_as_ubyte
#from skimage.morphology import dilation,disk
#import utils
from PIL import Image, ImageTk
from os import path
from math import ceil
from tkinter import Tk, Text, BOTH, NW,YES, HORIZONTAL, Toplevel, Canvas, IntVar
from tkinter import W as tkW
from tkinter import E as tkE
from tkinter import N as tkN
from tkinter import S as tkS
from tkinter import Frame, Button, Label, filedialog, Listbox, Checkbutton, messagebox
import pims
from os import path, rename
import json
from datetime import datetime
import imageAnalysis
import microscope
from experimentclass import agingExperiment
import motherCell_analysis
import utils

#%
class COUNTCELLDIVISION:
    def __init__(self, scope,folderName=None,positions = [],colors=['phase','GFP','iRFP']):
        namaPrefix = ''
        self.experiment = agingExperiment()
        self.scopeUsed = scope
        self.analysis = imageAnalysis.IMAGE_ANALYSIS(folderName,namaPrefix,\
            self.scopeUsed,'tif_noZstack',self.experiment)
        self.analysis.fileClass.setColors(colors)
        if not self.analysis.experimentObj.loadChLocations():
            self.analysis.experimentObj.getChLocAllPositions()
        if not self.analysis.loadRegistration():
            self.analysis.imRegistrationAllPosition()
        self.numb_rows = 3
        self.numb_cols = 20
        self.totalCounts = self.numb_cols*self.numb_rows

        self.numb_channels = self.analysis.scope.numChannels
        self.numbFrames = self.analysis.totalFrames
        self.folderName = folderName
        self.fileNames = []
        self.positions = positions
        self.currentPosition = 0
        self.all_motherTrackers = []
        self.updatePosition(positions[0])

        self.mainWindow = MAINWINDOW(self)
        self.mainWindow.mainloop()
    def continueLoad(self):
        for ch in self.numb_channels

    def updatePosition(self,position):
        self.currentPosition = position
        if hasattr(self, 'separatedImage'):
            self.separatedImage.clear()
        else:
            self.separatedImage = []
        self.getSeparatedImages()
        if self.all_motherTrackers: 
            for ch in range(self.numb_channels):
                self.all_motherTrackers[ch].updatePosition
        else:
            for ch in range(self.numb_channels):
                print(ch)
                self.all_motherTrackers.append(TRACKMOTHER(self, ch))

    def getSeparatedImages(self):
        self.separatedImage = self.analysis.experimentObj.output4Separated(\
            [self.currentPosition],color=1,outputTif=False)
    def saveResult(self):
        pass

class TRACKMOTHER:
    def __init__(self, motherTrackerManager, ch):
        self.motherTrackerManager = motherTrackerManager
        self.ch = ch
        self.motherLabelNumb = 12
        self.updatePosition()
    def updatePosition(self):
        self.posFolderName = path.join(self.motherTrackerManager.folderName,\
            'xy'+str(self.motherTrackerManager.currentPosition))
        self.numb_frames = 0
        self.height = 0
        self.width = 0
        self.getImages()
        self.getMask_ini()
        self.page_trackOrNot = np.zeros(self.numb_pages,dtype=bool)
        self.page=0
        self.completed = False
        self.saveOrNot = False
        self.motherLastFrame = self.numb_frames
        self.divisionTimes = []
        self.firstFrame = 1
        self.lastFrame = self.motherTrackerManager.numbFrames

        self.megaImgHeight = self.height*2*self.motherTrackerManager.numb_rows
        self.megaImgWidth = self.width*self.motherTrackerManager.numb_cols

    def saveResults(self):
        filePath = path.join(self.posFolderName,'lifespan.txt')
        if self.saveOrNot:
            pass
    def getMask_ini(self):
        # only for first page
        lastfr = self.motherTrackerManager.numb_cols*self.motherTrackerManager.numb_rows
        filePath = path.join(self.posFolderName, 'ch'+str(self.ch+1)+'mask_tracked.tif')
        filePath2 = path.join(self.posFolderName, 'ch'+str(self.ch+1)+'mask.tif')
        self.all_images = np.zeros((self.height, self.width, self.numb_frames),dtype=np.uint8)
        for fr in range(lastfr):
            try:
                self.all_images[:,:,fr] = tifread(filePath,key=fr)
            except:
                self.all_images[:,:,fr] = tifread(filePath2,key=fr)

    def getMask_continue(self):
        # only for first page
        filePath = path.join(self.posFolderName, 'ch'+str(self.ch+1)+'mask_tracked.tif')
        filePath2 = path.join(self.posFolderName, 'ch'+str(self.ch+1)+'mask.tif')
        self.all_images = np.zeros((self.height, self.width, self.numb_frames),dtype=np.uint8)
        for fr in range(self.motherTrackerManager.numbFrames):
            try:
                self.all_images[:,:,fr] = tifread(filePath,key=fr)
            except:
                self.all_images[:,:,fr] = tifread(filePath2,key=fr)
        self.all_images_orig = np.copy(self.all_images)
            
    def getImages(self):
        img = img_as_ubyte(equalize_hist(self.motherTrackerManager.separatedImage[self.ch]))
        self.numb_frames = img.shape[2]
        assert(self.numb_frames == self.motherTrackerManager.numbFrames, "number of frames don't match")
        print(img.shape)
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.numb_pages = ceil((self.numb_frames-1)/self.motherTrackerManager.totalCounts)
        self.all_images_phase = np.copy(img)
        for fr in range(self.numb_frames):
            self.all_images_phase[:,:,fr] = img[:,:,fr]

    def constructMegaImage(self,page):
        start_fr = page*self.motherTrackerManager.numb_cols*self.motherTrackerManager.numb_rows
        self.currentPageImgColor = np.zeros((self.megaImgHeight, self.megaImgWidth, 3),dtype=np.uint8)
        for ii in range(self.motherTrackerManager.numb_rows):
            rowsA = []
            rowsB = []
            idy = ii*2*self.height
            for jj in range(self.motherTrackerManager.numb_cols):
                ind = ii*self.motherTrackerManager.numb_cols+jj+start_fr
                if ind<self.motherTrackerManager.numbFrames:
                    rowsA.append(self.all_images_phase[:,:,ind])
                else:
                    rowsA.append(np.zeros((self.height,self.width),dtype=np.uint8))
            self.currentPageImgColor[idy:idy+self.height,:,:] = gray2rgb(np.hstack(tuple(rowsA)))
            for jj in range(self.motherTrackerManager.numb_cols):
                ind = ii*self.motherTrackerManager.numb_cols+jj+start_fr
                if ind<self.motherTrackerManager.numbFrames:
                    rowsB.append(self.all_images[:,:,ind])
                else:
                    rowsB.append(np.zeros((self.height,self.width),dtype=np.uint8))
            self.currentPageImgColor[idy+self.height:idy+2*self.height,:,:] = utils.label2rgb(np.hstack(tuple(rowsB)))*256

        if self.divisionTimes:
            for ii in np.unravel_index(self.divisionTimes,(self.motherTrackerManager.numb_rows,self.motherTrackerManager.numb_cols)):
                pass
                #self.currentPageImgColor[yy*self.height:(yy+1)*self.height,xx*self.width:(xx+1)*self.width,:]\
                #    = self.currentPageImgColor[yy*self.height:(yy+1)*self.height,xx*self.width:(xx+1)*self.width,:] * red_multiplier
        #print(self.currentPageImg.shape)

    def linkMultiFrames(self,start, N):  
        if start+N>=self.numb_frames:
            N = self.numb_frames - start - 1
        for fr in range(start,start+N):
            if fr==0:
                image1 = np.copy(self.all_images_orig[:,:,0])
                regions1 = regionprops(image1)
                if not regions1:# the first frame is empty
                    image1 = self.all_images_orig[:,:,1]
                    regions1 = regionprops(image1)
                motherLabel = image1[105,30]
                image1[image1==motherLabel] = self.motherLabelNumb
                self.all_images[:,:,fr] = image1
            elif fr==start:
                image1 = np.copy(self.all_images[:,:,start])
                if (image1==self.motherLabelNumb).any():
                    motherUndertrack = True
                else:
                    motherUndertrack = False

            regions1 = regionprops(image1)
            image2 = self.all_images_orig[fr+1]
            regions2 = regionprops(image2)

            image2,_, _,_,motherFind = utils.link_twoframes(image1,image2,regions1,regions2,self.motherLabelNumb)
            if (not motherFind) and motherUndertrack:
                image2[:,:] = self.all_images_orig[fr]
                regions2 = regionprops(image2)
                image2,_, _,_,motherFind = utils.link_twoframes(image1,image2,regions1,regions2,self.motherLabelNumb)
            motherUndertrack = motherFind

            self.all_images[:,:,fr+1]=image2[:,:]
            image1[:,:] = image2[:,:] # don't do image1=image2!

class DISPLAYCELLTRACKING(Frame):
    def __init__(self, master, motherTracker,**options):
        super().__init__(master,**options)
        self.master = master
        self.motherTracker = motherTracker
        self.config(width=self.motherTracker.width*self.motherTracker.motherTrackerManager.numb_cols)
        self.grid(rowspan=8)
        self.page = motherTracker.page
        self.motherTracker.constructMegaImage(self.page)
        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.motherTracker.currentPageImgColor))
        self.megaImageLabel = Label(self,image = self.img)
        self.megaImageLabel.grid(row=0, column=0, columnspan=10,sticky=tkW, pady=5, padx=5)
        self.megaImageLabel.bind("<Button-1>",self.getClickedFrame)
        self.megaImageLabel.bind("<Button-3>",self.assignLastFrame)

        self.pageDisplay = Label(self, text='Page 1/'+str(self.motherTracker.numb_pages))
        self.pageDisplay.grid(row=1,column=2,pady=5)

        self.pageUpButton = Button(self, text='Next Page')
        self.pageUpButton.bind("<Button-1>", self.pageUp)
        self.pageUpButton.grid(row=1,column=1,pady=5)
        self.pageDownButton = Button(self, text='Previous Page')
        self.pageDownButton.bind("<Button-1>", self.pageDown)
        self.pageDownButton.grid(row=1,column=0,pady=5)

    def setMotherTracker(self, motherTracker):
        self.motherTracker = motherTracker
        self.page = motherTracker.page

    def getClickedFrame(self,event):
        clickedFr = self.page*self.master.totalCounts + \
            event.x//self.motherTracker.width + \
                event.y//self.motherTracker.height*self.master.numb_cols
        self.motherTracker.divisionTimes.append(clickedFr)
        self.repaint()

    def assignLastFrame(self,event):
        clickedFr = self.page*self.master.totalCounts + \
            event.x//self.motherTracker.width + \
                event.y//self.motherTracker.height*self.master.numb_cols
        if clickedFr<self.motherTracker.numb_frames:
            clickedFr = clickedFr + 1
        self.motherTracker.motherLastFrame = clickedFr # it starts with 1, because the max is numb_frames
        #print(self.motherTracker.motherLastFrame)

    def repaint(self):
        self.motherTracker.constructMegaImage(self.page)
        img = ImageTk.PhotoImage(image=Image.fromarray(self.motherTracker.currentPageImgColor))
        self.megaImageLabel.configure(image=img)
        self.megaImageLabel.image = img
        self.pageDisplay.config(text='Page '+str(self.page+1)+'/'+str(self.motherTracker.numb_pages))

    def pageUp(self,event):
        if self.page < self.motherTracker.numb_pages-1:
            self.page = self.page+1
            self.motherTracker.page = self.motherTracker.page+1
            self.repaint()
            if self.page == self.motherTracker.numb_pages-1:
                self.motherTracker.completed = True

    def pageDown(self,event):
        if self.page>0:
            self.page = self.page-1
            self.motherTracker.page = self.motherTracker.page-1
            self.repaint()

class CHANNELLISTBOX(Listbox):
    def __init__(self,master,motherTrackerManager,**options):
        super().__init__(master,**options)
        self.motherTrackerManager = motherTrackerManager 
        self.updateFiles()
    def updateFiles(self):
        self.delete(0,'end')
        for ii, files in enumerate(self.motherTrackerManager.fileNames):
            self.insert(ii, files)
        self.select_set(0)

class popUpwindowSaving(Toplevel):
    def __init__(self, master, motherTrackerManager):
        super().__init__()
        self.master = master
        self.motherTrackerManager = motherTrackerManager
        self.checkbox_vars = []
        self.checkboxes = []
        numb_channels = self.motherTrackerManager.numb_channels
        for ii in range(numb_channels):
            status = int(self.motherTrackerManager.all_motherTrackers[ii].completed)
            self.checkbox_vars.append(IntVar(value=status))
            checkbox_text = 'Channel ' + str(ii+1)
            self.checkboxes.append(Checkbutton(self,text=checkbox_text,variable=self.checkbox_vars[ii],))
            self.checkboxes[ii].grid(row=ii,sticky=tkW)

        self.messageLabel = Label(self, text='Saving Results')
        self.messageLabel.grid(row = numb_channels)
        self.confirmButton = Button(self, text='Confirm')
        self.confirmButton.grid(row = numb_channels+1, column = 0)
        self.confirmButton.bind('<Button-1>', self.confirm)
        self.cancelButton = Button(self, text='Cancel')
        self.cancelButton.grid(row = numb_channels+1, column = 1)
        self.cancelButton.bind('<Button-1>', self.cancel)
    def confirm(self,event):
        for ch in range(self.motherTrackerManager.numb_channels):
            self.motherTrackerManager.all_motherTrackers[ch].saveOrNot = self.checkbox_vars[ch].get()
        self.motherTrackerManager.saveResults()
        self.destroy()
    def cancel(self,event):
        self.destroy()

class MAINWINDOW(Tk):
    def __init__(self, motherTrackerManager):
        super().__init__()
        self.motherTrackerManager = motherTrackerManager
        self.motherTracker = self.motherTrackerManager.all_motherTrackers[0]
        self.imageFrame = DISPLAYCELLTRACKING(self, self.motherTracker,relief='ridge',borderwidth=2)
        self.imageFrame.grid(row=0, column=0, columnspan=10)
        self.channelList = CHANNELLISTBOX(self,self.motherTrackerManager, relief='groove',borderwidth=2, selectmode='single')
        self.channelList.grid(row=1,column=10,padx=5)
        self.channelList.bind('<<ListboxSelect>>',self.CurSelect)

        self.saveButton = Button(self, text = 'Save Result', command = self.startSavingWindow)
        self.saveButton.grid(row=2,column=10,padx=5)
        self.selectFolderButton = Button(self, text = 'Select Folder', command = self.changeFolder)
        self.selectFolderButton.grid(row=0,column=10,padx=5)

        self.bind("<space>", self.imageFrame.pageUp)
        self.bind("<Right>", self.imageFrame.pageUp)
        self.bind("<Down>", self.imageFrame.pageUp)
        self.bind("<Next>", self.imageFrame.pageUp)
        self.bind("<Left>", self.imageFrame.pageDown)
        self.bind("<Up>", self.imageFrame.pageDown)
        self.bind("<Prior>", self.imageFrame.pageDown)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    def on_closing(self):
        if messagebox.askokcancel("Confirm Closing", "Make sure you have saved data before proceeding"):
            self.destroy()
    def changeFolder(self):
        if messagebox.askokcancel("Confirm To Change Folder", "Make sure you have saved data before proceeding"):
            self.getFolderPath()
    def getFolderPath(self):
        folder_selected = filedialog.askdirectory(initialdir = self.motherTrackerManager.folderName)
        if not folder_selected:
            pass
        else:
            self.motherTrackerManager.updateFolder(folder_selected)
            self.setMotherTracker(self.motherTrackerManager.all_motherTrackers[0])
            self.imageFrame.repaint()
    def setMotherTracker(self, motherTracker):
        self.motherTracker = self.motherTracker
        self.imageFrame.setMotherTracker(motherTracker)
    def startSavingWindow(self):
        self.savingWindow = popUpwindowSaving(self,self.motherTrackerManager)
    def CurSelect(self,event):
        if self.channelList.curselection():
            numb = self.channelList.curselection()[0]
        else:
            numb = 0
            self.channelList.select_set(0)
        self.setMotherTracker(self.motherTrackerManager.all_motherTrackers[numb])
        self.imageFrame.repaint()
#%%
scope = microscope.JulieMicroscope()
tm = COUNTCELLDIVISION(scope,'/home/yanfei/Julie_Aging/20191007/',positions=[2])

# %%


# %%
