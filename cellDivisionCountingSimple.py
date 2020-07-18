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
from tkinter import Tk, Text, BOTH, NW,YES, HORIZONTAL, Toplevel, Canvas, IntVar,BooleanVar,RIGHT, LEFT
from tkinter import W as tkW
from tkinter import E as tkE
from tkinter import N as tkN
from tkinter import S as tkS
from tkinter import X as tkX
from tkinter import Y as tkY
from tkinter import Frame, Button, Label, filedialog, Listbox, Checkbutton, messagebox, Scrollbar, Radiobutton
import pims
from os import path, rename, remove
import json
from datetime import datetime
import imageAnalysis
import microscope
from experimentclass import agingExperiment
import motherCell_analysis
import utils

#%
class COUNTCELLDIVISION:
    def __init__(self, scope,folderName=None,positions = [],colors=['phase','GFP','iRFP'], rows=5,cols=20):
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
        self.numb_rows = rows
        self.numb_cols = cols
        self.totalCounts = self.numb_cols*self.numb_rows

        self.numb_channels = self.analysis.scope.numChannels
        self.numbFrames = self.analysis.totalFrames
        self.folderName = folderName
        self.positions = positions
        self.currentPosition = 0
        self.currentPositionIndex = 0
        self.all_motherTrackers = []
        self.constructColorImages()
        self.updatePosition(positions[0])
        self.readResults()
        self.startWindow()
    def constructColorImages(self):
        self.img_red = np.zeros((self.scopeUsed.separateHeight,self.scopeUsed.separateWidth*2,3),dtype=np.uint8)
        self.img_green = np.copy(self.img_red)
        self.img_blue = np.copy(self.img_red) 
        self.img_red[:,:,0] = 255
        self.img_green[:,:,1] = 255
        self.img_blue[:,:,2] = 255

    def startWindow(self):
        self.mainWindow = MAINWINDOW(self)
        self.mainWindow.mainloop()

    def updatePosition(self,position):
        self.currentPosition = position
        self.currentPositionIndex = self.positions.index(position)
        self.posFolderName = path.join(self.folderName,'xy'+str(self.currentPosition))
        self.separatedImage = []
        self.fileNames = []
        self.getSeparatedImages()
        if self.all_motherTrackers: 
            for ch in range(self.numb_channels):
                print(ch)
                self.all_motherTrackers[ch].updatePosition()
                self.fileNames.append('trap'+str(ch+1))
        else:
            for ch in range(self.numb_channels):
                print(ch)
                self.all_motherTrackers.append(TRACKMOTHER(self, ch))
                self.fileNames.append('trap'+str(ch+1))

    def getSeparatedImages(self):
        self.separatedImage = self.analysis.experimentObj.output4Separated(\
            [self.currentPosition],color=1,outputTif=False)

    def saveResults(self):
        if path.exists(self.posFolderName+'_lifespan.txt'):
            remove(self.posFolderName+'_lifespan.txt')
        with open(self.posFolderName+'_lifespan.txt', 'a') as txt2write:
            for ch in range(self.numb_channels):
                self.all_motherTrackers[ch].saveTxt(txt2write)

    def readResults(self):
        if path.exists(self.posFolderName+'_lifespan.txt'):
            with open(self.posFolderName+'_lifespan.txt', 'r') as txt2read:
                contents = txt2read.readlines()
            for line in contents:
                word = [int(ii)-1 for ii in line.split()]
                trap = word[0]
                self.all_motherTrackers[trap].deathType = word[1]+1
                self.all_motherTrackers[trap].firstFrame = word[2]
                self.all_motherTrackers[trap].lastFrame = word[3]
                divisionTimes = word[4:]
                self.all_motherTrackers[trap].divisionTimes = divisionTimes
                start = 0
                ending = self.totalCounts
                self.all_motherTrackers[trap].divisionTimesCurrentPage = \
                    [i for i in divisionTimes if i>=start and i<ending]

class TRACKMOTHER:
    def __init__(self, motherTrackerManager, ch):
        self.motherTrackerManager = motherTrackerManager
        self.ch = ch
        self.updatePosition()

    def updatePosition(self):
        self.numb_frames = 0
        self.height = 0
        self.width = 0
        self.getImages()
        self.page_trackOrNot = np.zeros(self.numb_pages,dtype=bool)
        self.page=0
        self.saveOrNot = True
        self.motherLastFrame = self.numb_frames
        self.divisionTimes = set()
        self.divisionTimesCurrentPage = []
        self.firstFrame = 0
        self.lastFrame = self.motherTrackerManager.numbFrames-1
        self.deathType = 0

        self.megaImgHeight = self.height*self.motherTrackerManager.numb_rows
        self.megaImgWidth = self.width*self.motherTrackerManager.numb_cols

    def saveTxt(self,txt2write):
        if self.saveOrNot:
            txt2write.write(str(self.ch+1)+' ')
            txt2write.write(str(self.deathType)+' ')
            txt2write.write(str(self.firstFrame+1)+' ')
            txt2write.write(str(self.lastFrame+1)+' ')
            for dv in sorted(self.divisionTimes):
                txt2write.write(str(dv+1)+' ')
            txt2write.write("\n")
            
    def getImages(self):
        img = img_as_ubyte(equalize_hist(self.motherTrackerManager.separatedImage[self.ch]))
        self.numb_frames = img.shape[2]
        assert(self.numb_frames == self.motherTrackerManager.numbFrames, "number of frames don't match")
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.numb_pages = ceil((self.numb_frames)/self.motherTrackerManager.totalCounts)
        self.all_images_phase = np.copy(img)
        for fr in range(self.numb_frames):
            self.all_images_phase[:,:,fr] = img[:,:,fr]

    def constructMegaImage(self,page):
        start_fr = page*self.motherTrackerManager.numb_cols*self.motherTrackerManager.numb_rows
        self.currentPageImgColor = np.zeros((self.megaImgHeight, self.megaImgWidth, 3),dtype=np.uint8)
        for ii in range(self.motherTrackerManager.numb_rows):
            rowsA = []
            idy = ii*self.height
            for jj in range(self.motherTrackerManager.numb_cols):
                ind = ii*self.motherTrackerManager.numb_cols+jj+start_fr
                if ind<self.motherTrackerManager.numbFrames:
                    rowsA.append(self.all_images_phase[:,:,ind])
                else:
                    rowsA.append(np.zeros((self.height,self.width),dtype=np.uint8))
            self.currentPageImgColor[idy:idy+self.height,:,:] = gray2rgb(np.hstack(tuple(rowsA)))

        self.drawRedEdge(self.firstFrame,'green')
        if self.divisionTimes:
            for ii in self.divisionTimes:
                self.drawRedEdge(ii,'red')
        self.drawRedEdge(self.lastFrame,'blue')

    def drawRedEdge(self,fr,color):
        clicked = fr%self.motherTrackerManager.totalCounts
        page = fr//self.motherTrackerManager.totalCounts
        if page==self.page:
            ind = np.unravel_index(clicked,(self.motherTrackerManager.numb_rows,self.motherTrackerManager.numb_cols))
            yy = ind[0]
            xx = ind[1]
            img = self.currentPageImgColor[yy*self.height:(yy+1)*self.height,xx*self.width:(xx+1)*self.width,:]
            if color=='red':
                img2 = self.motherTrackerManager.img_red
            elif color=='green':
                img2 = self.motherTrackerManager.img_green
            elif color=='blue':
                img2 = self.motherTrackerManager.img_blue
            img2[5:-5,5:-5,:] = img[5:-5,5:-5,:]
            self.currentPageImgColor[yy*self.height:(yy+1)*self.height,xx*self.width:(xx+1)*self.width,:] = img2
        
class DISPLAYCELLTRACKING(Frame):
    def __init__(self, master, motherTracker,**options):
        super().__init__(master,**options)

        self.master = master
        self.motherTracker = motherTracker
        self.config(width=self.motherTracker.width*self.motherTracker.motherTrackerManager.numb_cols)
        self.config(bg='black')
        self.grid(rowspan=8)
        self.page = motherTracker.page
        self.motherTracker.constructMegaImage(self.page)
        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.motherTracker.currentPageImgColor))
        self.megaImageLabel = Label(self,image = self.img)
        self.megaImageLabel.grid(row=0, column=0, columnspan=10,sticky=tkW, pady=5, padx=5)
        self.megaImageLabel.bind("<Button-1>",self.getClickedFrame)
        self.megaImageLabel.bind("<Shift-Button-1>",self.assignFirstFrame)
        self.megaImageLabel.bind("<Button-3>",self.assignLastFrame)

        self.pageDisplay = Label(self, text='Page 1/'+str(self.motherTracker.numb_pages),bg='black',fg='yellow')
        self.pageDisplay.grid(row=1,column=2,pady=5)

        self.pageUpButton = Button(self, text='Next Page',bg='black',fg='yellow')
        self.pageUpButton.bind("<Button-1>", self.pageUp)
        self.pageUpButton.grid(row=1,column=1,pady=5)
        self.pageDownButton = Button(self, text='Previous Page',bg='black',fg='yellow')
        self.pageDownButton.bind("<Button-1>", self.pageDown)
        self.pageDownButton.grid(row=1,column=0,pady=5)

        self.chkValue = BooleanVar() 
        self.chkValue.set(True)
        self.saveOrNotCheckBox = Checkbutton(self, text='save this trap',variable=self.chkValue,command=self.changeSave) 
        self.saveOrNotCheckBox.grid(row=1,column=3)

        self.deathTypeValue = IntVar() 
        self.deathTypeValue.set(0)
        self.radioButtonDt1 = Radiobutton(self, text='Death Type 1',variable=self.deathTypeValue,value=1,command=self.setDeathType) 
        self.radioButtonDt1.grid(row=1,column=4)
        self.radioButtonDt2 = Radiobutton(self, text='Death Type 2',variable=self.deathTypeValue,value=2,command=self.setDeathType) 
        self.radioButtonDt2.grid(row=1,column=5)
        self.repaint()

    def changeSave(self):
        self.motherTracker.saveOrNot = self.chkValue.get()

    def setDeathType(self):
        self.motherTracker.deathType = self.deathTypeValue.get()

    def setMotherTracker(self, motherTracker):
        self.motherTracker = motherTracker
        self.page = motherTracker.page

    def getClickedFrame(self,event):
        clickedFr = self.page*self.master.motherTrackerManager.totalCounts + \
            event.x//self.motherTracker.width + \
                event.y//self.motherTracker.height*self.master.motherTrackerManager.numb_cols
        if clickedFr in self.motherTracker.divisionTimes:
            self.motherTracker.divisionTimes.remove(clickedFr)
        else:
            self.motherTracker.divisionTimes.add(clickedFr)
        self.repaint()

    def assignFirstFrame(self,event):
        clickedFr = self.page*self.master.motherTrackerManager.totalCounts + \
            event.x//self.motherTracker.width + \
                event.y//self.motherTracker.height*self.master.motherTrackerManager.numb_cols
        self.motherTracker.firstFrame = clickedFr # it starts with 1, because the max is numb_frames
        self.repaint()

    def assignLastFrame(self,event):
        clickedFr = self.page*self.master.motherTrackerManager.totalCounts + \
            event.x//self.motherTracker.width + \
                event.y//self.motherTracker.height*self.master.motherTrackerManager.numb_cols
        self.motherTracker.lastFrame = clickedFr # it starts with 1, because the max is numb_frames
        self.motherTracker.saveOrNot = True
        self.repaint()

    def repaint(self):
        self.motherTracker.constructMegaImage(self.page)
        img = ImageTk.PhotoImage(image=Image.fromarray(self.motherTracker.currentPageImgColor))
        self.megaImageLabel.configure(image=img)
        self.megaImageLabel.image = img
        self.deathTypeValue.set(self.motherTracker.deathType)
        self.pageDisplay.config(text='Page '+str(self.page+1)+'/'+str(self.motherTracker.numb_pages))

    def pageUp(self,event):
        if self.page < self.motherTracker.numb_pages-1:
            self.page = self.page+1
            self.motherTracker.page = self.motherTracker.page+1
            if self.motherTracker.divisionTimes:
                start = self.page*self.motherTracker.motherTrackerManager.totalCounts
                ending = (self.page+1)*self.motherTracker.motherTrackerManager.totalCounts
                self.motherTracker.divisionTimesCurrentPage = [i for i in self.motherTracker.divisionTimes if i>=start and i<ending]
            self.repaint()

    def pageDown(self,event):
        if self.page>0:
            self.page = self.page-1
            self.motherTracker.page = self.motherTracker.page-1
            if self.motherTracker.divisionTimes:
                start = self.page*self.motherTracker.motherTrackerManager.totalCounts
                ending = (self.page+1)*self.motherTracker.motherTrackerManager.totalCounts
                self.motherTracker.divisionTimesCurrentPage = [i for i in self.motherTracker.divisionTimes if i>=start and i<ending]
            self.repaint()

class postionFrame(Frame):
    def __init__(self, master,motherTrackerManager,**options):
        super().__init__(master,**options)
        self.master = master
        self.motherTrackerManager = motherTrackerManager
        self.configure(bg='black')
        self.nextPositionBution = Button(self, text = 'Next Position', command = self.nextPos,bg='black',fg='yellow')
        self.previousPositionBution = Button(self, text = 'Previous Position', command = self.previousPos,bg='black',fg='yellow')
        self.posDisplay = Label(self, text='Current at 1/'+str(len(self.master.motherTrackerManager.positions)),bg='black',fg='yellow')
        #self.nextPositionBution.grid(row=0,column=0,padx=5)
        #self.previousPositionBution.grid(row=1,column=0,padx=5)
        #self.posDisplay.grid(row=2,column=0,pady=5)
        self.nextPositionBution.pack(fill=tkX,pady=5)
        self.previousPositionBution.pack(fill=tkX,pady=5)
        self.posDisplay.pack()

    def nextPos(self):
        posIndex = self.motherTrackerManager.currentPositionIndex
        if posIndex < len(self.motherTrackerManager.positions)-1:
            if messagebox.askokcancel('confirm','Have you chosen death type? Chosen saving or not'):
                pos = self.motherTrackerManager.positions[posIndex+1]
                self.motherTrackerManager.updatePosition(pos)
                self.master.setMotherTracker(self.motherTrackerManager.all_motherTrackers[0])
                self.posDisplay.config(text='Current at '+str(posIndex+2)+'/'+str(len(self.motherTrackerManager.positions)))
                self.master.imageFrame.repaint()

    def previousPos(self):
        posIndex = self.motherTrackerManager.currentPositionIndex
        if posIndex > 0 :
            if messagebox.askokcancel('confirm','Have you chosen death type? Chosen saving or not'):
                pos = self.motherTrackerManager.positions[posIndex-1]
                self.motherTrackerManager.updatePosition(pos)
                self.master.setMotherTracker(self.motherTrackerManager.all_motherTrackers[0])
                self.posDisplay.config(text='Current at '+str(posIndex)+'/'+str(len(self.motherTrackerManager.positions)))
                self.master.imageFrame.repaint()


class CHANNELLISTBOX(Listbox):
    def __init__(self,master,motherTrackerManager,**options):
        super().__init__(master,**options)
        self.configure(bg='black',fg='yellow')
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
            status = int(self.motherTrackerManager.all_motherTrackers[ii].saveOrNot)
            if not self.motherTrackerManager.all_motherTrackers[ii].divisionTimes:
                status = 0
            deathType = self.motherTrackerManager.all_motherTrackers[ii].deathType
            self.checkbox_vars.append(IntVar(value=status))
            checkbox_text = 'Channel ' + str(ii+1) + '-Death Type: ' + str(deathType)
            if deathType==0: 
                self.checkboxes.append(Checkbutton(self,text=checkbox_text,fg='red', variable=self.checkbox_vars[ii],))
            else:
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

        self.channelList = CHANNELLISTBOX(self,self.motherTrackerManager, relief='groove',\
            selectbackground='blue',borderwidth=2, selectmode='single',exportselection=False)
        self.channelList.grid(row=1,column=10,padx=5)
        self.channelList.bind('<<ListboxSelect>>',self.CurSelect)

        self.poslayout = postionFrame(self,self.motherTrackerManager)
        self.poslayout.grid(row=3,column=10,pady=5)

        self.saveButton = Button(self, text = 'Save Result', command = self.startSavingWindow,bg='black',fg='yellow')
        self.saveButton.grid(row=5,column=10,padx=5)

        self.bind("<space>", self.imageFrame.pageUp)
        self.bind("<Right>", self.imageFrame.pageUp)
        self.bind("<Down>", self.imageFrame.pageUp)
        self.bind("<Next>", self.imageFrame.pageUp)
        self.bind("<Left>", self.imageFrame.pageDown)
        self.bind("<Up>", self.imageFrame.pageDown)
        self.bind("<Prior>", self.imageFrame.pageDown)

        self.configure(bg='black')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        if messagebox.askokcancel("Confirm Closing", "Make sure you have saved data before proceeding"):
            self.destroy()
    def setMotherTracker(self, motherTracker):
        self.motherTracker = motherTracker
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
        if self.motherTracker.saveOrNot:
            self.imageFrame.saveOrNotCheckBox.select()
        else:
            self.imageFrame.saveOrNotCheckBox.deselect()
        if self.motherTracker.deathType==1:
            self.imageFrame.radioButtonDt1.select()
        else:
            self.imageFrame.radioButtonDt2.select()
        self.imageFrame.repaint()
#%%
#scope = microscope.JulieMicroscope()
#tm = COUNTCELLDIVISION(scope,'/home/yanfei/Julie_Aging/20191007/',positions=[2,3])

# %%
#tm.startWindow()



# %%
