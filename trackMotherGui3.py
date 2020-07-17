# colors are all inherited from first frame
#%%
import numpy as np
from tifffile import imread as tifread
from tifffile import imwrite as tifwrite
from skimage.measure import regionprops
#from skimage.morphology import dilation,disk
import utils
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
#%%
class TRACKMOTHER_MANAGER:
    def __init__(self, folderName=None,numb_channels=13):
        self.numb_channels = numb_channels
        self.updateFolder(folderName)
        self.mainWindow = MAINWINDOW(self)
        self.mainWindow.mainloop()
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
            self.all_motherTrackers[ch].saveResults()
            data2save.append(self.all_motherTrackers[ch].data)
        with open(self.json_path,'w') as datafile:
            json.dump({"data2save":data2save},datafile, indent=2)

class TRACKMOTHER:
    def __init__(self, motherTrackerManager, ch):
        self.motherTrackerManager = motherTrackerManager
        self.filePath = self.motherTrackerManager.filesPath[ch]
        self.ch = ch
        self.readImages()

        self.removeOverlap = False
        self.page_trackOrNot = np.zeros(self.numb_pages,dtype=bool)
        self.motherLabelNumb = 12
        self.page=0
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
        self.data["tracked"] = self.completed
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

        self.numb_pages = ceil((self.numb_frames-1)/20)
        self.all_images = np.zeros([self.height, self.width,self.numb_frames], dtype=np.uint8)
        #self.all_images_orig = pims.open(self.filePath)
        self.all_images_orig = pims.TiffStack_pil(self.filePath)

    def constructMegaImage(self,page):
        start_fr = page*20
        images = []
        for ii in range(21):
            images.append(self.all_images[:,:,ii+start_fr])
        row1 = np.hstack(tuple(images))
        array = (utils.label2rgb(row1)*255).astype(np.uint8)
        return array
    def constructMegaImageSmall(self,fr):
        image1 = self.all_images[:,:,fr]
        image2 = self.all_images[:,:,fr+1]
        return self.constructMegaImageTwoImages(image1, image2)
    def constructMegaImageTwoImages(self, image1,image2):
        row1 = np.hstack((image1,image2))
        array = (utils.label2rgb(row1)*255).astype(np.uint8)
        return array
    def linkPages(self, page):
        start_fr = page*20
        N = 21
        self.linkMultiFrames(start_fr,N)
        self.page_trackOrNot[page] = True
    def linkMultiFrames(self,start, N):  
        if start+N>=self.numb_frames:
            N = self.numb_frames - start - 1
        for fr in range(start,start+N):
            if fr==0:
                #image1 = np.copy(self.all_images_orig[:,:,0])
                image1 = np.copy(self.all_images_orig[0])
                regions1 = regionprops(image1)
                if not regions1:# the first frame is empty
                    #image1 = self.all_images_orig[:,:,1]
                    image1 = self.all_images_orig[1]
                    regions1 = regionprops(image1)
                if self.removeOverlap:
                    image1 = utils.removeOverlapSmallObj(image1)
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
            motherUndertrack = motherFind

            self.all_images[:,:,fr+1]=image2[:,:]
            image1[:,:] = image2[:,:] # don't do image1=image2!

class DISPLAYCELLTRACKING(Frame):
    def __init__(self, master, motherTracker,**options):
        super().__init__(master,**options)
        self.master = master
        self.motherTracker = motherTracker
        self.config(width=self.motherTracker.width*21)
        self.grid(rowspan=3)
        self.page = motherTracker.page
        self.motherTracker.linkPages(self.page)
        self.array = self.motherTracker.constructMegaImage(self.page)
        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.array))
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
        clickedFr = self.page*20 + event.x//self.motherTracker.width
        popUpWindow(clickedFr, self, self.motherTracker)
    def assignLastFrame(self,event):
        clickedFr = self.page*20 + event.x//self.motherTracker.width
        if clickedFr<self.motherTracker.numb_frames:
            clickedFr = clickedFr + 1
        image = self.motherTracker.all_images[:,:,clickedFr]
        allLabels = set(np.unique(image))
        availableLabels = set(range(13)) - allLabels
        image[image==self.motherTracker.motherLabelNumb] = availableLabels.pop()
        self.motherTracker.linkMultiFrames(clickedFr, 20)
        self.motherTracker.page_trackOrNot[self.page+1:self.motherTracker.numb_pages] = False
        self.motherTracker.completed = True
        self.motherTracker.tracking_status = 'AssignedLast'
        self.motherTracker.motherLastFrame = clickedFr # it starts with 1, because the max is numb_frames
        print('lastframe')
        print(self.motherTracker.motherLastFrame)
        self.repaint()

    def repaint(self):
        if not self.motherTracker.page_trackOrNot[self.page]:
            self.motherTracker.linkPages(self.page)
        array = self.motherTracker.constructMegaImage(self.page)
        img = ImageTk.PhotoImage(image=Image.fromarray(array))
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
                self.motherTracker.tracking_status = 'TrackedLast'

    def pageDown(self,event):
        if self.page>0:
            self.page = self.page-1
            self.motherTracker.page = self.motherTracker.page-1
            self.repaint()

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

class popUpWindow(Toplevel):
    def __init__(self, clickedFr, master, motherTracker):
        super().__init__()
        self.master = master
        self.motherTracker = motherTracker
        self.fr = clickedFr
        #self.geometry = "120x160" 
        array = self.motherTracker.constructMegaImageSmall(clickedFr)
        self.image1 = np.copy(self.motherTracker.all_images[:,:,clickedFr])
        self.image2 = np.copy(self.motherTracker.all_images[:,:,clickedFr+1])
        self.img2 = ImageTk.PhotoImage(image=Image.fromarray(array))
        self.megaImageCanvas = Canvas(self,width=self.motherTracker.width*2,height=self.motherTracker.height)
        self.megaImageCanvas.pack(expand=YES, fill=BOTH)
        self.canvasImage = self.megaImageCanvas.create_image(0,0,image=self.img2,anchor=NW)
        self.megaImageCanvas.bind ('<Button-1>', self.getInitPoint)
        self.megaImageCanvas.bind ('<B1-Motion>', self.setVel)
        self.megaImageCanvas.bind ('<ButtonRelease-1>', self.getEndPoint)

        self.megaImageCanvas.pack()
        self.protocol("WM_DELETE_WINDOW", self.close_window)
    def setMotherTracker(self, motherTracker):
        self.motherTracker = motherTracker
    def close_window(self):
        print("clicked fr " + str(self.fr))
        self.motherTracker.linkMultiFrames(self.fr+1, 20)
        self.master.repaint()
        self.destroy()
        
    def getInitPoint(self, event):
        self.rx = event.x
        self.ry = event.y
        objLabel = self.image1[self.ry,self.rx]
        if objLabel==self.motherTracker.motherLabelNumb:
            self.startWithMother = True
        else:
            self.startWithMother = False
    def getEndPoint(self, event):
        objLabel = self.image2[event.y,event.x-self.motherTracker.width]
        if not objLabel == self.motherTracker.motherLabelNumb:
            self.motherTracker.page_trackOrNot[self.master.page+1:self.motherTracker.numb_pages] = False
            self.image2 = self.replaceNewMother(self.image1,self.image2,objLabel)
            array = self.motherTracker.constructMegaImageTwoImages(self.image1, self.image2)
            self.img2 = ImageTk.PhotoImage(image=Image.fromarray(array))
            self.megaImageCanvas.itemconfig(self.canvasImage, image = self.img2)
            self.motherTracker.all_images[:,:,self.fr+1] = self.image2[:,:]
        
    def replaceNewMother(self, image1, image2, objLabel):
        img1 = np.copy(image1)
        img2 = np.copy(image2)
        mask1 = img1==self.motherTracker.motherLabelNumb
        img1[mask1] = 0
        regions1 = regionprops(img1)
        allLabels = set(np.unique(img2))
        availableLabels = set(range(13)) - allLabels - {self.motherTracker.motherLabelNumb}
        img2[img2==self.motherTracker.motherLabelNumb] = availableLabels.pop()
        mask2 = img2==objLabel
        img2[mask2] = 0
        regions2 = regionprops(img2)
        img2,_,_,_,_ = utils.link_twoframes(img1,img2,regions1,regions2,self.motherTracker.motherLabelNumb)
        img2[mask2] = self.motherTracker.motherLabelNumb
        return img2

    def setVel(self, event):
        if hasattr(self.megaImageCanvas, 'lastline'):
            self.megaImageCanvas.delete(self.megaImageCanvas.lastline)
        if self.startWithMother:
            xm, ym = event.x, event.y
            self.megaImageCanvas.lastline = self.megaImageCanvas.create_line\
                (self.rx,self.ry,xm,ym,width=3,fill='red')

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

tm = TRACKMOTHER_MANAGER('/home/yanfei/Julie_Aging/20191007/2/')




# %%
