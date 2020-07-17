#%%
import imageAnalysis
import microscope
from experimentclass import agingExperiment
from os import path, mkdir
from matplotlib import pyplot as plt
#%%
filePath = '/home/nas2/yang/Julie/20180704/data'
namaPrefix = ''
experiment = agingExperiment()
scopeUsed = microscope.JulieMicroscope()
# the fourth argument could be 'ND', 'tif_noZstack'
analysis = imageAnalysis.IMAGE_ANALYSIS(filePath,namaPrefix,scopeUsed,'tif_noZstack',experiment)
#%%
analysis.experimentObj.setAttr(analysis)
# analysis.experimentObj.startSegmentMethod()
# for tif files, have to assign colors
analysis.fileClass.setColors(['phase','GFP','iRFP'])
#analysis.fileClass.setColors(['phase','GFP','mCherry','iRFP'])

#%%
#analysis.fileClass.NDimages.metadata['fields_of_view'] = range(0,4)
if not analysis.experimentObj.loadChLocations():
    analysis.experimentObj.getChLocAllPositions()
#%%
if not analysis.loadRegistration():
    analysis.imRegistrationAllPosition()

# %%
folder = path.join(analysis.experimentPath, 'output')
if not path.exists(folder):
    mkdir(folder)
analysis.experimentObj.output4Training(folder, '20180704',range(17,27), range(1,300,50), channels=range(1,11))

# %%
