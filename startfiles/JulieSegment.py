
#%% user input
filePath = '/home/nas2/yang/Julie/20200609/data'
colors = ['phase','GFP','iRFP']
positions2seg = range(1,17)
#positions2seg = [1,2]

#%%
import imageAnalysis
import microscope
from experimentclass import agingExperiment
from matplotlib import pyplot as plt
#%%
namaPrefix = ''
experiment = agingExperiment()
experiment.setSegmentMethod('Adarsh2')
scopeUsed = microscope.JulieMicroscope()
# the fourth argument could be 'ND', 'tif_noZstack'
analysis = imageAnalysis.IMAGE_ANALYSIS(filePath,namaPrefix,scopeUsed,'tif_noZstack',experiment)
analysis.segmentation_options = {'equalize':True}
#%%
analysis.experimentObj.startSegmentMethod()
analysis.fileClass.setColors(colors)
#analysis.fileClass.NDimages.metadata['fields_of_view'] = range(0,4) #this is a step has to perform for truncated ND


#%%
if not analysis.experimentObj.loadChLocations():
    analysis.experimentObj.getChLocAllPositions()
if not analysis.loadRegistration():
    analysis.imRegistrationAllPosition()

#%%
analysis.segmentPositionTimeZ(positions2seg,frames=None,Zs=None)

# %%
