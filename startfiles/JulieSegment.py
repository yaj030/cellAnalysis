
#%%
import imageAnalysis
import microscope
from experimentclass import agingExperiment
from matplotlib import pyplot as plt
#%%
filePath = '/home/nas2/yang/Julie/20200110/data'
namaPrefix = ''
experiment = agingExperiment()
experiment.setSegmentMethod('Adarsh2')
scopeUsed = microscope.JulieMicroscope()
# the fourth argument could be 'ND', 'tif_noZstack'
analysis = imageAnalysis.IMAGE_ANALYSIS(filePath,namaPrefix,scopeUsed,'tif_noZstack',experiment)
analysis.segmentation_options = {'equalize':True}
#%%
analysis.experimentObj.startSegmentMethod()
analysis.fileClass.setColors(['phase','GFP','iRFP'])
#analysis.fileClass.setColors(['phase','GFP','mCherry','iRFP'])


#%%
#analysis.fileClass.NDimages.metadata['fields_of_view'] = range(0,4)
if not analysis.experimentObj.loadChLocations():
    analysis.experimentObj.getChLocAllPositions()
if not analysis.loadRegistration():
    analysis.imRegistrationAllPosition()

#%%
#analysis.segmentPositionTimeZ([2],range(1,342),Zs=None)
analysis.segmentPositionTimeZ(range(1,1),frames=None,Zs=None)

# %%