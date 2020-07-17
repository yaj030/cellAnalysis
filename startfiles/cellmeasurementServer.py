#%%
import imageAnalysis
import microscope
from experimentclass import agingExperiment
from matplotlib import pyplot as plt
import motherCell_analysis
filePath = '/home/nas2/yang/Julie/20180704/data'
namaPrefix = ''
experiment = agingExperiment()
scopeUsed = microscope.JulieMicroscope()
# the fourth argument could be 'ND', 'tif_noZstack'
analysis = imageAnalysis.IMAGE_ANALYSIS(filePath,namaPrefix,scopeUsed,'tif_noZstack',experiment)
#analysis.experimentObj.setAttr(analysis)
analysis.fileClass.setColors(['phase','GFP','iRFP'])
#analysis.fileClass.setColors(['phase','GFP','iRFP'])

#analysis.fileClass.NDimages.metadata['fields_of_view'] = range(0,4)
if not analysis.experimentObj.loadChLocations():
    analysis.experimentObj.getChLocAllPositions()
if not analysis.loadRegistration():
    analysis.imRegistrationAllPosition()

# %%
measureGFP = motherCell_analysis.measureFluorInt(analysis,positions=range(17,25),color='GFP')
measureGFP.measure()
measureiRFP = motherCell_analysis.measureFluorInt(analysis,positions=range(17,25),color='iRFP')
measureiRFP.measure()

# %%

measureGFPfoci = motherCell_analysis.measureFoci(analysis,positions=range(17,25),color='GFP',cutoff=1.7)
measureGFPfoci.measure()

print('haha')
# %%
