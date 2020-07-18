
#%%
import imageAnalysis
import microscope
from experimentclass import agingExperiment
from matplotlib import pyplot as plt
#%%
filePath = '/home/yanfei/Julie_Aging/20191007'
namaPrefix = ''
experiment = agingExperiment()
scopeUsed = microscope.JulieMicroscope()
# the fourth argument could be 'ND', 'tif_noZstack'
analysis = imageAnalysis.IMAGE_ANALYSIS(filePath,namaPrefix,scopeUsed,'tif_noZstack',experiment)
analysis.segmentation_options = {'equalize':False}
#%%
#analysis.experimentObj.setAttr(analysis)
analysis.experimentObj.startSegmentMethod()
analysis.fileClass.setColors(['phase','GFP','iRFP'])


#%%
#analysis.fileClass.NDimages.metadata['fields_of_view'] = range(0,4)
if not analysis.experimentObj.loadChLocations():
    analysis.experimentObj.getChLocAllPositions()
#%%
if not analysis.loadRegistration():
    analysis.imRegistrationAllPosition()

# %%
fr = 1
data, masks = analysis.segmentOneSlice(2,fr,1)
#%%
analysis.segmentPositionTimeZ([2],range(1,342),Zs=None)

# %%
ch = 3
figure, (ax1, ax2) = plt.subplots(1,2)
image = analysis.fileClass.getOneSlice(2,fr,1)
separated = analysis.experimentObj.getChannelImage(image,ch, 2,fr,1)
from skimage import exposure
import numpy as np
p1, p99 = np.percentile(separated, (1, 99))
separated = exposure.rescale_intensity(separated, (p1,p99))
ax1.imshow(separated)
cellnumb = 0
#ax2.imshow(data[ch][0]['masks'][:,:,cellnumb]) 
ax2.imshow(masks[ch])
plt.show

# %%
from os import path, mkdir
folder = path.join(analysis.experimentPath, 'output')
if not path.exists(folder):
    mkdir(folder)
analysis.experimentObj.output4Training(folder, '20191007', [3], range(336,340), channels=range(1,5),equalize=True)

# %%
from os import path, mkdir
folder = path.join(analysis.experimentPath, 'output')
if not path.exists(folder):
    mkdir(folder)
analysis.experimentObj.output4Separated([2])

# %%
