#%%
import imageAnalysis
import microscope
from experimentclass import agingExperiment
from matplotlib import pyplot as plt
#%%
filePath = '/home/yanfei/Julie_Aging'
namaPrefix = 'sirhapDEL4'
experiment = agingExperiment()
scopeUsed = microscope.JulieMicroscope()
analysis = imageAnalysis.IMAGE_ANALYSIS(filePath,namaPrefix,scopeUsed,'ND',experiment)
#%%
#analysis.experimentObj.setAttr(analysis)
analysis.experimentObj.startSegmentMethod()

#%%
analysis.experimentObj.getChLocAllPositions()
#%%
analysis.fileClass.saveTiffFiles([4])
#%%
#analysis.fileClass.NDimages.metadata['fields_of_view'] = range(0,4)
if not analysis.loadRegistration():
    analysis.imRegistrationAllPosition()

# %%
fr = 300
data, masks = analysis.segmentOneSlice(1,fr,1)
#%%
analysis.segmentPositionTimeZ([1,2],[7,8,9],[1])

# %%

# %%
# %%
ch = 8
figure, (ax1, ax2) = plt.subplots(1,2)
image = analysis.fileClass.getOneSlice(1,fr,1)
separated = analysis.experimentObj.getChannelImage(image,ch, 1,fr,1)
ax1.imshow(separated)
ax2.imshow(masks[ch])
plt.show



# %%
