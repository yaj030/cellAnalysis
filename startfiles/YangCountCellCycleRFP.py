#%%
import microscope
import cellDivisionCounting
scope = microscope.YangMicroscope()
folder = '/home/yanfei/Julie_Aging/20191007/'
positions = range(2,4)
#positions = [2,3]
#colors = ['phase','mcherry','GFP', 'iRFP']
colors = ['phase','GFP', 'iRFP']
tm = cellDivisionCounting.COUNTCELLDIVISION(scope,folder,positions,colors,color2use="iRFP",
    contrast = (1,99.9), rows = 4, cols=20)



# %%