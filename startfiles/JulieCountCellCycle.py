#%%
# user input: folder, color and positions
folder = '/home/yanfei/Julie_Aging/20191007/'
#colors = ['phase','mcherry','GFP', 'iRFP']
colors = ['phase','GFP', 'iRFP']
positions = [2,3]
#positions = range(17,33)

import microscope
import cellDivisionCountingSimple
scope = microscope.JulieMicroscope()
tm = cellDivisionCountingSimple.COUNTCELLDIVISION(scope,folder,positions,colors, rows = 4, cols=20)



# %%
