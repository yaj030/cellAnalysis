#%%
import microscope
import cellDivisionCountingSimple
scope = microscope.JulieMicroscope()
folder = '/home/nas2/yang/Julie/20200110/data'
positions = range(17,33)
#positions = [2,3]
#colors = ['phase','mcherry','GFP', 'iRFP']
colors = ['phase','GFP', 'iRFP']
tm = cellDivisionCountingSimple.COUNTCELLDIVISION(scope,folder,positions,colors, rows = 4, cols=20)



# %%