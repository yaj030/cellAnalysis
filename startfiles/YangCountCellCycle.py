
# %%
import microscope
import cellDivisionCountingSimple
scope = microscope.YangMicroscope()
folder = '/home/yanfei/YangAgingData'
#folder = '/home/nas2/yang/Julie/20200110/data'
#positions = range(17,33)
positions = [1,2]
#colors = ['phase','mcherry','GFP', 'iRFP']
colors = ['phase','GFP', 'iRFP']
tm = cellDivisionCountingSimple.COUNTCELLDIVISION(scope,folder,positions,colors, rows = 4, cols=20)