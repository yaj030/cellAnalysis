#%%
folder = '/home/nas2/yang/Julie/20200717/data'
#positions = range(1,17)
positions = [16]

from trackMotherAuto import TRACKMOTHER_MANAGER_AUTO
tm = TRACKMOTHER_MANAGER_AUTO(folder,13,positions)
tm.trackAllPositions()

# %%
