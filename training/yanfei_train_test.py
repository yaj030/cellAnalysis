#%%
from mrcnn import utils
from mrcnn import visualize
import numpy as np
from samples.balloon import cellmodel
from skimage import exposure
from os.path import join
#%%
config = cellmodel.AgingConfig()
CELL_DIR = '/home/yanfei/YangTraining/newOnes'
print(CELL_DIR)

dataset = cellmodel.AgingDataset(labelMethod='SegmentPolygon')
dataset.load_cell(CELL_DIR, "train")
dataset.prepare()
#%% 
print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))
#%%
# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    image = exposure.equalize_adapthist(image)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
#%%
model_path = '/home/yanfei/agingAnalysis/09132020_35.h5'
#model_path = 'last'
cellmodel.train(CELL_DIR,model_path,'/home/yanfei/Julie_Aging/20191007/output/logs/',labelMethod='SegmentPolygon')



# %%
