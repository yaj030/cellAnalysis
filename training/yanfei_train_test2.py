#%%
import os
os.chdir('/home/yanfei/adarsh/MRCNN_cell/Mask_RCNN-master')

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model as modellib
from mrcnn.model import log

import numpy as np

from samples.balloon import balloon


# In[9]:
config = balloon.BalloonConfig()
CELL_DIR = '/home/yanfei/Julie_Aging/20191007/output'

# In[10]:
dataset = balloon.BalloonDataset()
dataset.load_balloon(CELL_DIR, "train")
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# In[11]:
image_ids = np.random.choice(dataset.image_ids, 2)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

# In[ ]:
model_path = '/home/yanfei/Julie_Aging/20191007/output/20191007_test1.h5'
#model_path = 'last'
cellmodel.train(CELL_DIR,model_path,'/home/yanfei/Julie_Aging/20191007/output/logs/')
