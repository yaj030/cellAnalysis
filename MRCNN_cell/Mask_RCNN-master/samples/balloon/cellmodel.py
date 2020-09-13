# This is a neater version of Adarsh's ballon.py
import os
import sys
import json
import skimage
import numpy as np
# Adarsh's imports
from tqdm import tqdm
from imgaug import augmenters as iaa
# Root directory of the project

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils
ROOT_DIR = os.path.abspath("../../")

# this following line is add path, so the above two import could be possible
# if the mrcnn folder is not in the python enviromment folder
# don't need it 
#sys.path.append(ROOT_DIR)  # To find local version of the library

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
BALLOON_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon.h5")
############################################################
#  Configurations
############################################################

class AgingConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Aging"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + cell

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200

    # Skip detections with < 70% confidence
    # the ballon example used 90%
    DETECTION_MIN_CONFIDENCE = 0.7

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_ANCHOR_RATIOS = [0.3, 0.6, 1.2]
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 256

############################################################
#  Dataset
############################################################

class AgingDataset(utils.Dataset):
    # Adarsh's way to setup labeling is different to yanfei
    # Yanfei, use segmentation, 
    # Adarsh uses classification then add object
    def __init__(self, labelMethod = 'Adarsh'): 
        # method can be SegmentPolygon or Adarsh
        super().__init__()
        self.labelMethod = labelMethod
        self.dataset_dir = ''

    def load_cell(self, dataset_dir, subset):
        """Load a subset of the cell dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val or test, test is what Adarsh added
        """
        # Add classes. We have only one class to add.
        self.add_class("cell", 1, "cell")

        # Train or validation dataset?
        assert subset in ["train", "val","test"]
        self.dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # Get the x, y coordinaets of points of the polygons that make up
        # the outline of each object instance. 
        # The annotation from labelbox is different to VIA used in the balloon example
        data = json.load(open(os.path.join(self.dataset_dir, "region_data.json")))
        if self.labelMethod=='SegmentPolygon':
            self.load_cell_SegmentPolygon(data)
        elif self.labelMethod=='Adarsh':
            self.load_cell_Adarsh(data)

    def load_cell_SegmentPolygon(self,data):
        # This deal with the json output by Yanfei's labeling
        for img in data:
            polygons = []
            for region in img['Label']['objects']:
                xs = []
                ys = []
                for xys in region['polygon']:
                    xs.append(xys['x'])
                    ys.append(xys['y'])
                polygons.append({'all_points_x':xs, 'all_points_y': ys})
            # remember labeling was done with jpg files, training is tif
            self.addImage(img, polygons)
        
    def load_cell_Adarsh(self,data):
        for img in data:
            image_path = os.path.join(self.dataset_dir, img['External ID'][0:-4] +'.tif')
            if not os.path.exists(image_path):
                continue
            polygons = []    
            if 'Masks' in img:
                try:
                    img['Label']['cell']
                    flag = 'cell'
                except:
                    flag = 'Cell'
                for region in img['Label'][flag]:
                    all_points_x = []
                    all_points_y = []
                    #print('region:', region)
                    for xy in region['geometry']:
                        all_points_x.append(xy['x'])
                        all_points_y.append(xy['y'])
                    temp_dict = {'all_points_x': all_points_x, 'all_points_y': all_points_y, 'name': 'polygon'}
                    polygons.append(temp_dict)
            # remember labeling was done with jpg files, training is tif
            self.addImage(img, polygons)

    def addImage(self, img, polygons):
        image_path = os.path.join(self.dataset_dir, img['External ID'][0:-4] +'.tif')
        try:
            image = skimage.io.imread(image_path)
            image = skimage.img_as_float(image)
        except:
            print(image_path)
            raise('something is wrong with reading image')
        height, width = image.shape[:2]

        self.add_image("cell", image_id=img['External ID'][0:-4] +'.tif',  # use file name as a unique image id
            path=image_path, width=width, height=height, polygons=polygons)
                   
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cell":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

def train(data_directory,model_path,log_path):
    """Train the model."""
    # Training dataset.
    dataset_train = AgingDataset()
    dataset_train.load_cell(data_directory, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = AgingDataset()
    dataset_val.load_cell(data_directory, "val")
    dataset_val.prepare()

    config = AgingConfig()
    
    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=log_path)

    #select weights file to load
    if model_path == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif model_path == "balloon":
        weights_path = BALLOON_WEIGHTS_PATH
    elif model_path == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif model_path == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = model_path

    # Load weights
    print("Loading weights ", weights_path)
    if model_path == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Change brightness of images (50-150% of original value).
    # This is what Adarsh added, atually it does not change brightness
    # it flip some of the images
    augmentation = iaa.SomeOf((0, 1), [iaa.Fliplr(0.5), ])
    
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    # It looks Adarsh kept the layers='heads' option
    print("Training network heads")
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=50,augmentation=augmentation,
        layers='heads')

