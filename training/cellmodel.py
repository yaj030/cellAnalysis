# This is a neater version of Adarsh's ballon.py
import os
import sys
import json
# Adarsh's imports
from tqdm import tqdm
from imgaug import augmenters as iaa
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

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
    STEPS_PER_EPOCH = 100

    # Skip detections with < 70% confidence
    # the ballon example used 90%
    DETECTION_MIN_CONFIDENCE = 0.7


############################################################
#  Dataset
############################################################

class AgingDataset(utils.Dataset):
    def load_cell(self, dataset_dir, subset):
        """Load a subset of the cell dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val or test, test is what Adarsh added
        """
        # Add classes. We have only one class to add.
        self.add_class("cell", 1, "cell")

        # Train or validation dataset?
        assert subset in ["train", "val","test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # Get the x, y coordinaets of points of the polygons that make up
        # the outline of each object instance. 
        # The annotation from labelbox is different to VIA used in the balloon example
        if subset == "train" or subset =="val":
            data = json.load(open(os.path.join(dataset_dir, "region_data.json")))
            for img in data:
                #print(img['External ID'][0:-4] +'.tif')
                #remember the label files was jpg
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
                image_path = os.path.join(dataset_dir, img['External ID'][0:-4] +'.tif')
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                self.add_image("cell",
                    image_id=img['External ID'][0:-4] +'.tif',  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons)
                   
        elif subset == "test":
            # Adarsh add this part, it looks like just load the image from the folder, no label, so polygons=None
            test_ids = next(os.walk(dataset_dir))[2]
            for id_ in tqdm((test_ids), total=len(test_ids)):
                image_path = os.path.join(dataset_dir,id_)
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                self.add_image("cell",
                    image_id=id_,  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    polygons=None)

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
    weights_path = model_path

    # Load weights
    model.load_weights(weights_path, by_name=True)

    # Change brightness of images (50-150% of original value).
    # This is what Adarsh added
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


