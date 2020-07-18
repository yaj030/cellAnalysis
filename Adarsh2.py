from os.path import join
from mrcnn import model as modellib
from samples.balloon import balloon
import numpy as np
from os.path import join
from segmentation import SegmentationMethods
import microscope
from tifffile import imshow
from skimage import exposure ,img_as_ubyte

#from os import environ
#environ["OMP_NUM_THREADS"] = "4"
import tensorflow as tf
#config = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
session = tf.Session()
#
#from keras import backend as K
#K.set_session(session)


class Adarsh2(SegmentationMethods):
    # compare to Adarsh, this methods deal with contrast-equalized images
    # take a look of line 85, and this option is not default, has to manually set 
    # analysis.segmentation_options = {'equalize':True}
    def __init__(self, imAnalysis):
        print('This method is only for aging experiment')
        self.imAnalysis = imAnalysis
        self._device = "/cpu:0"
        self.getPath()
    def methodPreparation(self):
        self.importFiles()
        self.createModel()
    def importFiles(self):
        # all files need to be put in the same folder with segmentation.py
        if isinstance(self.imAnalysis.scope, microscope.JulieMicroscope):
            self.weightPath = join(self.currentPath, 'JulieScopeNew.h5')
        elif isinstance(self.imAnalysis.scope, microscope.YangMicroscope):
            self.weightPath = join(self.currentPath, 'YangMicroscopeNew.h5')
        else:
            print('no weight path selected')

    def createModel(self):
        self.config = balloon.BalloonConfig()
        class InferenceConfig(self.config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        self.config = InferenceConfig()
        self.config.display()
        # Inspect the model in training or inference modes
        # values: 'inference' or 'training'

        with tf.device(self.device):
            self.model = modellib.MaskRCNN(mode="inference", model_dir=self.imAnalysis.experimentPath,config=self.config)
            self.model.load_weights(self.weightPath,by_name=True)

    @property
    def device(self):
        return self._device
    @device.setter
    def device(self, inputDevice):
        if inputDevice == 'cpu':
            self._device = "/cpu:0"
        elif inputDevice == 'gpu':
            self._device = "/gpu:0"


    def segment(self,image,position,frame):
        # it needs the position and frame number to get frame shift information and channel positions
        #image[0,:,:] = exposure.equalize_hist(np.uint16(image[0,:,:]))

        shifts = self.imAnalysis.registration[position][frame-1]
        ChLocations_y = self.imAnalysis.experimentObj.ChLocations_y[position]
        ChLocations_x = self.imAnalysis.experimentObj.ChLocations_x[position]
        separatedImage = self.imAnalysis.scope.separateChannels(image,shifts, ChLocations_x, ChLocations_y)
        results = []
        masks = []
        for ii in range(0,self.imAnalysis.scope.numChannels):
        #for ii in range(0,2):
            channelImage = separatedImage[ii]
            phaseImage = channelImage[self.imAnalysis.colors.index('phase'),:,:]
            if self.imAnalysis.segmentation_options['equalize']:
                phaseImage = exposure.equalize_adapthist(phaseImage)
                phaseImage = img_as_ubyte(phaseImage)
            self.imageunder = phaseImage # this for testing purpose, 
            # dupliate to make the image has three channels
            inputImage = np.stack([phaseImage,phaseImage,phaseImage],axis=2)
            result, mask = self.segmentOneChannel(inputImage)
            results.append(result)
            masks.append(mask)
        masks = dict(zip(range(1,self.imAnalysis.scope.numChannels+1),masks))
        return results, masks

    def segmentOneChannel(self,image):
        # Adarsh resize the image, I don't see it is necessary
        #image_resized, _, scale, padding, _ = utils.resize_image(image, \
        #    min_dim=self.config.IMAGE_MIN_DIM,max_dim=self.config.IMAGE_MAX_DIM, \
        #        mode=self.config.IMAGE_RESIZE_MODE)
        #segmentResults = self.model.detect([image_resized], verbose=0)
        segmentResults = self.model.detect([image], verbose=0)
        shape = segmentResults[0]['masks'].shape
        mask_label = np.zeros([shape[0], shape[1]], np.uint8)
        print(shape)
        for ii in range(0,segmentResults[0]['masks'].shape[2]):
            mask = segmentResults[0]['masks'][:,:,ii]
            mask_label[mask] = ii+1

        return segmentResults, mask_label
