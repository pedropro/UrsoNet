"""
Main network configuration and hyperparameters

Copyright (c) Pedro F. Proenza
"""

import math
import numpy as np


# Base Configuration Class

class Config:
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # Number of filters used in the last Convolution layer
    BOTTLENECK_WIDTH = 128

    # Size of branch input
    BRANCH_SIZE = 1024

    # Input image resizing
    # Generally, use the "square" resizing mode for training and inferencing
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "pad64"
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 512
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0

    NR_IMAGE_CHANNELS = 3 # DONT TOUCH THIS IF WE ARE USING PRE-TRAINED WEIGHTS

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Option to use cyclical learning rate
    CLR = False
    MAX_LEARNING_RATE = 0.0005#0.0015 # 0.0015
    BASE_LEARNING_RATE = 0.0001#0.00005 # 0.00005
    CLR_STEP_SIZE = 4000

    # Perform regression or classification
    REGRESS_ORI = True
    REGRESS_LOC = True
    REGRESS_KEYPOINTS = False

    # Performs camera rotation perturbations
    ROT_AUG = True

    # Performs blurring, brightness change, RGB2GRAY
    SIM2REAL_AUG = False

    # In-plane rotation augmentation
    ROT_IMAGE_AUG = False

    # Option only used for regression
    ORIENTATION_PARAM = 'quaternion'

    DECOUPLE_ORIENTATION = False

    # Only used for classification
    # Number of bins used to discretize a dimension
    LOC_BINS_PER_DIM = 16
    ORI_BINS_PER_DIM = 32
    # Soft assignement parameter to scale gaussian stddev
    BETA = 6.0

    OPTIMIZER = 'SGD'

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Sets Keras datatype to 16 bit instead of 32 bit (only works with RTX GPU family)
    F16 = False

    # If true, learns loss weights by minimizing the negative log likelihood
    LEARNABLE_LOSS_WEIGHTS = False

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "loc_loss": 1.,
        "ori_loss": 1.,
        "k2_loss": 1.,
        "k3_loss": 1.
    }

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when inferencing
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    def update(self):
        '''Set rest of parameters based on configuration values'''
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, self.NR_IMAGE_CHANNELS])
        elif self.IMAGE_RESIZE_MODE == "pad64":
            # Assumes wide images
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MAX_DIM, self.NR_IMAGE_CHANNELS])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, self.NR_IMAGE_CHANNELS])
        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + self.NR_IMAGE_CHANNELS + 3 + 4 + 1


    def __init__(self):
        self.update()

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def write_to_file(self, filepath):
        import json
        import os

        config_dict = {}

        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and \
                    not isinstance(getattr(self, a),np.ndarray):
                config_dict[a] = getattr(self, a)

        directory = os.path.dirname(filepath)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        with open(filepath, 'w+') as f:
            f.write(json.dumps(config_dict))

