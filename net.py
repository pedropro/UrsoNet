"""
This is the main implementation of UrsonNet.

Disclaimer:
Part of this code was adapted from
https://github.com/matterport/Mask_RCNN
Copyright (c)  2017 Matterport, INC.
Licenced under the MIT Licence

TODO:
- layer_regex replace 'fpn' and 'pose_'

"""

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import skimage.transform
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import imgaug as ia
from imgaug import augmenters as iaa

import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion

assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when inferencing
        """
        return super(self.__class__, self).call(inputs, training=training)

############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layres
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layres
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                                           '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layres
    """
    assert architecture in ["resnet50", "resnet101"]

    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # tmp
    #C1 = x = KL.Conv2D(64, (3, 3), strides=(2, 2), name='conv2')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]

############################################################
#  Shallow Resnet
############################################################

# Code adopted from:
# https://github.com/qubvel/classification_models/blob/master/classification_models/resnet/builder.py

def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name

def residual_basic_block(input_tensor, filters, stage, block, strides=(1, 1), cut='pre', use_bias=False, train_bn=True):

    # get names of layers
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

    # defining shortcut connection
    if cut == 'pre':
        shortcut = input_tensor
    elif cut == 'post':
        shortcut = KL.Conv2D(filters, (1, 1), name=sc_name, strides=strides, use_bias=use_bias)(input_tensor)
    else:
        raise ValueError('Cut type not in ["pre", "post"]')

    # Two 3x3 convolution layers
    x = KL.ZeroPadding2D(padding=(1, 1))(input_tensor)
    x = KL.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name + '2')(x, training=train_bn)
    x = KL.Activation('relu', name=relu_name + '1')(x)
    x = KL.ZeroPadding2D(padding=(1, 1))(x)
    x = KL.Conv2D(filters, (3, 3), name=conv_name + '2', use_bias=use_bias)(x)

    # add residual connection
    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name=relu_name + '2')(x)
    return x

def resnet_shallow_graph(input_image, architecture, train_bn=True):
    '''

    N.b: Currently convolutions do not use the bias term (unlike the 'deeper' resnet_graph)
     to keep compatibility with pre-trained weights
    '''

    assert architecture in ["resnet18", "resnet34"]

    nr_init_filters = 64

    # Resnet bottom
    x = KL.ZeroPadding2D(padding=(3, 3))(input_image)
    x = KL.Conv2D(nr_init_filters, (7, 7), strides=(2, 2), name='conv0', use_bias=False)(x)
    x = BatchNorm(name='bn_conv0')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # TODO: Allow more architectures
    if architecture == 'resnet18':
        repetitions = [2, 2, 2, 2]
    else:
        # This is fo 34 layers
        repetitions = (3, 4, 6, 3)

    for stage, rep in enumerate(repetitions):
        for block in range(rep):

            nr_filters = nr_init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = residual_basic_block(x, nr_filters, stage, block, strides=(1, 1), cut='post', train_bn=train_bn)

            elif block == 0:
                x = residual_basic_block(x, nr_filters, stage, block, strides=(2, 2),cut='post',train_bn=train_bn)

            else:
                x = residual_basic_block(x, nr_filters, stage, block, strides=(1, 1), cut='pre', train_bn=train_bn)

    return x

############################################################
#  Network Heads
############################################################

def build_loc_graph(feature_map, config, nr_features):
    """Builds the computation graph for location estimation on top of the Feature Network.
    Options: (1) XYZ regression (default), (2) 3D Keypoint regression and (3) classification (experimental)
    Returns: Location [batch, N]
    """

    nr_fc_layers = config.NR_DENSE_LAYERS
    assert nr_fc_layers in range(3)

    # TODO: Move this outside the function (redundancy)
    x = KL.Reshape((nr_features,))(feature_map)

    for i in range(nr_fc_layers):
        intermediate_fc_layer_name = 'loc_dense_' + str(i)
        x = KL.Dense(config.BRANCH_SIZE, name =intermediate_fc_layer_name)(x)

        if config.TRAIN_BN:
            bn_name = 'loc_bn_' + str(i)
            x = BatchNorm(name =bn_name)(x)
        x = KL.Activation('relu')(x)

    if config.REGRESS_KEYPOINTS:
        k1 = KL.Dense(3, activation='linear', name="k1_final")(x)
        k2 = KL.Dense(3, activation='linear', name="k2_final")(x)
        k3 = KL.Dense(3, activation='linear', name="k3_final")(x)
        loc = [k1,k2,k3]
    else:
        if config.REGRESS_LOC:
            loc = KL.Dense(3, activation='linear', name="loc_final")(x)
        else:
            loc = KL.Dense(config.LOC_BINS_PER_DIM**3, activation='relu', name="loc_final")(x)

    return loc

def build_ori_graph(feature_map, config, nr_features):
    """Builds the computation graph for orientation estimation on top of the Feature Network.
    Options: (1) 4D -vector regression (e.g. quaternion), (2) 3D-vector regression (e.g. angle-axis) and (3) classification
    Returns: Orientation [batch, N]
    """

    nr_fc_layers = config.NR_DENSE_LAYERS
    assert nr_fc_layers in range(3)

    # TODO: Move this outside the function (redundancy)
    x = KL.Reshape((nr_features,))(feature_map)

    for i in range(nr_fc_layers):
        intermediate_fc_layer_name = 'ori_dense_' + str(i)
        x = KL.Dense(config.BRANCH_SIZE, name =intermediate_fc_layer_name)(x)

        if config.TRAIN_BN:
            bn_name = 'ori_bn_' + str(i)
            x = BatchNorm(name =bn_name)(x)
        x = KL.Activation('relu')(x)

    if config.REGRESS_ORI:
        if config.ORIENTATION_PARAM == 'quaternion':
            q = KL.Dense(4, activation='linear', name="ori_q")(x)
            q = KL.Lambda(lambda x: K.l2_normalize(q, axis=-1))(q)
        else:
            q = KL.Dense(3, activation='linear', name="ori_final")(x)
    else:
        q = KL.Dense(config.ORI_BINS_PER_DIM**3, activation='relu', name="ori_final")(x)

    return q

############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id):
    """Load an image + object pose and apply augmentation pipeline (if necessary)

    Returns:
    image: [height, width, n]
    shape: the original shape of the image before resizing and cropping.
    loc: [x,y,z]
    ori: orientation representation
    """
    # Load and resize image
    image = dataset.load_image(image_id)

    if config.REGRESS_LOC:
        loc = dataset.load_location(image_id)
    else:
        loc = dataset.load_location_encoded(image_id)

    if config.REGRESS_KEYPOINTS:
        keypoints = dataset.load_keypoints(image_id)
        k1 = keypoints[0]
        k2 = keypoints[1]

    if config.REGRESS_KEYPOINTS or config.REGRESS_ORI:
        if config.ORIENTATION_PARAM == 'quaternion':
            ori = dataset.load_quaternion(image_id)
        elif config.ORIENTATION_PARAM == 'euler_angles':
            ori = dataset.load_euler_angles(image_id)
        elif config.ORIENTATION_PARAM == 'angle_axis':
            ori = dataset.load_angle_axis(image_id)
    else:
        ori = dataset.load_orientation_encoded(image_id)

    if config.SIM2REAL_AUG:
        image_gray = 0.2126*image[:,:,0]+0.7152*image[:,:,1]+0.0722*image[:,:,2]
        image[:, :, 0] = image_gray
        image[:, :, 1] = image_gray
        image[:, :, 2] = image_gray
        if np.random.rand(1) > 0.5:
            # Image Augmentation Pipeline
            aug_pipeline = iaa.Sequential([
                iaa.AdditiveGaussianNoise(scale=0.01 * 255),
                iaa.GaussianBlur(sigma=(0.0,1.5)),
                iaa.Add((-20, 20)),
                iaa.Multiply((0.5,2.0)),
                iaa.CoarseDropout([0.0, 0.03], size_percent=(0.02,0.1))
            ], random_order=True)

            det = aug_pipeline.to_deterministic()
            image = det.augment_image(image)


    if config.ROT_AUG or config.ROT_IMAGE_AUG:
        assert config.REGRESS_LOC
        assert config.ORIENTATION_PARAM == 'quaternion'

        # TODO: The 2 rotation augmentation operations are so far applied with mutual exclusion. Arbitrary may lead to more variation.

        dice = np.random.rand(1)

        # Camera orientation perturbation half the time
        if config.ROT_AUG and dice > 0.5:
            if config.REGRESS_KEYPOINTS or config.REGRESS_ORI:
                image, loc, ori = utils.rotate_cam(image, loc, ori, dataset.camera.K, 20)
                k1, k2 = utils.encode_as_keypoints(ori, loc)
            else:
                ori = dataset.load_quaternion(image_id)
                image, loc, ori = utils.rotate_cam(image, loc, ori, dataset.camera.K, 20)

                # Update encoded orientation
                ori = utils.encode_ori_fast(ori, config.BETA, dataset.ori_histogram_map, dataset.ori_output_mask)

        elif config.ROT_IMAGE_AUG and dice <= 0.5:
            if config.REGRESS_KEYPOINTS or config.REGRESS_ORI:
                image, loc, ori = utils.rotate_image(image, loc, ori, dataset.camera.K)
                k1, k2 = utils.encode_as_keypoints(ori, loc)
            else:
                ori = dataset.load_quaternion(image_id)
                image, loc, ori = utils.rotate_image(image, loc, ori, dataset.camera.K)

                # Update encoded orientation
                ori = utils.encode_ori_fast(ori, config.BETA, dataset.ori_histogram_map, dataset.ori_output_mask)

    original_shape = image.shape

    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale)

    if config.REGRESS_KEYPOINTS:
        return image, image_meta, loc, k1.T, k2.T
    else:
        return image, image_meta, loc, ori

def data_generator(dataset, config, shuffle=True, batch_size=1):
    """A generator that returns images and corresponding groundtruth.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    batch_size: How many images to return in each call

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - gt_locs: [batch, N]
    - gt_oris: [batch, N]
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    tensor_dtype = np.float32
    # For modern GPUs
    if config.F16:
        tensor_dtype = np.float16

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT for image.
            image_id = image_ids[image_index]
            if config.REGRESS_KEYPOINTS:
                image, image_meta, gt_loc, gt_k1, gt_k2 = load_image_gt(dataset, config, image_id)
            else:
                image, image_meta, gt_loc, gt_ori = load_image_gt(dataset, config, image_id)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=tensor_dtype)

                if config.REGRESS_LOC:
                    batch_gt_locs = np.zeros((batch_size, 3), dtype=tensor_dtype)
                else:
                    batch_gt_locs = np.zeros((batch_size, config.LOC_BINS_PER_DIM ** 3), dtype=tensor_dtype)

                if config.REGRESS_KEYPOINTS:
                    batch_gt_k1 = np.zeros((batch_size, 3), dtype=tensor_dtype)
                    batch_gt_k2 = np.zeros((batch_size, 3), dtype=tensor_dtype)
                else:
                    if config.REGRESS_ORI:
                        if config.ORIENTATION_PARAM == 'quaternion':
                            batch_gt_oris = np.zeros((batch_size, 4), dtype=tensor_dtype)
                        else:
                            batch_gt_oris = np.zeros((batch_size, 3), dtype=tensor_dtype)
                    else:
                        batch_gt_oris = np.zeros((batch_size, config.ORI_BINS_PER_DIM ** 3), dtype=tensor_dtype)

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_images[b] = mold_image(image.astype(tensor_dtype), config)
            batch_gt_locs[b] = gt_loc

            if config.REGRESS_KEYPOINTS:
                batch_gt_k1[b] = gt_k1
                batch_gt_k2[b] = gt_k2
            else:
                batch_gt_oris[b] = gt_ori

            b += 1

            # Batch full?
            if b >= batch_size:
                if config.REGRESS_KEYPOINTS:
                    inputs = [batch_images, batch_image_meta, batch_gt_locs, batch_gt_k1, batch_gt_k2]
                else:
                    inputs = [batch_images, batch_image_meta, batch_gt_locs, batch_gt_oris]


                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


############################################################
#  NNetwork Class and Graph Initialization
############################################################

class UrsoNet():

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build UrsoNet architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Change Keras backend to use f16 precision
        if config.F16:
            K.set_floatx('float16')
            # default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
            K.set_epsilon(1e-4)

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(shape=[None, None, config.NR_IMAGE_CHANNELS], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")

        tensor_dtype = tf.float32
        if config.F16:
            tensor_dtype = tf.float16

        if mode == "training":

            if config.REGRESS_LOC:
                input_gt_loc = KL.Input(shape=[3], name="input_gt_loc", dtype=tensor_dtype)
            else:
                input_gt_loc = KL.Input(shape=[config.LOC_BINS_PER_DIM**3], name="input_gt_loc", dtype=tensor_dtype)

            if config.REGRESS_KEYPOINTS:
                input_gt_k2 = KL.Input(shape=[3], name="input_gt_k2", dtype=tensor_dtype)
                input_gt_k3 = KL.Input(shape=[3], name="input_gt_k3", dtype=tensor_dtype)
            else:
                if config.REGRESS_ORI:
                    if config.ORIENTATION_PARAM == 'quaternion':
                        input_gt_ori = KL.Input(shape=[4], name="input_gt_ori", dtype=tensor_dtype)
                    else:
                        input_gt_ori = KL.Input(shape=[3], name="input_gt_ori", dtype=tensor_dtype)
                else:
                    input_gt_ori = KL.Input(shape=[config.ORI_BINS_PER_DIM ** 3], name="input_gt_ori", dtype=tensor_dtype)

        # Backbone architecture
        if config.BACKBONE in ['resnet50', 'resnet101']:
            _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN)
        else:
            C5 = resnet_shallow_graph(input_image, config.BACKBONE, train_bn=config.TRAIN_BN)

        # Original Resnet uses a 7x7 average pooling:
        # C6 = KL.GlobalAveragePooling2D()(C5)
        # but because we care about resolution, instead we perform here a convolution

        C6 = KL.Conv2D(config.BOTTLENECK_WIDTH, (3, 3), padding='SAME', strides=(2, 2), name='bottleneck_layer')(C5)
        nr_features = int(config.BOTTLENECK_WIDTH * config.IMAGE_SHAPE[0] * config.IMAGE_SHAPE[1] / (64 ** 2))

        loc_pred = build_loc_graph(C6, config, nr_features)
        ori_pred = build_ori_graph(C6, config, nr_features)

        if mode == "training":

            # Experimental feature
            if config.LEARNABLE_LOSS_WEIGHTS:
                self.ori_weight = K.variable(-2.3, name= 'ori_weight')
                self.loc_weight = K.variable(0.0, name= 'loc_weight')
            else:
                # Default
                self.ori_weight = K.variable(0.0, name= 'ori_weight')
                self.loc_weight = K.variable(0.0)

            if config.REGRESS_KEYPOINTS:
                loc_loss = KL.Lambda(lambda x: self.mse_loss_graph(*x), name="loc_loss")([input_gt_loc, loc_pred[0]])
                k2_loss = KL.Lambda(lambda x: self.mse_loss_graph(*x), name="k2_loss")([input_gt_k2, loc_pred[1]])
                k3_loss = KL.Lambda(lambda x: self.mse_loss_graph(*x), name="k3_loss")([input_gt_k3, loc_pred[2]])
            else:
                if config.REGRESS_LOC:
                    loc_loss = KL.Lambda(lambda x: self.rel_loss_graph(*x), name="loc_loss")([input_gt_loc, loc_pred])
                else:
                    loc_loss = KL.Lambda(lambda x: self.softmax_loss_graph(*x), name="loc_loss")([input_gt_loc, loc_pred])

                if config.REGRESS_ORI:
                    ori_loss = KL.Lambda(lambda x: self.one_minus_dot_prod_graph(*x), name="ori_loss")([input_gt_ori, ori_pred])
                else:
                    ori_loss = KL.Lambda(lambda x: self.softmax_loss_graph(*x), name="ori_loss")([input_gt_ori, ori_pred])

            # Model
            if config.REGRESS_KEYPOINTS:
                inputs = [input_image, input_image_meta, input_gt_loc, input_gt_k2, input_gt_k3]
            else:
                inputs = [input_image, input_image_meta, input_gt_loc, input_gt_ori]

            if config.REGRESS_KEYPOINTS:
                outputs = [loc_pred[0], loc_pred[1], loc_pred[2], loc_loss, k2_loss, k3_loss]
            else:
                outputs = [loc_pred, ori_pred, loc_loss, ori_loss]

            model = KM.Model(inputs, outputs, name='urso_net')

            # Workaround to make weights trainable
            if config.LEARNABLE_LOSS_WEIGHTS:
                model.layers[-1].trainable_weights.extend([self.ori_weight, self.loc_weight])
        else:
            if config.REGRESS_KEYPOINTS:
                model = KM.Model(input_image, [loc_pred[0], loc_pred[1], loc_pred[2]], name='urso_net')
            else:
                model = KM.Model(input_image, [loc_pred, ori_pred], name='urso_net')


        # Add multi-GPU support.
        # if config.GPU_COUNT > 1:
        #     from mrcnn.parallel_model import ParallelModel
        #     model = ParallelModel(model, config.GPU_COUNT)

        return model

    ############################################################
    #  Loss Functions
    ############################################################

    def softmax_loss_graph(self, y_gt, y_pred):
        """Loss for classification prediction.
        """
        # Experimental: Adaptive weighting based on Laplace likelihood (Kendall & Cipolla)
        # loss = tf.losses.softmax_cross_entropy(y_gt, y_pred)/tf.exp(self.ori_weight) + self.ori_weight
        loss = tf.losses.softmax_cross_entropy(y_gt, y_pred)
        return loss

    def arcos_graph(self, y_true, y_pred):
        """Implements rotation error
        y_true and y_pred are typicallly: [N, 4], but could be any shape.
        """
        loss = tf.acos(K.abs(K.sum(y_true * y_pred, axis=-1, keepdims=True)))
        # Experimental: Adaptive weighting based on Laplace likelihood (Kendall & Cipolla)
        # loss = loss/tf.exp(self.ori_weight) + self.ori_weight
        loss_mean = K.mean(loss)

        return loss_mean

    def one_minus_dot_prod_graph(self, y_true, y_pred):
        """Implements 1-dot-product.
        y_true and y_pred are typicallly: [N, 4], but could be any shape.
        """
        loss = 1 - K.abs(K.sum(y_true * y_pred, axis=-1, keepdims=True))
        # Experimental: Adaptive weighting based on Laplace likelihood (Kendall & Cipolla)
        # loss = loss / tf.exp(self.ori_weight) + self.ori_weight
        loss_mean = K.mean(loss)

        return loss_mean

    def mse_loss_graph(self, y_gt, y_pred):
        """Loss for regression prediction.
        e.g.
        pose_gt: [batch, (x,y,z)]
        pose_pred: [batch, (x,y,z)]
        """
        loss = K.square(y_gt - y_pred)

        # Experimental: Adaptive weighting based on Laplace likelihood (Kendall & Cipolla)
        # loss_mse = K.square(y_gt - y_pred)
        # loss = loss_mse/tf.exp(self.loc_weight) + self.loc_weight
        loss_mean = K.mean(loss)

        return loss_mean

    def rel_loss_graph(self, y_gt, y_pred):
        """Loss for regression prediction.
        e.g.
        pose_gt: [batch, (x,y,z)]
        pose_pred: [batch, (x,y,z)]
        """

        loss = tf.norm((y_gt - y_pred) / tf.norm(y_gt))

        # Experimental: Adaptive weighting based on Laplace likelihood (Kendall & Cipolla)
        # loss = loss/tf.exp(self.loc_weight) + self.loc_weight
        loss_mean = K.mean(loss)
        return loss_mean

    ############################################################
    #  Weights Loading Functions
    ############################################################

    def get_last_checkpoint(self,model_name):
        """Finds the last checkpoint file of a selected trained model in the
                model directory.
                Returns:
                    log_dir: The directory where events and weights are saved
                    checkpoint_path: the path to the last checkpoint file
                """
        dir_names = next(os.walk(self.model_dir))[1]

        assert model_name in dir_names

        model_path = os.path.join(self.model_dir, model_name)
        checkpoints = next(os.walk(model_path))[2]
        checkpoints = filter(lambda f: f.startswith("weights"), checkpoints)
        checkpoints = sorted(checkpoints)

        if not checkpoints:
            return model_path, None
        checkpoint = os.path.join(model_path, checkpoints[-1])

        return model_path, checkpoint


    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("weights"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, weights_in_path, weights_out_path, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to exclude
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(weights_in_path, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(weights_out_path)

    def get_imagenet_weights(self, architecture):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file

        if architecture in ['resnet50', 'resnet101']:
            TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/' \
                                     'releases/download/v0.2/' \
                                     'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        elif architecture == 'resnet18':

            TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/qubvel/classification_models/'\
                                     'releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5'
            weights_path = get_file('resnet18_imagenet_1000_no_top.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='318e3ac0cd98d51e917526c9f62f0b50')
        elif architecture == 'resnet34':

            TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/qubvel/classification_models/'\
                                     'releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5'
            weights_path = get_file('resnet34_imagenet_1000_no_top.h5',
                                   TF_WEIGHTS_PATH_NO_TOP,
                                   cache_subdir='models',
                                   md5_hash='8caaa0ad39d927cb8ba5385bf945d582')
        return weights_path



    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        if model_path:
            # Directory for training logs
            self.log_dir = os.path.dirname(model_path)
            self.epoch = int(model_path[-6:-3])
        else:
            self.epoch = 0
            now = datetime.datetime.now()
            self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
                self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "weights_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    ############################################################
    #  Weights Loading Functions
    ############################################################

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """

        # Optimizer object
        if self.config.OPTIMIZER == 'SGD':
            optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                             clipnorm=self.config.GRADIENT_CLIP_NORM)
        else:
            optimizer = keras.optimizers.Adam(learning_rate, amsgrad=True, clipnorm=self.config.GRADIENT_CLIP_NORM)

        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        if self.config.REGRESS_KEYPOINTS:
            loss_names = ["loc_loss", "k2_loss", "k3_loss"]
        else:
            loss_names = ["loc_loss", "ori_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        tensor_dtype = tf.float32
        if self.config.F16:
            tensor_dtype = tf.float16

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tensor_dtype)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue

            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        # All options except 'all' are only currently valid for resnet50 and renet101 models
        layer_regex = {
            # all layers but the backbone
            "heads": r"(ori\_.*)|(loc\_.*)|(fpn\_.*)|(bottleneck_layer)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(loc\_.*)|(ori\_.*)|(fpn\_.*)|(bottleneck_layer)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(loc\_.*)|(ori\_.*)|(fpn\_.*)|(bottleneck_layer)",
            "5+": r"(res5.*)|(bn5.*)|(loc\_.*)|(ori\_.*)|(fpn\_.*)|(bottleneck_layer)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Callbacks
        class BatchLogger(tf.keras.callbacks.Callback):
            def __init__(self):
                self.ori_loss_acc = []
                self.loc_loss_acc = []

            def on_batch_end(self, batch, logs={}):
                self.ori_loss_acc.append(logs.get('ori_loss'))
                self.loc_loss_acc.append(logs.get('loc_loss'))

        history_full = BatchLogger()

        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
            history_full
        ]

        if self.config.CLR:
            import clr_callback

            clr_triangular = clr_callback.CyclicLR(self.config.BASE_LEARNING_RATE, self.config.MAX_LEARNING_RATE,
                                                   self.config.CLR_STEP_SIZE, mode='triangular')
            callbacks.append(clr_triangular)


        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # TODO: print('Total FLOPs',get_flops(self))

        # print('Orientation var:', K.eval(K.exp(self.ori_weight)))
        # print('Location var:', K.eval(K.exp(self.loc_weight)))

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        hist = self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )

        self.epoch = max(self.epoch, epochs)

        return history_full

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale)
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."
        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
        # Run object detection
        if self.config.REGRESS_KEYPOINTS:
            loc_pred, k1_pred, k2_pred = self.keras_model.predict(molded_images, verbose=0)
            # Process detections
            results = []
            for i, image in enumerate(images):
                results.append({
                    "loc": loc_pred[i],
                    "k1": k1_pred[i],
                    "k2": k2_pred[i],
                })
        else:
            loc_pred, ori_pred = self.keras_model.predict(molded_images, verbose=0)
            # Process detections
            results = []
            for i, image in enumerate(images):
                results.append({
                    "loc": loc_pred[i],
                    "ori": ori_pred[i],
                })
        return results

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +  # size=3
        list(window) +  # size=4 (y1, x1, y2, x2) in image coords
        [scale] # size=1
    )
    return meta

def mold_image(image, config):
    """Subtract the mean pixel and converts it to float.
    """

    tensor_dtype = np.float32
    if config.F16:
        tensor_dtype = np.float16

    if image.shape[-1]==3:
        return image.astype(tensor_dtype) - config.MEAN_PIXEL
    else:
        return image.astype(tensor_dtype) - np.mean(config.MEAN_PIXEL)


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original.
    TODO: This does not accept grayscale"""

    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)

############################################################
#  Profiling Functions
############################################################

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.

