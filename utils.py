"""
Utility functions

"""

import sys
import os
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
import se3lib
import random
import itertools
import cv2
from scipy import stats
import matplotlib.pyplot as plt

def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def rotate_cam(image, t, q, K, magnitude):
    """ Apply warping corresponding to a random camera rotation
    Arguments:
     - image: Input image
     - t, q: Object pose (location,orientation)
     - K: Camera calibration matrix
     - magnitude: 2 * maximum perturbation per angle in deg
    Return:
        - image_warped: Output image
        - t_new, q_new: Updated object pose
    """

    pyr_change = (np.random.rand(3)-0.5)*magnitude

    R_change = se3lib.euler2SO3_left(pyr_change[0], pyr_change[1], pyr_change[2])

    # Construct warping (perspective) matrix
    M = K*R_change*np.linalg.inv(K)

    height, width = np.shape(image)[:2]
    image_warped = cv2.warpPerspective(image, M, (width, height), cv2.WARP_INVERSE_MAP)

    # Update pose
    t_new = np.array(np.matrix(t)*R_change.T)[0]
    q_change = se3lib.SO32quat(R_change)
    q_new = np.array(se3lib.quat_mult(q_change, q))[0]

    return image_warped, t_new, q_new

def rotate_image(image, t, q, K):
    """ Apply warping corresponding to a random in-plane rotation
     Arguments:
      - image: Input image
      - t, q: Object pose (location,orientation)
      - K: Camera calibration matrix
      - magnitude: 2 * maximum perturbation per roll
     Return:
         - image_warped: Output image
         - t_new, q_new: Updated object pose
    """

    change = (np.random.rand(1)-0.5)*170

    R_change = se3lib.euler2SO3_left(0, 0, change[0])

    # Construct warping (perspective) matrix
    M = K*R_change*np.linalg.inv(K)

    height, width = np.shape(image)[:2]
    image_warped = cv2.warpPerspective(image, M, (width, height), cv2.WARP_INVERSE_MAP)

    # Update pose
    t_new = np.array(np.matrix(t)*R_change.T)[0]
    q_change = se3lib.SO32quat(R_change)
    q_new = np.array(se3lib.quat_mult(q_change, q))[0]

    return image_warped, t_new, q_new

def polar_plot(q1,q2):
    """Plot two orientations as Euler angles on polar plots
         Arguments:
          - q1,q2: Input quaternions
    """
    fig = plt.figure(figsize=(1, 4))

    pyr_1 = se3lib.quat2euler(q1)
    pyr_2 = se3lib.quat2euler(q2)

    pyr_1 = np.array(pyr_1)*np.pi/180
    pyr_2 = np.array(pyr_2)*np.pi/180

    ax = plt.subplot(3, 1, 1, projection='polar')
    ax.plot([pyr_1[0],pyr_1[0]],[0,1],'r-')
    ax.plot([pyr_2[0],pyr_2[0]],[0,1],'b--')
    ax.set_rticks([])
    ax = plt.subplot(3, 1, 2, projection='polar')
    ax.plot([pyr_1[1],pyr_1[1]],[0,1],'r-')
    ax.plot([pyr_2[1],pyr_2[1]],[0,1],'b--')
    ax.set_rticks([])
    ax = plt.subplot(3, 1, 3, projection='polar')
    ax.plot([pyr_1[2],pyr_1[2]],[0,1],'r-')
    ax.plot([pyr_2[2],pyr_2[2]],[0,1],'b--')
    ax.set_rticks([])

    plt.show()


def visualize_weights(gt_pmf, est_pmf, nr_bins_per_dim):
    """ Shows the encoding weights of two 3D soft classification structures as 2D image stacks.
    This is used to show the network output activation for soft classification and the respective encoded orientation groundtruth
    """
    fig = plt.figure(figsize=(12, 2))

    ax = []
    max_activation = np.max(gt_pmf)
    for z in range(nr_bins_per_dim):
        slice = np.zeros(shape=(nr_bins_per_dim, nr_bins_per_dim), dtype=np.float32)
        for i in range(nr_bins_per_dim):
            for j in range(nr_bins_per_dim):
                slice[j, i] = gt_pmf[i * nr_bins_per_dim * nr_bins_per_dim + j * nr_bins_per_dim + z]

        ax.append(fig.add_subplot(2, nr_bins_per_dim, z + 1))
        if z == 0:
            plt.ylabel('GT')
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])
        plt.imshow(slice, vmin=0, vmax=max_activation)

    # plt.title('GT PMF')
    max_activation = np.max(est_pmf)
    for z in range(nr_bins_per_dim):
        slice = np.zeros(shape=(nr_bins_per_dim, nr_bins_per_dim), dtype=np.float32)
        for i in range(nr_bins_per_dim):
            for j in range(nr_bins_per_dim):
                slice[j, i] = est_pmf[i * nr_bins_per_dim * nr_bins_per_dim + j * nr_bins_per_dim + z]

        ax.append(fig.add_subplot(2, nr_bins_per_dim, nr_bins_per_dim + z + 1))
        if z == 0:
            plt.ylabel('Est.')
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])
        plt.imshow(slice, vmin=0, vmax=max_activation)


def visualize_axes(ax, q, C, K, scale):
    """ Shows an object pose by overlapping the object axes on a matplotlib input figure
    Arguments:
        - ax: Axis returned by a matplotlib subplot
        - q: Quaternion
      """

    # Create arrow points
    P = np.matrix([[1,0,0],
                   [0,-1,0],
                   [0,0,1]])

    # Rotate arrow points
    R = se3lib.quat2SO3(q)
    P_r = R*P

    # Translate to loc_gt
    P_t = np.asarray(P_r)+np.transpose([C])

    # Project to image
    p = P_t/P_t[-1,:]
    c = C/C[-1]
    p = K*p
    c = K*np.matrix(c).transpose()

    v = p-c
    v = scale*v/np.linalg.norm(v)

    ax.arrow(c[0, 0], c[1, 0], v[0, 0], v[1, 0], head_width=10, color='r')
    ax.arrow(c[0, 0], c[1, 0], v[0, 1], v[1, 1], head_width=10, color='g')
    ax.arrow(c[0, 0], c[1, 0], v[0, 2], v[1, 2], head_width=10, color='b')

def plot_axes(img, q, C, K, scale):
    """ (same as above) Shows an object pose by overlapping the object axes on an opencv image
    """

    # Create arrow points
    P = np.matrix([[1,0,0],
                   [0,-1,0],
                   [0,0,1]])*scale

    # Rotate arrow points
    R = se3lib.quat2SO3(q)
    P_r = R*P

    # Translate to loc_gt
    P_t = np.asarray(P_r)+np.transpose([C])

    # Project to image
    p = P_t/P_t[-1,:]
    c = C/C[-1]

    p = K*p
    c = K*np.matrix(c).transpose()

    v = p#-c
    #v = scale*v/np.linalg.norm(v)

    c = c.astype(np.int)
    v = p.astype(np.int)

    cv2.arrowedLine(img, (c[0, 0], c[1, 0]), (v[0, 0], v[1, 0]), (0, 0, 255), 2)
    cv2.arrowedLine(img, (c[0, 0], c[1, 0]), (v[0, 1], v[1, 1]), (0, 255, 0), 2)
    cv2.arrowedLine(img, (c[0, 0], c[1, 0]), (v[0, 2], v[1, 2]), (255, 0, 0), 2)


def encode_as_keypoints(oris, centroids, scale=1.0):
    '''Encode pose as 2 virtual 3D keypoints (aligned with axes) '''
    if np.ndim(oris) == 1:
        R = se3lib.quat2SO3(oris)
        c = np.asmatrix(centroids).T
        K1 = R * scale * np.matrix([0, 0, 1]).T + c
        K2 = R * scale * np.matrix([0, 1, 0]).T + c

    else:
        nr_examples = np.size(oris, 0)

        K1 = np.zeros((nr_examples, 3), dtype=np.float32)
        K2 = np.zeros((nr_examples, 3), dtype=np.float32)

        for i in range(nr_examples):
            R = se3lib.quat2SO3(oris[i, :])

            c = np.asmatrix(centroids[i, :]).T
            k1 = R * scale * np.matrix([0, 0, 1]).T + c
            k2 = R * scale * np.matrix([0, 1, 0]).T + c

            K1[i, :] = k1.T
            K2[i, :] = k2.T

    return K1, K2

def encode_ori(oris, nr_bins_per_dim, beta, min_lim, max_lim):
    '''Soft assignment of orientations to a quantized space
    Take input vectors as Gaussian random variables and quantize these using softmax
    Arguments:
        - oris: An array [n,4] of quaternions to be encoded
        - nr_bins_per_dim: Number of bins used for each axis/angle of the encoding structure. Assuming the same number of
        bins for all axes actually leads to different resolution per Euler angle. 
        - beta: hyperparameter used to scale the kernel width.
        - min_lim, max_lim: Limits of Euler angles as two 3D arrays.
    Returns:
        - ori_encoded: Encoded orientations (oris)
        - H_quat: Map of quaternions represented by the encoding bins
        - Redundant_flags: Binary mask for bins that represent the same orientation.
    '''

    nr_examples = np.size(oris, 0)
    d = 3
    nr_total_bins = nr_bins_per_dim**d

    # Use the variance of Gaussian approximation of a uniform distribution to scale distances
    # For details check the Chapter 2 of my PhD thesis
    delta = beta / nr_bins_per_dim
    var = delta ** 2 / 12
    print('Variance used for encoding: ', var)

    # Construct histogram structure
    bins_loc_per_dim = np.linspace(0.0, 1.0, nr_bins_per_dim)
    H_loc_list = list(itertools.product(bins_loc_per_dim, repeat=d))
    H_ori = np.asarray(H_loc_list * (max_lim - min_lim) + min_lim)
    H_quat = np.zeros(shape=(nr_total_bins, 4), dtype=np.float32)
    for i in range(nr_total_bins):
        H_quat[i, :] = se3lib.euler2quat(H_ori[i, 0], H_ori[i, 1], H_ori[i, 2]).T

    # Find redundant bins
    eps = np.cos(0.5*np.pi/180)

    print('Pruning redundant bins.')
    # A) Brute force way (super slow)
    # Boundary_flags = np.any(np.logical_or((H_ori == min_lim),(H_ori == max_lim)),1)
    #
    # for i in range(nr_total_bins):
    #     if Boundary_flags[i]:
    #         q1 = H_quat[i, :]
    #         for j in range(i+1,nr_total_bins):
    #             if Boundary_flags[j]:
    #                 q2 = H_quat[j, :]
    #                 if np.abs(np.sum(q1*q2, axis=-1)) > eps and not Redundant_flags[j]:
    #                     Redundant_flags[j] = True
    #
    # B) Efficient way
    # Mark redundant boundary bins
    Boundary_flags = np.logical_or(H_ori[:,0] == max_lim[0], H_ori[:,2] == max_lim[2])
    # Mark redundant bins due to the two singularities at y = -+ 90 deg
    Gymbal_flags = np.logical_and(np.abs(H_ori[:,1]) == max_lim[1], H_ori[:,0] != min_lim[0])
    Redundant_flags = np.logical_or(Boundary_flags, Gymbal_flags)

    print('Encoding.')
    ori_encoded = np.zeros(shape=(nr_examples, nr_bins_per_dim ** d), dtype=np.float32)

    # Sampling pdf for each
    for i in range(nr_examples):

        #1. Compute Kernel function outputs based on scaled angular errors [0,1]
        H_prbs = np.exp(-2 * (np.arccos(np.minimum(1.0,np.abs(np.sum(oris[i,:] * H_quat, axis=-1))))/np.pi) ** 2 / var)

        for j in range(nr_total_bins):
            if Redundant_flags[j]:
                H_prbs[j] = 0

        ori_encoded[i, :] = H_prbs / np.sum(H_prbs)

    return ori_encoded, H_quat, Redundant_flags

def encode_ori_fast(oris, beta, H_quat, Redundant_flags):
    '''Soft assignment of orientations to a quantized space
    Take input vectors as Gaussian random variables and quantize these using softmax
    and takes pre-constructed quantization structure
    Arguments:
        - oris: An array [n,4] of quaternions to be encoded
        - H_quat: Map of quaternions represented by the encoding bins
        - Redundant_flags: Binary mask for bins that represent the same orientation.
    '''

    nr_examples = 1
    d = 3
    nr_total_bins = len(H_quat)
    nr_bins_per_dim = round(nr_total_bins**(1./3))

    # Use the variance of Gaussian approximation of a uniform distribution to scale distances
    # For details check the Chapter 2 of my PhD thesis
    delta = beta / nr_bins_per_dim
    var = delta ** 2 / 12

    # 1. Compute Kernel function outputs based on scaled angular errors [0,1]
    H_prbs = np.exp(-2 * (np.arccos(np.minimum(1.0,np.abs(np.sum(oris * H_quat, axis=-1)))) / np.pi) ** 2 / var)

    for j in range(nr_total_bins):
        if Redundant_flags[j]:
            H_prbs[j] = 0

    return H_prbs / np.sum(H_prbs)


def encode_loc(locs, nr_bins_per_dim, beta, max_lim, min_lim):
    '''Soft assignment of location coordinates.
    Take input vectors as Gaussian random variables and quantize these using a probability
    mass function approximation'''

    nr_examples = np.size(locs, 0)
    d = 1
    if locs.ndim > 1:
        d = np.size(locs, 1)

    # Use variance of uniform distribution to scale distances
    # based on sigma_uniform =  delta^2/12 where delta (the pdf width) is 1/nr_steps_per_dim
    delta = beta*(max_lim[-1]-min_lim[-1])/nr_bins_per_dim
    # tmp
    delta = beta/nr_bins_per_dim
    cov = np.identity(3, dtype=float)*delta**2/12

    # Histogram structure
    bin_step_size = 1.0/(nr_bins_per_dim-1)
    bins_loc_per_dim = np.linspace(0.0,1.0,nr_bins_per_dim)
    H_loc_list = list(itertools.product(bins_loc_per_dim, repeat=d))
    H_loc_3D = np.asarray(H_loc_list*(max_lim-min_lim)+min_lim)
    H_loc_3D[:,0] = H_loc_3D[:,0] * H_loc_3D[:,2]
    H_loc_3D[:,1] = H_loc_3D[:,1] * H_loc_3D[:,2]

    loc_encoded = np.zeros(shape=(nr_examples, nr_bins_per_dim ** d), dtype=np.float32)

    # Sampling pdf for each
    for i in range(nr_examples):

        # Retrieve world XYZ coords for example_i
        Z = locs[i,2]
        X = locs[i,0]*Z
        Y = locs[i,1]*Z

        # Tmp
       # S_prbs = stats.multivariate_normal.pdf(np.array(S_loc), mean=[X, Y, Z], cov=cov)
        H_prbs = stats.multivariate_normal.pdf(H_loc_3D, mean=[X,Y,Z], cov=cov)

        #H_prbs = np.zeros(nr_bins_per_dim ** d)

        # Accumulate densities per bin
        # for j in range(nr_samples):
        #     H_prbs[S_bin_ids[j]] += S_prbs[j]

        loc_encoded[i,:] = H_prbs/np.sum(H_prbs)

    return loc_encoded, H_loc_3D

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """

    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        scale = min_dim / min(h, w)
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode != "crop":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = skimage.transform.resize(
            image, (round(h * scale), round(w * scale)),
            order=1, mode="constant", preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        if len(image.shape)>2:
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        else:
            padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0

        if len(image.shape) > 2:
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        else:
            padding = [(top_pad, bottom_pad), (left_pad, right_pad)]

        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop

############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")

def average_images(dataset_dir):
    ''' Get average pixel intensity from dataset'''
    import pandas as pd

    set_filename = dataset_dir + '/train_images.csv'
    rgb_list_df = pd.read_csv(set_filename, names=['filename'], header=-1)
    rgb_list = list(rgb_list_df['filename'])
    nr_images = len(rgb_list)

    mean = np.zeros(3)

    for file_name in rgb_list:
        rgb_path = os.path.join(dataset_dir, file_name)

        image = skimage.io.imread(rgb_path)

        mean += np.mean(np.mean(image,0),0)[0:3]

    mean /= nr_images

    print('Dataset pixel mean: ',mean)


import json
def split_speed(dataset_dir, val_percentage):
    ''' Split SPEED training set into train/val subset'''

    # Load annotations
    json_path = os.path.join(dataset_dir,'train.json')
    with open(json_path, 'r') as f:
        dataset = json.loads(f.read())

    random.shuffle(dataset)

    nr_instances = len(dataset)
    nr_val_instances = nr_instances*val_percentage

    train_new_set = []
    val_set = []

    for i, ann in enumerate(dataset):
        if i<nr_val_instances:
            val_set.append(ann)
        else:
            train_new_set.append(ann)

    # Write dataset splits

    train_out_path = os.path.join(dataset_dir,'train_no_val.json')
    val_out_path   = os.path.join(dataset_dir,'val.json')

    with open(train_out_path, 'w+') as f:
        f.write(json.dumps(train_new_set))

    with open(val_out_path, 'w+') as f:
        f.write(json.dumps(val_set))

def merge_speed(dataset_dir_1, dataset_dir_2, dataset_dir_3):
    ''' Split two SPEED subsets'''

    # Load annotations
    with open(dataset_dir_1, 'r') as f:
        dataset_1 = json.loads(f.read())

    with open(dataset_dir_2, 'r') as f:
        dataset_2 = json.loads(f.read())

    new_set = []

    for i, ann in enumerate(dataset_1):
            new_set.append(ann)

    for i, ann in enumerate(dataset_2):
            new_set.append(ann)

    # Write dataset splits

    with open(dataset_dir_3, 'w+') as f:
        f.write(json.dumps(new_set))
