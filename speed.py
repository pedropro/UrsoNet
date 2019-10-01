'''
Class to handle Speed dataset
'''
import os
import numpy as np
import os.path
import skimage
import pandas as pd
import se3lib
import utils
from dataset import Dataset
import itertools
import json

class Camera:
    fwx = 0.0176  # focal length[m]
    fwy = 0.0176  # focal length[m]
    width = 1920  # number of horizontal[pixels]
    height = 1200  # number of vertical[pixels]
    ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fx = fwx / ppx  # horizontal focal length[pixels]
    fy = fwy / ppy  # vertical focal length[pixels]

    K = np.matrix([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])

class Speed(Dataset):

    def load_dataset(self, dataset_dir, config, subset):
        """Load a subset of the dataset.
        dataset_dir: The root directory of the dataset.
        subset: What to load (train, val, test)
        """

        assert subset in ['train', 'train_no_val', 'val', 'test', 'real', 'real_test', 'train_total']

        self.name = 'Speed'

        # Set camera params
        self.camera = Camera()


        if not os.path.exists(dataset_dir):
            print("Image directory '" + dataset_dir + "' not found.")
            return None

        set_filename = dataset_dir + '/' + subset + '.json'
        with open(set_filename, 'r') as f:
            dataset = json.load(f)

        images_list = []
        nr_instances = len(dataset)

        print('Loading', nr_instances, 'images')

        if subset not in ['test', 'real_test']:
            q_array = np.zeros((nr_instances, 4), dtype=np.float32)
            t_array = np.zeros((nr_instances, 3), dtype=np.float32)
            i = 0
            for image_ann in dataset:
                images_list.append(image_ann['filename'])
                t_array[i, :] = image_ann['r_Vo2To_vbs_true']

                # Load quaternion, enforce north hemisphere and put scalar part at the end for consistency
                q = image_ann['q_vbs2tango']
                sign = np.sign(q[0])
                q_rectified = sign * np.array([q[1], q[2], q[3], q[0]]) # Do not forget to revert this before submission
                q_array[i, :] = q_rectified
                i += 1

            # Encode orientation using soft assignment
            if not config.REGRESS_ORI:
                print('Encoding orientations using soft assignment..')
                ori_encoded, ori_histogram_map, ori_output_mask = utils.encode_ori(q_array, config.ORI_BINS_PER_DIM, config.BETA,
                                                              np.array([-180, -90, -180]), np.array([180, 90, 180]))
                self.ori_histogram_map = ori_histogram_map
                self.ori_output_mask = ori_output_mask

            K1, K2 = utils.encode_as_keypoints(q_array, t_array)

            i = 0
            for file_name in images_list:

                q = [q_array[i,0], q_array[i,1], q_array[i,2], q_array[i,3]]

                # Convert to angle-axis
                v, theta = se3lib.quat2angleaxis(q)

                # Convert to euler angles
                pyr = np.asarray(se3lib.quat2euler(q))

                if subset in ['train_no_val', 'val']:
                    subdir = 'train'
                else:
                    subdir = subset

                if config.REGRESS_ORI:
                    ori_encoded_i = []
                else:
                    ori_encoded_i = ori_encoded[i, :]

                img_path = os.path.join(dataset_dir, 'images', subdir, file_name)
                self.add_image(
                    "SPEED",
                    image_id=i,
                    path=img_path,
                    location=t_array[i,:],
                    keypoints=[K1[i,:],K2[i,:]],
                    location_map=[],
                    quaternion=q_array[i,:],
                    angleaxis=[v[0] * theta, v[1] * theta, v[2] * theta],
                    pyr=pyr,
                    ori_map=ori_encoded_i
                )
                i = i + 1

        else:

            # First construct necessary orientation structure

            min_lim = np.array([-180, -90, -180])
            max_lim = np.array([180, 90, 180])
            nr_total_bins = config.ORI_BINS_PER_DIM**3

            bins_loc_per_dim = np.linspace(0.0, 1.0, config.ORI_BINS_PER_DIM)
            H_loc_list = list(itertools.product(bins_loc_per_dim, repeat=3))
            H_ori = np.asarray(H_loc_list * (max_lim - min_lim) + min_lim)
            H_quat = np.zeros(shape=(nr_total_bins, 4), dtype=np.float32)
            for i in range(nr_total_bins):
                H_quat[i, :] = se3lib.euler2quat(H_ori[i, 0], H_ori[i, 1], H_ori[i, 2]).T

            self.ori_histogram_map = H_quat
            self.ori_output_mask = np.full(config.ORI_BINS_PER_DIM ** 3, False) # Bogus

            # Load just image paths
            i = 0
            for image_ann in dataset:
                img_path = os.path.join(dataset_dir, 'images', subset, image_ann['filename'])
                self.add_image(
                    "SPEED",
                    image_id=i,
                    path=img_path
                )
                i += 1

        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,1] Numpy array.
        """
        # Load grayscale image
        image = skimage.io.imread(self.image_info[image_id]['path'])

        # Converting to RGB for compatibility
        image = skimage.color.gray2rgb(image)
        return image