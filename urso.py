'''
Class to handle URSO datasets
'''
import os
import numpy as np
import os.path
import skimage
import pandas as pd
import se3lib
import utils
from dataset import Dataset

class Camera:
    fov_x = 90.0 * np.pi / 180
    fov_y = 73.7 * np.pi / 180
    width = 1280  # number of horizontal[pixels]
    height = 960  # number of vertical[pixels]
    # Focal lengths
    fx = width / (2 * np.tan(fov_x / 2))
    fy = - height / (2 * np.tan(fov_y / 2))

    K = np.matrix([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])

# Image mean (RGB)
MEAN_PIXEL = np.array([45, 49, 52])

class Urso(Dataset):

    def load_dataset(self, dataset_dir, config, subset):
        """Load a subset of the dataset.
        dataset_dir: The root directory of the dataset.
        subset: What to load (train, val, test)
        """

        self.name = 'Urso'

        if not os.path.exists(dataset_dir):
            print("Image directory '" + dataset_dir + "' not found.")
            return None

        set_filename = dataset_dir + '/' + subset + '_images.csv'
        rgb_list_df = pd.read_csv(set_filename, names=['filename'], header=-1)
        rgb_list = list(rgb_list_df['filename'])

        # Set camera params
        self.camera = Camera()

        print('Loading poses')
        poses = pd.read_csv(dataset_dir + '/' + subset + '_poses_gt.csv')

        nr_instances = len(rgb_list)


        q_array = np.zeros((nr_instances, 4), dtype=np.float32)
        t_array = np.zeros((nr_instances, 3), dtype=np.float32)
        # Enforce injectivity, i.e., Keep quaternions only in the north hemisphere (useful for regression)
        for i in range(nr_instances):
            if poses['q4'][i] < 0:
                q_array[i, :] = np.asarray([-poses['q1'][i], -poses['q2'][i], -poses['q3'][i], -poses['q4'][i]])
            else:
                q_array[i, :] = np.asarray([poses['q1'][i], poses['q2'][i], poses['q3'][i], poses['q4'][i]])

            t_array[i, :] = [poses['x'][i], poses['y'][i], poses['z'][i]]


        # Encode orientation using soft assignment
        if not config.REGRESS_ORI:

            print('Encoding orientations using soft assignment..')
            ori_encoded, ori_histogram_map, ori_output_mask = utils.encode_ori(q_array, config.ORI_BINS_PER_DIM, config.BETA,
                                                        np.array([-180, -90, -180]), np.array([180, 90, 180]))
            self.ori_histogram_map = ori_histogram_map
            self.ori_output_mask = ori_output_mask

        if not config.REGRESS_LOC:
            print('Encoding locations using soft assignment..')

            # Obtain location as: (image_x, image_y, depth)
            img_x_array = poses['y'] / poses['x']  # simultaneously converting to cam ref frame
            img_y_array = poses['z'] / poses['x']  # simultaneously converting to cam ref frame
            z_array = poses['x']

            # Compute location limits based on camera FOV and dataset range
            theta_x = self.camera.fov_x * np.pi / 360
            theta_y = self.camera.fov_y * np.pi / 360
            x_max = np.tan(theta_x)
            y_max = np.tan(theta_y)
            z_min = min(z_array)
            z_max = max(z_array)

            loc_encoded, loc_histogram_map = utils.encode_loc(np.stack((img_x_array, img_y_array, z_array), axis=1),
                                                        config.LOC_BINS_PER_DIM, config.BETA,
                                                        np.array([-x_max, -y_max, z_min]), np.array([x_max, y_max, z_max]))

            # Store physical structure of histogram for later inference
            self.histogram_3D_map = loc_histogram_map

        if not rgb_list:
            print('No files found')
            return None

        K1, K2 = utils.encode_as_keypoints(q_array, t_array, 3.0)

        i = 0
        for file_name in rgb_list:

            q = q_array[i, :]

            # Convert to angle-axis
            v, theta = se3lib.quat2angleaxis(q)

            # Convert to euler angles
            pyr = np.asarray(se3lib.quat2euler(q))

            if config.REGRESS_ORI:
                ori_encoded_i = []
            else:
                ori_encoded_i = ori_encoded[i, :]

            if config.REGRESS_LOC:
                loc_encoded_i = []
            else:
                loc_encoded_i = loc_encoded[i, :]

            rgb_path = os.path.join(dataset_dir, file_name)
            self.add_image(
                "URSO",
                image_id=i,
                path=rgb_path,
                keypoints=[K1[i, :], K2[i, :]],
                location=[poses['x'][i], poses['y'][i], poses['z'][i]],
                location_map=loc_encoded_i,
                quaternion=q,
                angleaxis=[v[0] * theta, v[1] * theta, v[2] * theta],
                pyr=pyr,
                ori_map=ori_encoded_i
            )
            i = i + 1

        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image