import argparse
import pandas as pd
import random
import glob
import os
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='Split dataset.')
parser.add_argument('--dataset_dir',
                    required=True,
                    metavar='dir/to/dataset',
                    help='Relative path to dataset dir')
parser.add_argument('--test_percentage',
                    type=int,
                    default=10,
                    help='Percentage of images used as a test set')
parser.add_argument('--val_percentage',
                    type=int,
                    default=10,
                    help='Percentage of images used as a validation set')
args = parser.parse_args()

rgb_list = glob.glob(os.path.join(args.dataset_dir, '*rgb.png'))
nr_images = len(rgb_list)

poses = pd.read_csv(os.path.join(args.dataset_dir, 'gt.csv'))

assert nr_images == len(poses)

# Create random list for shuffling
shuffle_ids = np.arange(nr_images)
random.shuffle(shuffle_ids)

nr_testing_images = int(nr_images*args.testing_percentage*0.01+0.5)
nr_nontraining_images = int(nr_images*(args.testing_percentage+args.validation_percentage)*0.01+0.5)

# Split poses according to shuffle
poses_test = poses.loc[shuffle_ids[0:nr_testing_images]]
poses_val = poses.loc[shuffle_ids[nr_testing_images:nr_nontraining_images]]
poses_train = poses.loc[shuffle_ids[nr_nontraining_images:nr_images]]

poses_test.to_csv(os.path.join(args.dataset_dir,'test_poses_gt.csv'), index=False)
poses_val.to_csv(os.path.join(args.dataset_dir,'val_poses_gt.csv'), index=False)
poses_train.to_csv(os.path.join(args.dataset_dir,'train_poses_gt.csv'), index=False)

# Split images according to shuffle
training_img_list = []
testing_img_list = []
val_img_list = []
for i in range(nr_images):
    img_name = str(shuffle_ids[i]) + "_rgb.png"
    if i<nr_testing_images:
        testing_img_list.append(img_name)
    elif i<nr_nontraining_images:
        val_img_list.append(img_name)
    else:
        training_img_list.append(img_name)

# Writing img lists
with open(os.path.join(args.dataset_dir,'test_images.csv'), 'w') as f:
    for img_name in testing_img_list:
        f.write(img_name)
        f.write('\n')

with open(os.path.join(args.dataset_dir,'train_images.csv'), 'w') as f:
    for img_name in training_img_list:
        f.write(img_name)
        f.write('\n')

with open(os.path.join(args.dataset_dir,'val_images.csv'), 'w') as f:
    for img_name in val_img_list:
        f.write(img_name)
        f.write('\n')

