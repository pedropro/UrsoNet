"""
This is the main script you should call to train and test UrsoNet

Copyright (c) Pedro F. Proenza
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: Check README

"""

import os
import numpy as np
import os.path
import skimage
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import se3lib

import utils
import net
from config import Config

import urso
import speed

# Models directory (where weights are stored)
MODEL_DIR = os.path.abspath("./models")
DEFAULT_LOGS_DIR = os.path.join(MODEL_DIR, "logs")

# Dataset directory
DATA_DIR = os.path.abspath("./datasets")

# Path to trained weights file of Mask-RCNN on COCO
COCO_WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

OrientationParamOptions = ['quaternion', 'euler_angles', 'angle_axis']

def fit_GMM_to_orientation(q_map, pmf, nr_iterations, var, nr_max_modes=4):
    ''' Fits multiple quaternions to a PMF using Expectation Maximization'''

    nr_total_bins = len(pmf)
    scores = []

    # Sorting bins per probability
    pmf_sorted_indices = pmf.argsort()[::-1]

    for N in range(1, nr_max_modes):

        # 1. Initialize Gaussians
        Q_mean = np.zeros((N,4), np.float32)
        Q_var = np.ones(N, np.float32)*var
        priors = np.ones(N, np.float32)/N

        # Initialize Gaussian means by picking up the strongest bins
        check_q_mask = np.zeros_like(pmf)>0

        ptr = 0
        for k in range(N):

            # Select bin
            for i in range(ptr, nr_total_bins):
                if not check_q_mask[i]:
                    check_q_mask[i] = True
                    q_max = q_map[pmf_sorted_indices[i], :]
                    Q_mean[k, :] = q_max
                    ptr = i + 1
                    break

            # Mask out neighbours
            for i in range(nr_total_bins):
                q_i = q_map[pmf_sorted_indices[i], :]
                if not check_q_mask[i]:
                    #d_i = (1 - np.sum(q_i * q_max)) ** 2
                    d_i = (se3lib.angle_between_quats(q_i, q_max) / 180) ** 2
                    if d_i < 9 * var:
                        check_q_mask[i] = 1


        # 2. Expectation Maximization loop
        for it in range(nr_iterations):

            # Expectation step

            # Normalized angular distance
            Distances = np.asarray(se3lib.angle_between_quats(q_map, Q_mean))/180

            # Compute p(X|Theta)
            eps = 1e-18
            p_X_given_models = eps + np.divide(np.exp(np.divide(-Distances ** 2, 2.0 * Q_var)),
                                                 np.sqrt(2.0 * np.pi * Q_var))

            # Compute p(Theta|X) by applying Bayes rule
            # Get marginal likelihood
            p_X_given_models_times_priors = p_X_given_models*priors
            p_X = np.sum(p_X_given_models_times_priors, axis=1)
            p_models_given_X = p_X_given_models_times_priors/p_X[:,np.newaxis]

            # Maximization step

            # Compute weights
            W = p_models_given_X * pmf[:, np.newaxis]
            Z = np.sum(W, axis=0)
            W_n = W / Z

            # Compute average quaternions
            for k in range(N):

                q_mean_k, _ = se3lib.quat_weighted_avg(q_map, W_n[:, k])
                Q_mean[k, :] = q_mean_k
                Q_var[k] = 0
                Distances = np.asarray(se3lib.angle_between_quats(q_map,q_mean_k)/180)**2
                for i in range(nr_total_bins):
                    Q_var[k] += W_n[i, k] * Distances[i]

            # print('New mixture means:\n', Q_mean)
            # print('New mixture priors:\n', priors)
            # print('New mixture var:\n', Q_var)
            # print('\n')

            # Compute priors
            priors = Z

            if N == 1 and it == 1:
                break

        # Check model likelihood by reusing last iteration state
        score = np.sum(pmf * np.log(p_X))

        if len(scores)==0 or score > scores[-1]+0.005:
            # Update best model
            Q_mean_best = Q_mean
            Q_var_best = Q_var
            Q_priors_best = priors
            scores.append(score)
        else:
            # Stop model searching to return last state
            break

    # TODO: Sort by likelihood
    sorting_indices = Q_priors_best.argsort()[::-1]

    Q_mean_best = Q_mean_best[sorting_indices]
    Q_priors_best = Q_priors_best[sorting_indices]
    Q_var_best = Q_var_best[sorting_indices]

    print('Q priors:',Q_priors_best)
    print('Q :', Q_mean_best)
    print('Scores:', scores)

    return Q_mean_best, Q_var_best, Q_priors_best, scores

def evaluate_image(model, dataset, image_id):

    # Load pose in all formats
    loc_gt = dataset.load_location(image_id)
    q_gt = dataset.load_quaternion(image_id)
    image = dataset.load_image(image_id)
    I, I_meta, loc_encoded_gt, ori_encoded_gt = \
        net.load_image_gt(dataset, model.config, image_id)

    results = model.detect([image], verbose=1)

    # Retrieve location
    if model.config.REGRESS_LOC:
        loc_est = results[0]['loc']
    else:
        loc_pmf = utils.stable_softmax(results[0]['loc'])

        # Compute location mean according to first moment
        loc_est = np.asmatrix(loc_pmf) * np.asmatrix(dataset.histogram_3D_map)

        # Compute loc encoding error
        loc_decoded_gt = np.asmatrix(loc_encoded_gt) * np.asmatrix(dataset.histogram_3D_map)
        loc_encoded_err = np.linalg.norm(loc_decoded_gt - loc_gt)

    # Retrieve orientation
    if model.config.REGRESS_ORI:

        if model.config.ORIENTATION_PARAM == 'quaternion':
            q_est = results[0]['ori']
        elif model.config.ORIENTATION_PARAM == 'euler_angles':
            q_est = se3lib.SO32quat(
                se3lib.euler2SO3_left(results[0]['ori'][0], results[0]['ori'][1], results[0]['ori'][2]))
        elif model.config.ORIENTATION_PARAM == 'angle_axis':
            theta = np.linalg.norm(results[0]['ori'])
            if theta < 1e-6:
                v = [0, 0, 0]
            else:
                v = results[0]['ori'] / theta
            q_est = se3lib.angleaxis2quat(v, theta)
    else:
        ori_pmf = utils.stable_softmax(results[0]['ori'])

        # Compute mean quaternion
        q_est, q_est_cov = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_pmf)

        # Compute encoded error
        q_encoded_gt, _ = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_encoded_gt)
        ori_encoded_err = 2 * np.arccos(
            np.abs(np.asmatrix(q_encoded_gt) * np.asmatrix(q_gt).transpose())) * 180 / np.pi

    # Compute errors
    angular_err = 2 * np.arccos(np.abs(np.asmatrix(q_est) * np.asmatrix(q_gt).transpose()))
    # angular_err_in_deg = angular_err* 180 / np.pi
    loc_err = np.linalg.norm(loc_est - loc_gt)
    loc_rel_err = loc_err / np.linalg.norm(loc_gt)

    # Compute ESA score
    esa_score = loc_rel_err + angular_err

    return loc_err, angular_err, loc_rel_err, esa_score

def test_and_submit(model, dataset_virtual, dataset_real):
    """ Evaluates model on ESA challenge test-set (no labels)
    and outputs submission file in a format compatible with the ESA server (probably down by now)
    """

    # ESA API
    from submission import SubmissionWriter
    submission = SubmissionWriter()

    # TODO: Make the next 2 loops a nested loop

    # Synthetic test set
    for image_id in dataset_virtual.image_ids:

        print('Image ID:', image_id)

        image = dataset_virtual.load_image(image_id)
        info = dataset_virtual.image_info[image_id]

        results = model.detect([image], verbose=1)

        # Retrieve location
        if model.config.REGRESS_LOC:
            loc_est = results[0]['loc']
        else:
            loc_pmf = utils.stable_softmax(results[0]['loc'])

            # Compute location mean according to first moment
            loc_est = np.asmatrix(loc_pmf) * np.asmatrix(dataset_virtual.histogram_3D_map)

        # Retrieve orientation
        if model.config.REGRESS_ORI:

            if model.config.ORIENTATION_PARAM == 'quaternion':
                q_est = results[0]['ori']
            elif model.config.ORIENTATION_PARAM == 'euler_angles':
                q_est = se3lib.SO32quat(se3lib.euler2SO3_left(results[0]['ori'][0], results[0]['ori'][1], results[0]['ori'][2]))
            elif model.config.ORIENTATION_PARAM == 'angle_axis':
                theta = np.linalg.norm(results[0]['ori'])
                if theta < 1e-6:
                    v = [0,0,0]
                else:
                    v = results[0]['ori']/theta
                q_est = se3lib.angleaxis2quat(v,theta)
        else:
            ori_pmf = utils.stable_softmax(results[0]['ori'])

            # Compute mean quaternion
            q_est, q_est_cov = se3lib.quat_weighted_avg(dataset_virtual.ori_histogram_map, ori_pmf)

        # Change quaternion order
        q_rect = [q_est[3], q_est[0], q_est[1], q_est[2]]

        submission.append_test(info['path'].split('/')[-1], q_rect, loc_est)

    # Real test set

    for image_id in dataset_real.image_ids:

        print('Image ID:', image_id)

        image = dataset_real.load_image(image_id)
        info = dataset_real.image_info[image_id]

        results = model.detect([image], verbose=1)

        # Retrieve location
        if model.config.REGRESS_LOC:
            loc_est = results[0]['loc']
        else:
            loc_pmf = utils.stable_softmax(results[0]['loc'])

            # Compute location mean according to first moment
            loc_est = np.asmatrix(loc_pmf) * np.asmatrix(dataset_real.histogram_3D_map)

        # Retrieve orientation
        if model.config.REGRESS_ORI:

            if model.config.ORIENTATION_PARAM == 'quaternion':
                q_est = results[0]['ori']
            elif model.config.ORIENTATION_PARAM == 'euler_angles':
                q_est = se3lib.SO32quat(se3lib.euler2SO3_left(results[0]['ori'][0], results[0]['ori'][1], results[0]['ori'][2]))
            elif model.config.ORIENTATION_PARAM == 'angle_axis':
                theta = np.linalg.norm(results[0]['ori'])
                if theta < 1e-6:
                    v = [0,0,0]
                else:
                    v = results[0]['ori']/theta
                q_est = se3lib.angleaxis2quat(v,theta)
        else:
            ori_pmf = utils.stable_softmax(results[0]['ori'])

            # Compute mean quaternion
            q_est, q_est_cov = se3lib.quat_weighted_avg(dataset_real.ori_histogram_map, ori_pmf)

        # Change quaternion order
        q_rect = [q_est[3], q_est[0], q_est[1], q_est[2]]

        submission.append_real_test(info['path'].split('/')[-1], q_rect, loc_est)

    submission.export(suffix='debug')
    print('Submission exported.')


def evaluate(model, dataset):
    """ Evaluates model on all dataset images. Assumes all images have corresponding pose labels.
    """

    loc_err_acc = []
    loc_encoded_err_acc = []
    ori_err_acc = []
    ori_encoded_err_acc = []
    distances_acc = []
    esa_scores_acc = []

    # Variance used only for prob. orientation estimation
    delta = model.config.BETA / model.config.ORI_BINS_PER_DIM
    var = delta ** 2 / 12

    for image_id in dataset.image_ids:

        print('Image ID:', image_id)

        # Load pose in all formats
        loc_gt = dataset.load_location(image_id)
        q_gt = dataset.load_quaternion(image_id)
        image = dataset.load_image(image_id)

        results = model.detect([image], verbose=1)

        if model.config.REGRESS_KEYPOINTS:
            # Experimental

            I, I_meta, loc_gt, k1_gt, k2_gt = \
                net.load_image_gt(dataset, model.config, image_id)

            loc_est = results[0]['loc']
            k1_est = results[0]['k1']
            k2_est = results[0]['k2']

            # Prepare keypoint matches
            # TODO: take scale into account and get rid of magic numbers
            P1 = np.zeros((3, 3))
            P1[2,0] = 3.0
            P1[1,1] = 3.0

            P2 = np.zeros((3, 3))
            P2[:, 0] = k1_est
            P2[:, 1] = k2_est
            P2[:, 2] = loc_est

            t, R = se3lib.pose_3Dto3D(np.asmatrix(P1),np.asmatrix(P2))
            q_est = se3lib.SO32quat(R.T)

        else:

            I, I_meta, loc_encoded_gt, ori_encoded_gt = \
                net.load_image_gt(dataset, model.config, image_id)

            # Retrieve location
            if model.config.REGRESS_LOC:
                loc_est = results[0]['loc']
            else:
                loc_pmf = utils.stable_softmax(results[0]['loc'])

                # Compute location mean according to first moment
                loc_est = np.asmatrix(loc_pmf) * np.asmatrix(dataset.histogram_3D_map)

                # Compute loc encoding error
                loc_decoded_gt = np.asmatrix(loc_encoded_gt) * np.asmatrix(dataset.histogram_3D_map)
                loc_encoded_err = np.linalg.norm(loc_decoded_gt - loc_gt)
                loc_encoded_err_acc.append(loc_encoded_err)

            # Retrieve orientation
            if model.config.REGRESS_ORI:

                if model.config.ORIENTATION_PARAM == 'quaternion':
                    q_est = results[0]['ori']
                elif model.config.ORIENTATION_PARAM == 'euler_angles':
                    q_est = se3lib.SO32quat(se3lib.euler2SO3_left(results[0]['ori'][0], results[0]['ori'][1], results[0]['ori'][2]))
                elif model.config.ORIENTATION_PARAM == 'angle_axis':
                    theta = np.linalg.norm(results[0]['ori'])
                    if theta < 1e-6:
                        v = [0,0,0]
                    else:
                        v = results[0]['ori']/theta
                    q_est = se3lib.angleaxis2quat(v,theta)
            else:

                ori_pmf = utils.stable_softmax(results[0]['ori'])

                # Compute mean quaternion
                q_est, q_est_cov = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_pmf)

                # Multimodal estimation
                # Uncomment this block to try the EM framework
                # nr_EM_iterations = 5
                # Q_mean, Q_var, Q_priors, model_scores = fit_GMM_to_orientation(dataset.ori_histogram_map, ori_pmf,
                #                                                                nr_EM_iterations, var)
                #
                # print('Err:', angular_err)
                # angular_err = 2*np.arccos(np.abs(np.asmatrix(Q_mean)*np.asmatrix(q_gt).transpose()))*180/np.pi
                #
                # # Select best of two
                # if len(angular_err) == 1 or angular_err[0]<angular_err[1]:
                #     q_est = Q_mean[0, :]
                # else:
                #     q_est = Q_mean[1, :]
                #
                # print('Err:',angular_err)

                # Compute encoded error
                q_encoded_gt, _ = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_encoded_gt)
                ori_encoded_err = 2*np.arccos(np.abs(np.asmatrix(q_encoded_gt)*np.asmatrix(q_gt).transpose()))*180/np.pi
                ori_encoded_err_acc.append(ori_encoded_err)

        # 3. Angular error
        angular_err = 2*np.arccos(np.abs(np.asmatrix(q_est)*np.asmatrix(q_gt).transpose()))*180/np.pi
        ori_err_acc.append(angular_err.item(0))

        # 4. Loc error
        loc_err = np.linalg.norm(loc_est - loc_gt)
        loc_err_acc.append(loc_err)

        print('Loc Error: ', loc_err)
        print('Ori Error: ', angular_err)

        # Compute ESA score
        esa_score = loc_err/np.linalg.norm(loc_gt) + 2*np.arccos(np.abs(np.asmatrix(q_est)*np.asmatrix(q_gt).transpose()))
        esa_scores_acc.append(esa_score)

        # Store depth
        distances_acc.append(loc_gt[2])

    print('Mean est. location error: ', np.mean(loc_err_acc))
    print('Mean est. orientation error: ', np.mean(ori_err_acc))
    print('ESA score: ', np.mean(esa_scores_acc))
    print('Mean encoded location error: ', np.mean(loc_encoded_err_acc))

    # Dump results
    pd.DataFrame(np.asarray(ori_err_acc)).to_csv("ori_err.csv")
    pd.DataFrame(np.asarray(loc_err_acc)).to_csv("loc_err.csv")
    pd.DataFrame(np.asarray(distances_acc)).to_csv("dists_err.csv")


def detect_dataset(model, dataset, nr_images):
    """ Tests model on N random images of the dataset
     and shows the results.
    """

    # Variance used only for prob. orientation estimation
    delta = model.config.BETA / model.config.ORI_BINS_PER_DIM
    var = delta ** 2 / 12

    for i in range(nr_images):
        image_id = random.choice(dataset.image_ids)

        # Load pose in all formats
        loc_gt = dataset.load_location(image_id)
        q_gt = dataset.load_quaternion(image_id)
        I, I_meta, loc_encoded_gt, ori_encoded_gt = \
            net.load_image_gt(dataset, model.config, image_id)
        image_ori = dataset.load_image(image_id)

        info = dataset.image_info[image_id]

        # Run detection
        results = model.detect([image_ori], verbose=1)

        # Retrieve location
        if model.config.REGRESS_LOC:
            loc_est = results[0]['loc']
        else:
            loc_pmf = utils.stable_softmax(results[0]['loc'])

            # Compute location mean according to first moment
            loc_est = np.asmatrix(loc_pmf) * np.asmatrix(dataset.histogram_3D_map)

            # Compute loc encoding error
            loc_encoded_gt = np.asmatrix(loc_encoded_gt) * np.asmatrix(dataset.histogram_3D_map)
            loc_encoded_err = np.linalg.norm(loc_encoded_gt - loc_gt)

        # Retrieve orientation
        if model.config.REGRESS_ORI:

            if model.config.ORIENTATION_PARAM == 'quaternion':
                q_est = results[0]['ori']
            elif model.config.ORIENTATION_PARAM == 'euler_angles':
                q_est = se3lib.SO32quat(se3lib.euler2SO3_left(results[0]['ori'][0], results[0]['ori'][1], results[0]['ori'][2]))
            elif model.config.ORIENTATION_PARAM == 'angle_axis':
                theta = np.linalg.norm(results[0]['ori'])
                if theta < 1e-6:
                    v = [0,0,0]
                else:
                    v = results[0]['ori']/theta
                q_est = se3lib.angleaxis2quat(v,theta)
        else:
            ori_pmf = utils.stable_softmax(results[0]['ori'])

            # Compute mean quaternion
            q_est, q_est_cov = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_pmf)

            # Multimodal estimation
            # Uncomment this block to try the EM framework
            # nr_EM_iterations = 5
            # Q_mean, Q_var, Q_priors, model_scores = fit_GMM_to_orientation(dataset.ori_histogram_map, ori_pmf, nr_EM_iterations, var)
            # print('Multimodal errors',2 * np.arccos(np.abs(np.asmatrix(Q_mean) * np.asmatrix(q_gt).transpose())) * 180 / np.pi)
            #
            # q_est_1 = Q_mean[0, :]
            # q_est_2 = Q_mean[1, :]
            # utils.polar_plot(q_est_1, q_est_2)

        # Compute Errors
        angular_err = 2 * np.arccos(np.abs(np.asmatrix(q_est) * np.asmatrix(q_gt).transpose())) * 180 / np.pi
        loc_err = np.linalg.norm(loc_est - loc_gt)

        print('GT location: ', loc_gt)
        print('Est location: ', loc_est)
        print('Processed Image:', info['path'])
        print('Est orientation: ', q_est)
        print('GT_orientation: ', q_gt)

        print('Location error: ', loc_err)
        print('Angular error: ', angular_err)

        # Visualize PMFs
        if not model.config.REGRESS_ORI:

            nr_bins_per_dim = model.config.ORI_BINS_PER_DIM
            utils.visualize_weights(ori_encoded_gt,ori_pmf,nr_bins_per_dim)

        # Show image
        fig, (ax_1, ax_2) = plt.subplots(1,2,figsize=(12, 8))
        ax_1.imshow(image_ori)
        ax_1.set_xticks([])
        ax_1.set_yticks([])
        ax_2.imshow(image_ori)
        ax_2.set_xticks([])
        ax_2.set_yticks([])

        height_ori = np.shape(image_ori)[0]
        width_ori = np.shape(image_ori)[1]

        # Recover focal lengths
        fx = dataset.camera.fx
        fy = dataset.camera.fy

        K = np.matrix([[fx,0,width_ori/2],[0,fy,height_ori/2],[0,0,1]])

        # Speed  labels expresses q_obj_cam whereas
        # Urso labels expresses q_cam_obj
        if dataset.name == 'Speed':
            q_est = se3lib.quat_inv(q_est)
            q_gt = se3lib.quat_inv(q_gt)

        utils.visualize_axes(ax_1, q_gt, loc_gt, K, 100)
        utils.visualize_axes(ax_2, q_est, loc_est, K, 100)

        utils.polar_plot(q_gt,q_est)

        # Location overlap visualization
        fig, ax = plt.subplots()
        ax.imshow(image_ori)

        # Project 3D coords for visualization
        x_est = loc_est[0] / loc_est[2]
        y_est = loc_est[1] / loc_est[2]

        x_gt = loc_gt[0] / loc_gt[2]
        y_gt = loc_gt[1] / loc_gt[2]

        if not model.config.REGRESS_LOC:
            x_decoded_gt = loc_encoded_gt[0, 0] / loc_encoded_gt[0, 2]
            y_decoded_gt = loc_encoded_gt[0, 1] / loc_encoded_gt[0, 2]

            circ = Circle((x_decoded_gt * fx + width_ori / 2, height_ori / 2 + y_decoded_gt * fy), 7, facecolor='b',
                          label='encoded')
            ax.add_patch(circ)

        # Plot locations
        circ_gt = Circle((x_gt*fx + width_ori/2, height_ori/2 + y_gt*fy), 15, facecolor='r', label='gt')
        ax.add_patch(circ_gt)

        circ = Circle((x_est*fx + width_ori/2, height_ori/2 + y_est*fy), 10, facecolor='g',label='pred')
        ax.add_patch(circ)

        ax.legend(loc='upper right', shadow=True, fontsize='x-small')
        plt.show()

def detect_video(model, dataset, video_path):
    ''' Experimental'''

    import cv2

    # Video capture
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    # Camera projection mat
    width = dataset.camera.width/2  # TODO: work on original image size not 1/2
    height = dataset.camera.height/2
    fov_horizontal = np.pi / 2
    fx = width / (2 * np.tan(dataset.camera.fov_x / 2))
    fy = - height / (2 * np.tan(dataset.camera.fov_y / 2))
    K = np.matrix([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])

    R_cam_unreal = np.matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    # Define codec and create video writer
    vwriter = cv2.VideoWriter("video_real.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (int(width), int(height)))

    count = 0
    pose_est_acc = []
    success = True
    while success:
        print("frame: ", count)
        count += 1
        # Read next image
        success, image = vcapture.read()
        if success and count>16900:
            # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            image = image[:,1:-150,:] # crop
            image = np.pad(image, [(400, 400), (400, 400), (0, 0)], mode='constant', constant_values=0)
            image[:,:,0] = 0.21*image[:,:,0]+0.72*image[:,:,1]+0.07*image[:,:,2]
            image[:, :, 1] = image[:,:,0]
            image[:, :, 2] = image[:, :, 0]

            # Resize to network input shape
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=model.config.IMAGE_MIN_DIM,
                min_scale=model.config.IMAGE_MIN_SCALE,
                max_dim=model.config.IMAGE_MAX_DIM,
                mode=model.config.IMAGE_RESIZE_MODE)

            # Detect objects
            results = model.detect([image], verbose=0)[0]

            loc_est = results['loc']

            ori_pmf = utils.stable_softmax(results['ori'])
            q_est, q_est_cov = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_pmf)

            z = loc_est[2]
            x = loc_est[0]
            y = loc_est[1]
            print(str(z) + " " + str(x) + " " + str(y))

            # Recover Unreal orientation: R_wo
            R_co = se3lib.quat2SO3(q_est)
            R_co = R_cam_unreal.T * R_co
            R_wc = se3lib.euler2SO3_unreal(0, 0, 0)
            R_wo = R_wc*R_co
            roll, pitch, yaw = se3lib.SO32euler(R_wo)
            #
            print(str(-pitch) + " " + str(yaw) + " " + str(-roll))

            # Stack frame gt
            pose_est = np.array([loc_est[2], loc_est[0], loc_est[1], -pitch, yaw, -roll])
            pose_est_acc.append(pose_est)

            # Crop and resize image to match original input size
            margin = (model.config.IMAGE_MAX_DIM - 480) // 2
            image = molded_image[margin:model.config.IMAGE_MAX_DIM-margin, :, :]

            # Show image
            #fig, ax_1 = plt.subplots(1, 1, figsize=(12, 8))

            utils.plot_axes(image, q_est, loc_est, K, 5.0)
            # ax_1.imshow(image)
            # ax_1.set_xticks([])
            # ax_1.set_yticks([])

            nr_bins_per_dim = model.config.ORI_BINS_PER_DIM
            utils.visualize_weights(ori_pmf, ori_pmf, nr_bins_per_dim)

            # plt.show(block=True)
            # Add image to video writer
            vwriter.write(image)

        if count > 17200:
            success = False

    vwriter.release()

    # Connect to simulator and load estimated poses

    # from unrealcv.automation import UE4Binary
    # from unrealcv.util import read_png, read_npy
    # from unrealcv import client
    #
    # client.connect()
    #
    # # Define codec and create video writer
    # vwriter2 = cv2.VideoWriter("video_virtual.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (1280, 960))
    #
    # # Rotation between reference frames
    # # Set up camera
    # command = 'vset /camera/0/location ' + str(0) + " " + str(0) + " " + str(0)
    # client.request(command)
    # command = 'vset /camera/0/rotation ' + str(0) + " " + str(0) + " " + str(0)
    # client.request(command)
    #
    # object_name = 'Soyuz_HP_10'
    # object_set_loc_command_prefix = 'vset /object/' + object_name + '/location '
    # object_set_ori_command_prefix = 'vset /object/' + object_name + '/rotation '
    #
    # for pose_est in pose_est_acc:
    #
    #     # Translate object
    #     command = object_set_loc_command_prefix + str(pose_est[0]*100.0) + " " + str(pose_est[1]*100.0) + " " + str(pose_est[2]*100.0)
    #     client.request(command)
    #
    #     # Rotate object
    #     command = object_set_ori_command_prefix + str(pose_est[3]) + " " + str(pose_est[4]) + " " + str(pose_est[5])
    #     client.request(command)
    #
    #     # Load and save rgb
    #     res = client.request('vget /camera/0/lit png')
    #     im = read_png(res)
    #
    #     # Convert to opencv and record video frame
    #     img_cv = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #     vwriter2.write(img_cv)
    #
    # vwriter2.release()

def train(model, dataset_train, dataset_val):
    """Train the model."""

    model.config.STEPS_PER_EPOCH = min(1000,int(len(dataset_train.image_ids)/model.config.BATCH_SIZE))

    # Write config to disk
    config_filename = 'config_' + str(model.epoch) + '.json'
    config_filepath = os.path.join(model.log_dir, config_filename)
    model.config.write_to_file(config_filepath)

    print("Training")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=config.EPOCHS, layers='all')

############################################################
#  Main
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="'train' or 'evaluate'")
    parser.add_argument('--backbone', required=False, default='resnet50',help='Backbone architecture')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--epochs', required=False, default=100, type=int, help='Number of epochs')
    parser.add_argument('--image_scale', required=False, default=1.0, type=float, help='Resize scale')
    parser.add_argument('--ori_weight', required=False, default=1.0, type=float, help='Loss weight')
    parser.add_argument('--loc_weight', required=False, default=1.0, type=float, help='Loss weight')
    parser.add_argument('--bottleneck', required=False, default=32, type=int, help='Bottleneck width')
    parser.add_argument('--branch_size', required=False, default=1024, type=int, help='Branch input size')
    parser.add_argument('--learn_rate', required=False, default=0.001, type=float, help='Learning rate')
    parser.add_argument('--batch_size', required=False, default=4, type=int, help='Number of images per GPU')
    parser.add_argument('--rot_aug', dest='rot_aug', action='store_true')
    parser.set_defaults(rot_aug=False)
    parser.add_argument('--rot_image_aug', dest='rot_image_aug', action='store_true')
    parser.set_defaults(rot_image_aug=False)
    parser.add_argument('--classify_ori', dest='regress_ori', action='store_false')
    parser.add_argument('--regress_ori', dest='regress_ori', action='store_true')
    parser.set_defaults(regress_ori=False)
    parser.add_argument('--classify_loc', dest='regress_loc', action='store_false')
    parser.add_argument('--regress_loc', dest='regress_loc', action='store_true')
    parser.set_defaults(regress_loc=True)
    parser.add_argument('--regress_keypoints', dest='regress_keypoints', action='store_true') # Experimental: Overides options above
    parser.set_defaults(regress_keypoints=False)
    parser.add_argument('--sim2real', dest='sim2real', action='store_true')
    parser.set_defaults(sim2real=False)
    parser.add_argument('--clr', dest='clr', action='store_true')
    parser.set_defaults(clr=False)
    parser.add_argument('--f16', dest='f16', action='store_true')
    parser.set_defaults(f16=False)
    parser.add_argument('--ori_param', required=False, default='quaternion', help="'quaternion' 'euler_angles' 'angle_axis'")
    parser.add_argument('--ori_resolution', required=False, default=16, type=int, help="Number of bins assigned to each angle")
    parser.add_argument('--weights', required=True, help="Path to weights .h5 file or 'coco' or 'imagenet' for coco pre-trained weights")
    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False, metavar="path or URL to image", help='Image to evaluate')
    parser.add_argument('--video', required=False, metavar="path or URL to video", help='Video to evaluate')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    assert args.ori_param in OrientationParamOptions

    # Set up configuration
    config = Config()
    config.ORIENTATION_PARAM = args.ori_param # only used in regression mode
    config.ORI_BINS_PER_DIM = args.ori_resolution # only used in classifcation mode
    config.NAME = args.dataset
    config.EPOCHS = args.epochs
    config.NR_DENSE_LAYERS = 1 # Number of fully connected layers used on top of the feature network
    config.LEARNING_RATE = args.learn_rate # 0.001
    config.BOTTLENECK_WIDTH = args.bottleneck
    config.BRANCH_SIZE = args.branch_size
    config.BACKBONE = args.backbone
    config.ROT_AUG = args.rot_aug
    config.F16 = args.f16
    config.SIM2REAL_AUG = args.sim2real
    config.CLR = args.clr
    config.ROT_IMAGE_AUG = args.rot_image_aug
    config.OPTIMIZER = "SGD"
    config.REGRESS_ORI = args.regress_ori
    config.REGRESS_LOC = args.regress_loc
    config.REGRESS_KEYPOINTS = args.regress_keypoints
    config.LOSS_WEIGHTS['loc_loss'] = args.loc_weight
    config.LOSS_WEIGHTS['ori_loss'] = args.ori_weight

    # Set up resizing & padding if needed
    config.IMAGE_RESIZE_MODE = 'pad64' # 'square' #
    if args.dataset == "speed":
        width_original = speed.Camera.width
        height_original = speed.Camera.height
    else:
        width_original = urso.Camera.width
        height_original = urso.Camera.height

    config.IMAGE_MAX_DIM = round(width_original * args.image_scale)

    if config.IMAGE_MAX_DIM % 64 > 0:
        raise Exception("Scale problem. Image maximum dimension must be dividable by 2 at least 6 times.")

    # n.b: assumes height is less than width
    height_scaled = round(height_original * args.image_scale)
    if height_scaled % 64 > 0:
        config.IMAGE_MIN_DIM = height_scaled - height_scaled%64 + 64
    else:
        config.IMAGE_MIN_DIM = height_scaled

    # Uncomment this if the model is trained from scratch
    # if args.dataset == "speed":
    #     config.NR_IMAGE_CHANNELS = 1

    if args.command == "train":
        config.IMAGES_PER_GPU = args.batch_size
    else:
        config.IMAGES_PER_GPU = 1
    config.BATCH_SIZE = config.IMAGES_PER_GPU * config.GPU_COUNT

    config.update()
    config.display()

    # Create model
    if args.command == "train":
        model = net.UrsoNet(mode="training", config=config,
                             model_dir=args.logs)
    else:
        model = net.UrsoNet(mode="inference", config=config,
                             model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        _, weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights(config.BACKBONE)
    elif args.weights.lower() != "none":
        _, weights_path = model.get_last_checkpoint(args.weights)

    # Load weights
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, None, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    elif args.weights.lower() == "imagenet":
        model.load_weights(weights_path, None, by_name=True)
    elif args.weights.lower() != "none":
        model.load_weights(weights_path, weights_path, by_name=True)
        #model.load_weights(weights_path, weights_path, by_name=True, exclude=["ori_final"]) # tmp

    dataset_dir = os.path.join(DATA_DIR, args.dataset)

    # Train or evaluate
    if args.command == "train":

        # Load training and validation set
        if args.dataset != "speed":
            dataset_train = urso.Urso()
            dataset_train.load_dataset(dataset_dir, model.config, "train")
            dataset_val = urso.Urso()
            dataset_val.load_dataset(dataset_dir, model.config, "val")
        else:
            dataset_train = speed.Speed()
            dataset_train.load_dataset(dataset_dir, model.config, "train_no_val") # 'train_total') #
            dataset_val = speed.Speed()
            dataset_val.load_dataset(dataset_dir, model.config, "val")

        train(model, dataset_train, dataset_val)

    elif args.command == "test":

        if args.image:
            detect_image(model, args.image)
        elif args.video:
            dataset = urso.Urso()
            dataset.load_dataset(dataset_dir, config, "test")
            detect_video(model, dataset, args.video)
        else:
            # Load validation dataset
            if args.dataset != "speed":
                dataset = urso.Urso()
                dataset.load_dataset(dataset_dir, config, "test")
            else:
                dataset = speed.Speed()
                dataset.load_dataset(dataset_dir, config, "val")

            detect_dataset(model, dataset, 10)

    elif args.command == "evaluate":

        # Test dataset
        if args.dataset != "speed":
            dataset_test = urso.Urso()
            dataset_test.load_dataset(dataset_dir, config, "test")
        else:
            dataset_test = speed.Speed()
            dataset_test.load_dataset(dataset_dir, config, "val")

        evaluate(model, dataset_test)

    elif args.command == "submit":
        assert args.dataset == "speed"
        dataset_real = speed.Speed()
        dataset_real.load_dataset(dataset_dir, config, "real_test")
        dataset_virtual = speed.Speed()
        dataset_virtual.load_dataset(dataset_dir, config, "test")

        test_and_submit(model, dataset_virtual, dataset_real)

    else:
        print("wrong command")


