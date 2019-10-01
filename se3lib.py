"""
Copyright (c) 2019 Pedro F. Proenza
"""

import numpy as np
import math as math

def euler2SO3_unreal(pitch, yaw, roll):
    """Convert euler angles in degrees to a rotation matrix using unreal engine order (valid)"""
    cos_pitch = np.cos(pitch*np.pi/180)
    sin_pitch = np.sin(pitch*np.pi/180)
    cos_yaw = np.cos(yaw*np.pi/180)
    sin_yaw = np.sin(yaw*np.pi/180)
    cos_roll = np.cos(roll*np.pi/180)
    sin_roll = np.sin(roll*np.pi/180)

    R = np.matrix([[cos_pitch * cos_yaw, cos_pitch*sin_yaw, sin_pitch],
                   [sin_roll*sin_pitch*cos_yaw - cos_roll*sin_yaw, sin_roll*sin_pitch*sin_yaw + cos_roll*cos_yaw, -sin_roll*cos_pitch],
                   [-(cos_roll*sin_pitch*cos_yaw+sin_roll*sin_yaw), cos_yaw*sin_roll - cos_roll*sin_pitch*sin_yaw, cos_roll*cos_pitch]])

    return np.transpose(R)

def euler2SO3(pitch, yaw, roll):
    """Convert euler angles in degrees to a rotation matrix using XYZ order"""
    cos_pitch = np.cos(pitch*np.pi/180)
    sin_pitch = np.sin(pitch*np.pi/180)
    cos_yaw = np.cos(yaw*np.pi/180)
    sin_yaw = np.sin(yaw*np.pi/180)
    cos_roll = np.cos(roll*np.pi/180)
    sin_roll = np.sin(roll*np.pi/180)

    R = np.matrix([[cos_yaw * cos_roll, - sin_pitch*sin_yaw*cos_roll - cos_pitch * sin_roll, - cos_pitch * sin_yaw * cos_roll + sin_pitch * sin_roll],
        [cos_yaw * sin_roll, - sin_pitch * sin_yaw * sin_roll + cos_pitch * cos_roll, - cos_pitch * sin_yaw * sin_roll - sin_pitch * cos_roll],
        [sin_yaw, sin_pitch * cos_yaw, cos_pitch * cos_yaw]])

    return R

def euler2SO3_left(pitch, yaw, roll):
    """ Convert euler angles in degrees to a rotation matrix using XYZ order (valid)"""
    cos_pitch = np.cos(pitch*np.pi/180)
    sin_pitch = np.sin(pitch*np.pi/180)
    cos_yaw = np.cos(yaw*np.pi/180)
    sin_yaw = np.sin(yaw*np.pi/180)
    cos_roll = np.cos(roll*np.pi/180)
    sin_roll = np.sin(roll*np.pi/180)

    R = np.matrix([[cos_yaw*cos_roll, sin_pitch*sin_yaw*cos_roll - cos_pitch*sin_roll, cos_pitch*sin_yaw*cos_roll + sin_pitch*sin_roll],
        [cos_yaw*sin_roll, sin_pitch*sin_yaw*sin_roll + cos_pitch*cos_roll, cos_pitch*sin_yaw*sin_roll - sin_pitch*cos_roll],
        [-sin_yaw, sin_pitch*cos_yaw, cos_pitch*cos_yaw]])

    return R

def euler2quat(pitch, yaw, roll):
    """Convert euler angles in degrees to a quaternion"""
    cos_pitch = np.cos(pitch*np.pi/360)
    sin_pitch = np.sin(pitch*np.pi/360)
    cos_yaw = np.cos(yaw*np.pi/360)
    sin_yaw = np.sin(yaw*np.pi/360)
    cos_roll = np.cos(roll*np.pi/360)
    sin_roll = np.sin(roll*np.pi/360)

    q = np.matrix([[sin_yaw*sin_roll*cos_pitch - cos_yaw*cos_roll*sin_pitch],
                   [-sin_yaw*cos_roll*cos_pitch - cos_yaw*sin_roll*sin_pitch],
                   [-cos_yaw*sin_roll*cos_pitch + sin_yaw*cos_roll*sin_pitch],
                   [cos_yaw*cos_roll*cos_pitch + sin_yaw*sin_roll*sin_pitch]])

    return q

def composeSE3(R,t):
    """Takes translation vector and rotation matrix and returns transformation matrix"""
    T = np.matrix([[R[0,0], R[0,1], R[0,2], t[0]],
                   [R[1,0], R[1,1], R[1,2], t[1]],
                   [R[2,0], R[2,1], R[2,2], t[2]],
                   [0, 0, 0, 1]])
    return T

def SO32quat(R):
    """ Convert rotation matrix to left-handed quaternion (JPL convention)
    according to Trawny and Roumeliotis "Indirect kalman filter for 3d attitude estimation"
    """

    q = [0, 0, 0, 0]

    # R trace
    tr = R[0,0] + R[1,1] + R[2,2]

    # At least one of the solutions below is non-singular
    # Singularities arise when S <= 0

    if tr > 0:
        Z = math.sqrt(tr + 1) * 2
        q[3] = 0.25 * Z
        q[0] = (R[1, 2] - R[2, 1]) / Z
        q[1] = (R[2, 0] - R[0, 2]) / Z
        q[2] = (R[0, 1] - R[1, 0]) / Z
    elif (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2]):
        Z = math.sqrt(1.0 + 2 * R[0, 0] - tr) * 2
        q[3] = (R[1, 2] - R[2, 1]) / Z
        q[0] = 0.25 * Z
        q[1] = (R[0, 1] + R[1, 0]) / Z
        q[2] = (R[0, 2] + R[2, 0]) / Z
    elif R[1, 1] > R[2, 2]:
        Z = math.sqrt(1.0 + 2 * R[1, 1] - tr) * 2
        q[3] = (R[2, 0] - R[0, 2]) / Z
        q[0] = (R[0, 1] + R[1, 0]) / Z
        q[1] = 0.25 * Z
        q[2] = (R[1, 2] + R[2, 1]) / Z
    else:
        Z = math.sqrt(1.0 + 2 * R[2, 2] - tr) * 2
        q[3] = (R[0, 1] - R[1, 0]) / Z
        q[0] = (R[0, 2] + R[2, 0]) / Z
        q[1] = (R[1, 2] + R[2, 1]) / Z
        q[2] = 0.25 * Z

    return q

def SO32euler(R):
    """Convert SO3 to a euler angles (Note: there are generally two solutions, this only retrieves one)"""

    # Checks for gymbal lock
    if R[2,0] > 0.998:
        yaw = -np.pi/2
        roll = 0
        pitch = np.arctan2(R[0,1], R[0,2])
    elif R[2,0] < -0.998:
        yaw = np.pi/2
        roll = 0
        pitch = np.arctan2(R[0,1], R[0,2])
    else:
        yaw = np.arcsin(-R[2,0])
        pitch = np.arctan2(R[2,1],R[2,2])
        roll = np.arctan2(R[1,0],R[0,0])
    return pitch*180/np.pi, yaw*180/np.pi, roll*180/np.pi


def quat2SO3(q):
    """ Convert left-handed quaternion (JPL convention) to SO3 according to
        Trawny and Roumeliotis "Indirect kalman filter for 3d attitude estimation" (valid)"""

    R = np.matrix([[1 - 2*q[1]**2 - 2*q[2]**2, 2*(q[0]*q[1] + q[2]*q[3]), 2*(q[0]*q[2] - q[1]*q[3])],
                   [2*(q[0]*q[1] - q[2]*q[3]), 1 - 2*q[0]**2 - 2*q[2]**2, 2*(q[1]*q[2] + q[0]*q[3])],
                   [2*(q[0]*q[2] + q[1]*q[3]), 2*(q[1]*q[2] - q[0]*q[3]), 1 - 2*q[0]**2 - 2*q[1]**2]])

    return R

def quat2angleaxis(q):
    """ Convert quaternion to angle-axis representation"""
    theta = 2 * np.arccos(q[3])

    # Test singularity
    if abs(q[3]) == 1:
        v = [0, 0, 1]
    else:
        den = np.sin(theta/2)
        v = [q[0] / den, q[1] / den, q[2] / den]

    return v, theta

def angleaxis2quat(v,theta):
    """ Convert angle-axis to quaternion"""
    sin_half_theta = np.sin(theta/2)
    return [v[0]*sin_half_theta, v[1]*sin_half_theta, v[2]*sin_half_theta, np.cos(theta/2)]

def quat_mult(a,b):
   """ Multiply 2 quaternions """
   c = np.matrix([[a[3], a[2], -a[1], a[0]],
                  [-a[2], a[3], a[0], a[1]],
                  [a[1], -a[0], a[3], a[2]],
                  [-a[0], -a[1], -a[2], a[3]]])

   if np.shape(b)[0] == 1:
       result = c*b
   else:
       result = b*c.T

    # Enforcing quaternion unit for sanity
   result = result / np.linalg.norm(result)

   return result

def quat_inv(q):
    q_inv = [-q[0], -q[1], -q[2], q[3]]
    return q_inv

def quat2euler(q):
    """ Convert left-handed quaternion to euler angles (X,Y,Z) (valid)"""

    sqx = q[0] * q[0]
    sqy = q[1] * q[1]
    sqz = q[2] * q[2]
    test = q[0]*q[2] + q[1]*q[3]
    if test > 0.499: # singularity at north pole
        pitch = 2 * np.arctan2(q[0], q[3])
        yaw = - np.pi / 2
        roll = 0
    elif test < -0.499: # singularity at south pole
        pitch = -2 * np.arctan2(q[0], q[3])
        yaw = np.pi / 2
        roll = 0
    else:
        pitch = np.arctan2(2*(q[1]*q[2] - q[0]*q[3]), 1-2*sqx-2*sqy)
        yaw = np.arcsin(-2*(q[0]*q[2]+q[1]*q[3]))
        roll = np.arctan2(2*(q[0]*q[1] - q[2]*q[3]), 1-2*sqy-2*sqz)

    # Keeps pitch between [-180, 180] under singularities
    if pitch > np.pi:
        pitch = 2*np.pi - pitch
    if pitch < -np.pi:
        pitch = 2*np.pi + pitch

    return pitch*180/np.pi, yaw*180/np.pi, roll*180/np.pi

def angle_between_quats(q1,q2):

    return 2 * np.arccos(np.clip(np.abs(np.asmatrix(q1) * np.asmatrix(q2).transpose()),0.0,1.0)) * 180 / np.pi

def quat_weighted_avg(Q,W):
    """Compute the average quaternion q of a set of quaternions Q,
    based on a Linear Least Squares Solution of the form: Ax = 0

    The sum of squared dot products between quaternions:
        L(q) = sum_i w_i(Q_i^T*q)^T(Q_i^T*q)^T

    achieves its maximum when its derivative wrt q is zero, i.e.:

        Aq = 0 where A = sum_i (Q_i*Q_i^T)

    Therefore, the optimal q is simply the right null space of A.

    For more solutions check:
    Markley, F. Landis, et al. "Averaging quaternions." Journal of Guidance, Control, and Dynamics (2007)

    Arguments:
        Q: The set of quaternions
        W: The respective weights
    Returns:
        q_avg: The solution
        H_inv: The uncertainty in the maximum likelihood sense

    """

    N = np.size(Q,0)

    # Compute A TODO: vectorize
    A = np.zeros(shape=(4, 4), dtype=np.float32)
    for i in range(N):
        a = np.matrix([Q[i, 0], Q[i, 1], Q[i, 2], Q[i, 3]])
        A += a.transpose() * a * W[i]

    s, v = np.linalg.eig(A)
    idx = np.argsort(s)

    q_avg = v[:, idx[-1]] # 0

    # Due to numerical errors, we need to enforce normalization
    q_avg = q_avg / np.linalg.norm(q_avg)

    H_inv = np.linalg.inv(A)

    return q_avg, H_inv

def rodrigues(x):
    """Rodrigues formula: Converts 3D angle-axis vector to SO3 through exponential map"""

    theta = np.linalg.norm(x)

    if theta<np.finfo(np.float32).eps:
        R = np.eye(3)
    else:
        e = x/theta
        e_skew = [[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]]
        R = np.eye(3) + e_skew*np.sin(theta) + e_skew*np.transpose(e_skew)*(1-np.cos(theta))

    return R


def pose_3Dto3D(P1, P2, t=None):
    """ Closed-form solution to pose from 3D keypoint matches """


    if t is None:
        # Compute centroids
        C1 = np.mean(P1, 1)
        C2 = np.mean(P2, 1)

        # Compute Covariance H
        H = (P1 - C1) * (P2 - C2).T

        # Optimal rotation from SVD(H)
        U, S, Vh = np.linalg.svd(H)
        Aux = np.identity(3)
        Aux[-1, -1] = np.linalg.det(U) * np.linalg.det(Vh.T)
        R = U * Aux * Vh

        # Obtain translation
        t = C2 - R * C1

    else:
        C1 = P1[:,2]
        C2 = t

        P2_shifted = (P2 - C2)

        P2_shifted[:,0] = P2_shifted[:,0]/np.linalg.norm(P2_shifted[:,0])
        P2_shifted[:, 1] = P2_shifted[:, 1] / np.linalg.norm(P2_shifted[:, 1])

        print(P2_shifted)

        # Compute Covariance H
        H = (P1 - C1) * P2_shifted.T

        # Optimal rotation from SVD(H)
        U, S, Vh = np.linalg.svd(H)
        Aux = np.identity(3)
        Aux[-1, -1] = np.linalg.det(U) * np.linalg.det(Vh.T)
        R = U * Aux * Vh

    return t, R





