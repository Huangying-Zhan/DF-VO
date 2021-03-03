''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-03-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-27
@LastEditors: Huangying Zhan
@Description: This file contains functions related to GRIC computation
'''

import numpy as np


def compute_fundamental_residual(F, kp1, kp2):
    """ 
    Compute fundamental matrix residual

    Args:
        F (array, [3x3]): Fundamental matrix (from view-1 to view-2)
        kp1 (array, [Nx2]): keypoint 1
        kp2 (array, [Nx2]): keypoint 2
    
    Returns:
        res (array, [N]): residual
    """
    # get homogeneous keypoints (3xN array)
    m0 = np.ones((3, kp1.shape[0]))
    m0[:2] = np.transpose(kp1, (1,0))
    m1 = np.ones((3, kp2.shape[0]))
    m1[:2] = np.transpose(kp2, (1,0))

    Fm0 = F @ m0 #3xN
    Ftm1 = F.T @ m1 #3xN

    m1Fm0 = (np.transpose(Fm0, (1,0)) @ m1).diagonal()
    res = m1Fm0**2 / (np.sum(Fm0[:2]**2, axis=0) + np.sum(Ftm1[:2]**2, axis=0))
    return res


def compute_homography_residual(H_in, kp1, kp2):
    """ 
    Compute homography matrix residual

    Args:
        H (array, [3x3]): homography matrix (Transformation from view-1 to view-2)
        kp1 (array, [Nx2]): keypoint 1
        kp2 (array, [Nx2]): keypoint 2
    
    Returns:
        res (array, [N]): residual
    """
    n = kp1.shape[0]
    H = H_in.flatten()

    # get homogeneous keypoints (3xN array)
    m0 = np.ones((3, kp1.shape[0]))
    m0[:2] = np.transpose(kp1, (1,0))
    m1 = np.ones((3, kp2.shape[0]))
    m1[:2] = np.transpose(kp2, (1,0))


    G0 = np.zeros((3, n))
    G1 = np.zeros((3, n))

    G0[0]= H[0] - m1[0] * H[6]
    G0[1]= H[1] - m1[0] * H[7]
    G0[2]=-m0[0] * H[6] - m0[1] * H[7] - H[8]

    G1[0]= H[3] - m1[1] * H[6]
    G1[1]= H[4] - m1[1] * H[7]
    G1[2]=-m0[0] * H[6] - m0[1] * H[7] - H[8]

    magG0=np.sqrt(G0[0]*G0[0] + G0[1]*G0[1] + G0[2]*G0[2])
    magG1=np.sqrt(G1[0]*G1[0] + G1[1]*G1[1] + G1[2]*G1[2])
    magG0G1=G0[0]*G1[0] + G0[1]*G1[1]

    alpha=np.arccos(magG0G1 /(magG0*magG1))

    alg = np.zeros((2, n))
    alg[0]=   m0[0]*H[0] + m0[1]*H[1] + H[2] - \
       m1[0]*(m0[0]*H[6] + m0[1]*H[7] + H[8])
    
    alg[1]=   m0[0]*H[3] + m0[1]*H[4] + H[5] - \
       m1[1]*(m0[0]*H[6] + m0[1]*H[7] + H[8])
    
    D1=alg[0]/magG0
    D2=alg[1]/magG1

    res = (D1*D1 + D2*D2 - 2.0*D1*D2*np.cos(alpha))/np.sin(alpha)
   
    return res


def calc_GRIC(res, sigma, n, model):
    """Calculate GRIC

    Args:
        res (array, [N]): residual
        sigma (float): assumed variance of the error
        n (int): number of residuals
        model (str): model type
            - FMat
            - EMat
            - HMat
    """
    R = 4
    sigmasq1 = 1./ sigma**2 

    K = {
        "FMat": 7,
        "EMat": 5,
        "HMat": 8,
    }[model]
    D = {
        "FMat": 3,
        "EMat": 3,
        "HMat": 2,
    }[model]
    
    lam3RD=2.0 * (R-D)

    sum_ = 0
    for i in range(n):
        tmp=res[i] * sigmasq1
        if tmp<=lam3RD:
            sum_ += tmp
        else:
            sum_ += lam3RD
    
    sum_ += n * D * np.log(R) + K * np.log(R*n)

    return sum_
