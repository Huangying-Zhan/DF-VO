''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-07
@LastEditors: Huangying Zhan
@Description: this file contains different correspondence selection methods
'''

import math
import numpy as np


def convert_idx_to_global_coord(local_idx, local_kp_list, x0):
    """Convert pixel index on a local window to global pixel index

    Args: 
        local_idx (list): K indexs of selected pixels 
        local_kp_list (4xN): 
        x0 (list): top-left offset of the local window [y, x]

    Returns:
        coord (array, [4xK]): selected pixels on global image coordinate 
    """
    coord = [local_kp_list[0][local_idx], local_kp_list[1][local_idx], local_kp_list[2][local_idx], local_kp_list[3][local_idx]]
    coord = np.asarray(coord)
    coord[1] += x0[0] # h
    coord[2] += x0[1] # w
    return coord


def bestN_flow_kp(kp1, kp2, ref_data, cfg, outputs):
    """select best-N keypoints with least flow inconsistency
    
    Args:
        kp1 (array, [1xHxWx2]): keypoints on view-1
        kp2 (array, [1xHxWx2]): keypoints on view-2
        ref_data (dict): data of reference view, a dictionary containing

            - **id** (int): index
            - **flow_diff** (array, [HxWx1]): flow difference
        cfg (edict): configuration dictionary
        outputs (dict): output data dictionary
    
    Returns:
        outputs (dict): output data dictionary. New data added

            - **kp1_best** (array, [Nx2]): keypoints on view-1
            - **kp2_best** (array, [Nx2]): keypoints on view-
    """
    bestN_cfg = cfg.kp_selection.bestN

    # initialization
    N = bestN_cfg.num_bestN

    # get data
    flow_diff = np.expand_dims(ref_data['flow_diff'], 0)

    # kp selection
    tmp_kp_list = np.where(flow_diff >= 0) # select all points as intialization
    sel_list = np.argpartition(flow_diff[tmp_kp_list], N)[:N]
    sel_kps = convert_idx_to_global_coord(sel_list, tmp_kp_list, [0, 0])

    kp1_best = kp1[:, sel_kps[1], sel_kps[2]]
    kp2_best = kp2[:, sel_kps[1], sel_kps[2]]

    outputs['kp1_best'] = kp1_best
    outputs['kp2_best'] = kp2_best
    outputs['fb_flow_mask'] = flow_diff[0,:,:,0]
    return outputs


def local_bestN(kp1, kp2, ref_data, cfg, outputs):
    """select best-N filtered keypoints from uniformly divided regions
    
    Args:
        kp1 (array, [1xHxWx2]): keypoints on view-1
        kp2 (array, [1xHxWx2]): keypoints on view-2
        ref_data (dict): data of reference view, a dictionary containing

            - **id** (int): index
            - **flow_diff** (array, [HxWx1]): flow difference
            - **depth_diff** (array, [HxWx1]): depth difference
        cfg (edict): configuration dictionary
        outputs (dict): output data dictionary
    
    Returns:
        outputs (dict): output data dictionary. New data added

            - **kp1_best** (array, [Nx2]): keypoints on view-1
            - **kp2_best** (array, [Nx2]): keypoints on view-2

    """
    # configuration setup
    kp_cfg = cfg.kp_selection
    bestN_cfg = cfg.kp_selection.local_bestN

    # initialization
    num_row = bestN_cfg.num_row
    num_col = bestN_cfg.num_col
    N = bestN_cfg.num_bestN
    score_method = bestN_cfg.score_method
    flow_diff_thre = bestN_cfg.thre
    depth_diff_thre = kp_cfg.depth_consistency.thre
    good_region_cnt = 0

    h, w, _ = ref_data['flow_diff'].shape
    
    outputs['kp1_best'] = {}
    outputs['kp2_best'] = {}

    n_best = math.floor(N/(num_col*num_row))
    sel_kps = []

    # get data
    flow_diff = np.expand_dims(ref_data['flow_diff'], 0)
    if kp_cfg.depth_consistency.enable:
        depth_diff = ref_data['depth_diff'].reshape(1, h, w, 1)
    
    # Insufficent keypoint case 1
    if (flow_diff[0,:,:,0] < flow_diff_thre).sum() < N * 0.1 :
        print("Cannot find enough good keypoints!")
        outputs['good_kp_found'] = False
        return outputs

    for row in range(num_row):
        for col in range(num_col):
            x0 = [int(h/num_row*row), int(w/num_col*col)] # top_left
            x1 = [int(h/num_row*(row+1))-1, int(w/num_col*(col+1))-1] # bottom right

            # computing masks
            tmp_flow_diff = flow_diff[:, x0[0]:x1[0], x0[1]:x1[1]].copy()

            if score_method == "flow":
                flow_mask = tmp_flow_diff < flow_diff_thre
            elif score_method == "flow_ratio":
                tmp_flow = np.expand_dims(ref_data['flow'][:, x0[0]:x1[0], x0[1]:x1[1]], 0)
                tmp_flow = np.transpose(tmp_flow, (0, 2, 3, 1))
                tmp_flow_diff_ratio = tmp_flow_diff / np.linalg.norm(tmp_flow, axis=3, keepdims=True)
                flow_mask = tmp_flow_diff_ratio < flow_diff_thre

            valid_mask = flow_mask

            if kp_cfg.depth_consistency.enable:
                tmp_depth_diff = depth_diff[:, x0[0]:x1[0],x0[1]:x1[1]].copy()
                depth_mask = tmp_depth_diff < depth_diff_thre
                valid_mask *= depth_mask
            
            # computing scores
            if score_method == 'flow':
                score = tmp_flow_diff
            elif score_method == 'flow_depth':
                score = tmp_flow_diff * tmp_depth_diff
            elif score_method == 'flow_ratio':
                score = tmp_flow_diff_ratio

            # kp selection
            tmp_kp_list = np.where(valid_mask)

            num_to_pick = min(n_best, len(tmp_kp_list[0]))

            if num_to_pick != 0:
                good_region_cnt += 1
            
            if num_to_pick <= n_best:
                sel_list = np.argpartition(score[tmp_kp_list], num_to_pick-1)[:num_to_pick]
            else:
                sel_list = np.argpartition(score[tmp_kp_list], num_to_pick)[:num_to_pick]
            
            sel_global_coords = convert_idx_to_global_coord(sel_list, tmp_kp_list, x0)
            for i in range(sel_global_coords.shape[1]):
                sel_kps.append(sel_global_coords[:, i:i+1])
    
    # Insufficent keypoint case 2
    if good_region_cnt < (num_row * num_col) * 0.1:
        print("Cannot find enough good keypoints from diversed regions!")
        outputs['good_kp_found'] = False
        return outputs

    # reshape selected keypoints
    sel_kps = np.asarray(sel_kps)
    sel_kps = np.transpose(sel_kps, (1, 0, 2))
    sel_kps = np.reshape(sel_kps, (4, -1))

    kp1_best = kp1[:, sel_kps[1], sel_kps[2]]
    kp2_best = kp2[:, sel_kps[1], sel_kps[2]]

    outputs['kp1_best'] = kp1_best
    outputs['kp2_best'] = kp2_best

    # mask generation
    if score_method == 'flow_ratio':
        flow = np.expand_dims(ref_data['flow'], 0)
        flow = np.transpose(flow, (0, 2, 3, 1))
        flow_diff_ratio = flow_diff / np.linalg.norm(flow, axis=3, keepdims=True)
        outputs['fb_flow_mask'] = flow_diff_ratio[0,:,:,0]
    elif score_method == 'flow':
        outputs['fb_flow_mask'] = flow_diff[0,:,:,0]
    return outputs


def opt_rigid_flow_kp(kp1, kp2, ref_data, cfg, outputs, score_method):
    """select best-N filtered keypoints from uniformly divided regions 
    with rigid-flow mask
    
    Args:
        kp1 (array, [1xHxWx2]): keypoints on view-1
        kp2 (array, [1xHxWx2]): keypoints on view-2
        ref_data (dict):

            - **rigid_flow_diff** (array, [HxWx1]): rigid-optical flow consistency
            - **flow_diff** (array, [HxWx1]): forward-backward flow consistency
        cfg (edict): configuration dictionary
        outputs (dict): output data 
        method (str): [uniform, best]
        score_method (str): [opt_flow, rigid_flow]
    
    Returns:
        outputs (dict): output data with the new data

            - **kp1_depth** (array, [Nx2]): keypoints in view-1, best in terms of score_method
            - **kp2_depth** (array, [Nx2]): keypoints in view-2, best in terms of score_method
            - **kp1_depth_uniform** (array, [Nx2]): keypoints in view-1, uniformly sampled
            - **kp2_depth_uniform** (array, [Nx2]): keypoints in view-2, uniformly sampled
            - **rigid_flow_mask** (array, [HxW]): rigid-optical flow consistency 
    """
    kp_cfg = cfg.kp_selection
    bestN_cfg = cfg.kp_selection.rigid_flow_kp

    # initialization
    num_row = bestN_cfg.num_row
    num_col = bestN_cfg.num_col
    N = bestN_cfg.num_bestN
    rigid_flow_diff_thre = kp_cfg.rigid_flow_kp.rigid_flow_thre
    opt_flow_diff_thre = kp_cfg.rigid_flow_kp.optical_flow_thre

    n_best = math.floor(N/(num_col*num_row))
    sel_kps = []
    sel_kps_uniform = []

    # get data
    # flow diff
    rigid_flow_diff = ref_data['rigid_flow_diff']
    rigid_flow_diff = np.expand_dims(rigid_flow_diff, 0)
    _, h, w, _ = rigid_flow_diff.shape

    # optical flow diff
    opt_flow_diff = np.expand_dims(ref_data['flow_diff'], 0)

    for row in range(num_row):
        for col in range(num_col):
            x0 = [int(h/num_row*row), int(w/num_col*col)] # top_left
            x1 = [int(h/num_row*(row+1))-1, int(w/num_col*(col+1))-1] # bottom right

            # computing masks
            tmp_opt_flow_diff = opt_flow_diff[:, x0[0]:x1[0], x0[1]:x1[1]].copy()

            tmp_rigid_flow_diff = rigid_flow_diff[:, x0[0]:x1[0], x0[1]:x1[1]].copy()
            flow_mask = (tmp_rigid_flow_diff < rigid_flow_diff_thre) 
            
            flow_mask = flow_mask * (tmp_opt_flow_diff < opt_flow_diff_thre)
            valid_mask = flow_mask

            # computing scores
            if score_method == "rigid_flow":
                score = tmp_rigid_flow_diff
            elif score_method == "opt_flow":
                score = tmp_opt_flow_diff

            # kp selection
            tmp_kp_list = np.where(valid_mask)
            num_to_pick = min(n_best, len(tmp_kp_list[0]))
            
            # Pick uniform kps
            # if method == 'uniform':
            if num_to_pick > 0:
                step = int(len(tmp_kp_list[0]) / (num_to_pick))
                sel_list = np.arange(0, len(tmp_kp_list[0]), step)[:num_to_pick]
            else:
                sel_list = []
            sel_global_coords = convert_idx_to_global_coord(sel_list, tmp_kp_list, x0)
            for i in range(sel_global_coords.shape[1]):
                sel_kps_uniform.append(sel_global_coords[:, i:i+1])

            # elif method == 'best':
            if num_to_pick <= n_best:
                sel_list = np.argpartition(score[tmp_kp_list], num_to_pick-1)[:num_to_pick]
            else:
                sel_list = np.argpartition(score[tmp_kp_list], num_to_pick)[:num_to_pick]

            sel_global_coords = convert_idx_to_global_coord(sel_list, tmp_kp_list, x0)
            for i in range(sel_global_coords.shape[1]):
                sel_kps.append(sel_global_coords[:, i:i+1])

    # best
    sel_kps = np.asarray(sel_kps)
    assert sel_kps.shape[0]!=0, "sampling threshold is too small."
    sel_kps = np.transpose(sel_kps, (1, 0, 2))
    sel_kps = np.reshape(sel_kps, (4, -1))

    kp1_best = kp1[:, sel_kps[1], sel_kps[2]]
    kp2_best = kp2[:, sel_kps[1], sel_kps[2]]

    outputs['kp1_depth'] = kp1_best.copy()
    outputs['kp2_depth'] = kp2_best.copy()

    # uniform
    sel_kps = np.asarray(sel_kps_uniform)
    assert sel_kps.shape[0]!=0, "sampling threshold is too small."
    sel_kps = np.transpose(sel_kps, (1, 0, 2))
    sel_kps = np.reshape(sel_kps, (4, -1))

    kp1_best = kp1[:, sel_kps[1], sel_kps[2]]
    kp2_best = kp2[:, sel_kps[1], sel_kps[2]]

    outputs['kp1_depth_uniform'] = kp1_best.copy()
    outputs['kp2_depth_uniform'] = kp2_best.copy()



    # mask generation
    outputs['rigid_flow_mask'] = rigid_flow_diff[0,:,:,0]
    return outputs


def sampled_kp(kp1, kp2, ref_data, kp_list, cfg, outputs):
    """select sampled keypoints with given keypoint index list
    
    Args:
        kp1 (array, [1xHxWx2]): keypoints on view-1
        kp2 (array, [1xHxWx2]): keypoints on view-2
        ref_data (dict): data of reference view, a dictionary containing
        kp_list (list): list of keypoint index
        cfg (edict): configuration dictionary
        outputs (dict): output data dictionary
    
    Returns:
        outputs (dict): output data dictionary. New data added

            - **kp1_list** (array, [Nx2]): keypoints on view-1
            - **kp2_list** (array, [Nx2]): keypoints on view-2
    """
    kp_cfg = cfg.kp_selection
    img_crop = cfg.crop.flow_crop

    # initialization
    h, w = ref_data['depth'].shape
    n = 1

    outputs['kp1_list'] = {}
    outputs['kp2_list'] = {}

    y0, y1 = 0, h
    x0, x1 = 0, w

    # Get uniform sampled keypoints
    if img_crop is not None:
        y0, y1 = int(h*img_crop[0][0]), int(h*img_crop[0][1])
        x0, x1 = int(w*img_crop[1][0]), int(w*img_crop[1][1])

        kp1 = kp1[:, y0:y1, x0:x1]
        kp2 = kp2[:, y0:y1, x0:x1]

    kp1_list = kp1.reshape(n, -1, 2)
    kp2_list = kp2.reshape(n, -1, 2)
    kp1_list = np.transpose(kp1_list, (1,0,2))
    kp2_list = np.transpose(kp2_list, (1,0,2))

    # select kp from sampled kp_list
    kp1_list = kp1_list[kp_list]
    kp2_list = kp2_list[kp_list]
    kp1_list = np.transpose(kp1_list, (1,0,2))
    kp2_list = np.transpose(kp2_list, (1,0,2))

    outputs['kp1_list'] = kp1_list
    outputs['kp2_list'] = kp2_list
    return outputs
