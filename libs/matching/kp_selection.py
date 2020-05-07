# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file 
# which allows for non-commercial use only.

import math
import numpy as np

def convert_idx_to_global_coord(local_idx, local_kp_list, x0):
        coord = [local_kp_list[0][local_idx], local_kp_list[1][local_idx], local_kp_list[2][local_idx], local_kp_list[3][local_idx]]
        coord = np.asarray(coord)
        coord[1] += x0[0] # h
        coord[2] += x0[1] # w
        return coord


def uniform_filtered_bestN(kp1, kp2, ref_data, cfg, outputs):
    """select best-N filtered kps uniformly
    Args:
        kp1 (1xHxWx2)
        kp2 (1xHxWx2)
        cur_data (dict):
            -
        ref_data (dict):
            - flow_diff
            - depth_diff
        cfg (edict): cfg
        outputs (dict)
    Returns:
        outputs (dict)
    """
    kp_cfg = cfg.kp_selection
    bestN_cfg = cfg.kp_selection.uniform_filtered_bestN

    # initialization
    num_row = bestN_cfg.num_row
    num_col = bestN_cfg.num_col
    N = bestN_cfg.num_bestN
    score_method = bestN_cfg.score_method
    flow_diff_thre = kp_cfg.flow_consistency.thre
    depth_diff_thre = kp_cfg.depth_consistency.thre

    h, w = ref_data['depth'][ref_data['id'][0]].shape

    outputs['kp1_best'] = {}
    outputs['kp2_best'] = {}

    n_best = math.floor(N/(num_col*num_row))
    sel_kps = []

    for ref_id in ref_data['id']:
        # get data
        flow_diff = np.expand_dims(ref_data['flow_diff'][ref_id], 0)
        if kp_cfg.depth_consistency.enable:
            depth_diff = ref_data['depth_diff'][ref_id].reshape(1, h, w, 1)

        for row in range(num_row):
            for col in range(num_col):
                x0 = [int(h/num_row*row), int(w/num_col*col)] # top_left
                x1 = [int(h/num_row*(row+1))-1, int(w/num_col*(col+1))-1] # bottom right

                # computing masks
                tmp_flow_diff = flow_diff[:, x0[0]:x1[0], x0[1]:x1[1]].copy()
                flow_mask = tmp_flow_diff < flow_diff_thre
                valid_mask = flow_mask

                if kp_cfg.depth_consistency.enable:
                    tmp_depth_diff = depth_diff[:, x0[0]:x1[0],x0[1]:x1[1]].copy()
                    depth_mask = tmp_depth_diff < depth_diff_thre
                    valid_mask *= depth_mask
                
                # computing scores
                if score_method == "flow":
                    score = tmp_flow_diff
                elif score_method == "flow_depth":
                    score = tmp_flow_diff * tmp_depth_diff

                # kp selection
                tmp_kp_list = np.where(valid_mask)

                num_to_pick = min(n_best, len(tmp_kp_list[0]))
                if num_to_pick <= n_best:
                    sel_list = np.argpartition(score[tmp_kp_list], num_to_pick-1)[:num_to_pick]
                else:
                    sel_list = np.argpartition(score[tmp_kp_list], num_to_pick)[:num_to_pick]
                
                sel_global_coords = convert_idx_to_global_coord(sel_list, tmp_kp_list, x0)
                for i in range(sel_global_coords.shape[1]):
                    sel_kps.append(sel_global_coords[:, i:i+1])

        # DEBUG
        # obj = (ref_data['raw_depth'][ref_data['id'][0]])
        # norm_depth_weight = (obj - obj.min()) / (obj.max() - obj.min())
        # obj = flow_diff[0,:,:,0]
        # norm_flow_diff = (obj - obj.min()) / (obj.max()-obj.min())
        # obj = depth_diff[0,:,:,0]
        # norm_depth_diff = (obj - obj.min()) / (obj.max()-obj.min())
        # score_full = norm_depth_weight * norm_depth_diff * norm_flow_diff
        # from matplotlib import pyplot as plt
        # f, ax = plt.subplots(4,1)
        # ax[0].imshow(score_full, vmin=0, vmax=0.0001)
        # ax[1].imshow(norm_depth_weight, vmin=0, vmax=0.1)
        # ax[2].imshow(norm_flow_diff, vmin=0, vmax=0.1)
        # ax[3].imshow(norm_depth_diff, vmin=0, vmax=0.1)
        # plt.show()

        sel_kps = np.asarray(sel_kps)
        assert sel_kps.shape[0]!=0, "sampling threshold is too small."
        sel_kps = np.transpose(sel_kps, (1, 0, 2))
        sel_kps = np.reshape(sel_kps, (4, -1))

        kp1_best = kp1[:, sel_kps[1], sel_kps[2]]
        kp2_best = kp2[:, sel_kps[1], sel_kps[2]]

        outputs['kp1_best'] = kp1_best
        outputs['kp2_best'][ref_id] = kp2_best

        # mask generation
        outputs['flow_mask'] = flow_diff[0,:,:,0]
        valid_mask = (flow_diff < flow_diff_thre) * 1

        if kp_cfg.depth_consistency.enable:
            outputs['depth_mask'] = depth_diff[0,:,:,0]
            valid_mask = (flow_diff < flow_diff_thre) * (depth_diff < depth_diff_thre) * 1
        valid_mask = valid_mask[0,:,:,0]
        outputs['valid_mask'] = flow_diff[0,:,:,0] * (1/(valid_mask+1e-4))
    return outputs


def rigid_flow_kp(kp1, kp2, ref_data, cfg, outputs):
    """select best-N filtered kps with rigid-flow mask
    Args:
        kp1 (1xHxWx2)
        kp2 (1xHxWx2)
        cur_data (dict):
            -
        ref_data (dict):
            - flow_diff
            - depth_diff
        cfg (edict): cfg
        outputs (dict)
    Returns:
        outputs (dict)
    """
    kp_cfg = cfg.kp_selection
    bestN_cfg = cfg.kp_selection.rigid_flow_kp

    # initialization
    num_row = bestN_cfg.num_row
    num_col = bestN_cfg.num_col
    N = bestN_cfg.num_bestN
    score_method = bestN_cfg.score_method
    flow_diff_thre = kp_cfg.rigid_flow_kp.thre
    # depth_diff_thre = kp_cfg.depth_consistency.thre

    h, w = ref_data['depth'][ref_data['id'][0]].shape

    outputs['kp1_depth'] = {}
    outputs['kp2_depth'] = {}

    n_best = math.floor(N/(num_col*num_row))
    sel_kps = []

    for ref_id in ref_data['id']:
        # get data

        # flow ratio diff
        # flow_diff = ref_data['rigid_flow_diff'][ref_id] / np.expand_dims(np.linalg.norm(ref_data['flow'][ref_id], axis=0)**2, 2)
        # flow_diff = flow_diff * ref_data['rigid_flow_diff'][ref_id] 

        # flow diff
        flow_diff = ref_data['rigid_flow_diff'][ref_id]
        flow_diff = np.expand_dims(flow_diff, 0)

        # optical flow diff
        opt_flow_diff = np.expand_dims(ref_data['flow_diff'][ref_id], 0)

        for row in range(num_row):
            for col in range(num_col):
                x0 = [int(h/num_row*row), int(w/num_col*col)] # top_left
                x1 = [int(h/num_row*(row+1))-1, int(w/num_col*(col+1))-1] # bottom right

                # computing masks
                tmp_opt_flow_diff = opt_flow_diff[:, x0[0]:x1[0], x0[1]:x1[1]].copy()

                tmp_flow_diff = flow_diff[:, x0[0]:x1[0], x0[1]:x1[1]].copy()
                flow_mask = (tmp_flow_diff < flow_diff_thre) 
                
                flow_mask = flow_mask * (tmp_opt_flow_diff < 1)
                valid_mask = flow_mask

                # computing scores
                if score_method == "flow":
                    score = tmp_flow_diff #+ tmp_opt_flow_diff

                # kp selection
                tmp_kp_list = np.where(valid_mask)
                num_to_pick = min(n_best, len(tmp_kp_list[0]))
                
                # Pick random kps
                if num_to_pick > 0:
                    step = int(len(tmp_kp_list[0]) / (num_to_pick))
                    sel_list = np.arange(0, len(tmp_kp_list[0]), step)[:num_to_pick]
                else:
                    sel_list = []

                # Pick best depth consistent points
                # if num_to_pick <= n_best:
                #     sel_list = np.argpartition(score[tmp_kp_list], num_to_pick-1)[:num_to_pick]
                # else:
                #     sel_list = np.argpartition(score[tmp_kp_list], num_to_pick)[:num_to_pick]
                
                sel_global_coords = convert_idx_to_global_coord(sel_list, tmp_kp_list, x0)
                for i in range(sel_global_coords.shape[1]):
                    sel_kps.append(sel_global_coords[:, i:i+1])

        sel_kps = np.asarray(sel_kps)
        assert sel_kps.shape[0]!=0, "sampling threshold is too small."
        sel_kps = np.transpose(sel_kps, (1, 0, 2))
        sel_kps = np.reshape(sel_kps, (4, -1))

        kp1_best = kp1[:, sel_kps[1], sel_kps[2]]
        kp2_best = kp2[:, sel_kps[1], sel_kps[2]]

        outputs['kp1_depth'] = kp1_best
        outputs['kp2_depth'][ref_id] = kp2_best

        # mask generation
        outputs['rigid_flow_mask'] = flow_diff[0,:,:,0]
        # valid_mask = (flow_diff < flow_diff_thre) * 1

        # if kp_cfg.depth_consistency.enable:
        #     outputs['depth_mask'] = depth_diff[0,:,:,0]
        #     valid_mask = (flow_diff < flow_diff_thre) * (depth_diff < depth_diff_thre) * 1
        # valid_mask = valid_mask[0,:,:,0]
        # outputs['valid_mask'] = flow_diff[0,:,:,0] * (1/(valid_mask+1e-4))
    return outputs


def opt_rigid_flow_kp(kp1, kp2, ref_data, cfg, outputs):
    """select best-N filtered kps with good optical-rigid flow consistency
    Args:
        kp1 (1xHxWx2)
        kp2 (1xHxWx2)
        cur_data (dict):
            -
        ref_data (dict):
            - flow_diff
            - depth_diff
        cfg (edict): cfg
        outputs (dict)
    Returns:
        outputs (dict)
    """
    kp_cfg = cfg.kp_selection
    bestN_cfg = cfg.kp_selection.rigid_flow_kp

    # initialization
    num_row = bestN_cfg.num_row
    num_col = bestN_cfg.num_col
    N = bestN_cfg.num_bestN
    score_method = bestN_cfg.score_method
    rigid_flow_diff_thre = kp_cfg.rigid_flow_kp.thre
    opt_flow_diff_thre = kp_cfg.flow_consistency.thre

    # depth_diff_thre = kp_cfg.depth_consistency.thre

    h, w = ref_data['depth'][ref_data['id'][0]].shape

    outputs['kp1_rigid'] = {}
    outputs['kp2_rigid'] = {}

    n_best = math.floor(N/(num_col*num_row))
    sel_kps = []

    for ref_id in ref_data['id']:
        # flow diff
        rigid_flow_diff = ref_data['rigid_flow_diff'][ref_id]
        rigid_flow_diff = np.expand_dims(rigid_flow_diff, 0)

        # optical flow diff
        opt_flow_diff = np.expand_dims(ref_data['flow_diff'][ref_id], 0)

        for row in range(num_row):
            for col in range(num_col):
                x0 = [int(h/num_row*row), int(w/num_col*col)] # top_left
                x1 = [int(h/num_row*(row+1))-1, int(w/num_col*(col+1))-1] # bottom right

                # computing masks
                tmp_opt_flow_diff = opt_flow_diff[:, x0[0]:x1[0], x0[1]:x1[1]].copy()

                tmp_flow_diff = rigid_flow_diff[:, x0[0]:x1[0], x0[1]:x1[1]].copy()
                flow_mask = (tmp_flow_diff < rigid_flow_diff_thre) 
                
                flow_mask = flow_mask * (tmp_opt_flow_diff < opt_flow_diff_thre)
                valid_mask = flow_mask

                # computing scores
                if score_method == "flow":
                    score = tmp_flow_diff #+ tmp_opt_flow_diff

                # kp selection
                tmp_kp_list = np.where(valid_mask)
                num_to_pick = min(n_best, len(tmp_kp_list[0]))
                
                if num_to_pick <= n_best:
                    sel_list = np.argpartition(score[tmp_kp_list], num_to_pick-1)[:num_to_pick]
                else:
                    sel_list = np.argpartition(score[tmp_kp_list], num_to_pick)[:num_to_pick]

                sel_global_coords = convert_idx_to_global_coord(sel_list, tmp_kp_list, x0)
                for i in range(sel_global_coords.shape[1]):
                    sel_kps.append(sel_global_coords[:, i:i+1])

        sel_kps = np.asarray(sel_kps)
        assert sel_kps.shape[0]!=0, "sampling threshold is too small."
        sel_kps = np.transpose(sel_kps, (1, 0, 2))
        sel_kps = np.reshape(sel_kps, (4, -1))

        kp1_best = kp1[:, sel_kps[1], sel_kps[2]]
        kp2_best = kp2[:, sel_kps[1], sel_kps[2]]

        outputs['kp1_rigid'] = kp1_best
        outputs['kp2_rigid'][ref_id] = kp2_best

        # mask generation
        outputs['rigid_flow_mask'] = rigid_flow_diff[0,:,:,0]
        outputs['flow_mask'] = opt_flow_diff[0,:,:,0]
        # valid_mask = (flow_diff < flow_diff_thre) * 1

        # if kp_cfg.depth_consistency.enable:
        #     outputs['depth_mask'] = depth_diff[0,:,:,0]
        #     valid_mask = (flow_diff < flow_diff_thre) * (depth_diff < depth_diff_thre) * 1
        # valid_mask = valid_mask[0,:,:,0]
        # outputs['valid_mask'] = flow_diff[0,:,:,0] * (1/(valid_mask+1e-4))
    return outputs



def bestN_flow_kp(kp1, kp2, ref_data, cfg, outputs):
    """select best-N kps
    Args:
        kp1 (1xHxWx2)
        kp2 (1xHxWx2)
        ref_data (dict):
            - flow_diff
        cfg (edict): cfg.kp_selection
        outputs (dict)
    Returns:
        outputs (dict)
    """
    bestN_cfg = cfg.kp_selection.bestN

    # initialization
    N = bestN_cfg.num_bestN

    h, w = ref_data['depth'][ref_data['id'][0]].shape

    outputs['kp1_best'] = {}
    outputs['kp2_best'] = {}

    for ref_id in ref_data['id']:
        # get data
        flow_diff = np.expand_dims(ref_data['flow_diff'][ref_id], 0)

        # kp selection
        tmp_kp_list = np.where(flow_diff >= 0) # select all points as intialization
        sel_list = np.argpartition(flow_diff[tmp_kp_list], N)[:N]
        sel_kps = convert_idx_to_global_coord(sel_list, tmp_kp_list, [0, 0])

        kp1_best = kp1[:, sel_kps[1], sel_kps[2]]
        kp2_best = kp2[:, sel_kps[1], sel_kps[2]]

        outputs['kp1_best'] = kp1_best
        outputs['kp2_best'][ref_id] = kp2_best
        outputs['flow_mask'] = flow_diff[0,:,:,0]
    return outputs


def sampled_kp(kp1, kp2, ref_data, kp_list, cfg, outputs):
    """select sampled kps
    Args:
        kp1 (1xHxWx2)
        kp2 (1xHxWx2)
        ref_data (dict):
        kp_list (int list): list of keypoint index
        cfg (edict): cfg
        outputs (dict)
    Returns:
        outputs (dict)
    """
    kp_cfg = cfg.kp_selection
    sampled_cfg = kp_cfg.sampled_kp
    img_crop = cfg.crop.flow_crop

    # initialization
    h, w = ref_data['depth'][ref_data['id'][0]].shape
    n = 1

    outputs['kp1_list'] = {}
    outputs['kp2_list'] = {}

    y0, y1 = 0, h
    x0, x1 = 0, w
    for ref_id in ref_data['id']:
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
        outputs['kp2_list'][ref_id] = kp2_list
    return outputs

