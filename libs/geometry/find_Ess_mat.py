import cv2

def find_Ess_mat(inputs):
    # inputs
    kp_cur = inputs['kp_cur']
    kp_ref = inputs['kp_ref']
    H_inliers = inputs['H_inliers']
    cfg = inputs['cfg']
    cam_intrinsics = inputs['cam_intrinsics']

    # initialization
    valid_cfg = cfg.compute_2d2d_pose.validity
    principal_points = (cam_intrinsics.cx, cam_intrinsics.cy)
    fx = cam_intrinsics.fx

    # compute Ess
    E, inliers = cv2.findEssentialMat(
                        kp_cur,
                        kp_ref,
                        focal=fx,
                        pp=principal_points,
                        method=cv2.RANSAC,
                        prob=0.99,
                        threshold=cfg.compute_2d2d_pose.ransac.reproj_thre,
                        )
    # check homography inlier ratio
    if valid_cfg.method == "homo_ratio":
        H_inliers_ratio = H_inliers.sum()/(H_inliers.sum()+inliers.sum())
        valid_case = H_inliers_ratio < valid_cfg.thre
    elif valid_cfg.method == "flow":
        cheirality_cnt, R, t, _ = cv2.recoverPose(E, kp_cur, kp_ref,
                                focal=cam_intrinsics.fx,
                                pp=principal_points,)
        valid_case = cheirality_cnt > kp_cur.shape[0]*0.05
    
    # gather output
    outputs = {}
    outputs['E'] = E
    outputs['valid_case'] = valid_case
    outputs['inlier_cnt'] = inliers.sum()
    outputs['inlier'] = inliers

    return outputs

