import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hd3_ops import *


class LossCalculator(object):

    def __init__(self, task):
        assert task in ['flow', 'stereo']
        self.task = task
        self.dim = 1 if task == 'stereo' else 2

    def __call__(self, ms_prob, ms_pred, gt, corr_range, ds=6):
        B, C, H, W = gt.size()
        lv = len(ms_prob)
        criterion = nn.KLDivLoss(reduction='batchmean').cuda()
        losses = {}
        kld_loss = 0
        for l in range(lv):
            scaled_gt, valid_mask = downsample_flow(gt, 1 / 2**(ds - l))
            if self.task == 'stereo':
                scaled_gt = scaled_gt[:, 0, :, :].unsqueeze(1)
            if l > 0:
                scaled_gt = scaled_gt - F.interpolate(
                    ms_pred[l - 1],
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True)
            scaled_gt = scaled_gt / 2**(ds - l)
            gt_dist = vector2density(scaled_gt, corr_range[l],
                                     self.dim) * valid_mask
            kld_loss += 4**(ds - l) / (H * W) * criterion(
                F.log_softmax(ms_prob[l], dim=1), gt_dist.detach())

        losses['total'] = kld_loss
        for loss_type, loss_value in losses.items():
            losses[loss_type] = loss_value.reshape(1)
        return losses


def EndPointError(output, gt):
    # output: [B, 1/2, H, W], stereo or flow prediction
    # gt: [B, C, H, W], 2D ground-truth annotation which may contain a mask
    # NOTE: To benchmark the result, please ensure the ground-truth keeps
    # its ORIGINAL RESOLUTION.
    if output.size(1) == 1:  # stereo
        output = disp2flow(output)
    output = resize_dense_vector(output, gt.size(2), gt.size(3))
    error = torch.norm(output - gt[:, :2, :, :], 2, 1, keepdim=False)
    if gt.size(1) == 3:
        mask = (gt[:, 2, :, :] > 0).float()
    else:
        mask = torch.ones_like(error)
    epe = (error * mask).sum() / mask.sum()
    return epe.reshape(1)
