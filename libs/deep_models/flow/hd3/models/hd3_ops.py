import math
import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    'flow_warp', 'vector2density', 'density2vector', 'prob_gather',
    'disp2flow', 'downsample_flow', 'resize_dense_vector'
]


def flow_warp(x, flo, mul=True):
    """
    inverse warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(x.device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(x.device)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid = torch.stack([
        2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0,
        2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    ],
                        dim=1)

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, padding_mode='border')
    mask = torch.ones(x.size(), device=x.device)
    mask = F.grid_sample(mask, vgrid, padding_mode='zeros')

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    if mul:
        return output * mask
    else:
        return output, mask


def vector2density(vect, c, dim):
    # vect: point estimate
    # c: support size
    # dim: dimension (stereo: 1 / flow: 2)
    assert vect.size(1) == dim
    if dim == 2:
        return _flow2distribution(vect, c)
    else:
        dist = _flow2distribution(disp2flow(vect), c)
        return dist[:, c * (2 * c + 1):(c + 1) * (2 * c + 1), :, :]


def density2vector(prob, dim, normalize=True):
    # prob: density
    # dim: dimension (stereo: 1 / flow: 2)
    # normalize: if true, prob will be normalized via softmax
    if dim == 2:
        flow = _prob2flow(prob, normalize)
        return flow
    else:
        prob_padded = _disp_prob2flow_prob(prob, normalize)
        flow = _prob2flow(prob_padded, False)
        disp = flow[:, 0, :, :].unsqueeze(1)
        return disp


def prob_gather(prob, normalize=True, dim=2, return_indices=False):
    # gather probability for confidence map visualization
    # return shape: out, indice [B,1,H,W]
    if normalize:
        prob = F.softmax(prob, dim=1)
    if dim == 1:
        prob = _disp_prob2flow_prob(prob, False)
    B, C, H, W = prob.size()
    d = int(math.sqrt(C))
    pr = prob.reshape(B, d, d, -1).permute(0, 3, 1, 2)
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    max_pool = nn.MaxPool2d(kernel_size=d - 1, stride=1, return_indices=True)
    out, indice = max_pool(avg_pool(pr))
    out = 4 * out.squeeze().reshape(B, 1, H, W)
    if not return_indices:
        return out
    else:
        indice += indice / (d - 1)
        indice = indice.squeeze().reshape(B, 1, H, W)
        return out, indice


def disp2flow(disp):
    assert disp.size(1) == 1
    padder = torch.zeros(disp.size(), device=disp.device)
    return torch.cat([disp, padder], dim=1)


def downsample_flow(flo, scale_factor):
    # note the value is not scaled accordingly here
    # return downsampled flow field and valid mask
    assert scale_factor <= 1
    B, C, H, W = flo.size()
    if flo.size(1) == 2:
        # dense format
        flo = F.interpolate(
            flo,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=True)
        mask = torch.ones((B, 1, int(H * scale_factor), int(W * scale_factor)),
                          dtype=torch.float,
                          device=flo.device)
    else:
        # sparse format
        flo = F.avg_pool2d(flo, int(1 / scale_factor))
        mask = (flo[:, 2, :, :].unsqueeze(1) > 0).float()
        flo = flo[:, :2, :, :] / (flo[:, 2, :, :].unsqueeze(1) + 1e-9)
    return flo, mask


def resize_dense_vector(vec, des_height, des_width):
    ratio_height = float(des_height / vec.size(2))
    ratio_width = float(des_width / vec.size(3))
    vec = F.interpolate(
        vec, (des_height, des_width), mode='bilinear', align_corners=True)
    if vec.size(1) == 1:
        vec = vec * ratio_width
    else:
        vec = torch.stack(
            [vec[:, 0, :, :] * ratio_width, vec[:, 1, :, :] * ratio_height],
            dim=1)
    return vec


def _flow2distribution(flo, c):
    # flo: [B,2,H,W]
    # return [B,d*d,H,W], d=2c+1
    B, _, H, W = flo.size()
    flo = torch.clamp(flo, min=-c, max=c)
    x = flo[:, 0, :, :]
    y = flo[:, 1, :, :]
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0_safe = torch.clamp(x0, min=-c, max=c)
    y0_safe = torch.clamp(y0, min=-c, max=c)
    x1_safe = torch.clamp(x1, min=-c, max=c)
    y1_safe = torch.clamp(y1, min=-c, max=c)

    wt_x0 = (x1 - x) * torch.eq(x0, x0_safe).float()
    wt_x1 = (x - x0) * torch.eq(x1, x1_safe).float()
    wt_y0 = (y1 - y) * torch.eq(y0, y0_safe).float()
    wt_y1 = (y - y0) * torch.eq(y1, y1_safe).float()

    wt_tl = wt_x0 * wt_y0
    wt_tr = wt_x1 * wt_y0
    wt_bl = wt_x0 * wt_y1
    wt_br = wt_x1 * wt_y1

    mask_tl = torch.eq(x0, x0_safe).float() * torch.eq(y0, y0_safe).float()
    mask_tr = torch.eq(x1, x1_safe).float() * torch.eq(y0, y0_safe).float()
    mask_bl = torch.eq(x0, x0_safe).float() * torch.eq(y1, y1_safe).float()
    mask_br = torch.eq(x1, x1_safe).float() * torch.eq(y1, y1_safe).float()

    wt_tl *= mask_tl
    wt_tr *= mask_tr
    wt_bl *= mask_bl
    wt_br *= mask_br

    out = torch.zeros((B, (2 * c + 1)**2, H, W), device=flo.device)
    label_tl = (y0_safe + c) * (2 * c + 1) + x0_safe + c
    label_tr = (y0_safe + c) * (2 * c + 1) + x1_safe + c
    label_bl = (y1_safe + c) * (2 * c + 1) + x0_safe + c
    label_br = (y1_safe + c) * (2 * c + 1) + x1_safe + c

    out.scatter_add_(1, label_tl.unsqueeze(1).long(), wt_tl.unsqueeze(1))
    out.scatter_add_(1, label_tr.unsqueeze(1).long(), wt_tr.unsqueeze(1))
    out.scatter_add_(1, label_bl.unsqueeze(1).long(), wt_bl.unsqueeze(1))
    out.scatter_add_(1, label_br.unsqueeze(1).long(), wt_br.unsqueeze(1))

    return out


def _prob2cornerflow(prob, normalize=True):
    # prob: [B,C,H,W]
    def indice2flow(ind, d):
        # ind [B,1,H,W]
        return torch.cat([ind % d - d // 2, ind / d - d // 2], 1)

    if normalize:
        normalizer = nn.Softmax(dim=1)
        prob = normalizer(prob)
    B, C, H, W = prob.size()
    d = int(math.sqrt(C))
    pr = prob.reshape(B, d, d, -1).permute(0, 3, 1, 2)
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    max_pool = nn.MaxPool2d(kernel_size=d - 1, stride=1, return_indices=True)
    out, indice = max_pool(avg_pool(pr))
    indice += indice / (d - 1)  # in original coordinate
    indice = indice.squeeze().reshape(B, H, W).unsqueeze(1)
    lt_prob = torch.gather(prob, 1, indice)
    lt_flow = indice2flow(indice, d).float()
    rt_prob = torch.gather(prob, 1, indice + 1)
    rt_flow = indice2flow(indice + 1, d).float()
    lb_prob = torch.gather(prob, 1, indice + d)
    lb_flow = indice2flow(indice + d, d).float()
    rb_prob = torch.gather(prob, 1, indice + d + 1)
    rb_flow = indice2flow(indice + d + 1, d).float()
    return [lt_prob, rt_prob, lb_prob,
            rb_prob], [lt_flow, rt_flow, lb_flow, rb_flow]


def _cornerflow2expectation(cor_prob, cor_flow):
    cor_prob_sum = sum(cor_prob)
    cor_prob_n = [prob / cor_prob_sum for prob in cor_prob]
    out = torch.cat([
        cor_flow[1][:, 0, :, :].unsqueeze(1) - cor_prob_n[0] - cor_prob_n[2],
        cor_flow[2][:, 1, :, :].unsqueeze(1) - cor_prob_n[0] - cor_prob_n[1]
    ], 1)
    return out


def _prob2flow(prob, normalize=True):
    cor_prob, cor_flow = _prob2cornerflow(prob, normalize)
    out = _cornerflow2expectation(cor_prob, cor_flow)
    return out


def _disp_prob2flow_prob(prob, normalize=True):
    ###### return NORMALIZED FLOW PROBABILITY ######
    if normalize:
        normalizer = nn.Softmax(dim=1)
        prob = normalizer(prob)
    B, d, H, W = prob.size()
    padding = torch.zeros((B, d * (d - 1) // 2, H, W), device=prob.device)
    prob_padded = torch.cat([padding, prob, padding], dim=1)
    return prob_padded
