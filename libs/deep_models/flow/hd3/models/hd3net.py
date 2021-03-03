import torch
import torch.nn as nn
import torch.nn.functional as F
from . import correlation
from .hd3_ops import *
from .dla_up import DLAEncoder as dlaup_encoder
from .vgg import VGGEncoder as vgg_encoder
from . import decoder as PreDecoder

BatchNorm = nn.BatchNorm2d


class Context(nn.Module):

    def __init__(self, inplane, classes):
        super(Context, self).__init__()
        self.num_convs = 7
        ch = [inplane, 128, 128, 128, 128, 128, 128, 128]
        dilations = [1, 1, 2, 4, 8, 16, 1]
        for i in range(self.num_convs):
            setattr(
                self, 'dc_conv_{}'.format(i),
                nn.Sequential(
                    nn.Conv2d(
                        ch[i],
                        ch[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=dilations[i],
                        dilation=dilations[i],
                        bias=False), BatchNorm(ch[i + 1]),
                    nn.ReLU(inplace=True)))
        self.cls = nn.Conv2d(
            ch[-1], classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = x
        for i in range(self.num_convs):
            dc_conv = getattr(self, 'dc_conv_' + str(i))
            out = dc_conv(out)
        out = self.cls(out)
        return out, None


class Decoder(nn.Module):

    def __init__(self, inplane, block, classes, up_classes):
        super(Decoder, self).__init__()
        self.mapping = block(inplane, 128)
        self.cls = nn.Sequential(
            BatchNorm(128), nn.ReLU(inplace=True),
            nn.Conv2d(
                128, classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.up = None
        if up_classes > 0:
            self.up = nn.Sequential(
                BatchNorm(128), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    128,
                    up_classes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False), BatchNorm(up_classes), nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.mapping(x)
        prob = self.cls(out)
        up_feat = self.up(out) if self.up else None
        return prob, up_feat


class HD3Net(nn.Module):

    def __init__(self, task, encoder, decoder, corr_range, context=False,
                 ds=6):
        super(HD3Net, self).__init__()

        self.task = task
        self.dim = 1 if task == 'stereo' else 2
        self.levels = len(corr_range)
        self.ds = ds  # downsample ratio of the coarsest level
        if self.task == 'flow':
            self.classes = [(2 * d + 1)**2 for d in corr_range]
        else:
            self.classes = [2 * d + 1 for d in corr_range]

        if encoder == 'vgg':
            pyr_channels = [16, 32, 64, 96, 128, 196]
            assert self.levels <= len(pyr_channels)
            self.encoder = vgg_encoder(pyr_channels)
        elif encoder == 'dlaup':
            pyr_channels = [16, 32, 64, 128, 256, 512, 512]
            self.encoder = dlaup_encoder(pyr_channels)
        else:
            raise ValueError('Unknown encoder {}'.format(encoder))

        if decoder == 'resnet':
            dec_block = PreDecoder.ResnetDecoder
        elif decoder == 'hda':
            dec_block = PreDecoder.HDADecoder
        else:
            raise ValueError('Unknown decoder {}'.format(decoder))

        feat_d_offset = pyr_channels[::-1]
        feat_d_offset[0] = 0
        up_d_offset = [0] + self.classes[1:]

        for l in range(self.levels):
            setattr(self, 'cost_bn_{}'.format(l), BatchNorm(self.classes[l]))
            input_d = self.classes[l] + feat_d_offset[l] + up_d_offset[
                l] + self.dim * (
                    l > 0)
            if l < self.levels - 1:
                up_classes = self.classes[l + 1]
            else:
                up_classes = -1
            if context and l == self.levels - 1:
                setattr(self, 'Decoder_{}'.format(l),
                        Context(input_d, self.classes[l]))
            else:
                setattr(
                    self, 'Decoder_{}'.format(l),
                    Decoder(
                        input_d,
                        dec_block,
                        self.classes[l],
                        up_classes=up_classes))

        for m in self.modules():
            classname = m.__class__.__name__
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def shift(self, x, vect):
        if vect.size(1) < 2:
            vect = disp2flow(vect)
        return flow_warp(x, vect)

    def forward(self, x):
        # extract pyramid features
        bs = x.size(0)
        feat_list = self.encoder(
            torch.cat([x[:, :3, :, :], x[:, 3:, :, :]], 0))
        fp_0 = [f[:bs, :, :, :] for f in feat_list[::-1]]
        fp_1 = [f[bs:, :, :, :] for f in feat_list[::-1]]

        ms_pred = []
        for l in range(self.levels):
            ref_feat = fp_0[l]
            tar_feat = fp_1[l]

            if l == 0:
                tar_feat_corr = tar_feat
            else:
                tar_feat_corr = self.shift(tar_feat, up_curr_vect)

            cost_vol = correlation.FunctionCorrelation(
                tensorFirst=ref_feat, tensorSecond=tar_feat_corr)
            if self.task == 'stereo':
                c = self.classes[l] // 2
                cost_vol = cost_vol[:, c * (2 * c + 1):(c + 1) *
                                    (2 * c + 1), :, :].contiguous()
            cost_bn = getattr(self, 'cost_bn_' + str(l))
            cost_vol = cost_bn(cost_vol)

            if l == 0:
                decoder_input = cost_vol
            else:
                decoder_input = torch.cat(
                    [cost_vol, ref_feat, ms_pred[-1][-1], up_curr_vect], 1)

            decoder = getattr(self, 'Decoder_' + str(l))
            prob_map, up_feat = decoder(decoder_input)

            curr_vect = density2vector(prob_map, self.dim, True)
            if l > 0:
                curr_vect += up_curr_vect
            if self.task == 'stereo':
                curr_vect = torch.clamp(curr_vect, max=0)
            ms_pred.append([prob_map, curr_vect * 2**(self.ds - l), up_feat])

            if l < self.levels - 1:
                up_curr_vect = 2 * F.interpolate(
                    curr_vect,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True)

        ms_prob = [l[0] for l in ms_pred]
        ms_vect = [l[1] for l in ms_pred]

        return ms_prob, ms_vect
