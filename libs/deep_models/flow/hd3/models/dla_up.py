import math
import numpy as np
import torch
from torch import nn
from . import dla

BatchNorm = nn.BatchNorm2d


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):

    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(
                    nn.Conv2d(c, out_dim, kernel_size=1, stride=1, bias=False),
                    BatchNorm(out_dim), nn.ReLU(inplace=True))
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(
                    out_dim,
                    out_dim,
                    f * 2,
                    stride=f,
                    padding=f // 2,
                    output_padding=0,
                    groups=out_dim,
                    bias=False)
                fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = nn.Sequential(
                nn.Conv2d(
                    out_dim * 2,
                    out_dim,
                    kernel_size=node_kernel,
                    stride=1,
                    padding=node_kernel // 2,
                    bias=False), BatchNorm(out_dim), nn.ReLU(inplace=True))
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():
            classname = m.__class__.__name__
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x, y


class DLAUp(nn.Module):

    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self, 'ida_{}'.format(i),
                IDAUp(3, channels[j], in_channels[j:],
                      scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        ms_feat = [layers[-1]]
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])  # y : aggregation nodes
            layers[-i - 1:] = y
            ms_feat.append(x)
        return ms_feat  # x


class DLAUpEncoder(nn.Module):

    def __init__(self, planes):
        super(DLAUpEncoder, self).__init__()
        self.first_level = 1
        self.base = dla.dla34(planes)
        scales = [2**i for i in range(len(planes[self.first_level:]))]
        self.dla_up = DLAUp(planes[self.first_level:], scales=scales)

    def forward(self, x):
        x = self.base(x)
        y = self.dla_up(x[self.first_level:])
        return y[::-1]


def DLAEncoder(planes):
    model = DLAUpEncoder(planes)
    return model


def test():
    net = dla34up([16, 32, 64, 128, 256, 512, 512])
    y = net(torch.randn(1, 3, 384, 448))
    for t in y:
        print(t.size())


if __name__ == '__main__':
    test()
