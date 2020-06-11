import torch.nn as nn

BatchNorm = nn.BatchNorm2d


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            bias=False)
        self.bn1 = BatchNorm(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False)
        self.bn2 = BatchNorm(out_planes)
        self.conv3 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False)
        self.bn3 = BatchNorm(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


class VGG(nn.Module):

    def __init__(self, block, planes):
        super(VGG, self).__init__()
        self.levels = len(planes)
        channels = [3] + planes
        for i in range(self.levels):
            setattr(self, 'block_{}'.format(i),
                    block(channels[i], channels[i + 1]))

        for m in self.modules():
            classname = m.__class__.__name__
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = []
        for i in range(self.levels):
            x = getattr(self, 'block_' + str(i))(x)
            out.append(x)
        return out


def VGGEncoder(planes):
    return VGG(BasicBlock, planes)
