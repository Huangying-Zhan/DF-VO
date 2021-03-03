import torch
import torch.nn as nn

BatchNorm = nn.BatchNorm2d


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, normalize=True):
        super(PreActBlock, self).__init__()
        if normalize:
            self.bn1 = BatchNorm(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False))

    def forward(self, x):
        out = self.relu(self.bn1(x)) if hasattr(self, 'bn1') else x
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out += shortcut
        return out


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), BatchNorm(self.expansion * planes))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResnetDecoder(nn.Module):

    def __init__(self, inplane, outplane):
        super(ResnetDecoder, self).__init__()
        self.block1 = PreActBlock(inplane, outplane, normalize=False)
        self.block2 = PreActBlock(outplane, outplane, normalize=True)

    def forward(self, x):
        x = self.block1(x)
        out = self.block2(x)
        return out


class HDADecoder(nn.Module):

    def __init__(self, inplane, outplane):
        super(HDADecoder, self).__init__()
        self.block1 = PreActBlock(inplane, outplane, normalize=False)
        self.block2 = PreActBlock(outplane, outplane, normalize=True)
        self.root = nn.Sequential(
            BatchNorm(outplane * 2), nn.ReLU(inplace=True),
            nn.Conv2d(
                outplane * 2, outplane, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(y1)
        out = self.root(torch.cat([y1, y2], 1))
        return out
