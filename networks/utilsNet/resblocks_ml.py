import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from torch.nn.modules import conv
sys.path.append("../../")
from networks.utilsNet.simplecnn import conv1x1_bn_relu, conv3x3_optional
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import sys
sys.path.append("../../")
from deform.modules.modulated_deform_conv import ModulatedDeformConvPack
from deform.modules import DeformConv
import time

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, leaky_relu=True):
        """StereoNet uses leaky relu (alpha = 0.2)"""
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_optional(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        self.conv2 = conv3x3_optional(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Simple Residual Block
class Simple_Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Simple_Residual_Block, self).__init__()
        self.conv1 = conv3x3_optional(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_optional(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# Simple Residual Block without relu last
class Simple_Residual_Block_No_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Simple_Residual_Block_No_Relu, self).__init__()
        self.conv1 = conv3x3_optional(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_optional(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        return out

class Simple_Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Simple_Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1_bn_relu(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_optional(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1_bn_relu(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ResBlock in EDNet
class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = residual + out
        out = self.relu(out)
        return out

# Defromable Resblock
class DeconBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, use_residual=True):
        super(DeconBlock, self).__init__()
        self.conv1 = ResBlock(in_planes, out_planes, stride=stride)
        self.deconv = ModulatedDeformConvPack(out_planes, out_planes, kernel_size=(3, 3), stride=1, 
                                              padding=1, deformable_groups=2)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(0.1, True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        if in_planes != out_planes or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.downsample = None

        self.use_residual = use_residual
    
    def forward(self, x):
        if self.use_residual:
            residual = x
            if self.downsample is not None:
                residual = self.downsample(residual)
        else:
            residual = 0
        out = self.conv1(x)
        out = self.deconv(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = residual + out
        
        return out