import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.append("../..")
from deform.modules.modulated_deform_conv import ModulatedDeformConvPack
from networks.submodule import ResBlock
from networks.utilsNet.Disparity_warper import LRwarp_error

'''   Basic convolution Module '''
def conv1x1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, with_bn_relu=False, leaky_relu=False):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    if with_bn_relu:
        relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        conv = nn.Sequential(conv,
                             nn.BatchNorm2d(out_planes),
                             relu)
    return conv

'''   Basic ResNet Module '''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, leaky_relu=True):
        """StereoNet uses leaky relu (alpha = 0.2)"""
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        if stride!=1 or inplanes!= planes:
            if self.downsample is None:
                self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes)
                )

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


'''   Basic ResNet Module '''
class BasicBlockV2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, leaky_relu=True):
        """StereoNet uses leaky relu (alpha = 0.2)"""
        super(BasicBlockV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        if stride!=1 or inplanes!= planes:
            if self.downsample is None:
                self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes)
                )

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


class BasicBlock_WORELU(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, leaky_relu=True):
        """StereoNet uses leaky relu (alpha = 0.2)"""
        super(BasicBlock_WORELU, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        if stride!=1 or inplanes!= planes:
            if self.downsample is None:
                self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes)
                )

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
        return out

# Simple Deforcov : 1 simple CONV +1 deformable CONV+ BN + relu
'''
Conv
deformable CONV
BN
RELU
'''
class DeforConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, deformable_groups):
        super(DeforConv, self).__init__()
        self.conv1 = ResBlock(in_planes, out_planes, stride=stride)
        self.conv2 = ModulatedDeformConvPack(out_planes, out_planes, kernel_size=kernel_size, stride=1,
                                             padding=padding, deformable_groups=deformable_groups)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

# deformable Block: Without RELU at the last
'''
Resblock
deformable conv
BN +relu
conv
BN + relu
residual add
'''
class DeconBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,stride, use_residual=True):
        super(DeconBlock, self).__init__()
        self.conv1 = ResBlock(in_planes, out_planes, stride=stride)
        self.deconv = ModulatedDeformConvPack(out_planes, out_planes, kernel_size=(kernel_size, kernel_size), stride=1, 
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

# Deformable ResNet Block used in AANET
'''
CONV
BN +relu
deformable
BN+ relu
CONV
BN 
relu(residual + identity)
'''
class DeformBottleneck(nn.Module):
    def __init__(self,inplanes,plane,kernel_size,padding,stride=1,downsample=None,groups=1,
                base_width =32,dilation=1,norm_layer=None):
        super(DeformBottleneck,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(plane*base_width/32.0) *groups
        self.conv1 = conv1x1(inplanes,width)
        self.bn1 = norm_layer(width)
        self.conv2 = ModulatedDeformConvPack(width,width,kernel_size=kernel_size,dilation=dilation,groups=groups,stride=stride,padding=padding)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width,plane)
        self.bn3= norm_layer(width)
        self.relu =nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if self.downsample is None:
            if stride!=1 or inplanes!=plane:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=inplanes,out_channels=plane,stride=stride,padding=0,kernel_size=1,dilation=1),
                    norm_layer(plane)
                )

    def forward(self,x):
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
