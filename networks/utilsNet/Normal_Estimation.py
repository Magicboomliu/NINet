
import torch
from torch._C import set_flush_denormal
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../")
from networks.utilsNet.resblocks import DeformBottleneck,DeconBlock
from torch.nn.init import kaiming_normal

# Conv2d
def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.1, inplace=True))

# conv2d + BN
def convbn(in_planes,out_planes,kernel_size,stride,pad,dilation):
    return nn.Sequential(nn.Conv2d(in_planes,out_planes,kernel_size=kernel_size,stride=stride,
                                                                        padding= dilation if dilation >1 else pad, dilation=dilation,bias=False) ,
                                                                        nn.BatchNorm2d(out_planes))

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
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

# How about use dilation Here
class NormalEstimation(nn.Module):
    def __init__(self,is_first=0,use_deform=False):
        super(NormalEstimation,self).__init__()
        self.is_first= is_first
        self.use_deform = use_deform
        self.disp_conv1 = conv2d(1,16)
        self.disp_conv2 = conv2d(16,32)
        if self.is_first==0:
            self.feature_conv1 = conv2d(56,32)
        if self.is_first==1:
            self.feature_conv2 = conv2d(32,16)
        if self.is_first==2:
            self.feature_conv = conv2d(32,16)
        if self.is_first==3:
            self.feature_conv = conv2d(32,16)
        
        self.left_image_conv= conv2d(3,16)
        if self.is_first==0:
            self.norm_conv1 = self.make_layer_simple(ResidualBlock, 80, 64, 1, 1)  # H/4, C =64
        else:
            self.norm_conv1 = self.make_layer_simple(ResidualBlock, 64, 64, 1, 1)  # H/4, C =64
        
        self.norm_conv2 = self.make_layer_simple(ResidualBlock,64,32,2,1) # H/4, C =32
        if self.use_deform:
            self.norm_conv3 = DeformBottleneck(inplanes=32,plane=32,kernel_size=3,padding=1,stride=1,dilation=1)
        else:
            self.norm_conv3 = self.make_layer_simple(ResidualBlock,32,32,1,1) # H/4, C =32
        self.normal = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False)  # H/4, C=3

    def make_layer_simple(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self,features,disp,residual=False,left_image=None):
        # First Decode the disparity
        disp_feature1 = self.disp_conv1(disp)
        disp_feature = self.disp_conv2(disp_feature1)
        # Then encode the features
        if self.is_first==0:
            features = self.feature_conv1(features)
            cur_left_image = F.interpolate(left_image,scale_factor=1./pow(2,3-self.is_first),mode='bilinear',
            align_corners=False)
            left_image_feature = self.left_image_conv(cur_left_image)
        elif self.is_first==1:
            features = self.feature_conv2(features)
            cur_left_image = F.interpolate(left_image,scale_factor=1./pow(2,3-self.is_first),mode='bilinear',
            align_corners=False)
            left_image_feature = self.left_image_conv(cur_left_image)
        else:
            features = self.feature_conv(features)
            cur_left_image = F.interpolate(left_image,scale_factor=1./pow(2,3-self.is_first),mode='bilinear',
            align_corners=False)
            left_image_feature = self.left_image_conv(cur_left_image)

        # Concated the feature together
        inputs_ = torch.cat((disp_feature,features,left_image_feature),dim=1)
        norm_conv1 = self.norm_conv1(inputs_)
        norm_conv2 = self.norm_conv2(norm_conv1)
        norm_conv3 = self.norm_conv3(norm_conv2)
        normal = self.normal(norm_conv3)
        
        if residual:
            return normal,norm_conv3
        else:
            normal = F.normalize(normal,dim=1)
            return normal,norm_conv3


if __name__=="__main__":
    features = torch.randn(1,56,36,72).cuda()
    disp= torch.randn(1,1,36,72).cuda()
    left_image = torch.randn(1,3,288,576).cuda()
    normal_estimation = NormalEstimation(is_first=0,use_deform=True).cuda()
    surface_normal,_ = normal_estimation(features,disp,False,left_image)
    print(surface_normal.shape)
    
