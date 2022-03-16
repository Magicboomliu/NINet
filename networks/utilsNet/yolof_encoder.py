import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_channels, out_channels,stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,bias=False),
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


class Dilation_Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1,
                 dilation=1, norm_layer=None, leaky_relu=False):
        super(Dilation_Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, inplanes, stride=stride)
        self.bn3 = norm_layer(inplanes)

        self.stride = stride

    def forward(self, x):
        identity = x

        # first is one-by-one convolution for reduction
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Then is a 3-by-3 convolution with dilation
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Then is a one-by-one convolution to recover the dim
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out



# Residual Block we want
class Yolof_Block(nn.Module):
    def __init__(self,in_channels,reduce_channels,dilation_list=None,stride=1):
        '''
        Args: dilation : used for conv dilataion
                project_stride : reduction dimenson when project the feature.
        #TODO
        reduce channels of the project layer
        reduce and recover channels in the diliation layer

        '''
        super(Yolof_Block,self).__init__()
        self.in_channels = in_channels

        self.project_layer = nn.Sequential(
            conv1x1(in_channels=self.in_channels,out_channels=reduce_channels),
            nn.Conv2d(reduce_channels, reduce_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduce_channels),
            nn.ReLU(True)
        )
        self.dilation_list = dilation_list
        if self.dilation_list is None:
          self.dilation_list = [2,4,6,8]
        dilation_blocks = nn.ModuleList()

        for idx in range(len(self.dilation_list)):
          dilation_blocks.append(Dilation_Block(inplanes=reduce_channels,planes=reduce_channels,
                                                dilation=self.dilation_list[idx],
                                                leaky_relu=False))

        self.dilation_blocks = nn.Sequential(*dilation_blocks)

    def forward(self,features):
        '''
        :param features: Input Features
        :return: Features
        '''
        project_feature = self.project_layer(features)

        new_feature = self.dilation_blocks(project_feature)

        return new_feature


if __name__=="__main__":

    features = torch.randn(1,64,320,640).cuda()

    yolo_block = Yolof_Block(64,64,dilation_list=[2,4,6,8]).cuda()

    new_features = yolo_block(features)
    print(new_features.shape)



