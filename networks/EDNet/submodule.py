import sys
from torch.functional import norm
from torch.nn.modules.conv import Conv2d
sys.path.append("../..")
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from networks.utilsNet.Disparity_warper import disp_warp
from networks.utilsNet.attention.CBAM import CBAMBlock
from deform.modules.modulated_deform_conv import ModulatedDeformConvPack
from networks.utilsNet.resblocks import DeconBlock,DeforConv,DeformBottleneck
from utils.normal_cal import get_normal
from networks.utilsNet.resblocks import BasicBlockV2
from networks.utilsNet.yolof_encoder import Yolof_Block


# conv3x3 + BN + relu
def conv(in_planes, out_planes, kernel_size=3, stride=1, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )
# conv3x3 + BN
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
        nn.BatchNorm2d(out_planes)
    )

# simple conv3x3 only    
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, stride=stride, kernel_size=3, padding=1, bias=False)

# deconv : upsample to double
def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )
# conv + relu
def conv_Relu(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False),
        nn.ReLU(True)
    )
# conv3d + BatchNorm
def convbn_3d(in_planes, out_planes, kernel_size=3, stride=1, pad=1):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))
# Correlation Cost Volume 
def build_corr(img_left, img_right, max_disp=40):
    B, C, H, W = img_left.shape
    volume = img_left.new_zeros([B, max_disp, H, W])
    for i in range(max_disp):
        if i > 0:
            volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, :-i]).mean(dim=1)
        else:
            volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

    volume = volume.contiguous()
    return volume
# Concated Cost Volume
def form_cost_volume(ref_feature, tar_feature, disp):
    B, C, H, W = ref_feature.shape
    cost = Variable(torch.FloatTensor(B, C*2, disp, H, W).zero_()).cuda()
    for i in range(disp):
        if i > 0:
            cost[:, :C, i, :, i:] = ref_feature[:, :, :, i:]
            cost[:, C:, i, :, i:] = tar_feature[:, :, :, :-i]
        else:
            cost[:, :C, i, :, :] = ref_feature
            cost[:, C:, i, :, :] = tar_feature
    cost = cost.contiguous()
    return cost
# Image warped
def image_rec(img, flow, padding_mode='border'):

    bs, _, h, w = flow.size()
    u = flow[:,0,:,:]
    # 网格化处理
    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    
    X = grid_x + u
    Y = grid_y
    
    X = 2*(X/(w-1.0) - 0.5)
    Y = 2*(Y/(h-1.0) - 0.5)
    
    grid_tf = torch.stack((X, Y), dim=3)
    img_tf = torch.nn.functional.grid_sample(img, grid_tf, padding_mode=padding_mode)

    return img_tf

# Using disparity candidate to do disparity regression
class D_Regression(nn.Module):
    def __init__(self, maxdisp):
        super(D_Regression, self).__init__()
        self.disp = torch.FloatTensor(
            np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])
        )

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3]).cuda()
        out = torch.sum(x*disp, 1)
        return out

# Simple resblock :(conv+bn) + (conv+bn) + Skip Connection
class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride = 1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = kernel_size, stride = stride, padding = 1)
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
        
        out += residual
        out = self.relu(out)
        return out

# BasicBlock: without relu
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, norm_layer=None,
                 bn_eps=1e-5, bn_momentum=0.1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu1 = nn.ReLU(True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride
        # self.leakyrelu2 = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class FeatureExtraction(nn.Module):
    def __init__(self, channal_num=32):
        super(FeatureExtraction, self).__init__()
        self.inplanes = channal_num
        self.firstconv = nn.Sequential(                 # 1/2
            convbn(3, 32, 3, 2, 1, 1),
            nn.ReLU(True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(True)
        )

        self.layer1 = self._make_layer(BasicBlock, blocks=3, inplanes=32, outplanes=32)
        self.layer2 = self._make_layer(BasicBlock, blocks=16, inplanes=32, outplanes=64, stride=2)
        self.layer3 = self._make_layer(BasicBlock, blocks=3, inplanes=64, outplanes=128)
        self.layer4 = self._make_layer(BasicBlock, blocks=3, inplanes=128, outplanes=128, stride=2)     # 1/8
        
        self.branch1 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(True)
        )

        self.branch2 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(True)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d((4, 4), stride=(4, 4)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d((2, 2), stride=(2, 2)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(True)
        )
        
        self.lastconv = nn.Sequential(
            convbn(256, 128, 3, 1, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def _make_layer(self, block, blocks, inplanes=32, outplanes=32, norm_layer=nn.BatchNorm2d, stride=1, bn_eps=1e-5, bn_momentum=0.1):
        downsample = None
        if stride != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                norm_layer(outplanes)
            )
        layers = []
        layers.append(block(inplanes, outplanes, stride, norm_layer, bn_eps,
                            bn_momentum, downsample))
        for i in range(1, blocks):
            layers.append(block(outplanes, outplanes, norm_layer=norm_layer, bn_eps=bn_eps, 
                                bn_momentum=bn_momentum))
        
        return nn.Sequential(*layers)

    def forward(self, input):
        blocks = []
        output = self.firstconv(input)
        
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        branch1 = self.branch1(output)
        branch1 = F.interpolate(branch1, size=(output.size()[2], output.size()[3]), mode='bilinear')

        branch2 = self.branch2(output)
        branch2 = F.interpolate(branch2, size=(output.size()[2], output.size()[3]), mode='bilinear')

        branch3 = self.branch3(output)
        branch3 = F.interpolate(branch3, size=(output.size()[2], output.size()[3]), mode='bilinear')

        branch4 = self.branch4(output)
        branch4 = F.interpolate(branch4, size=(output.size()[2], output.size()[3]), mode='bilinear')

        output = torch.cat((output, branch1, branch2, branch3, branch4), dim=1)
        output = self.lastconv(output)
        return output

class res_submodule_attention(nn.Module):

    def __init__(self, scale, input_layer, out_planes=64):
        super(res_submodule_attention, self).__init__()

        self.pool = nn.AvgPool2d(2**scale, 2**scale)
        
        #Spatial Attention Module here
        self.attention = SA_Module(input_nc=10)

        # input convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(out_planes*4, out_planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*4, out_planes*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2)
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*2, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.redir1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.res = nn.Conv2d(out_planes, 1, 1, 1, bias=False)
    
    def forward(self, left, right, disp, feature):
        scale = left.size()[2] / disp.size()[2]
        left = self.pool(left)
        right = self.pool(right)

        # dummy_flow = torch.autograd.Variable(torch.zeros(disp.data.shape).cuda())
        disp_ = disp / scale            # align the disparity to the proper scale
        # flow = torch.cat((disp_, dummy_flow), dim=1)
  
        # call new warp function
        left_rec ,mask= disp_warp(right,disp_)
    
        error_map = left_rec - left

        # This is the attention's input
        query = torch.cat((left, right, error_map, disp_), dim=1)
        # Attention Here
        attention_map = self.attention(query)
        # attention feature
        attented_feature = attention_map * torch.cat((feature,query), dim=1)

        # ResBlocks
        conv1 = self.conv1(attented_feature)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(attented_feature), inplace=True)

        res = self.res(conv6) * scale
        return res

class res_submodule_deform_only(nn.Module):

    def __init__(self, scale, input_layer, out_planes=64):
        super(res_submodule_deform_only, self).__init__()

        self.pool = nn.AvgPool2d(2**scale, 2**scale)
        
        #Spatial Attention Module here
        self.attention = SA_Module(input_nc=10)

        # input convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True),
            ModulatedDeformConvPack(out_planes*2,out_planes*2,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            ModulatedDeformConvPack(out_planes*4,out_planes*4,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True),
            nn.Conv2d(out_planes*4, out_planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*4, out_planes*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2)
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*2, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.redir1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.res = nn.Conv2d(out_planes, 1, 1, 1, bias=False)
    
    def forward(self, left, right, disp, feature):
        scale = left.size()[2] / disp.size()[2]
        left = self.pool(left)
        right = self.pool(right)

        # dummy_flow = torch.autograd.Variable(torch.zeros(disp.data.shape).cuda())
        disp_ = disp / scale            # align the disparity to the proper scale
        # flow = torch.cat((disp_, dummy_flow), dim=1)
  
        # call new warp function
        left_rec ,mask= disp_warp(right,disp_)
    
        error_map = left_rec - left

        # This is the attention's input
        query = torch.cat((left, right, error_map, disp_), dim=1)
        # Attention Here
        attention_map = self.attention(query)
        # attention feature
        attented_feature = attention_map * torch.cat((feature,query), dim=1)

        # ResBlocks
        conv1 = self.conv1(attented_feature)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(attented_feature), inplace=True)

        res = self.res(conv6) * scale
        return res


class res_submodule_attention_noerror(nn.Module):
    def __init__(self, scale, input_layer, out_planes=64):
        super(res_submodule_attention_noerror, self).__init__()
        # self.resample = Resample2d()
        self.pool = nn.AvgPool2d(2**scale, 2**scale)
        
        self.attention = SA_Module(input_nc=7)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+7, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(out_planes*4, out_planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*4, out_planes*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2)
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*2, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.redir1 = nn.Sequential(
            nn.Conv2d(input_layer+7, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.res = nn.Conv2d(out_planes, 1, 1, 1, bias=False)
    
    def forward(self, left, right, disp, feature):
        scale = left.size()[2] / disp.size()[2]
        left = self.pool(left)
        right = self.pool(right)

        # dummy_flow = torch.autograd.Variable(torch.zeros(disp.data.shape).cuda())
        disp_ = disp / scale            # align the disparity to the proper scale
        # flow = torch.cat((disp_, dummy_flow), dim=1)
        # left_rec = self.resample(right, -flow)
        # error_map = left - left_rec

        query = torch.cat((left, right, disp_), dim=1)
        # query_feature = self.query_feature(query.detach())
        # attention_map = self.attention(torch.cat((query_feature, feature), dim=1))
        attention_map = self.attention(query)
        attented_feature = attention_map * torch.cat((feature,query), dim=1)

        conv1 = self.conv1(attented_feature)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(attented_feature), inplace=True)

        res = self.res(conv6) * scale
        return res

# Simple Residual Prediction: Left-Right Images, Disparity Warped Error
class res_submodule(nn.Module):
    def __init__(self, scale, input_layer, out_planes=64):
        super(res_submodule, self).__init__()
        #self.resample = Resample2d()
        
        # Avgerage Pooling
        self.pool = nn.AvgPool2d(2**scale, 2**scale)
        # Left + Right + Warped Error + Coarse Disparity : 10
        # self.conv1: Downsample to input's 1/2, channels doubled
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )
        # Self.conv2: aggregation: Size not change
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )
        # Self.conv3: Downsample to input's 1/4, channels doubled again
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )
        # Self.conv4: aggregation: Size not change
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_planes*4, out_planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )

        # Self.conv5: Upsample to former's 1/2, channels half
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*4, out_planes*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2)
        )

        # Self.conv6; Upsample to input's Size, channels half
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*2, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        # Skip connection1
        self.redir1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        # Skip connection2 
        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        # Residual prediction
        self.res = nn.Conv2d(out_planes, 1, 1, 1, bias=False)
    
    def forward(self, left, right, disp, feature):
        '''
        left: left Image
        right :right image
        disp: disparity
        feature : left image feature
        '''
        
        # Current Scale for interpolation
        scale = left.size()[2] / disp.size()[2]
        # Left right Image Pooling
        left = self.pool(left)
        right = self.pool(right)

        # Align disparity to the current scale for image warped error computaion

        disp_ = disp / scale            # align the disparity to the proper scale
        # flow = torch.cat((disp_, dummy_flow), dim=1)
        left_rec,mask = disp_warp(right,disp_)
        # Calculate the disparity error
        error_map = left_rec -left
         
        # Downsample to 1/2
        conv1 = self.conv1(torch.cat((left, right, disp_, error_map, feature), dim=1))
        # Aggregation
        conv2 = self.conv2(conv1)
        # Downsample to 1/4
        conv3 = self.conv3(conv2)
        # Aggregation
        conv4 = self.conv4(conv3)
        # Upsample+Skip Connection to 1/2
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        # Upsample + Skip Conncetion to 1/2
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((left, right, disp_, error_map, feature), dim=1)))
        # Recover to the Original Scale
        res = self.res(conv6) * scale
        return res

# Residual Prediction with Surface Normal
class res_submodule_with_normal_deform(nn.Module):
    def __init__(self, scale, input_layer, out_planes=64):
        super(res_submodule_with_normal_deform, self).__init__()
        #self.resample = Resample2d()
        self.pool = nn.AvgPool2d(2**scale, 2**scale)

        self.attention = SA_Module(input_nc=13)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+13, out_planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True),

        )
        
        self.conv2 = nn.Sequential(
            ModulatedDeformConvPack(out_planes*2,out_planes*2,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True),
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )
        
        self.conv4 = nn.Sequential(
            ModulatedDeformConvPack(out_planes*4,out_planes*4,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True),
            nn.Conv2d(out_planes*4, out_planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*4, out_planes*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2)
        )

        self.conv6 = nn.Sequential(
            
            nn.ConvTranspose2d(out_planes*2, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.redir1 = nn.Sequential(
            nn.Conv2d(input_layer+13, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.res = nn.Conv2d(out_planes, 1, 1, 1, bias=False)
    
    def forward(self, left, right, disp, surface_normal,feature):
     
        scale = left.size()[2] / disp.size()[2]
        left = self.pool(left)
        right = self.pool(right)

        # dummy_flow = torch.autograd.Variable(torch.zeros(disp.data.shape).cuda())
        disp_ = disp / scale            # align the disparity to the proper scale
        # flow = torch.cat((disp_, dummy_flow), dim=1)
        left_rec,mask = disp_warp(right,disp_)
        error_map = left_rec -left

        query = torch.cat((left, right, disp_,error_map,surface_normal), dim=1)
        attention = self.attention(query)
        attention_feature = attention * torch.cat((left, right, disp_, error_map, surface_normal,feature), dim=1)
        conv1 = self.conv1(attention_feature)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((left, right, disp_, error_map, surface_normal,feature), dim=1)))
        res = self.res(conv6) * scale
        return res

# Residual Prediction with surface Normal Refinement
class res_submodule_with_norm_occlu(nn.Module):
    def __init__(self,scale, input_layer, out_planes=64):
        super(res_submodule_with_norm_occlu,self).__init__()
        self.pool = nn.AvgPool2d(2**scale, 2**scale)
        # The Attention Occlusion Mask
        #dynamic attention
        self.attention = SA_Module(input_nc=10)
        self.dynamic_occlusion_mask = SA_Module(input_nc=1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),

        )
        
        self.conv2 = nn.Sequential(
            ModulatedDeformConvPack(out_planes,out_planes,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
            nn.Conv2d(out_planes, out_planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            ModulatedDeformConvPack(out_planes*2,out_planes*2,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True),
            nn.Conv2d(out_planes*2, out_planes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )
        
        self.conv4 = nn.Sequential(
            ModulatedDeformConvPack(out_planes*4,out_planes*4,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True),
            nn.Conv2d(out_planes*4, out_planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*4, out_planes*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2)
        )

        self.conv6 = nn.Sequential(
            
            nn.ConvTranspose2d(out_planes*2, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.redir1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.res = nn.Conv2d(out_planes, 1, 1, 1, bias=False)
    
    def forward(self, left, right, disp, surface_normal,feature,occlusion_mask):
        '''
        left and right is the image input
        feature is the aggregated feature
        disp is refined disparity
        occlusion mask is the occlusion mask
        '''

        scale = left.size()[2] / disp.size()[2]
        left = self.pool(left)
        right = self.pool(right)

        disp_ = disp / scale  # align the disparity to the proper scale
        
        # Error Map: Consist of two parts:
        #(1) ill-prediction areas  ---> increase the attention
        #(2) occlusion areas   ---> reduce the attention: use propagation to refine this areas
        left_rec,mask = disp_warp(right,disp_)
        # Abundant information here
        error_map = left_rec -left
        error_map = error_map
        # Get remove of the occlusion areas
        ill_prediction_error_map = error_map * occlusion_mask

        # feature dynamic adjustment
        dynamic_occlusion_mask = self.dynamic_occlusion_mask(occlusion_mask.float())
        feature = feature*dynamic_occlusion_mask

        # This is the query
        query = torch.cat((left, disp_,ill_prediction_error_map,surface_normal), dim=1)
        dynamic_attention = self.attention(query)
        attention_feature = dynamic_attention * torch.cat((left, disp_, error_map, surface_normal,feature), dim=1)
        
        conv1 = self.conv1(attention_feature)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((left, disp_, error_map, surface_normal,feature), dim=1)))
        res = self.res(conv6) * scale
        return res



#SubTest Number : 1
class res_submodule_with_norm_wo_right(nn.Module):
    def __init__(self,scale, input_layer, out_planes=64):
        super(res_submodule_with_norm_wo_right,self).__init__()
        # Left + Disp + Error Map + Surface Normal = 3+1+3+3 = 10
        self.pool = nn.AvgPool2d(2**scale, 2**scale)
        self.attention = SA_Module(input_nc=10)
        
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True))
        self.conv2 = nn.Sequential(
            ModulatedDeformConvPack(out_planes,out_planes,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
            nn.Conv2d(out_planes, out_planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True))
        self.conv3 = nn.Sequential(
            ModulatedDeformConvPack(out_planes*2,out_planes*2,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True),
            nn.Conv2d(out_planes*2, out_planes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True))
        
        self.conv4 = nn.Sequential(
            ModulatedDeformConvPack(out_planes*4,out_planes*4,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True),
            nn.Conv2d(out_planes*4, out_planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True))
        # Decoder
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*4, out_planes*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2))

        self.conv6 = nn.Sequential(
            
            nn.ConvTranspose2d(out_planes*2, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes))

        self.redir1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=1, stride=1, padding=0, bias=False))

        self.res = nn.Conv2d(out_planes, 1, 1, 1, bias=False)
    
    def forward(self, left, right, disp, surface_normal,feature,occlusion_mask):
        '''
        left and right is the image input
        feature is the aggregated feature
        disp is refined disparity
        occlusion mask is the occlusion mask
        '''

        scale = left.size()[2] / disp.size()[2]
        left = self.pool(left)
        right = self.pool(right)

        disp_ = disp / scale  # align the disparity to the proper scale

        left_rec,mask = disp_warp(right,disp_)
        # Abundant information here
        error_map = left_rec -left
        query = torch.cat((left, disp_,error_map,surface_normal), dim=1)
        dynamic_attention = self.attention(query)
        attention_feature = dynamic_attention * torch.cat((left, disp_, error_map, surface_normal,feature), dim=1)
        
        conv1 = self.conv1(attention_feature)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((left, disp_, error_map, surface_normal,feature), dim=1)))
        res = self.res(conv6) * scale
        return res


class ResBlock_Deform(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride = 1):
        super(ResBlock_Deform, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = kernel_size, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = ModulatedDeformConvPack(n_out, n_out, kernel_size = 3, padding = 1,stride=1)
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

        out += residual
        out = self.relu(out)
        return out




# Spatial Attention
class SA_Module(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=16):
        super(SA_Module, self).__init__()
        self.attention_value = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, output_nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_value = self.attention_value(x)
        return attention_value
