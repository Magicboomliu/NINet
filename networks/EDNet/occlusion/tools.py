import sys
sys.path.append("../../..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from deform.modules.modulated_deform_conv import ModulatedDeformConvPack
from networks.utilsNet.attention.CBAM import CBAMBlock
from networks.utilsNet.resblocks import DeconBlock,DeforConv,DeformBottleneck

from networks.utilsNet.Disparity_warper import disp_warp

from pac.pac import PacConv2d,PacConvTranspose2d

def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def conv_bn_relu_df(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(ModulatedDeformConvPack(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers

def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
                  bn=True, relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


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