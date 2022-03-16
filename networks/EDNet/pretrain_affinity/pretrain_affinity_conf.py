import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../..")
from networks.utilsNet.Disparity_warper import LRwarp_error
from networks.utilsNet.spatial_propagation import Affinity_propagation_with_conf
from networks.EDNet.submodule import *
from torch.nn.init import kaiming_normal
from networks.utilsNet.devtools import print_tensor_shape
from networks.EDNet.submodule import *
from networks.utilsNet.confidenceEstimation import DisparityConfEstimation


# Get Surface Normal From disparity

class EDNet(nn.Module):
    def __init__(self, batchNorm=False, max_disp=192, input_channel=3, res_type='normal',
                 squeezed_volume=False):
        super(EDNet, self).__init__()
        self.max_disp = max_disp
        self.squeezed_volume = squeezed_volume
        self.res_type = res_type
        
        self.conv1 = conv(3, 64, 7, 2)                      # 1/2
        self.conv2 = ResBlock(64, 128, stride=2)            # 1/4
        self.conv3 = ResBlock(128, 256, stride=2)           # 1/8
        
        self.conv_redir = ResBlock(256, 32, stride=1)       # skip connection
        
        # squeezed 3d Convolution
        if squeezed_volume:
            self.conv3d = nn.Sequential(
                convbn_3d(32*2, 32, 3, 1, 1),
                nn.ReLU(True),
                convbn_3d(32, 32, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.conv_compress = ResBlock(256, 32, stride=1)    # 1/8

            self.conv3_1 = ResBlock(80, 256)                # 192 / 8 = 24 -> correlation + squeezed volume -> 24 * 2 + 32
        else:
            self.conv3_1 = ResBlock(56, 256)
        
        # Downsample 

        self.conv4 = ResBlock(256, 512, stride=2)           # 1/16
        self.conv4_1 = ResBlock(512, 512)
        self.conv5 = ResBlock(512, 512, stride=2)           # 1/32
        self.conv5_1 = ResBlock(512, 512)
        self.conv6 = ResBlock(512, 1024, stride=2)          # 1/64
        self.conv6_1 = ResBlock(1024, 1024)

        self.iconv5 = nn.ConvTranspose2d(1024, 512, 3, 1, 1) # Just change the channels, Size not change
        self.iconv4 = nn.ConvTranspose2d(768, 256, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(384, 128, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(192,64, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(96, 32, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(19, 32, 3, 1, 1)
        
        # Deconvoluton : nn.ConvTranspose2d + Relu
        self.upconv5 = deconv(1024, 512) # Channel changes, size double
        self.upconv4 = deconv(512, 256)
        self.upconv3 = deconv(256, 128)
        self.upconv2 = deconv(128, 64)
        self.upconv1 = deconv(64, 32)
        self.upconv0 = deconv(32, 16)

        # disparity estimation
        self.disp3 = nn.Conv2d(128,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu0 = nn.ReLU(inplace=True)


        # Affinity Spatail Propagation
        self.affinity_propagation_list=nn.ModuleList()
        for _ in range(4):
            self.affinity_propagation_list.append(Affinity_propagation_with_conf())

        # Confidence Estimation 
        self.confidence_branch3 = DisparityConfEstimation(input_channels=128,hidden_layer=32)
        self.confidence_branch2 = DisparityConfEstimation(input_channels=64,hidden_layer=32)
        self.confidence_branch1 = DisparityConfEstimation(input_channels=32,hidden_layer=32)
        self.confidence_branch0 = DisparityConfEstimation(input_channels=32,hidden_layer=32)

        # residual learning
        if self.res_type == 'attention':
            residual = res_submodule_attention
        elif self.res_type == 'normal':
            residual = res_submodule
        elif self.res_type == 'noerror':
            residual = res_submodule_attention_noerror
        elif self.res_type =='deform_norm':
            residual = res_submodule_with_normal_deform
        else:
            raise NotImplementedError("Wrong residual type")
        
        self.res_submodule_2 = residual(scale=2, input_layer=64, out_planes=32)
        self.res_submodule_1 = residual(scale=1, input_layer=32, out_planes=32)
        self.res_submodule_0 = residual(scale=0, input_layer=32, out_planes=32)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img_left, img_right,assisted_normal,training=False):

        # split left image and right image

        conv1_l = self.conv1(img_left)          # 64 1/2
        conv2_l = self.conv2(conv1_l)           # 128 1/4
        conv3_l = self.conv3(conv2_l)           # 256 1/8

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3_r = self.conv3(conv2_r)           # 1/8

        # build corr 
        # 24 channels probility cost volume
        out_corr = build_corr(conv3_l,conv3_r, self.max_disp//8)
        # 32 channels cost volume
        out_conv3a_redir = self.conv_redir(conv3_l)
        # Concated Cost Volume
        in_conv3b = torch.cat((out_conv3a_redir, out_corr), dim=1)         # 24+32=56
        # 3D cost volume
        if self.squeezed_volume:
            conv_compress_left = self.conv_compress(conv3_l)
            conv_compress_right = self.conv_compress(conv3_r)

            cost_volume = form_cost_volume(conv_compress_left, conv_compress_right, self.max_disp//8)
            cost_volume = self.conv3d(cost_volume)
            cost_volume = torch.squeeze(cost_volume, dim=1)
            in_conv3b = torch.cat((in_conv3b, cost_volume), dim=1)

        conv3b = self.conv3_1(in_conv3b)    # 56 ---> 256
        conv4a = self.conv4(conv3b)    #Downsample to 1/16     
        conv4b = self.conv4_1(conv4a)       # 512 1/16: Simple ResBlock
        conv5a = self.conv5(conv4b)    #Downsample to 1/32
        conv5b = self.conv5_1(conv5a)       # 512 1/32：Simple ResBlock
        conv6a = self.conv6(conv5b)    #Downsample to 1/64
        conv6b = self.conv6_1(conv6a)       # 1024 1/64:Simple ResBlock

        upconv5 = self.upconv5(conv6b)      # Upsample to 1/32 : 512 1/32
        concat5 = torch.cat((upconv5, conv5b), dim=1)   # 1024 1/32
        iconv5 = self.iconv5(concat5)       # 1024-->512

        upconv4 = self.upconv4(iconv5)      # Upsample to 1/16: 256 1/16
        concat4 = torch.cat((upconv4, conv4b), dim=1)   #256+512: 768 1/16
        iconv4 = self.iconv4(concat4)       # 768-->256 1/16

        upconv3 = self.upconv3(iconv4)      # Upsample to 1/8: 128 1/8
        concat3 = torch.cat((upconv3, conv3b), dim=1)    # 128+256=384 1/8
        iconv3 = self.iconv3(concat3)       # 128

        # Get 1/8 Disparity Here
        pr3 = self.disp3(iconv3)
        pr3 = self.relu3(pr3) # Use Simple CNN to do disparity regression

        # Get 1/8 Confidence Here
        confidence3 = self.confidence_branch3(pr3,img_left,img_right,iconv3)
        cur_normal = F.interpolate(assisted_normal,scale_factor=1./8,mode='bilinear',align_corners=False)
    
        pr3 = self.affinity_propagation_list[0](cur_normal,img_left,img_right,pr3,confidence3)

        # 1/4 Feature
        upconv2 = self.upconv2(iconv3)      # Upsample to 1/4 :64 1/4
        concat2 = torch.cat((upconv2, conv2_l), dim=1)  # 64+128=192 1/4
        iconv2 = self.iconv2(concat2) #192-->64

        '''Here Beigin the Disparity Residual refinement'''
        # 1/4 Disparity Refinement
        # Upsample the 1/8 Disparity to coarse 1/4 Disparity
        pr2 = F.interpolate(pr3, size=(pr3.size()[2] * 2, pr3.size()[3] * 2), mode='bilinear')
        # Stacked Hourglass to do disparity residual 
        '''1/4 Feature Spatial Propagation Here'''
        # Get 1/4 Confidence Here
        confidence2 = self.confidence_branch2(pr2,img_left,img_right,iconv2)
        cur_normal = F.interpolate(assisted_normal,scale_factor=1./4,mode='bilinear',align_corners=False)
        
        pr2 = self.affinity_propagation_list[1](cur_normal,img_left,img_right,pr2,confidence2)
        res2 = self.res_submodule_2(img_left, img_right, pr2,cur_normal,iconv2)

        pr2 = pr2 + res2
        pr2 = self.relu2(pr2)

        # 1/2 Disparity Refinement
        upconv1 = self.upconv1(iconv2)      # Upsample to 1/2 :32 1/2
        concat1 = torch.cat((upconv1, conv1_l), dim=1)  #32+64=96
        iconv1 = self.iconv1(concat1)       # 32 1/2
        # pr1 = self.upflow2to1(pr2)
        pr1 = F.interpolate(pr2, size=(pr2.size()[2] * 2, pr2.size()[3] * 2), mode='bilinear')
        # Get 1/2 Confidence Here
        confidence1 = self.confidence_branch1(pr1,img_left,img_right,iconv1)
        '''1/2 Feature Spatial Propagation Here'''
        cur_normal = F.interpolate(assisted_normal,scale_factor=1./2,mode='bilinear',align_corners=False)
    
        pr1 = self.affinity_propagation_list[2](cur_normal,img_left,img_right,pr1,confidence1)
        res1 = self.res_submodule_1(img_left, img_right, pr1,cur_normal,iconv1) #
        pr1 = pr1 + res1
        pr1 = self.relu1(pr1)

        # Full Scale Disparity refinemnts
        upconv1 = self.upconv0(iconv1)      # 16 1
        concat0 = torch.cat((upconv1, img_left), dim=1)     # 16+3=19 1
        iconv0 = self.iconv0(concat0)       # 16 1
        # pr0 = self.upflow1to0(pr1)
        pr0 = F.interpolate(pr1, size=(pr1.size()[2] * 2, pr1.size()[3] * 2), mode='bilinear')
        # Get full Confidence Here
        confidence0 = self.confidence_branch0(pr0,img_left,img_right,iconv0)
        
        '''Full Scale Feature Spatial Propagation Here'''
        cur_normal = assisted_normal
        
        pr0 = self.affinity_propagation_list[3](cur_normal,img_left,img_right,pr0,confidence0)
        res0 = self.res_submodule_0(img_left, img_right, pr0,cur_normal,iconv0)
        pr0 = pr0 + res0
        pr0 = self.relu0(pr0)

        if training:
            return [pr0, pr1, pr2, pr3],[confidence0,confidence1,confidence2,confidence3]
        else: 
            return pr0,confidence0


if __name__=="__main__":
    left_image = torch.randn(1,3,320,640).cuda()
    right_image = torch.randn(1,3,320,640).cuda()
    assisted_normal = torch.randn(1,3,320,640).cuda()

    mynet = EDNet(res_type='deform_norm',squeezed_volume=True)
    mynet = torch.nn.DataParallel(mynet, device_ids=[0]).cuda()

    disparity_pyramid,conf_pyramid = mynet(left_image,right_image,assisted_normal,True)
    print_tensor_shape(disparity_pyramid)
    print("--------")
    print_tensor_shape(conf_pyramid)