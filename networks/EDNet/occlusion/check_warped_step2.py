import sys
sys.path.append("../../..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal
from networks.EDNet.submodule import *
from networks.EDNet.occlusion.tools import res_submodule_with_norm_wo_right
from networks.utilsNet.devtools import print_tensor_shape
from occlusion_ht import get_occlusion_mask
from networks.EDNet.occlusion.tools import conv_bn_relu
from deform.modules.modulated_deform_conv import ModulatedDeformConvPack
from networks.EDNet.occlusion.propagation import OSNLP,Propagation_Args
from networks.EDNet.occlusion.aggregation import CrossAggregationCostVolume

'''Try 1/4 and 1/8 Cost Aggregation'''

class OSNet(nn.Module):
    def __init__(self, batchNorm=False, max_disp=192, input_channel=3, res_type='normal',args=None,
                    prop_layer_list=[1,1,1,1]):
        super(OSNet, self).__init__()
        self.max_disp = max_disp
        self.res_type = res_type
        self.args = args
        self.num_neighbors = self.args.prop_kernel* self.args.prop_kernel -1
        self.prop_layer_list = prop_layer_list

        self.conv1 = conv(3, 64, 7, 2)                      # 1/2
        self.conv2 = ResBlock(64, 128, stride=2)            # 1/4
        self.conv3 = ResBlock(128, 256, stride=2)           # 1/8

        self.cross_aggregation = CrossAggregationCostVolume(num_blocks=2,
                                                            max_disp_list=[self.max_disp//4,self.max_disp//8]
                                                            ,deform=True)
        
        self.conv3_1 = DeconBlock(56, 256,kernel_size=3,stride=1)
        self.conv_redir = ResBlock(256, 32, stride=1)       # skip connection

    

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

        # Residual learning
        if self.res_type =='deform_norm':
            residual = res_submodule_with_normal_deform
        elif self.res_type =='deform_occlus':
            residual = res_submodule_with_norm_occlu
        elif self.res_type =="wr":
            residual = res_submodule_with_norm_wo_right
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

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f
    
    def forward(self, img_left, img_right,assisted_normal,gt_disparity,training=False):
        
        # The Occlusion Mask
        occlusion_mask_list = []
        for i in range(4):
            cur_disparity = F.interpolate(gt_disparity,scale_factor=1./pow(2,i),mode='bilinear',align_corners=False)
            occlus_mask = get_occlusion_mask(cur_disparity,img_left,img_left,mode='mask')
            occlusion_mask_list.append(occlus_mask)

        occlusion_mask_list=occlusion_mask_list[::-1]
        
        # Split left image and right image
        conv1_l = self.conv1(img_left)          # 64 1/2
        conv2_l = self.conv2(conv1_l)           # 128 1/4
        conv3_l = self.conv3(conv2_l)           # 256 1/8

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3_r = self.conv3(conv2_r)           # 1/8

        # Build Corr 
        out_corr_1 = build_corr(conv2_l,conv2_r,self.max_disp//4)
        # 24 Channels Probility Cost Volume
        out_corr_0 = build_corr(conv3_l,conv3_r, self.max_disp//8)
        out_corr = self.cross_aggregation(out_corr_1,out_corr_0)

        # 32 channels cost volume
        out_conv3a_redir = self.conv_redir(conv3_l)
        # Concated Cost Volume
        in_conv3b = torch.cat((out_conv3a_redir, out_corr), dim=1)         # 24+32=56

        # Here use deformable convolution for aggregation
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
        
        # Get the 1/8 Surface Normal Here
        cur_normal = F.interpolate(assisted_normal,scale_factor=1./8,mode='bilinear',align_corners=False)
        
        upconv2 = self.upconv2(iconv3)      # Upsample to 1/4 :64 1/4
        concat2 = torch.cat((upconv2, conv2_l), dim=1)  # 64+128=192 1/4
        iconv2 = self.iconv2(concat2) #192-->64

        '''Here Beigin the Disparity Residual refinement'''
        # 1/4 Disparity Refinement
        # Upsample the 1/8 Disparity to coarse 1/4 Disparity
        pr2 = F.interpolate(pr3, size=(pr3.size()[2] * 2, pr3.size()[3] * 2), mode='bilinear')
        # Stacked Hourglass to do disparity residual 
        cur_normal = F.interpolate(assisted_normal,scale_factor=1./4,mode='bilinear',align_corners=False)
        if self.res_type != 'deform_norm':
            res2 = self.res_submodule_2(img_left, img_right, pr2,cur_normal,iconv2,occlusion_mask_list[1])
        else:
            res2 = self.res_submodule_2(img_left, img_right, pr2,cur_normal,iconv2)
        
        pr2 = pr2 + res2
        pr2 = self.relu2(pr2)

         

        # 1/2 Disparity Refinement
        upconv1 = self.upconv1(iconv2)      # Upsample to 1/2 :32 1/2
        concat1 = torch.cat((upconv1, conv1_l), dim=1)  #32+64=96
        iconv1 = self.iconv1(concat1)       # 32 1/2

        pr1 = F.interpolate(pr2, size=(pr2.size()[2] * 2, pr2.size()[3] * 2), mode='bilinear')
        cur_normal = F.interpolate(assisted_normal,scale_factor=1./2,mode='bilinear',align_corners=False)
        if self.res_type !='deform_norm':
            res1 = self.res_submodule_1(img_left, img_right, pr1,cur_normal,iconv1,occlusion_mask_list[2])
        else:
            res1 = self.res_submodule_1(img_left, img_right, pr1,cur_normal,iconv1) 
        pr1 = pr1 + res1
        pr1 = self.relu1(pr1)

        # Full Scale Disparity refinemnts
        upconv1 = self.upconv0(iconv1)      # 16 1
        concat0 = torch.cat((upconv1, img_left), dim=1)     # 16+3=19 1
        iconv0 = self.iconv0(concat0)       # 16 1


        pr0 = F.interpolate(pr1, size=(pr1.size()[2] * 2, pr1.size()[3] * 2), mode='bilinear')
        cur_normal = assisted_normal
        if self.res_type !='deform_norm':
            res0 = self.res_submodule_0(img_left, img_right, pr0,cur_normal,iconv0,occlusion_mask_list[3])
        else:
            res0 = self.res_submodule_0(img_left, img_right, pr0,cur_normal,iconv0)
        pr0 = pr0 + res0
        pr0 = self.relu0(pr0)
  

        if training:
            return [pr0, pr1, pr2, pr3]
        else: 
            return pr0


if __name__=="__main__":

    # Test For inference
    input_tensor = torch.randn(1,3,320,640).cuda()
    assisted_normal = torch.ones(1,3,320,640).cuda()
    disparity_gt = torch.abs(torch.randn(1,1,320,640)).cuda()
    # Propagation args
    prop_args = Propagation_Args(prop_time=3,affinity="ASS",affinity_gamma=0.5,prop_kernel=3,conf_prop=True,
    preserve_input=True)
    
    mynet = OSNet(res_type='wr',args=prop_args,prop_layer_list=[1,1,1,1]).cuda()
    disp_pyramid = mynet(input_tensor,input_tensor,assisted_normal,disparity_gt,True)
    print_tensor_shape(disp_pyramid)
