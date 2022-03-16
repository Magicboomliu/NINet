import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../..")
from networks.utilsNet.Disparity_warper import LRwarp_error
from networks.EDNet.submodule import *
from networks.EDNet.occlusion.Residual_Types import SA_Module2, res_submodule_with_normal_deform_S
from torch.nn.init import kaiming_normal
from networks.utilsNet.devtools import print_tensor_shape
from occlusion_ht import get_occlusion_mask
from networks.EDNet.occlusion.tools import conv_bn_relu
from networks.EDNet.occlusion.affinity_propagation import Args,NLSPN
# Get Surface Normal From disparity


class OSNet(nn.Module):
    def __init__(self, batchNorm=False, max_disp=192, input_channel=3, res_type='normal',args=None,prop_list=[1,1,1,1]):
        super(OSNet, self).__init__()
        self.max_disp = max_disp
        self.res_type = res_type
        self.num_neighbors =args.kernel_size* args.kernel_size -1 
        self.args = args
        self.prop_list = prop_list
        
        self.conv1 = conv(3, 64, 7, 2)                      # 1/2
        self.conv2 = ResBlock(64, 128, stride=2)            # 1/4
        self.conv3 = ResBlock(128, 256, stride=2)           # 1/8
        
        self.conv_redir = ResBlock_Deform(256, 32, stride=1)       # skip connection
        
        self.conv3_1 = ResBlock(56, 256)
        
        # Downsample 

        self.conv4 = ResBlock(256, 512, stride=2)           # 1/16
        self.conv4_1 = ResBlock_Deform(512, 512)
        self.conv5 = ResBlock(512, 512, stride=2)           # 1/32
        self.conv5_1 = ResBlock_Deform(512, 512)
        self.conv6 = ResBlock(512, 1024, stride=2)          # 1/64
        self.conv6_1 = ResBlock_Deform(1024, 1024)

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

        
        if self.prop_list[0]==1:
            # Confidence Branch
            self.confidence3 = nn.Sequential(
                conv_bn_relu(128+1,64,3,1,1),
                nn.Conv2d(64,1,3,1,1),
                nn.Sigmoid())
            # Guidance Branch
            self.guide_info3 = nn.Sequential(
                conv_bn_relu(6,64,3,1,1),
                conv_bn_relu(64,64,3,1,1))
            self.guidance3 = nn.Sequential(
                ResBlock(128+64,128,3,1),
                conv_bn_relu(128,self.num_neighbors,3,1,1))

            

        if self.prop_list[1]==1:
            self.confidence2 = nn.Sequential(
                conv_bn_relu(64+1,32,3,1,1),
                nn.Conv2d(32,1,3,1,1),
                nn.Sigmoid())
            self.guide_info2 = nn.Sequential(
                conv_bn_relu(6,32,3,1,1),
                conv_bn_relu(32,32,3,1,1))
            self.guidance2 = nn.Sequential(
                ResBlock(64+32,64,3,1),
                conv_bn_relu(64,self.num_neighbors,3,1,1))


        if self.prop_list[2]==1:
            self.confidence1 = nn.Sequential(
                conv_bn_relu(32+1,32,3,1,1),
                nn.Conv2d(32,1,3,1,1),
                nn.Sigmoid())
            self.guide_info1 = nn.Sequential(
                conv_bn_relu(6,32,3,1,1),
                conv_bn_relu(32,32,3,1,1))
            self.guidance1 = nn.Sequential(
                ResBlock(32+32,32,3,1),
                conv_bn_relu(32,self.num_neighbors,3,1,1))


        if self.prop_list[3]==1:
            self.confidence0 = nn.Sequential(
                conv_bn_relu(32+1,32,3,1,1),
                nn.Conv2d(32,1,3,1,1),
                nn.Sigmoid()
            )
            self.guide_info0 = nn.Sequential(
                conv_bn_relu(6,32,3,1,1),
                conv_bn_relu(32,32,3,1,1))
            self.guidance0 = nn.Sequential(
                ResBlock(32+32,32,3,1),
                conv_bn_relu(32,self.num_neighbors,3,1,1)
        )

        
        # residual learning
        if self.res_type =='deform_norm':
            residual = res_submodule_with_normal_deform_S
        else:
            raise NotImplementedError("Wrong residual type")
        
        self.res_submodule_2 = residual(scale=2, input_layer=64, out_planes=32)
        self.res_submodule_1 = residual(scale=1, input_layer=32, out_planes=32)
        self.res_submodule_0 = residual(scale=0, input_layer=32, out_planes=32)

        #Propagation Branch
        self.prop_layer3 = NLSPN(args=self.args,guide_channel=8,depth_channel=1,prop_kernel=3)
        self.dynamic_occ3 = SA_Module2(input_nc=1)
        self.prop_layer2 = NLSPN(args=self.args,guide_channel=8,depth_channel=1,prop_kernel=3)
        self.dynamic_occ2 = SA_Module2(input_nc=1)
        self.prop_layer1 = NLSPN(args=self.args,guide_channel=8,depth_channel=1,prop_kernel=3)
        self.dynamic_occ1 = SA_Module2(input_nc=1)
        self.prop_layer0 = NLSPN(args=self.args,guide_channel=8,depth_channel=1,prop_kernel=3)
        self.dynamic_occ0 = SA_Module2(input_nc=1)


        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img_left, img_right,assisted_normal,gt_disparity,training=False):
        
        # The Occlusion Mask
        occlusion_mask_list = []
        for i in range(4):
            cur_disparity = F.interpolate(gt_disparity,scale_factor=1./pow(2,i),mode='bilinear',align_corners=False)
            occlus_mask = get_occlusion_mask(cur_disparity,img_left,img_left,mode='mask')
            occlusion_mask_list.append(occlus_mask)

        occlusion_mask_list=occlusion_mask_list[::-1]

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
        cur_normal = F.interpolate(assisted_normal,scale_factor=1./8,mode='bilinear',align_corners=False)
        
        
        if self.prop_list[0]==1:
            # Get 1/8 Confidence Here: Confidence 3
            cur_left3 = F.interpolate(img_left,scale_factor=1./8,align_corners=False,mode='bilinear')
            cur_right3 = F.interpolate(img_right,scale_factor=1./8,align_corners=False,mode='bilinear')
            cur_disp3 = pr3 *1.0/8.0
            warped_left3,_ = disp_warp(cur_right3,cur_disp3)
            error_map3 = torch.abs(cur_left3-warped_left3)
            error_map_squeeze3 = torch.sum(error_map3,dim=1).unsqueeze(1)
            confidence_feature3 = torch.cat((error_map_squeeze3,iconv3),dim=1)
            confidence3 = self.confidence3(confidence_feature3)
            # Get the Guidance Information Here
            guide_info3 = self.guide_info3(torch.cat((cur_left3,cur_normal),dim=1))
            guide_feature3 = self.guidance3(torch.cat((iconv3,guide_info3),dim=1))
            '''Spatail Propagation Here at 1/8 Scale'''
            occlusion_mask3 = self.dynamic_occ3(occlusion_mask_list[0].float())
            pr3, disp_inter,offset,aff,aff_const = self.prop_layer3(pr3,occlusion_mask3,confidence3,guide_feature3)

        upconv2 = self.upconv2(iconv3)      # Upsample to 1/4 :64 1/4
        concat2 = torch.cat((upconv2, conv2_l), dim=1)  # 64+128=192 1/4
        iconv2 = self.iconv2(concat2) #192-->64
 

        '''Here Beigin the Disparity Residual refinement'''
        # 1/4 Disparity Refinement
        # Upsample the 1/8 Disparity to coarse 1/4 Disparity
        pr2 = F.interpolate(pr3, size=(pr3.size()[2] * 2, pr3.size()[3] * 2), mode='bilinear',align_corners=False)
        # Stacked Hourglass to do disparity residual 
        cur_normal = F.interpolate(assisted_normal,scale_factor=1./4,mode='bilinear',align_corners=False)
        res2 = self.res_submodule_2(img_left, img_right, pr2,cur_normal,iconv2)
        pr2 = pr2 + res2
        pr2 = self.relu2(pr2)


        if self.prop_list[1]==1:
            # Get 1/4 Confidence Here
            cur_left2 = F.interpolate(img_left,scale_factor=1./4,align_corners=False,mode='bilinear')
            cur_right2 = F.interpolate(img_right,scale_factor=1./4,align_corners=False,mode='bilinear')
            cur_disp2 = pr2 *1.0/4.0
            warped_left2,_ = disp_warp(cur_right2,cur_disp2)
            error_map2 = torch.abs(cur_left2-warped_left2)
            error_map_squeeze2 = torch.sum(error_map2,dim=1).unsqueeze(1)
            confidence_feature2 = torch.cat((error_map_squeeze2,iconv2),dim=1)
            confidence2 = self.confidence2(confidence_feature2)
            # Get the Guidance Information Here
            guide_info2 = self.guide_info2(torch.cat((cur_left2,cur_normal),dim=1))
            guide_feature2 = self.guidance2(torch.cat((iconv2,guide_info2),dim=1))
            '''Spatail Propagation Here at 1/4 Scale'''
            occlusion_mask2 = self.dynamic_occ2(occlusion_mask_list[1].float())
            pr2, disp_inter,offset,aff,aff_const = self.prop_layer2(pr2,occlusion_mask2,confidence2,guide_feature2)


        # 1/2 Disparity Refinement
        upconv1 = self.upconv1(iconv2)      # Upsample to 1/2 :32 1/2
        concat1 = torch.cat((upconv1, conv1_l), dim=1)  #32+64=96
        iconv1 = self.iconv1(concat1)       # 32 1/2
        # pr1 = self.upflow2to1(pr2)
        pr1 = F.interpolate(pr2, size=(pr2.size()[2] * 2, pr2.size()[3] * 2), mode='bilinear',align_corners=False)
        '''1/2 Feature Spatial Propagation Here'''
        cur_normal = F.interpolate(assisted_normal,scale_factor=1./2,mode='bilinear',align_corners=False)
        res1 = self.res_submodule_1(img_left, img_right, pr1,cur_normal,iconv1) #
        pr1 = pr1 + res1
        pr1 = self.relu1(pr1)

        if self.prop_list[2]==1:
            # Get 1/2 Confidence Here
            cur_left1 = F.interpolate(img_left,scale_factor=1./2,align_corners=False,mode='bilinear')
            cur_right1 = F.interpolate(img_right,scale_factor=1./2,align_corners=False,mode='bilinear')
            cur_disp1 = pr1 *1.0/2.0
            warped_left1,_ = disp_warp(cur_right1,cur_disp1)
            error_map1 = torch.abs(cur_left1-warped_left1)
            error_map_squeeze1 = torch.sum(error_map1,dim=1).unsqueeze(1)
            confidence_feature1 = torch.cat((error_map_squeeze1,iconv1),dim=1)
            confidence1 = self.confidence1(confidence_feature1)
            # Get the Guidance Information Here
            guide_info1 = self.guide_info1(torch.cat((cur_left1,cur_normal),dim=1))
            guide_feature1 = self.guidance1(torch.cat((iconv1,guide_info1),dim=1))
            '''Spatail Propagation Here at 1/2 Scale'''
            occlusion_mask1 = self.dynamic_occ1(occlusion_mask_list[2].float())
            pr1, disp_inter,offset,aff,aff_const = self.prop_layer1(pr1,occlusion_mask1,confidence1,guide_feature1)
 


        # Full Scale Disparity refinemnts
        upconv1 = self.upconv0(iconv1)      # 16 1
        concat0 = torch.cat((upconv1, img_left), dim=1)     # 16+3=19 1
        iconv0 = self.iconv0(concat0)       # 16 1
        # pr0 = self.upflow1to0(pr1)
        pr0 = F.interpolate(pr1, size=(pr1.size()[2] * 2, pr1.size()[3] * 2), mode='bilinear',align_corners=False)
        '''Full Scale Feature Spatial Propagation Here'''
        cur_normal = assisted_normal
        
        res0 = self.res_submodule_0(img_left, img_right, pr0,cur_normal,iconv0)
        pr0 = pr0 + res0
        pr0 = self.relu0(pr0)

        if self.prop_list[3]==1:
            cur_disp0 = pr0 *1.0
            warped_left0,_ = disp_warp(img_right,cur_disp0)
            error_map0 = torch.abs(img_left-warped_left0)
            error_map_squeeze0 = torch.sum(error_map0,dim=1).unsqueeze(1)
            confidence_feature0 = torch.cat((error_map_squeeze0,iconv0),dim=1)
            confidence0 = self.confidence0(confidence_feature0)
            # Get the Guidance Information Here
            guide_info0 = self.guide_info0(torch.cat((img_left,cur_normal),dim=1))
            guide_feature0 = self.guidance0(torch.cat((iconv0,guide_info0),dim=1))
            '''Spatail Propagation Here at Full Sice Scale'''
            occlusion_mask0 = self.dynamic_occ0(occlusion_mask_list[3].float())
            pr0, disp_inter,offset,aff,aff_const = self.prop_layer0(pr0,occlusion_mask0,confidence0,guide_feature0)




        if training:
            return [pr0, pr1, pr2, pr3]
        else: 
            return pr0


if __name__=="__main__":
    input_tensor = torch.randn(1,3,320,640).cuda()
    assisted_normal = torch.ones(1,3,320,640).cuda()
    gt_disparity = torch.ones(1,1,320,640).cuda()

    nlsp_args = Args(kernel_size=3,propagate_time=10,affinity="TGASS",affinity_gamma=0.5)
    mynet = OSNet(res_type='deform_norm',args=nlsp_args,prop_list=[1,0,1,0]).cuda()
    disp_pyramid = mynet(input_tensor,input_tensor,assisted_normal,gt_disparity,True)
    print_tensor_shape(disp_pyramid)
