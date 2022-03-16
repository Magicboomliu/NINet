import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("../../..")
from networks.utilsNet.Disparity_warper import LRwarp_error
from networks.utilsNet.spatial_propagation import Affinity_propagation
from networks.EDNet.submodule import *
from torch.nn.init import kaiming_normal
from networks.utilsNet.devtools import print_tensor_shape

from networks.EDNet.submodule import *
from networks.EDNet.two_branches.submodule import normres_submodule

from torch.nn.init import kaiming_normal

# Modal Learning
class Modal_Learning(nn.Module):
    def __init__(self, batchNorm=False, max_disp=192, 
                input_channel=3, res_type='normal'):
        super(Modal_Learning, self).__init__()
        
        self.max_disp = max_disp
        self.res_type = res_type
        
        self.conv1 = conv(input_channel, 64, 7, 2)                      # 1/2
        self.conv2 = ResBlock(64, 128, stride=2)            # 1/4
        self.conv3 = ResBlock(128, 256, stride=2)           # 1/8
        
        self.conv_redir = ResBlock(256, 32, stride=1)       # skip connection

        # Here there are two branches: 
        # Branch One: Surface Normal Estimation
        # Branch Two : Disparity Estimation
        '''First Here is used for downsample'''
        num_of_tasks = 2

        self.task_branches = nn.ModuleList()
        for i in range(num_of_tasks):
            if i ==0:
                downsample_branch1 = nn.ModuleList()
                self.conv3_1_a = ResBlock(56, 256)
                downsample_branch1.append(self.conv3_1_a)
                self.conv4_a = ResBlock(256, 512, stride=2)           # 1/16
                downsample_branch1.append(self.conv4_a)
                self.conv4_1_a = ResBlock(512, 512)
                downsample_branch1.append(self.conv4_1_a)
                self.conv5_a = ResBlock(512, 512, stride=2)           # 1/32
                downsample_branch1.append(self.conv5_a)
                self.conv5_1_a = ResBlock(512, 512)
                downsample_branch1.append(self.conv5_1_a)
                self.conv6_a = ResBlock(512, 1024, stride=2)          # 1/64
                downsample_branch1.append(self.conv6_a)
                self.conv6_1_a = ResBlock(1024, 1024)
                downsample_branch1.append(self.conv6_1_a)
                # Branch One
                self.task_branches.append(downsample_branch1)
                
            if i ==1:
                downsample_branch2 = nn.ModuleList()
                self.conv3_1_b = ResBlock(56, 256)
                downsample_branch2.append(self.conv3_1_b)
                self.conv4_b = ResBlock(256, 512, stride=2)           # 1/16
                downsample_branch2.append(self.conv4_b)
                self.conv4_1_b = ResBlock(512, 512)
                downsample_branch2.append(self.conv4_1_b)
                self.conv5_b = ResBlock(512, 512, stride=2)           # 1/32
                downsample_branch2.append(self.conv5_b)
                self.conv5_1_b = ResBlock(512, 512)
                downsample_branch2.append(self.conv5_1_b)
                self.conv6_b = ResBlock(512, 1024, stride=2)          # 1/64
                downsample_branch2.append(self.conv6_b)
                self.conv6_1_b = ResBlock(1024, 1024)
                downsample_branch2.append(self.conv6_1_b)
                # Branch Two
                self.task_branches.append(downsample_branch2)
                
        self.upsample_branches = nn.ModuleList()
        for k in range(num_of_tasks):
            if k==0:
                upsample_branch1 = nn.ModuleList()
                self.iconv5_n = nn.ConvTranspose2d(1024, 512, 3, 1, 1) #0
                upsample_branch1.append(self.iconv5_n)
                self.iconv4_n = nn.ConvTranspose2d(768, 256, 3, 1, 1) #1
                upsample_branch1.append(self.iconv4_n)
                self.iconv3_n = nn.ConvTranspose2d(384, 128, 3, 1, 1) #2
                upsample_branch1.append(self.iconv3_n)
                self.iconv2_n = nn.ConvTranspose2d(192,64, 3, 1, 1) #3
                upsample_branch1.append(self.iconv2_n)
                self.iconv1_n = nn.ConvTranspose2d(96, 32, 3, 1, 1) #4
                upsample_branch1.append(self.iconv1_n)
                self.iconv0_n = nn.ConvTranspose2d(19, 32, 3, 1, 1) #5
                upsample_branch1.append(self.iconv0_n)

                # Deconvoluton : nn.ConvTranspose2d + Relu
                self.upconv5_n = deconv(1024, 512)
                upsample_branch1.append(self.upconv5_n) #6
                self.upconv4_n = deconv(512, 256)
                upsample_branch1.append(self.upconv4_n) #7
                self.upconv3_n = deconv(256, 128)
                upsample_branch1.append(self.upconv3_n) #8
                self.upconv2_n = deconv(128, 64)
                upsample_branch1.append(self.upconv2_n) #9
                self.upconv1_n = deconv(64, 32)
                upsample_branch1.append(self.upconv1_n) #10
                self.upconv0_n = deconv(32, 16)
                upsample_branch1.append(self.upconv0_n) #11

                self.upsample_branches.append(upsample_branch1)
            
            if k==1:
                upsample_branch2 = nn.ModuleList()
                self.iconv5_d = nn.ConvTranspose2d(1024, 512, 3, 1, 1) #0
                upsample_branch2.append(self.iconv5_d)
                self.iconv4_d = nn.ConvTranspose2d(768, 256, 3, 1, 1) #1
                upsample_branch2.append(self.iconv4_d)
                self.iconv3_d = nn.ConvTranspose2d(384, 128, 3, 1, 1) #2
                upsample_branch2.append(self.iconv3_d)
                self.iconv2_d = nn.ConvTranspose2d(192,64, 3, 1, 1) #3
                upsample_branch2.append(self.iconv2_d)
                self.iconv1_d = nn.ConvTranspose2d(96, 32, 3, 1, 1) #4
                upsample_branch2.append(self.iconv1_d)
                self.iconv0_d = nn.ConvTranspose2d(19, 32, 3, 1, 1) #5
                upsample_branch2.append(self.iconv0_d)

                # Deconvoluton : nn.ConvTranspose2d + Relu
                self.upconv5_d = deconv(1024, 512)
                upsample_branch2.append(self.upconv5_d) #6
                self.upconv4_d = deconv(512, 256)
                upsample_branch2.append(self.upconv4_d) #7
                self.upconv3_d = deconv(256, 128)
                upsample_branch2.append(self.upconv3_d) #8
                self.upconv2_d = deconv(128, 64)
                upsample_branch2.append(self.upconv2_d) #9
                self.upconv1_d = deconv(64, 32)
                upsample_branch2.append(self.upconv1_d) #10
                self.upconv0_d = deconv(32, 16)
                upsample_branch2.append(self.upconv0_d) #11

                self.upsample_branches.append(upsample_branch2)

        '''1/8 Normal and Disp Estimation'''
        self.disp3 = nn.Conv2d(128,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.normal3 = nn.Conv2d(128,3,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu0 = nn.ReLU(inplace=True)


        # Normal residual refinement
        normal_residual = normres_submodule
        self.res_submodule_2_n = normal_residual(scale=2, input_layer=64, out_planes=32)
        self.res_submodule_1_n = normal_residual(scale=1, input_layer=32, out_planes=32)
        self.res_submodule_0_n = normal_residual(scale=0, input_layer=32, out_planes=32)
        

        # Disparity residual refinement
        if self.res_type == 'attention':
            disp_residual = res_submodule_attention
        elif self.res_type == 'normal':
            disp_residual = res_submodule
        elif self.res_type == 'noerror':
            disp_residual = res_submodule_attention_noerror
        elif self.res_type =='fsp':
            disp_residual = res_submodule_feature_porpagation
        elif self.res_type =='deform_norm':
            disp_residual = res_submodule_with_normal_deform
        else:
            raise NotImplementedError("Wrong residual type")
        
        self.res_submodule_2_d = disp_residual(scale=2, input_layer=64, out_planes=32)
        self.res_submodule_1_d = disp_residual(scale=1, input_layer=32, out_planes=32)
        self.res_submodule_0_d = disp_residual(scale=0, input_layer=32, out_planes=32)


        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self,img_left,img_right,is_training= False):
        
        # Shared Feature Extraction
        conv1_l = self.conv1(img_left)          # 64 1/2
        conv2_l = self.conv2(conv1_l)           # 128 1/4
        conv3_l = self.conv3(conv2_l)           # 256 1/8

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3_r = self.conv3(conv2_r)           # 1/8

        # build Shared Corr Cost Volume 
        out_corr = build_corr(conv3_l,conv3_r, self.max_disp//8)
        out_conv3a_redir = self.conv_redir(conv3_l)
        in_conv3b = torch.cat((out_conv3a_redir, out_corr), dim=1)         # 24+32=56

        # Downsample to 1/32 Scale
        normal_branch1 = self.task_branches[0]
        disp_branch1 = self.task_branches[1]

        conv3b_n = normal_branch1[0](in_conv3b) # 256
        conv4a_n = normal_branch1[1](conv3b_n)           
        conv4b_n = normal_branch1[2](conv4a_n)  # 512 1/16
        conv5a_n = normal_branch1[3](conv4b_n)  
        conv5b_n = normal_branch1[4](conv5a_n)  # 512 1/32
        conv6a_n = normal_branch1[5](conv5b_n)               
        conv6b_n = normal_branch1[6](conv6a_n)  # 1024 1/64

        conv3b_d = disp_branch1[0](in_conv3b)  # 256
        conv4a_d = disp_branch1[1](conv3b_d)
        conv4b_d = disp_branch1[2](conv4a_d) # 512 1/16
        conv5a_d = disp_branch1[3](conv4b_d)
        conv5b_d = disp_branch1[4](conv5a_d) # 512 1/32
        conv6a_d = disp_branch1[5](conv5b_d)
        conv6b_d = disp_branch1[6](conv6a_d) # 1024 1/64

        normal_branch2 = self.upsample_branches[0]

        disp_branch2 = self.upsample_branches[1]

        upconv5_n = normal_branch2[6](conv6b_n)  # 512 1/32
        concat5_n = torch.cat((upconv5_n, conv5b_n), dim=1)   # 1024 1/32
        iconv5_n = normal_branch2[0](concat5_n)    # 512

        upconv4_n = normal_branch2[7](iconv5_n)   # 256 1/16
        concat4_n = torch.cat((upconv4_n, conv4b_n), dim=1)   # 768 1/16
        iconv4_n =  normal_branch2[1](concat4_n)       # 256 1/16

        upconv3_n = normal_branch2[8](iconv4_n)   # 128 1/8
        concat3_n = torch.cat((upconv3_n, conv3b_n), dim=1)    # 128+256=384 1/8
        iconv3_n = normal_branch2[2](concat3_n)     # 128
        
        # Normal at 1/8 Scale
        normal3 = self.normal3(iconv3_n)
        normal3 = F.normalize(normal3,dim=1)


        upconv5_d = disp_branch2[6](conv6b_d)  # 512 1/32
        concat5_d = torch.cat((upconv5_d, conv5b_d), dim=1)   # 1024 1/32
        iconv5_d = disp_branch2[0](concat5_d)    # 512

        upconv4_d = disp_branch2[7](iconv5_d)   # 256 1/16
        concat4_d = torch.cat((upconv4_d, conv4b_d), dim=1)   # 768 1/16
        iconv4_d =  disp_branch2[1](concat4_d)       # 256 1/16

        upconv3_d = disp_branch2[8](iconv4_d)   # 128 1/8
        concat3_d = torch.cat((upconv3_d, conv3b_d), dim=1)    # 128+256=384 1/8
        iconv3_d = disp_branch2[2](concat3_d)     # 128

        # Disp at 1/8 Scale
        disp3 = self.disp3(iconv3_d)
        disp3 = self.relu3(disp3)

        # Surface Normal Refinement at 1/4 Scale
        upconv2_n = normal_branch2[9](iconv3_n)     # 64 1/4
        concat2_n = torch.cat((upconv2_n, conv2_l), dim=1)  # 192 1/4
        iconv2_n = normal_branch2[3](concat2_n)

        normal2 = F.interpolate(normal3, size=(normal3.size()[2] * 2, normal3.size()[3] * 2), mode='bilinear')
        res2_n = self.res_submodule_2_n(img_left, img_right, normal2, iconv2_n)
        normal2 = normal2 + res2_n
        normal2 = F.normalize(normal2,dim=1)

        # Disparity Refinment at 1/4 Scale
        upconv2_d = disp_branch2[9](iconv3_d)      # 64 1/4
        concat2_d = torch.cat((upconv2_d, conv2_l), dim=1)  # 192 1/4
        iconv2_d = disp_branch2[3](concat2_d)    
    
        disp2 = F.interpolate(disp3, size=(disp3.size()[2] * 2, disp3.size()[3] * 2), mode='bilinear')
        if self.res_type=="deform_norm":
            res2_d = self.res_submodule_2_d(img_left, img_right, disp2, normal2.detach(),iconv2_d)
        else:
            res2_d = self.res_submodule_2_d(img_left, img_right, disp2, iconv2_d)
        
        disp2 = disp2 + res2_d
        disp2 = self.relu2(disp2)

        # Surface Normal Refinement at 1/2 Scale
        upconv1_n =normal_branch2[10](iconv2_n)      # 32 1/2
        concat1_n = torch.cat((upconv1_n, conv1_l), dim=1)  #32+64=96
        iconv1_n =normal_branch2[4](concat1_n)       # 32 1/2

        normal1 = F.interpolate(normal2, size=(normal2.size()[2] * 2, normal2.size()[3] * 2), mode='bilinear')
        res1_n = self.res_submodule_1_n(img_left, img_right, normal1, iconv1_n) #
        normal1 = normal1 + res1_n
        normal1 = F.normalize(normal1,dim=1)

        # Disparity Refinemnet at 1/2 Scale
        upconv1_d = disp_branch2[10](iconv2_d)     # 32 1/2
        concat1_d = torch.cat((upconv1_d, conv1_l), dim=1)  #32+64=96
        iconv1_d = disp_branch2[4](concat1_d)    # 32 1/2
    
        disp1 = F.interpolate(disp2, size=(disp2.size()[2] * 2, disp2.size()[3] * 2), mode='bilinear')
        if self.res_type=="deform_norm":
            res1_d = self.res_submodule_1_d(img_left, img_right, disp1, normal1.detach(),iconv1_d)
        else:
            res1_d = self.res_submodule_1_d(img_left, img_right, disp1, iconv1_d) #
        
        disp1 = disp1 + res1_d
        disp1 = self.relu1(disp1)

        # Surface Normal Estimation at Full Scale
        upconv1_n = normal_branch2[11](iconv1_n)    # 16 1
        concat0_n = torch.cat((upconv1_n, img_left), dim=1)     # 16+3=19 1
        iconv0_n =  normal_branch2[5](concat0_n)                # 16 1
        
        normal0 = F.interpolate(normal1, size=(normal1.size()[2] * 2, normal1.size()[3] * 2), mode='bilinear')
        res0_n = self.res_submodule_0_n(img_left, img_right, normal0, iconv0_n)
        normal0 = normal0 + res0_n
        normal0 = F.normalize(normal0,dim=1)

        # Disparity Refinemnt the Full Scale
        upconv1_d = disp_branch2[11](iconv1_d)          # 16 1
        concat0_d = torch.cat((upconv1_d, img_left), dim=1)     # 16+3=19 1
        iconv0_d = disp_branch2[5](concat0_d)               # 16 1
    
        disp0 = F.interpolate(disp1, size=(disp1.size()[2] * 2, disp1.size()[3] * 2), mode='bilinear')
        if self.res_type=="deform_norm":
            res0_d = self.res_submodule_0_d(img_left, img_right, disp0,normal0.detach(),iconv0_d)
        else:
            res0_d = self.res_submodule_0_d(img_left, img_right, disp0, iconv0_d)
        
        disp0 = disp0 + res0_d
        disp0 = self.relu0(disp0)

        if is_training:
            return [disp0,disp1,disp2,disp3],[normal0,normal1,normal2,normal3]
        else:
            return disp0,normal0

if __name__ =="__main__":
    input_tensor = torch.randn(1,3,320,640).cuda()
    liuzihuanet = Modal_Learning(res_type="deform_norm").cuda()
    disp_pyramid,surface_normal_pyramid = liuzihuanet(input_tensor,input_tensor,True)
    print_tensor_shape(disp_pyramid)
    print("--------------------------------")
    print_tensor_shape(surface_normal_pyramid)