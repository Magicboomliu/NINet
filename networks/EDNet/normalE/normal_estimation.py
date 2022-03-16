
'''Input: Left Right Image
  Output: Surface Normal'''
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("../../..")
from networks.EDNet.normalE.submodule import *
from torch.nn.init import kaiming_normal
from networks.utilsNet.devtools import print_tensor_shape

class NormalNet(nn.Module):
    def __init__(self, batchNorm=False,max_disp=192):
        super(NormalNet, self).__init__()

        self.max_disp = max_disp

        self.conv1 = conv(3, 64, 7, 2)                      # 1/2
        self.conv2 = ResBlock(64, 128, stride=2)            # 1/4
        self.conv3 = ResBlock(128, 256, stride=2)           # 1/8
        
        self.conv_redir = ResBlock(256, 32, stride=1)       # skip connection

        
        self.conv3_1 = ResBlock(56, 256)
        self.conv4 = ResBlock(256, 512, stride=2)           # 1/16
        self.conv4_1 = ResBlock(512, 512)
        self.conv5 = ResBlock(512, 512, stride=2)           # 1/32
        self.conv5_1 = ResBlock(512, 512)
        self.conv6 = ResBlock(512, 1024, stride=2)          # 1/64
        self.conv6_1 = ResBlock(1024, 1024)

        self.iconv5 = nn.ConvTranspose2d(1024, 512, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d(768, 256, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(384, 128, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(192,64, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(96, 32, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(19, 32, 3, 1, 1)
        
        # Deconvoluton : nn.ConvTranspose2d + Relu
        self.upconv5 = deconv(1024, 512)
        self.upconv4 = deconv(512, 256)
        self.upconv3 = deconv(256, 128)
        self.upconv2 = deconv(128, 64)
        self.upconv1 = deconv(64, 32)
        self.upconv0 = deconv(32, 16)

        # Normalization
        self.normal3 = nn.Conv2d(128,3,kernel_size=3,stride=1,padding=1,bias=False)


        residual = normres_submodule
        #residual = normres_submodule_deform
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


    def forward(self,img_left,img_right,is_training=False):

        
        
        conv1_l = self.conv1(img_left)          # 64 1/2
        conv2_l = self.conv2(conv1_l)           # 128 1/4
        conv3_l = self.conv3(conv2_l)           # 256 1/8

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3_r = self.conv3(conv2_r)           # 1/8
        
        # build corr 
        out_corr = build_corr(conv3_l,conv3_r, self.max_disp//8)
        out_conv3a_redir = self.conv_redir(conv3_l)
        in_conv3b = torch.cat((out_conv3a_redir, out_corr), dim=1)         # 24+32=56
        
        
        conv3b = self.conv3_1(in_conv3b)    # 256
        conv4a = self.conv4(conv3b)         
        conv4b = self.conv4_1(conv4a)       # 512 1/16
        conv5a = self.conv5(conv4b)
        conv5b = self.conv5_1(conv5a)       # 512 1/32
        conv6a = self.conv6(conv5b)
        conv6b = self.conv6_1(conv6a)       # 1024 1/64

        upconv5 = self.upconv5(conv6b)      # 512 1/32
        concat5 = torch.cat((upconv5, conv5b), dim=1)   # 1024 1/32
        iconv5 = self.iconv5(concat5)       # 512

        upconv4 = self.upconv4(iconv5)      # 256 1/16
        concat4 = torch.cat((upconv4, conv4b), dim=1)   # 768 1/16
        iconv4 = self.iconv4(concat4)       # 256 1/16

        upconv3 = self.upconv3(iconv4)      # 128 1/8
        concat3 = torch.cat((upconv3, conv3b), dim=1)    # 128+256=384 1/8
        iconv3 = self.iconv3(concat3)       # 128
        


        # Normal at 1/8 Scale
        normal3 = self.normal3(iconv3)
        normal3 = F.normalize(normal3,dim=1)

        # Normal at 1/4 Sclae
        upconv2 = self.upconv2(iconv3)      # 64 1/4
        concat2 = torch.cat((upconv2, conv2_l), dim=1)  # 192 1/4
        iconv2 = self.iconv2(concat2)
        # pr2 = self.upflow3to2(pr3)
        normal2 = F.interpolate(normal3, size=(normal3.size()[2] * 2, normal3.size()[3] * 2), mode='bilinear')
        res2 = self.res_submodule_2(img_left, img_right, normal2, iconv2)
        normal2 = normal2 + res2
        normal2 = F.normalize(normal2,dim=1)

        # Normal at 1/2 Scale
        upconv1 = self.upconv1(iconv2)      # 32 1/2
        concat1 = torch.cat((upconv1, conv1_l), dim=1)  #32+64=96
        iconv1 = self.iconv1(concat1)       # 32 1/2

        normal1 = F.interpolate(normal2, size=(normal2.size()[2] * 2, normal2.size()[3] * 2), mode='bilinear')
        res1 = self.res_submodule_1(img_left, img_right, normal1, iconv1) #
        normal1 = normal1 + res1
        normal1 = F.normalize(normal1,dim=1)

        upconv1 = self.upconv0(iconv1)      # 16 1
        concat0 = torch.cat((upconv1, img_left), dim=1)     # 16+3=19 1
        iconv0 = self.iconv0(concat0)       # 16 1
        
        normal0 = F.interpolate(normal1, size=(normal1.size()[2] * 2, normal1.size()[3] * 2), mode='bilinear')
        res0 = self.res_submodule_0(img_left, img_right, normal0, iconv0)
        normal0 = normal0 + res0
        normal0 = F.normalize(normal0,dim=1)
        
        

        
        if is_training:
          return[normal0,normal1,normal2,normal3]
        else:
          return normal0


if __name__ =="__main__":
  input_left = torch.randn(1,3,320,640).cuda()
  normal_estimation_net = NormalNet().cuda()  
  features = normal_estimation_net(input_left,input_left,True)
  print_tensor_shape(features)

