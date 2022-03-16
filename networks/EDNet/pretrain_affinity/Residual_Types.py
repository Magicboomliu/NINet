import sys

from torch._C import set_flush_denormal
from torch.nn.modules.activation import Tanh
from torch.nn.modules.loss import TripletMarginLoss
sys.path.append("../../..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from networks.utilsNet.Disparity_warper import disp_warp
from networks.utilsNet.attention.CBAM import CBAMBlock
from deform.modules.modulated_deform_conv import ModulatedDeformConvPack


# Residual Prediction with Surface Normal
class res_submodule_with_normal_deform_Simple(nn.Module):
    def __init__(self, scale, input_layer, out_planes=64):
        super(res_submodule_with_normal_deform_Simple, self).__init__()
        #self.resample = Resample2d()
        self.pool = nn.AvgPool2d(2**scale, 2**scale)

        self.attention = SA_Module2(input_nc=13)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+13, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),

        )
        
        self.conv2 = nn.Sequential(
            ModulatedDeformConvPack(out_planes,out_planes*2,kernel_size=3,stride=1,padding=1,bias=False),
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

        self.relu = nn.ReLU(inplace=True)
    
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

        disp = disp + res
        disp = self.relu(disp)

        return disp



class res_submodule_with_normal_deform_S(nn.Module):
    def __init__(self, scale, input_layer, out_planes=64):
        super(res_submodule_with_normal_deform_S, self).__init__()
        #self.resample = Resample2d()
        self.pool = nn.AvgPool2d(2**scale, 2**scale)

        self.attention = SA_Module2(input_nc=13)
        

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+13, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),

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
        



# Spatial Attention
class SA_Module2(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=16):
        super(SA_Module2, self).__init__()
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




# Test the Module
if __name__=="__main__":
    test_disparity = torch.abs(torch.randn(1,1,160,320).cuda())
    test_surface_normal = torch.randn(1,3,160,320).cuda()
    test_feature = torch.randn(1,64,160,320).cuda()
    left_image = torch.abs(torch.randn(1,3,320,640)).cuda()
    right_image = torch.abs(torch.randn(1,3,320,640)).cuda()

    residual_module = res_submodule_lstm(scale=1,input_layer=64,out_planes=64,iteration=2).cuda()
    cur_disp,cur_h,cur_c = residual_module(left_image,right_image,test_disparity,test_surface_normal,test_feature)

    print(cur_disp.shape)
    
    
