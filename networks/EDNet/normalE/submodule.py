import sys
sys.path.append("../..")
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from networks.utilsNet.Disparity_warper import disp_warp
from deform.modules.modulated_deform_conv import ModulatedDeformConvPack


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

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
        nn.BatchNorm2d(out_planes)
    )
    
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, stride=stride, kernel_size=3, padding=1, bias=False)

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )

def conv_Relu(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False),
        nn.ReLU(True)
    )

def convbn_3d(in_planes, out_planes, kernel_size=3, stride=1, pad=1):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

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

# def form_cost_volume(ref_feature, tar_feature, disp):
#     # B, C, H, W = ref_feature.shape
#     cost = torch.FloatTensor(ref_feature.size()[0], ref_feature.size()[1], disp, ref_feature.size()[2], ref_feature.size()[3]).zero_().cuda()
#     for i in range(disp):
#         if i > 0:
#             cost[:, :, i, :, i:] = torch.abs(ref_feature[:, :, :, i:] - tar_feature[:, :, :, :-i])
#         else:
#             cost[:, :, i, :, i:] = torch.abs(ref_feature - tar_feature)
#     cost = cost.contiguous()
#     return cost

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
        # 残差的输出是不使用relu激活函数的
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
        # image warp
        # self.resample = Resample2d()
        self.pool = nn.AvgPool2d(2**scale, 2**scale)
        
        self.attention = SA_Module(input_nc=10)

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

        dummy_flow = torch.autograd.Variable(torch.zeros(disp.data.shape).cuda())
        disp_ = disp / scale            # align the disparity to the proper scale
        flow = torch.cat((disp_, dummy_flow), dim=1)
  
        # call new warp function
        left_rec ,mask= disp_warp(right,disp_)
    
        error_map = left_rec - left

        query = torch.cat((left, right, error_map, disp_), dim=1)
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

class res_submodule(nn.Module):
    def __init__(self, scale, input_layer, out_planes=64):
        super(res_submodule, self).__init__()
        #self.resample = Resample2d()
        self.pool = nn.AvgPool2d(2**scale, 2**scale)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )
        
        self.conv2 = nn.Sequential(
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
        left_rec,mask = disp_warp(right,disp_)
        error_map = left_rec -left

        conv1 = self.conv1(torch.cat((left, right, disp_, error_map, feature), dim=1))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((left, right, disp_, error_map, feature), dim=1)))

        res = self.res(conv6) * scale
        return res

class res_submodule_with_normal(nn.Module):
    def __init__(self, scale, input_layer, out_planes=64):
        super(res_submodule_with_normal, self).__init__()
        #self.resample = Resample2d()
        self.pool = nn.AvgPool2d(2**scale, 2**scale)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+13, out_planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )
        
        self.conv2 = nn.Sequential(
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
    
    def forward(self, left, right, disp, surface_normal,feature):
     
        scale = left.size()[2] / disp.size()[2]
        left = self.pool(left)
        right = self.pool(right)

        # dummy_flow = torch.autograd.Variable(torch.zeros(disp.data.shape).cuda())
        disp_ = disp / scale            # align the disparity to the proper scale
        # flow = torch.cat((disp_, dummy_flow), dim=1)
        left_rec,mask = disp_warp(right,disp_)
        error_map = left_rec -left

        conv1 = self.conv1(torch.cat((left, right, disp_, surface_normal,error_map, feature), dim=1))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((left, right, disp_, error_map, feature), dim=1)))

        res = self.res(conv6) * scale
        return res







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



class normres_submodule(nn.Module):
    def __init__(self, scale, input_layer, out_planes=64):
        super(normres_submodule, self).__init__()
        #self.resample = Resample2d()
        '''
        Left+Right + normal =9
        '''
        self.pool = nn.AvgPool2d(2**scale, 2**scale)

        raw_feature_num =9

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+raw_feature_num, out_planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )
        
        self.conv2 = nn.Sequential(
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
            nn.Conv2d(input_layer+raw_feature_num, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.res = nn.Conv2d(out_planes, 3, 1, 1, bias=False)
    
    def forward(self, left, right, surface_normal, feature):
     
        left = self.pool(left)
        right = self.pool(right)

        conv1 = self.conv1(torch.cat((left, right, surface_normal, feature), dim=1))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((left, right, surface_normal, feature), dim=1)))

        res = self.res(conv6)
        return res


class normres_submodule_deform(nn.Module):
    def __init__(self, scale, input_layer, out_planes=64):
        super(normres_submodule_deform, self).__init__()
        #self.resample = Resample2d()
        '''
        Left+Right + normal =9
        '''
        self.pool = nn.AvgPool2d(2**scale, 2**scale)

        raw_feature_num =9

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+raw_feature_num, out_planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
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
            nn.Conv2d(input_layer+raw_feature_num, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.res = nn.Conv2d(out_planes, 3, 1, 1, bias=False)
    
    def forward(self, left, right, surface_normal, feature):
     
        left = self.pool(left)
        right = self.pool(right)

        conv1 = self.conv1(torch.cat((left, right, surface_normal, feature), dim=1))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((left, right, surface_normal, feature), dim=1)))

        res = self.res(conv6)
        return res