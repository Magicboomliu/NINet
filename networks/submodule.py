import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import sys
sys.path.append("../")
import time
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

def conv_Relu(in_planes, out_planes, kernerl_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False),
        nn.ReLU(True)
    )

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

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

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    
    X = grid_x + u
    Y = grid_y
    
    X = 2*(X/(w-1.0) - 0.5)
    Y = 2*(Y/(h-1.0) - 0.5)
    
    grid_tf = torch.stack((X, Y), dim=3)
    img_tf = torch.nn.functional.grid_sample(img, grid_tf, padding_mode=padding_mode)

    return img_tf


def get_grad(depth):
    edge_kernel_x = torch.from_numpy(np.array([[1/8, 0, -1/8],[1/4,0,-1/4],[1/8,0,-1/8]])).type_as(depth)
    edge_kernel_y = torch.from_numpy(np.array([[1/8, 1/4, 1/8],[0,0,0],[-1/8,-1/4,-1/8]])).type_as(depth)
    sobel_kernel = torch.cat((edge_kernel_x.view(1, 1, 3, 3), edge_kernel_y.view(1, 1, 3, 3)), dim=0)
    sobel_kernel.requires_grad = False
    grad_depth = torch.nn.functional.conv2d(depth, sobel_kernel, padding=1)
    return grad_depth


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
    def __init__(self, n_in, n_out, stride = 1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
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

        out = residual + out
        out = self.relu(out)
        return out


class DisparityEstimation(nn.Module):
    def __init__(self, max_disp):
        super(DisparityEstimation, self).__init__()
        self.max_disp = max_disp
    
    def forward(self, similarity_volume):

        assert similarity_volume.dim() == 4

        # compute probability across the disparity dimension
        prob_volume = F.softmax(similarity_volume, dim=1)
        disp_candidates = torch.arange(0, self.max_disp).type_as(prob_volume)
        disp_candidates = disp_candidates.view(1, self.max_disp, 1, 1)
        # output shape is BHW
        disp = torch.sum(prob_volume * disp_candidates, dim=1, keepdim=True)   
        return disp


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

        # self.layer1_0 = self._make_layer(BasicBlock, blocks=1, inplanes=32, outplanes=32)
        # self.layer1_1 = self._make_layer(BasicBlock, blocks=1, inplanes=32, outplanes=32)
        # self.layer1_2 = self._make_layer(BasicBlock, blocks=1, inplanes=32, outplanes=32)
        # self.layer1_3 = self._make_layer(BasicBlock, blocks=1, inplanes=32, outplanes=32)

        # self.layer2_0 = self._make_layer(BasicBlock, blocks=1, inplanes=32, outplanes=64, stride=2)     # 1/4
        # self.layer2_1 = self._make_layer(BasicBlock, blocks=1, inplanes=64, outplanes=64)
        # self.layer2_2 = self._make_layer(BasicBlock, blocks=1, inplanes=64, outplanes=64)
        # self.layer2_3 = self._make_layer(BasicBlock, blocks=1, inplanes=64, outplanes=64)

        # self.layer3_0 = self._make_layer(BasicBlock, blocks=1, inplanes=64, outplanes=128, stride=2)    # 1/8
        # self.layer3_1 = self._make_layer(BasicBlock, blocks=1, inplanes=128, outplanes=128)
        # self.layer3_2 = self._make_layer(BasicBlock, blocks=1, inplanes=128, outplanes=128)
        # self.layer3_3 = self._make_layer(BasicBlock, blocks=1, inplanes=128, outplanes=128)

        
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



class res_submodule(nn.Module):
    def __init__(self, scale, value_planes, out_planes):
        super(res_submodule, self).__init__()
        # self.resample = Resample2d()
        self.pool = nn.AvgPool2d(2**scale, 2**scale)

        self.conv1 = nn.Sequential(
            nn.Conv2d(value_planes+7, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*2, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(out_planes, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.redir1 = nn.Sequential(
            nn.Conv2d(value_planes+7, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.res = nn.Conv2d(out_planes, 1, 1, 1, bias=False)

    def forward(self, left, right, disp, feature):
        left = self.pool(left)
        right = self.pool(right)
        conv1 = self.conv1(torch.cat((left, right, disp, feature), dim=1))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((left, right, disp, feature), dim=1)))

        res = self.res(conv6)
        return res


class res_submodule_with_feature(nn.Module):
    def __init__(self, scale, value_planes, out_planes):
        super(res_submodule_with_feature, self).__init__()
        # self.resample = Resample2d()
        self.pool = nn.AvgPool2d(2**scale, 2**scale)

        self.conv1 = nn.Sequential(
            nn.Conv2d(value_planes+7, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*2, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(out_planes, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.redir1 = nn.Sequential(
            nn.Conv2d(value_planes+7, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.res = nn.Conv2d(out_planes, 1, 1, 1, bias=False)

    def forward(self, left, right, disp, feature):
        left = self.pool(left)
        right = self.pool(right)
        conv1 = self.conv1(torch.cat((left, right, disp, feature), dim=1))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((left, right, disp, feature), dim=1)))

        res = self.res(conv6)
        return res,conv6

class slant_residual(res_submodule):
    def __init__(self, scale, value_planes, out_planes):
        super(slant_residual, self).__init__(scale, value_planes, out_planes)


        self.conv1 = nn.Sequential(
            nn.Conv2d(value_planes+9, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
        self.redir1 = nn.Sequential(
            nn.Conv2d(value_planes+9, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.res = nn.Conv2d(out_planes, 3, 1, 1, bias=False)

    def forward(self, left, right, disp, feature):
        scale = left.size()[2] / disp.size()[2]
        #left = self.pool(left)
        #right = self.pool(right)
        grad = get_grad(disp)
        # disp_ = disp / scale

        conv1 = self.conv1(torch.cat((left, right, disp, grad, feature), dim=1))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((left, right, disp, grad, feature), dim=1)))

        res = self.res(conv6)
        res = (grad * res[:, :2, :, :]).sum(dim=1, keepdim=True) + res[:, 2:, :, :]
        return res


class slant_residual_stage2(nn.Module):
    def __init__(self, value_planes, out_planes):
        super(slant_residual_stage2, self).__init__()
        # self.conv1 = DeconBlock(value_planes+3+1, out_planes, 1, True)
        self.conv1 = ResBlock(value_planes+3+1+2, out_planes, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, True)
        )

        # self.conv3 = DeconBlock(out_planes, out_planes, 1, True)
        self.conv3 = ResBlock(out_planes, out_planes, 1)

        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(out_planes, out_planes, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, True)
        )
        self.residual = nn.Conv2d(out_planes, 2, 3, 1, 1, bias=False)
    
    def forward(self, feature, img_left, disp):

        grad = get_grad(disp)
        conv1 = self.conv1(torch.cat((feature, img_left, disp, grad), dim=1))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        res = self.residual(conv4)

        return (grad / 100 * res[:, :2, :, :]).sum(dim=1, keepdim=True)

class slant_residual_softmax(res_submodule):
    def __init__(self, scale, value_planes, out_planes):
        super(slant_residual_softmax, self).__init__(scale, value_planes, out_planes)


        self.conv1 = nn.Sequential(
            nn.Conv2d(value_planes+9, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
        self.redir1 = nn.Sequential(
            nn.Conv2d(9+value_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.res = nn.Conv2d(out_planes, 9, 1, 1, bias=False)

    def forward(self, left, right, disp, feature):
        grad = get_grad(disp)
        
        # residual range
        residual_range = torch.arange(-3.0, 4.0, 1.0).view(1, 7, 1, 1).repeat(left.size()[0], 1, left.size()[2], left.size()[3]).cuda()

        conv1 = self.conv1(torch.cat((left, right, disp, grad, feature), dim=1))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((left, right, disp, grad, feature), dim=1)))

        res = self.res(conv6)
        range_prob = F.softmax(res[:, 2:, :, :], dim=1)

        res = (grad * res[:, :2, :, :]).sum(dim=1, keepdim=True) + torch.sum(range_prob * residual_range, dim=1, keepdim=True)
        # res = (grad * res[:, :2, :, :]).sum(dim=1, keepdim=True) + scale * res[:, 2:, :, :]
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



