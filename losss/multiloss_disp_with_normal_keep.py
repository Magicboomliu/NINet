import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append("../")
from networks.submodule import image_rec
from utils.SSIM import SSIM
from utils.normal_cal import get_normal
def get_smooth_loss(disp, img):
    
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def EPE_normal(normal_infer,normal_gt):
    return F.l1_loss(normal_infer,normal_gt,size_average=True)

def EPE_Loss(disp_infer, disp_gt):
    mask = disp_gt < 192
    return F.l1_loss(disp_infer[mask], disp_gt[mask], size_average=True)

class MultiScaleLossN(nn.Module): # here make sure scales_list[1] is not 0
    def __init__(self,scales_list,weights_list,loss="Smooth_l1",mask=False,downsample =1, apply_rec=False):
        super(MultiScaleLossN,self).__init__()

        self.mask = mask 
        self.weights_list = weights_list   # 0 is disp, 1 is normal
        self.downsample = downsample
        assert(len(self.weights_list)==len(scales_list))
        self.loss = loss
        self.smoothl1 = nn.SmoothL1Loss(size_average=True)
        self.img_rec_loss = SSIM()
        # self.img_rec_loss = nn.L1Loss()
        self.apply_rec = apply_rec
    def forward(self,disp_infer,disp_gt,normal_infer,img_left,img_right):
        total_loss = 0
        disp_loss = 0
        normal_loss =0
        # For Disp LOSS
        if (type(disp_infer) is tuple) or (type(disp_infer) is list):
            for i, input_ in enumerate(disp_infer):
                if not self.apply_rec:
                    if i==0:
                        target_ = disp_gt
                    else:
                        target_ = F.interpolate(disp_gt,scale_factor=1./pow(2,i),mode='bilinear',align_corners=False)
                    traget_normal = get_normal(target_*1.0/pow(2,i))
                    pred_normal = normal_infer[i]
                    if self.mask:
                        # work for sparse
                        #TODO
                        pass
                    else:
                        mask = (target_ < 192) & (target_ > 0)
                        mask.detach_()
                        input_ = input_[mask]
                        target_ = target_[mask]
                    disp_loss += self.smoothl1(input_, target_) * self.weights_list[0][i]
                    normal_loss+= self.smoothl1(pred_normal,traget_normal) *self.weights_list[1][i]
                else:
                    img_rec = image_rec(self.mutilscales_disp[i](img_right), -input_ / (2 ** i))
                    disp_loss += self.img_rec_loss(self.mutilscales_disp[i](img_left), img_rec) * 10
        else:
            if self.mask:
                mask = (disp_gt > 0) & (disp_gt < 192)
                mask.detach_()
            else:
                mask = disp_gt < 192
                mask = mask.detach_()
            disp_loss = self.smoothl1(disp_infer[mask], disp_gt[mask])
            normal_loss = self.smoothl1(normal_infer,get_normal(disp_gt))
            

        total_loss = disp_loss * 1.0 + normal_loss * 1.0*192

        return [total_loss,disp_loss,normal_loss]

def multiscalelossN(scale_list =[4,4],downsample =1, weight_list =None,loss='Smooth_l1', sparse=False, mask=False, apply_rec=False):
    return MultiScaleLossN(scale_list,weight_list,loss,mask,downsample,apply_rec)




