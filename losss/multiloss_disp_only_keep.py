import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append("../")
from networks.submodule import image_rec
from utils.SSIM import SSIM


def EPE_Loss(disp_infer, disp_gt):
    mask = disp_gt < 192
    return F.l1_loss(disp_infer[mask], disp_gt[mask], size_average=True)

def get_smooth_loss(disp, img):
    
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

class MultiScaleLoss(nn.Module):
    def __init__(self, scales, weights=None, loss='Smooth_l1', mask=False, downsample=1, apply_rec=False):
        super(MultiScaleLoss, self).__init__()
        self.mask = mask
        self.weights = weights
        self.downsample = downsample
        assert(len(self.weights) == scales)         
        self.loss = loss
        self.smoothl1 = nn.SmoothL1Loss(size_average=True)
        self.img_rec_loss = SSIM()
        # self.img_rec_loss = nn.L1Loss()
        self.apply_rec = apply_rec
        if self.loss=='Smooth_l1':
            self.loss = self.smoothl1

    def forward(self, disp_infer, disp_gt, img_left, img_right):
        loss = 0
        if (type(disp_infer) is tuple) or (type(disp_infer) is list):
            for i, input_ in enumerate(disp_infer):
                if not self.apply_rec:
                    if i ==0:
                        target_ = disp_gt
                    else:
                        target_ = F.interpolate(disp_gt,scale_factor=1./pow(2,i),mode='bilinear',align_corners=False)
                    if self.mask:
                        # work for sparse
                        mask = disp_gt > 0
                        mask.detach_()
                        mask = mask.type(torch.cuda.FloatTensor)
                        pooling_mask = self.multiScales[i](mask)
                        # use unbalanced avg
                        target_ = target_ / pooling_mask
                        mask = (target_ > 0) & (target_ < 192)
                        mask.detach_()
                        input_ = input_[mask]
                        target_ = target_[mask]
                    else:
                        mask = (target_ < 192) & (target_ > 0)
                        mask.detach_()
                        input_ = input_[mask]
                        target_ = target_[mask]
                    loss += self.smoothl1(input_, target_) * self.weights[i]
                else:
                    img_rec = image_rec(self.multiScales[i](img_right), -input_ / (2 ** i))
                    loss += self.img_rec_loss(self.multiScales[i](img_left), img_rec) * 10
                    

        else:
            if self.mask:
                mask = (disp_gt > 0) & (disp_gt < 192)
                mask.detach_()
            else:
                mask = disp_gt < 192
                mask = mask.detach_()
            loss = self.loss(disp_infer[mask], disp_gt[mask])
        
        return loss
    
def multiscaleloss(scales=5, downscale=4, weights=None, loss='Smooth_l1', sparse=False, mask=False, apply_rec=False):
    if weights is None:
        weights = (0.005, 0.01, 0.02, 0.08, 0.32)
    if scales == 1 and type(weights) is not tuple:
        weights = (weights, )
    return MultiScaleLoss(scales, weights, loss, mask, downscale, apply_rec)



