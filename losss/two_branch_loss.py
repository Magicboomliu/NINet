from typing import MutableMapping
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append("../")
from utils.normal_cal import get_normal
from utils.angles_loss import get_costheta_loss

def NormalEpeLoss(surface_normal_infer,disp_gt):
    surface_normal_gt = get_normal(disp_gt)
    assert surface_normal_gt.size()==surface_normal_infer.size()
    loss = F.smooth_l1_loss(surface_normal_infer,surface_normal_gt,size_average=True)
    return loss

def DispEpeLoss(disp_infer, disp_gt):
    mask = disp_gt < 192
    return F.l1_loss(disp_infer[mask], disp_gt[mask], size_average=True)


class MultiScaleLoss(nn.Module):
    def __init__(self, scales, weights=None, loss='Smooth_l1'):
        super(MultiScaleLoss, self).__init__()
        
        self.weights = weights
        assert(len(self.weights) == scales)         
        self.loss = loss
        self.smoothl1 = nn.SmoothL1Loss(size_average=True)
    
        if self.loss=='Smooth_l1':
            self.loss = self.smoothl1

    def forward(self, disp_infer,surface_normal_infer, disp_gt):
        two_task_loss = 0
        disp_smooth_l1_loss = 0


        total_loss = 0
        smooth_l1_loss = 0
        angle_loss = 0
        sphere_coord_loss = 0
        # Surface Normal Loss
        if (type(surface_normal_infer) is tuple) or (type(surface_normal_infer) is list):
            for i, input_ in enumerate(surface_normal_infer):
                if i ==0:
                    target_ = disp_gt
                    supervised_gt_disp_ = disp_gt
                    target_normal = get_normal(target_)

                else:
                    supervised_gt_disp_ = F.interpolate(disp_gt,scale_factor=1./pow(2,i),mode='bilinear',align_corners=False)
                    target_ = disp_gt/ pow(2,i)
                    target_ = F.interpolate(target_,scale_factor=1./pow(2,i),mode='bilinear',align_corners=False)
                    target_normal = get_normal(target_)
                
                # Normal Prediction Result
                pred_normal = input_
                
                # Directly Loss
                smooth_l1_loss+=self.smoothl1(pred_normal,target_normal)*self.weights[i]

                # Angle Loss
                angle_loss+=get_costheta_loss(pred_normal,target_normal)*self.weights[i]

                # Disparity Loss
                disp_infer_ = disp_infer[i]
                mask = (supervised_gt_disp_ < 192) & (supervised_gt_disp_ > 0)
                mask.detach_()
                disp_infer_ = disp_infer_[mask]
                supervised_gt_disp_ = supervised_gt_disp_[mask]
                disp_smooth_l1_loss+=self.smoothl1(disp_infer_,supervised_gt_disp_)*self.weights[i]

        else:
            target_ = disp_gt
            target_normal = get_normal(target_)
            pred_normal = surface_normal_infer

            # Surface Normal Smooth L1 loss
            smooth_l1_loss+=self.smoothl1(pred_normal,target_normal)
            
            mask = (disp_gt > 0) & (disp_gt < 192)
            mask.detach_()
            # Disp Smooth l1 loss
            disp_smooth_l1_loss+=self.smoothl1(disp_infer[mask],disp_gt[mask])
            # Angle loss
            angle_loss+=get_costheta_loss(pred_normal,target_normal)
        
        smooth_l1_loss_value = smooth_l1_loss.item()
        angle_loss_value = angle_loss.item()
        disp_smooth_l1_loss_value = disp_smooth_l1_loss.item()

        # ratio = angle_loss_value*1.0/(smooth_l1_loss_value+1e-10)
        # total_loss = smooth_l1_loss*ratio+angle_loss*1.0
        # total_loss = total_loss*100

        if disp_smooth_l1_loss!=0:
            ratio = disp_smooth_l1_loss_value*1.0/(smooth_l1_loss_value+1e-10)
            total_loss = smooth_l1_loss*ratio +disp_smooth_l1_loss*1.1
        else:
            total_loss = smooth_l1_loss_value*192 + disp_smooth_l1_loss
        return total_loss, smooth_l1_loss,disp_smooth_l1_loss

def multiscaleloss(scales=5, weights=None, loss='Smooth_l1'):
    if weights is None:
        weights = (0.005, 0.01, 0.02, 0.08, 0.32)
    if scales == 1 and type(weights) is not tuple:
        weights = (weights, )
    return MultiScaleLoss(scales=scales,weights=weights,loss=loss)


if __name__=="__main__":
    multiscaleloss_ = multiscaleloss(4,weights=[1.0,1.0,1.0,0.6],loss='Smooth_l1')
    predict_disp = torch.ones(1,1,320,640)
    predict_normal = torch.randn(1,3,320,640)
    predict_disp_pyramid = [F.interpolate(predict_disp,scale_factor=1./pow(2,i),mode='bilinear',align_corners=False) for i in range(4)]
    predict_normal_pyramid = [F.interpolate(predict_normal,scale_factor=1./pow(2,i),mode='bilinear',align_corners=False) for i in range(4)]
    loss,normal_loss,disp_loss = multiscaleloss_(predict_disp_pyramid,predict_normal_pyramid,predict_disp)
    print(loss.item())