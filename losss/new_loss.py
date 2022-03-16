import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append("../")
# EPE Loss for model evaluation
def EPE_Loss(disp_infer, disp_gt):
    mask = ((disp_gt < 256) * (disp_gt >0))
    return F.l1_loss(disp_infer[mask], disp_gt[mask], size_average=True)

def TrueLoss(disp_infer,disp_gt,gt_mask,left_mask):
    error = torch.abs(disp_infer-disp_gt)
    B,C,H,W = error.shape
    error1 = error *gt_mask *1.0
    error2 = error * left_mask *0.6
    return (error1.sum()+error2.sum())/(B*C*H*W+1e-6)


# Confidence Loss
def get_confidence_loss(disp_infer,infer_disp_conf,disp_gt):
    '''
    Version One:
    Include Mask Here: Disparity Infer is dense, infer disparity conf is dense
    Disparity Gt is sparse.
    # Only consider the GT part confidence Loss, the other areas should be Zeros.
    which should not be optimized.
    '''   
    # if epe is too big, just ignore it, because the disparity is not good enough 
    #to do propagation
    epe = EPE_Loss(disp_infer,disp_gt)
    if epe>=5:
        return 0
    # Disparity Mask
    gt_disparity_mask = ((disp_gt>0) * (disp_gt<256))
    
    # Ones Shape Like Disparity
    condition_ones = torch.ones_like(disp_infer).type_as(disp_infer)
    condition_ones.requires_grad = False
    # Zeros Shape Like Disparity
    condition_zeros = torch.zeros_like(disp_infer).type_as(disp_infer)
    condition_zeros.requires_grad = False
    
    # Disparity Abs diff
    disp_abs_diff = torch.abs(disp_gt-disp_infer)
    
    # Threshold = 1.5 adn 0.5
    confidence_cond_upper = torch.where(disp_abs_diff>1.5,condition_ones,condition_zeros).type_as(disp_infer)
    confidence_cond_lowwer = torch.where(disp_abs_diff<0.5,condition_ones,condition_zeros).type_as(disp_infer)
    # 1-c
    loss1 = condition_ones - infer_disp_conf
    loss1 = loss1 * confidence_cond_lowwer * gt_disparity_mask
    #c
    loss2 = infer_disp_conf
    loss2 = loss2 * confidence_cond_upper * gt_disparity_mask

    # Total Confidence Loss
    total_loss = loss1 + loss2
    total_loss = torch.sum(total_loss) /(torch.sum(confidence_cond_lowwer*gt_disparity_mask)+torch.sum(confidence_cond_upper*gt_disparity_mask))
    return total_loss


# MultiScale Loss 
'''
KIITI 是稀疏的，所以注意加Mask
较低的尺度放大到高分辨率，进行Loss计算。
'''
class MultiScaleLoss(nn.Module):
    def __init__(self, scales, weights=None, loss='Smooth_l1', 
                mask=False, downsample=1):
        super(MultiScaleLoss, self).__init__()
        self.weights = weights
        self.downsample = downsample
        assert(len(self.weights) == scales)         
        self.loss = loss
        self.smoothl1 = nn.SmoothL1Loss(size_average=True)
        if self.loss=='Smooth_l1':
            self.loss = self.smoothl1
    def forward(self, disp_infer_pyramid, disp_gt, disp_conf_infer):
        '''Disparity Infer:dense
        Disparity GT: sparse
        disparity conf infer: dense
        img_left,img_right
        '''
        loss = 0
        confidence_loss = 0 
        if (type(disp_infer_pyramid) is tuple) or (type(disp_infer_pyramid) is list):
            for i, cur_scale_disp in enumerate(disp_infer_pyramid):
                target_disp = disp_gt
                scale = target_disp.size(-2)//cur_scale_disp.size(-2)
                
                if scale ==1:
                    confidence_ = disp_conf_infer[0]
                    recoverd_disp = cur_scale_disp
                    assert confidence_.shape == disp_gt.shape
                    assert recoverd_disp.shape == disp_gt.shape
                else:
                    # Resize Infer disparity here
                    recoverd_disp = F.interpolate(cur_scale_disp,scale_factor=scale,mode='bilinear',align_corners=False)
                    confidence_ = F.interpolate(disp_conf_infer[i],scale_factor=scale,mode='bilinear',align_corners=False)
                    confidence_ = torch.clamp(confidence_,min=0.0,max=1.0)
                # Get Loss Here
                disparity_gt_mask = ((disp_gt>0) *(disp_gt<256))
                disparity_gt_mask.detach_()
                # First Get Confidence Loss: current disparity,confidence, gt_disp
                conf_loss = get_confidence_loss(recoverd_disp,confidence_,target_disp)
                recoverd_disp = recoverd_disp[disparity_gt_mask]
                target_disp = disp_gt[disparity_gt_mask]
                loss += self.smoothl1(recoverd_disp,target_disp) * self.weights[i]
                confidence_loss+=conf_loss * self.weights[i]

        else:
            target_disp = gt_disp
            gt_disparity_mask = (disp_gt>0) * (disp_gt<256)
            gt_disparity_mask.detach_()
            loss = self.loss(disp_infer_pyramid[gt_disparity_mask], target_disp[gt_disparity_mask])
            confidence_loss = get_confidence_loss(disp_infer_pyramid,disp_conf_infer,target_disp)
        
        loss = loss *  4.0 + confidence_loss
        return loss
    
def multiscaleloss(scales=5, downscale=4, weights=None, loss='Smooth_l1'):
    if weights is None:
        weights = (0.005, 0.01, 0.02, 0.08, 0.32)
    if scales == 1 and type(weights) is not tuple:
        weights = (weights, )
    return MultiScaleLoss(scales=scales,weights=weights,loss=loss)

if __name__=="__main__":

    disp0 = torch.abs(torch.randn(1, 1, 320, 640))
    disp1 = torch.abs(torch.randn(1, 1, 160, 320))
    disp2 = torch.abs(torch.randn(1, 1, 80, 160))
    disp3 = torch.abs(torch.randn(1, 1, 40, 80))

    target_disp = torch.abs(torch.randn(1,1,320,640))

    confidence0 = torch.abs(torch.randn(1, 1, 320, 640))
    confidence0 = torch.sigmoid(confidence0)
    confidence1 = torch.abs(torch.randn(1, 1, 160, 320))
    confidence1 = torch.sigmoid(confidence1)
    confidence2 = torch.abs(torch.randn(1,1,80,160))
    confidence2 = torch.sigmoid(confidence2)
    confidence3 = torch.abs(torch.randn(1,1,40,80))
    confidence3 = torch.sigmoid(confidence3) 

    disp_pyramid = [disp0,disp1,disp2,disp3]
    confidence_pyramid = [confidence0,confidence1,confidence2,confidence3]

    criten = multiscaleloss(scales=4,weights=[1,1,1,1])
    total_loss = criten(disp_pyramid,target_disp,confidence_pyramid)

