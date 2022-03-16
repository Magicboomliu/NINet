import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
from disparity_plane import ToDisparityPlane,DisparityPlaneToDisparity,DisparityPlaneToSurfaceNormal

def GetDisparityPlanePixelLoss(disparity_infer,normal_infer,disparity_gt,normal_gt):
    '''
    Args: This is to use the plane information: (infered by PatchMatch) to provide
    extra supervision
    '''
    infer_disparity_plane = ToDisparityPlane(normal_infer=normal_infer,disparity_infer=disparity_infer)[0]
    gt_disparity_plane = ToDisparityPlane(normal_infer=normal_gt,disparity_infer=disparity_gt)[0]
    loss = F.smooth_l1_loss(infer_disparity_plane,gt_disparity_plane,size_average=True)

    return loss
