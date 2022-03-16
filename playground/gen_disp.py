import torch
import torch.nn as nn
import torch.nn.functional as F

def gen_disp(gt_disp,pseudo_disp):
    mask = gt_disp>0
    mask = mask.type(torch.cuda.FloatTensor)
    whole = torch.ones_like(mask).type_as(mask)
    left_mask = whole - mask
    final_disp = gt_disp* mask + pseudo_disp * left_mask
    return pseudo_disp,mask,left_mask
    