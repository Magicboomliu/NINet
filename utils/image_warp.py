# from __future__ import division
# import torch
# from torch.autograd import Variable



# def image_warp(img, disp_, padding_mode='zeros'):

#     """

#     Inverse warp a source image to the target image plane.



#     Args:

#         img: the source image (where to sample pixels) -- [B, 3, H, W]

#         flow: flow map of the target image -- [B, 2, H, W]

#     Returns:

#         Source image warped to the target image plane

#     """

#     bs, h, w = disp_.size()
#     disp_ = torch.unsqueeze(disp_, dim=1)

#     u = disp_[:,0,:,:]

#     grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]

#     grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]

#     X = grid_x + u

#     Y = grid_y

#     X = 2*(X/(w-1.0) - 0.5)

#     Y = 2*(Y/(h-1.0) - 0.5)

#     grid_tf = torch.stack((X,Y), dim=3)

#     img_tf = torch.nn.functional.grid_sample(img, grid_tf, padding_mode=padding_mode)

#     return img_tf

import torch
import torch.nn as nn
import torch.nn.functional as F

def image_warp(img, disp):

    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    # x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    disp = torch.squeeze(disp, dim=1)
    x_shifts = disp

    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                            padding_mode='zeros')

    return output
