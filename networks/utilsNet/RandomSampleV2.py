import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import cv2
from copy import deepcopy
import pdb
import math

class AdaptiveSample(nn.Module):
    def __init__(self, k_size=5, dilation=1, stride=1,
                 depth_max=192,
                 sample_num=15):
        '''
        :param k_size: 
        :param dilation: 
        :param stride: 
        :param depth_max: 
        :param sample_num: 
        :param area_type: 
        :param area_thred: 
        '''

        super(AdaptiveSample, self).__init__()
        self.k_size = k_size

        self.stride = stride
        self.dilation = dilation
        self.padding = (self.dilation * (self.k_size - 1) + 1 - self.stride + 1) // 2

        self.depth_max = depth_max

        self.sample_num = sample_num

        self.unford = torch.nn.Unfold(kernel_size=(self.k_size, self.k_size),
                                      padding=self.padding, stride=self.stride,
                                      dilation=self.dilation)

        self.normal_unford = torch.nn.Unfold(kernel_size=(self.k_size, self.k_size),
                                      padding=self.padding, stride=self.stride,
                                      dilation=self.dilation)

        self.unford_stride = torch.nn.Unfold(kernel_size=1, padding=0, stride=self.stride)

    # Get Pixels Coordinates
    def set_id_grid(self, depth):
        b, h, w = depth.size()
        i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth).to(depth.device)  # [1, H, W]
        j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth).to(depth.device)  # [1, H, W]
        ones = torch.ones(1, h, w).type_as(depth).to(depth.device)

        pixel_coords = torch.stack((j_range, i_range, ones), dim=1).type_as(depth).to(
            depth.device)  # [1, 3, H, W]

        return pixel_coords


    def select_index(self):
        '''
         Sampling from a Local Kernel
        :return: 
        '''
        num_upper_limit = self.k_size**2

        # Select Points from the kernel
        points = np.random.choice(num_upper_limit,int(self.sample_num),replace=True)
        np.random.shuffle(points)
        points = np.array(points)

        return points


    def forward(self, depth, surface_normal,features ,guide_weight=None,
                normal_only_weight_used=True, guided_feature_weight_used=True):
        '''
        :param depth: 
        :param surface_normal: 
        :param features: 
        :param guide_weight: 
        :param normal_only_weight: USE surface Normal ONLY 
        :param guided_feature_weight: Use Guided Feature ONLY
        :return: 
        '''

        device = depth.get_device()
        if device < 0:
            device = 'cpu'
        depth = depth.squeeze(1)

        b, c,h, w = features.shape

        #Guide Weight [B,H/s,W/s,Kw*Kh]
        if guide_weight is None:
            guide_weight = torch.ones([b,
                                       h // self.stride,
                                       w // self.stride,
                                       self.k_size * self.k_size]).type_as(depth).to(device)

        # Get the Coordinate[B,3,H,W]
        pixel_coords = self.set_id_grid(depth) #[B,3,H,W]

        # Get the 3D point

        # Count the Useful Points
        valid_condition = ((depth > 0) & (depth < self.depth_max)).type(torch.FloatTensor).to(device)
        valid_condition = valid_condition.unsqueeze(1)  # [B,1,H,W]

        # 进行卷积，但是不求和的运算
        # Kernel_size = 5
        # Dilation = 1
        # Stride = 1
        points_patches = self.unford(features) # [B, C* kH * kW, L] --->[B,3*5*5,320*640]

        #################### Patch Normals #####################
        # [B, C* kH * kW, L] --->[B,3*5*5,160*320]
        normal_patches = self.normal_unford(surface_normal)
        #[B,3,Kh*Kw,H,W]
        normal_patches = normal_patches.view(-1, 3,
                                             self.k_size * self.k_size,
                                             h // self.stride, w // self.stride)
        #[B,H,W,Kh*Kw,3]
        normal_patches = normal_patches.permute(0, 3, 4, 2, 1)


        #Convert to [B,C,Kh*Kw,H,W]
        points_patches = points_patches.view(-1, c,
                                             self.k_size * self.k_size,
                                             h // self.stride, w // self.stride)
        # Convert to [B, H, W, Kh*Kw, C]
        points_patches = points_patches.permute(0, 3, 4, 2, 1)

        # Select Points
        local_sample_index = self.select_index()
        #[sample_nums=n]
        local_sample_index = torch.from_numpy(local_sample_index).type(torch.LongTensor).to(device)

        local_sample_index_copy = local_sample_index.view(-1)  # [n]

        #[B,H,W,num_sample,C] --->[B,H,W,15,3]
        sampled_patches = torch.index_select(points_patches,dim=3,index=local_sample_index_copy)

        #[B,H,W,num_sample,3] ---> [B,H,W,15,3]
        normal_sample_patches = torch.index_select(normal_patches,dim=3,index=local_sample_index_copy)

        # Get Normal Weight Here
        '''
        INPUT LEVEL:
            surface normal:[B,3,H,W]
            Normal_Sample_Patches:[B,H,W,15,3]
        OUTPUT LEVEL = [B,H,W,15]
        '''
        inference_surface_normal = surface_normal.view(b,h,w,3)
        distance_list=[]
        for sample in range(self.sample_num):
            current_sample = normal_sample_patches[:,:,:,sample,:]
            diff = torch.sqrt(torch.sum(
                (current_sample - inference_surface_normal)*(current_sample-inference_surface_normal),
                dim=3))
            diff_cpu = diff.detach().cpu().numpy()
            weights = np.exp(-0.5*diff_cpu)
            distance_list.append(weights)
        distance_weight = np.stack(distance_list,axis=3)
        #[B,H,W,15]
        normal_sample_weights = torch.from_numpy(distance_weight).type_as(features).to(device)


        # Extract valid_condition_patches
        valid_condition_patches = self.unford(valid_condition)
        valid_condition_patches = valid_condition_patches.view(-1,
                                                               self.k_size * self.k_size,
                                                               h // self.stride,
                                                               w // self.stride)
        #[B,H,W,Kw*Kh]
        valid_condition_patches = valid_condition_patches.permute(0,2,3,1)

        #[B,H,W,num_samples]
        validation_sample_points = torch.index_select(valid_condition_patches,3,local_sample_index)

        final_weight_sampled = validation_sample_points

        # Guided weight
        guide_weight_sampled = torch.index_select(guide_weight, 3, local_sample_index)



        # Position Weight
        if normal_only_weight_used:
            final_weight_sampled = final_weight_sampled * normal_sample_weights   # (b, h, w, n)

        # Learned Guide Weight
        if guided_feature_weight_used:
            final_weight_sampled = final_weight_sampled * guide_weight_sampled

        final_weight_sampled = torch.softmax(final_weight_sampled, dim=-1)  # (b, h, w, n)

        # The Final Features

        feature_weight = torch.sum(sampled_patches* final_weight_sampled.unsqueeze(-1),
                                   dim=3,keepdim=False)

        return  feature_weight.permute(0, 3, 1, 2),features


if __name__ == '__main__':
    # Test
    normal_infer = torch.randn(1,3,160,320).cuda()
    depth_infer = torch.abs(torch.randn(1,1,160,320)).cuda()
    surface_normal = torch.randn(1,3,160,320).cuda()
    normalized_surface_normal = F.normalize(surface_normal,dim=1)
    #
    Features = torch.randn(1,8,160,320).cuda()
    # Network Build
    as_module = AdaptiveSample(k_size=5,dilation=1,stride=1,depth_max=192,sample_num=15).cuda()

    # Invoke at Here
    features_weight,features = as_module(depth_infer,surface_normal,Features,None,True,True)
