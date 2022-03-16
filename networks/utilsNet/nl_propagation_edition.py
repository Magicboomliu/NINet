import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("../../")
from deform.modules.modulated_deform_conv import ModulatedDeformConvPack,ModulatedDeformConvFunction
from networks.utilsNet.Disparity_warper import disp_warp

class Args(object):
    def __init__(self,prop_times=1,affinity='TGASS',nlk=3,affinity_gamma=1.0,
                    conf_prop=True,
                    fusion =True) -> None:
        self.prop_times = prop_times
        self.affinity = affinity
        self.nlk = nlk
        self.affinity_gamma = affinity_gamma
        self.conf_prop = conf_prop
        self.fusion = fusion

def compute_affinity_matrix(input_channel=9, neighbour_num=8,non_local_kernel=3):
    affinity_matrix = nn.Sequential(
        nn.Conv2d(input_channel, 32, non_local_kernel, 1, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.Conv2d(32, 64, non_local_kernel, 1, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.Conv2d(64, 3*neighbour_num, non_local_kernel, 1, 1, bias=False),
    )
    return affinity_matrix
        

class NonLocalSpatialPropagation(nn.Module):
    def __init__(self,args:Args,guidance_channel):
        super(NonLocalSpatialPropagation,self).__init__()
        # Configs
        self.args = args
        self.prop_times = self.args.prop_times
        self.affinity = self.args.affinity
        self.non_local_kernel = self.args.nlk
        self.num = self.non_local_kernel * self.non_local_kernel -1
        self.idx_ref = self.num // 2
        # input channels
        self.guidance_channel = guidance_channel

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            self.conv_offset_aff = compute_affinity_matrix(input_channel=self.guidance_channel,
            neighbour_num=self.num,non_local_kernel=self.non_local_kernel)

            if self.affinity == 'TC':
                self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            elif self.affinity == 'TGASS':
                self.aff_scale_const = nn.Parameter(
                    self.args.affinity_gamma * self.num * torch.ones(1))
            else:
                self.aff_scale_const = nn.Parameter(torch.ones(1))
                self.aff_scale_const.requires_grad = False
        else:
            raise NotImplementedError

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((1, 1, 3, 3)))
        self.b = nn.Parameter(torch.zeros(1))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = 1
        self.dilation = 1
        self.groups = 1
        self.deformable_groups = 1
        self.im2col_step = 64
    
    # Guidance ---> non-local affinity
    def _get_offset_affinity(self, guidance, confidence=None):
        B, _, H, W = guidance.shape

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            offset_aff = self.conv_offset_aff(guidance)
            o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

            # Add zero reference offset
            offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
            list_offset = list(torch.chunk(offset, self.num, dim=1))
            list_offset.insert(self.idx_ref,
                               torch.zeros((B, 1, 2, H, W)).type_as(offset))
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

            if self.affinity in ['AS', 'ASS']:
                pass
            elif self.affinity == 'TC':
                aff = torch.tanh(aff) / self.aff_scale_const
            elif self.affinity == 'TGASS':
                aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Apply confidence
        if self.args.conf_prop:
            list_conf = []
            offset_each = torch.chunk(offset, self.num + 1, dim=1)

            modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()

            for idx_off in range(0, self.num + 1):
                ww = idx_off % self.non_local_kernel
                hh = idx_off // self.non_local_kernel

                if ww == (self.non_local_kernel - 1) / 2 and hh == (self.non_local_kernel - 1) / 2:
                    continue

                offset_tmp = offset_each[idx_off].detach()

                # Confidence propagation
                conf_tmp = ModulatedDeformConvFunction.apply(
                    confidence, offset_tmp, modulation_dummy, self.w_conf,
                    self.b, self.stride, 0, self.dilation, self.groups,
                    self.deformable_groups, self.im2col_step)
                list_conf.append(conf_tmp)

            # confidence * affinity
            conf_aff = torch.cat(list_conf, dim=1)
            aff = aff * conf_aff.contiguous()

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        if self.affinity in ['ASS', 'TGASS']:
            aff_abs_sum[aff_abs_sum < 1.0] = 1.0

        if self.affinity in ['AS', 'ASS', 'TGASS']:
            aff = aff / aff_abs_sum

        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum

        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        return offset, aff

    def _propagate_once(self, feat, offset, aff):
        feat = ModulatedDeformConvFunction.apply(
            feat, offset, aff, self.w, self.b, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups, self.im2col_step
        )

        return feat
    
    def forward(self,coarse_disparity,normal_estimation,left_image,right_image,confidence=None,roi=False):
        # Current Scale of the disparity
        current_scale = left_image.size(-2)//coarse_disparity.size(-2)
        
        # resize the image
        cur_left_img = F.interpolate(left_image,size=[coarse_disparity.size(-2),
                                                 coarse_disparity.size(-1)], mode='bilinear',
                                        align_corners= False)
        cur_right_img = F.interpolate(right_image,size=[coarse_disparity.size(-2),
                                       coarse_disparity.size(-1)],mode='bilinear',
                                align_corners=False)
        disp_ = coarse_disparity / current_scale
        warped_right = disp_warp(cur_right_img,disp_)[0]
        warped_error = warped_right - cur_left_img
        # Disparity error + Left Image + right Image + surface normal = 12
        guidance = torch.cat((normal_estimation,cur_left_img,cur_right_img,warped_error),dim=1)
        if self.args.conf_prop:
            assert confidence is not None
        
        if self.args.conf_prop:
            offset, aff = self._get_offset_affinity(guidance,confidence)
        else:
            offset,aff = self._get_offset_affinity(guidance,None)
        
        
        inter_disp= coarse_disparity
        final_disp = coarse_disparity
        crop_height = 375
        crop_width = 1240
        temp = coarse_disparity
        cur_h = inter_disp.size(-2)
        cur_w = inter_disp.size(-1)
        
        '''crop the inter disparity
        crop the offset
        crop the affinity '''
        if roi:       
            inter_disp= inter_disp[:,:,cur_h-crop_height:,cur_w-crop_width:]
            coarse_disparity = inter_disp
            final_disp = inter_disp
            aff = aff[:,:,cur_h-crop_height:,cur_w-crop_width:]
            offset = offset[:,:,cur_h-crop_height:,cur_w-crop_width:]
            inter_disp = inter_disp.contiguous()
            aff = aff.contiguous()
            offset = offset.contiguous()
            
                
        # Propagation
        for k in range(1,self.prop_times+1):
            inter_disp = self._propagate_once(inter_disp,offset,aff)
            mask = inter_disp>=0
            inter_disp = inter_disp * mask
        
        inter_disp = torch.clamp(inter_disp,min=0)
        if self.args.fusion:
            final_disp = 0.7 * coarse_disparity + 0.3* inter_disp
        else:
            final_disp = inter_disp

        final_disp = torch.clamp(final_disp,min=0) 
        
        if roi:
            temp[:,:,cur_h-crop_height:,cur_w-crop_width:]= final_disp
            final_disp = temp
        return final_disp


if __name__=="__main__":
    left_image = torch.randn(2,3,384,1280).cuda()
    right_image = torch.randn(2,3,384,1280).cuda()
    corase_disp = torch.abs(torch.randn(2,1,100,100)).cuda()

    confidenceMap = torch.randn(2,1,100,100).cuda()
    normal_estimation = torch.randn(2,3,100,100).cuda()

    nl_prop = Args(prop_times=3,affinity='TGASS',nlk=3,affinity_gamma=0.5,conf_prop=True,fusion=True)

    nl_propagation= NonLocalSpatialPropagation(nl_prop,guidance_channel=12).cuda()

    refined_disp = nl_propagation(corase_disp,normal_estimation,left_image,right_image,confidenceMap)

    print(refined_disp.shape)
