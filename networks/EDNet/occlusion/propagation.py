import torch
import torch.nn as nn
import torch.nn.functional as F
from deform.modules.modulated_deform_conv import ModulatedDeformConvPack,ModulatedDeformConvFunction

class Propagation_Args(object):
    def __init__(self,prop_time=3,affinity="AS",affinity_gamma=1.20,prop_kernel=3,conf_prop=True,preserve_input=True) -> None:
        super().__init__()
        self.prop_kernel = prop_kernel
        self.conf_prop = conf_prop
        self.prop_time = prop_time
        self.affinity = affinity
        self.affinity_gamma = affinity_gamma
        self.preserve_input = preserve_input



class OSNLP(nn.Module):
    def __init__(self,args,ch_g,ch_f,k_g,k_f):
        super(OSNLP,self).__init__()
        "Input Guidance Feature: [B,ch_g,H,W]"
        'Feature:[B,ch_f,H,W]'

        #Here Assume the Feature is the disparity: ch_f =1
        assert ch_f == 1, 'only tested with ch_f == 1 but {}'.format(ch_f)
        assert (k_g % 2) == 1, \
            'only odd kernel is supported but k_g = {}'.format(k_g)
        pad_g = int((k_g - 1) / 2)
        assert (k_f % 2) == 1, \
            'only odd kernel is supported but k_f = {}'.format(k_f)
        pad_f = int((k_f - 1) / 2)

        self.args = args
        # How many iter times(nums) want to use
        self.prop_time = self.args.prop_time
        # The affnity normalization type
        self.affinity = self.args.affinity

        # Guide kernel
        self.ch_g = ch_g
        self.ch_f = ch_f

        # propagation kernel
        self.k_g = k_g
        self.k_f = k_f

        # Assume zero offset for center pixels
        self.num = self.k_f * self.k_f - 1
        self.idx_ref = self.num // 2  # Center Pixel ID

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            # Get the offset of each pixel
            self.conv_offset_aff = nn.Conv2d(
                self.ch_g, 3 * self.num, kernel_size=self.k_g, stride=1,
                padding=pad_g, bias=True
            )
            # Zero initialization
            self.conv_offset_aff.weight.data.zero_()
            self.conv_offset_aff.bias.data.zero_()
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
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64


    def _get_offset_affinity(self, guidance, confidence=None, rgb=None):
        
        B, _, H, W = guidance.shape
        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            # Get the affinity_offset and its affinity
            offset_aff = self.conv_offset_aff(guidance)
            # Split the result
            o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

            # Add zero reference offset
            #[B,2*neighbors,H,W]
            offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
            #len is neighbors: [B,2,H,W]
            list_offset = list(torch.chunk(offset, self.num, dim=1))
            # Center add Zeros for inference pixel
            list_offset.insert(self.idx_ref,
                               torch.zeros((B, 1, 2, H, W)).type_as(offset))
            # Back to shape [B,2*(neighbors+1),H,W]
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)
            # Here the affinity shape is [B,neighbors,H,W]
            
            # Affinity Normalization
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
        # TODO : Need more efficient way
        if self.args.conf_prop:
            list_conf = []
            offset_each = torch.chunk(offset, self.num + 1, dim=1)

            modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()
            
            # confidence adjustment
            for idx_off in range(0, self.num + 1):
                # Get neigbor's offset coordinate
                ww = idx_off % self.k_f
                hh = idx_off // self.k_f
                # if it is the center pixel: Skiped
                if ww == (self.k_f - 1) / 2 and hh == (self.k_f - 1) / 2:
                    continue

                # Get the offset mask at current pixel
                offset_tmp = offset_each[idx_off].detach()
                
                conf_tmp = ModulatedDeformConvFunction.apply(
                    confidence, offset_tmp, modulation_dummy, self.w_conf,
                    self.b, self.stride, 0, self.dilation, self.groups,
                    self.deformable_groups, self.im2col_step)
                list_conf.append(conf_tmp)
            
            conf_aff = torch.cat(list_conf, dim=1)
            # affity with confidence
            aff = aff * conf_aff.contiguous()

                # Affinity normalization
        
        # Affinity Normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        if self.affinity in ['ASS', 'TGASS']:
            aff_abs_sum[aff_abs_sum < 1.0] = 1.0
        if self.affinity in ['AS', 'ASS', 'TGASS']:
            aff = aff / aff_abs_sum
        
        # center reference
        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum

        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)
        
        
        return offset, aff
    
    # Use Deformable convolution for spatail propagation
    def _propagate_once(self, feat, offset, aff):
        feat = ModulatedDeformConvFunction.apply(
            feat, offset, aff, self.w, self.b, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups, self.im2col_step
        )

        return feat

    def forward(self, depth_init, guidance, confidence=None, depth_strict=None,
                rgb=None):
        '''
        Input:  
        (1) inital depth [B,1]
        (2) Affinity [B,nums_neigbhour]
        (3) Confidence [B,1]
        (4) Strict depth [B,1]
        (5) RGB image [B,3]
        
        '''
        assert self.ch_g == guidance.shape[1]

        assert self.ch_f == depth_init.shape[1]
        
        # Make Sure there is confidence map
        # To do confidence based propagation
        if self.args.conf_prop:
            assert confidence is not None
        if self.args.conf_prop:
            
            offset, aff = self._get_offset_affinity(guidance, confidence, rgb)
        else:
            offset, aff = self._get_offset_affinity(guidance, None, rgb)
        
        # Propagation
        if self.args.preserve_input:
            assert depth_init.shape == depth_strict.shape
            mask_depth = torch.sum(depth_strict > 0.0, dim=1, keepdim=True).detach()
            mask_strict = (mask_depth > 0.0).type_as(depth_strict)
        list_disparity = []
        propagated_result = depth_init

        for k in range(1, self.prop_time + 1):
            propagated_result = self._propagate_once(propagated_result,offset,aff)
            list_disparity.append(propagated_result)
            
        if self.args.preserve_input:
            propagated_result = (1.0-mask_strict)*propagated_result + mask_strict * depth_strict
        
        return propagated_result, list_disparity, offset, aff, self.aff_scale_const.data




