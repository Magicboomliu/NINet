import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../..")
from deform.modules.modulated_deform_conv import ModulatedDeformConvFunction,ModulatedDeformConvPack

class Args:
    def __init__(self,kernel_size,propagate_time,affinity,affinity_gamma) -> None:
        '''
        Propagation Kernel Size: kernel size
        '''
        self.kernel_size = kernel_size
        self.propagate_time = propagate_time
        self.affinity_type = affinity
        self.affinity_gamma = affinity_gamma
        

# Propagation Network
class NLSPN(nn.Module):
    def __init__(self, args,guide_channel,depth_channel,prop_kernel):
        super(NLSPN, self).__init__()
        '''
        input: 
        (1) Guidance Feature: channels = guide_channel
        (2) initial depth: channels = depth_channel
        (3) need to be included
        
        '''
        self.args = args
        self.prop_time = args.propagate_time
        self.guide_channel = guide_channel
        self.depth_channels = depth_channel
        self.prop_kernel = prop_kernel
        assert self.depth_channels == 1
        assert self.prop_kernel%2 ==1
        self.pad_prop = int((self.prop_kernel-1)/2)

        # Propagation numbers
        self.num = self.prop_kernel * self.prop_kernel - 1
        self.idx_ref = self.num // 2 # the currnet disparity

        # Offset-affinity prediction
        self.conv_offset_aff = nn.Conv2d(
                self.guide_channel, 3 * self.num, kernel_size=3, stride=1,
                padding=1, bias=True
            )
        
        self.conv_offset_aff.weight.data.zero_()
        self.conv_offset_aff.bias.data.zero_()

        # affinity Normalization Factor
        if self.args.affinity_type == "TC":
            self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
            self.aff_scale_const.requires_grad = False
        elif self.args.affinity_type =="TGASS":
            self.aff_scale_const = nn.Parameter(self.args.affinity_gamma * self.num * torch.ones(1))
        else:
            self.aff_scale_const = nn.Parameter(torch.ones(1))
            self.aff_scale_const.requires_grad = False



        # Dummy parameters for gathering: For deformable convolution
        self.w = nn.Parameter(torch.ones((self.depth_channels, 1, self.prop_kernel, self.prop_kernel)))
        self.b = nn.Parameter(torch.zeros(self.depth_channels))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False
        self.stride = 1
        self.padding = self.pad_prop
        self.dilation = 1
        self.groups = self.depth_channels
        self.deformable_groups = 1
        self.im2col_step = 64

    def _get_offset_affinity(self, guidance, confidence=None):
        B, _, H, W = guidance.shape
        
        # Get offset and affinity Here
        offset_aff = self.conv_offset_aff(guidance)
        o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

        # Add zero reference offset
        offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
        list_offset = list(torch.chunk(offset, self.num, dim=1))
        list_offset.insert(self.idx_ref,
                torch.zeros((B, 1, 2, H, W)).type_as(offset))
        offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

        # Affinity Activation
        if self.args.affinity_type == "TC":
            aff = torch.tanh(aff) / self.aff_scale_const
        elif self.args.affinity_type =="TGASS":
            aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
        else:
            pass

        # Confidence value propagation
        list_conf = []
        offset_each = torch.chunk(offset, self.num + 1, dim=1)
        modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()
        for idx_off in range(0, self.num + 1):
            ww = idx_off % self.prop_kernel
            hh = idx_off // self.prop_kernel
            # Skip the center parameter
            if ww == (self.prop_kernel - 1) / 2 and hh == (self.prop_kernel - 1) / 2:
                continue
            # offset current
            offset_tmp = offset_each[idx_off].detach()
            # confidence propagation
            conf_tmp = ModulatedDeformConvFunction.apply(
                    confidence, offset_tmp, modulation_dummy, self.w_conf,
                    self.b, self.stride, 0, self.dilation, self.groups,
                    self.deformable_groups, self.im2col_step)
            list_conf.append(conf_tmp)

        conf_aff = torch.cat(list_conf, dim=1)
        # Affinity_with_confidence
        aff = aff * conf_aff.contiguous()

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-5
        aff_abs_sum[aff_abs_sum < 1.0] = 1.0
        aff = aff / aff_abs_sum

        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum
        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        return offset,aff

    # Start Propagation
    def _propagate_once(self, feat, offset, aff):
        feat = ModulatedDeformConvFunction.apply(
            feat, offset, aff, self.w, self.b, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups, self.im2col_step
        )

        return feat


    def forward(self, initial_disp, occlus_mask,confidence ,guidance_feature):
        assert initial_disp.size(-2)==occlus_mask.size(-2)==guidance_feature.size(-2)
        
        # Get the non-local offset and its affinity
        offset,affinity = self._get_offset_affinity(guidance_feature, confidence)
        # inital result
        disp_result = initial_disp
        list_disp = []
        #Propagation
        for k in range(1, self.prop_time + 1):
            # Get the propagation result
            disp_result = self._propagate_once(disp_result, offset, affinity)
            disp_result = torch.clamp(disp_result,min=0)
            # only use non-propagated propagation result
            disp_result = (1-occlus_mask) * disp_result + occlus_mask * initial_disp
            
            list_disp.append(disp_result)
        disp_result = torch.clamp(disp_result,min=0)
            
        return disp_result, list_disp, offset, affinity, self.aff_scale_const.data
               


if __name__=="__main__":
    # parameters
    nlsp_args = Args(kernel_size=3,propagate_time=10,affinity="TAS",affinity_gamma=0.5)
    nlsp_model = NLSPN(args=nlsp_args,guide_channel=8,depth_channel=1,prop_kernel=3).cuda()

    init_disp = torch.randn(1,1,100,120).cuda()
    occlu_mask_f = torch.randn(1,1,100,120).cuda()
    occlu_mask = torch.sigmoid(occlu_mask_f)
    confidence = torch.sigmoid(torch.randn(1,1,100,120).cuda())
    guidence_feature = torch.randn(1,8,100,120).cuda()

    refine_disp, disp_inter,offset,aff,aff_const = nlsp_model(init_disp,occlu_mask,confidence,guidence_feature)
    
    # Remove negative depth
    print(refine_disp.shape)
