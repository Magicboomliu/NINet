import sys

from torch._C import set_flush_denormal  
sys.path.append("../..")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from networks.utilsNet.devtools import print_tensor_shape

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape  # B is the batch size

        # Change the dim 1 and dim 2 : x-->B,C,N
        x = x.transpose(1, 2).view(B, C, H, W)
        # convolution
        x = self.dwconv(x) #[B,C,H,W]
        # flatten(2) flatten from the third dimension -->[B,C,H*W]
        x = x.flatten(2).transpose(1, 2)#-->[B,H*W,C]-->[B,N,C]

        return x

# MLP Layer : Convolution Like Way
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Self-Attention Layer here
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, 
                   qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        '''
        dim: input dimension
        num_heads: mutil-head attention
        qk_scale: scale dot product as attention

        
        '''
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # Scale Attention : scale is 1/sqrt(head_dim)
        self.scale = qk_scale or head_dim ** -0.5
        
        

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # whether use Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #q-->[B,head,N,C//head]

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            # kv shape:[2,B,head,N,C//head]
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        #k shape :[B,head,N,C//head]
        #v shape:[B,head,N,C//head]

        # @ is the matrix multiplication
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) # softmax at last dimension
    
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
      

        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Step1: Layer Normal x, self attention ,Add
        # Step2 : Layer Normal x, Mlp,Add
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

# Image Embeddings
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


# Mix transformer
class MixVisionTransformer(nn.Module):
    def __init__(self, patch_size=[7,3,3,3], 
                 overlap_stride=[4,2,2,2],
                 in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], 
                 mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()

        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(patch_size=patch_size[0], 
                                            stride=overlap_stride[0], in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=patch_size[1], 
                                                stride=overlap_stride[1], in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=patch_size[2], 
                                                stride=overlap_stride[2], in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=patch_size[3], 
                                                stride=overlap_stride[3], in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            pass

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
    
        

        # stage 2
        x, H, W = self.patch_embed2(x)
        
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        # X here is still [B,N,C]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # Back to the [B,3,H,W]
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        return x


class MixVisionTransformerV2(nn.Module):
    def __init__(self, patch_size=[7,3,3,3], 
                 overlap_stride=[4,2,2,2],
                 in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], 
                 mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()

        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(patch_size=patch_size[0], 
                                            stride=overlap_stride[0], in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=patch_size[1], 
                                                stride=overlap_stride[1], in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=patch_size[2], 
                                                stride=overlap_stride[2], in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=patch_size[3], 
                                                stride=overlap_stride[3], in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        self.patch_embed5 = OverlapPatchEmbed(patch_size=patch_size[4], 
                                                stride=overlap_stride[4], in_chans=embed_dims[3],
                                              embed_dim=embed_dims[4])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        cur += depths[3]
        self.block5 = nn.ModuleList([Block(
            dim=embed_dims[4], num_heads=num_heads[4], mlp_ratio=mlp_ratios[4], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[4])
            for i in range(depths[4])])
        self.norm5 = norm_layer(embed_dims[4])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            pass

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]
        
        cur += self.depths[3]
        for i in range(self.depths[4]):
            self.block5[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        

        # stage 2
        x, H, W = self.patch_embed2(x)
        
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        # X here is still [B,N,C]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # Back to the [B,3,H,W]
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 5
        x, H, W = self.patch_embed5(x)
        for i, blk in enumerate(self.block5):
            x = blk(x, H, W)
        x = self.norm5(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        return x


# Mix transformer
class MixVisionTransformerV3(nn.Module):
    def __init__(self, patch_size=[7,3], 
                 overlap_stride=[4,2],
                 in_chans=3, embed_dims=[64, 128],
                 num_heads=[1, 2], 
                 mlp_ratios=[4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4], sr_ratios=[8, 4]):
        super().__init__()

        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(patch_size=patch_size[0], 
                                            stride=overlap_stride[0], in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=patch_size[1], 
                                                stride=overlap_stride[1], in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])


        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            pass

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]


    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        

        # stage 2
        x, H, W = self.patch_embed2(x)
        
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        # X here is still [B,N,C]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # Back to the [B,3,H,W]
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        

        return x


# Mix transformer
class MixVisionTransformerV4(nn.Module):
    def __init__(self, patch_size=[3,3,3], 
                 overlap_stride=[2,2,2],
                 in_chans=256, embed_dims=[256,512,1024],
                 num_heads=[4, 8,8], 
                 mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6], sr_ratios=[ 2, 1,1]):
        super().__init__()

        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(patch_size=patch_size[0], 
                                            stride=overlap_stride[0], in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=patch_size[1], 
                                                stride=overlap_stride[1], in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=patch_size[2], 
                                                stride=overlap_stride[2], in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            pass

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
    
        

        # stage 2
        x, H, W = self.patch_embed2(x)
        
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        # X here is still [B,N,C]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # Back to the [B,3,H,W]
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        return x



class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=[7,3,3,3], 
            overlap_stride=[4,2,2,2],
            embed_dims=[64, 128, 256, 512], 
            num_heads=[1, 2, 4, 8], 
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[2, 2, 2, 2], 
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, 
            drop_path_rate=0.1)


class LinearMLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

# Segformer 
class Segformer(nn.Module):
    def __init__(self,patch_size=[7,3,3,3],overlap_stride=[4,2,2,2],
                embed_dims=[64,128,256,512],num_heads=[1,2,4,8],
                mlp_ratios=[4,4,4,4],depth=[2,2,2,2],
                sr_ratios=[8,4,2,1],
                decoder_dim = 128,
                feature_fusion_type=None):
        super(Segformer,self).__init__()
        
        self.patch_size = patch_size
        self.overlap_stride = overlap_stride
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratios
        self.depth = depth
        self.sr_ratios = sr_ratios
        self.decoder_dim = decoder_dim

        self.feature_fusion_type = feature_fusion_type

        self.feature_extractor = MixVisionTransformer(
            patch_size=self.patch_size, 
            overlap_stride=self.overlap_stride,
            embed_dims=self.embed_dims, 
            num_heads=self.num_heads, 
            mlp_ratios=self.mlp_ratio,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=self.depth, 
            sr_ratios=self.sr_ratios,
            drop_rate=0.0, 
            drop_path_rate=0.1
        )
        if feature_fusion_type is not None:
            self.fuse_list = nn.ModuleList()
            if self.feature_fusion_type =="conv":
                fuse1 = nn.Sequential(
                    nn.Conv2d(in_channels=embed_dims[1],
                                out_channels=decoder_dim,
                                kernel_size=1,stride=1)
                )
                self.fuse_list.append(fuse1)
                fuse2 = nn.Sequential(
                    nn.Conv2d(in_channels=embed_dims[2],
                            out_channels=decoder_dim,
                            kernel_size=1,stride=1)
                )
                self.fuse_list.append(fuse2)
                fuse3 = nn.Sequential(
                    nn.Conv2d(in_channels=embed_dims[3],
                            out_channels=decoder_dim,
                            kernel_size=1,stride=1)
                )
                self.fuse_list.append(fuse3)
                self.fusion_layer = nn.Sequential(
                    nn.Conv2d(in_channels=decoder_dim*4-64,out_channels=decoder_dim,
                        kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(decoder_dim),
                        nn.ReLU(True)
                )

            elif self.feature_fusion_type=='mlp':
                fuse1 = LinearMLP(input_dim=embed_dims[1],embed_dim=decoder_dim)
                self.fuse_list.append(fuse1)
                fuse2 = LinearMLP(input_dim=embed_dims[2],embed_dim=decoder_dim)
                self.fuse_list.append(fuse2)
                fuse3 = LinearMLP(input_dim=embed_dims[3],embed_dim=decoder_dim)
                self.fuse_list.append(fuse3)
                self.fusion_layer = nn.Sequential(
                    nn.Conv2d(in_channels=decoder_dim*4-64,out_channels=decoder_dim,
                        kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(decoder_dim),
                        nn.ReLU(True))
            else:
                raise NotImplementedError
                
    def forward(self,x):
        '''
        Features 1/4
        Features 1/8
        Features 1/16
        '''
        features = self.feature_extractor(x)
        
        if self.feature_fusion_type is None:
            return features
        else:
            feature_list = []
            if self.feature_fusion_type=="conv":
                target_H = features[0].size(-2)
                target_W = features[0].size(-1)
                feature_list.append(features[0])
                for idx,feature in enumerate(features):
                    if feature.size(-2)!=target_H:
                        feature = self.fuse_list[idx-1](feature)
                        feature = F.interpolate(feature,size=[target_H,target_W],mode='bilinear',align_corners=False)
                        feature_list.append(feature)
                aggregated_features = torch.cat((feature_list[0],feature_list[1],feature_list[2],feature_list[3]),dim=1)
                
                
                aggregated_feature = self.fusion_layer(aggregated_features)
            elif self.feature_fusion_type=="mlp":
                
                target_batch = features[0].size(0)
                target_H = features[0].size(-2)
                target_W = features[0].size(-1)
                feature_list.append(features[0])
                for idx,feature in enumerate(features):
                    if idx==0:
                        continue
                    else:
                        feature = self.fuse_list[idx-1](feature).permute(0,2,1).reshape(target_batch, -1, feature.shape[2], feature.shape[3])
                        feature = F.interpolate(feature,size=[target_H,target_W],mode='bilinear',align_corners=False)
                        feature_list.append(feature)
                aggregated_features = torch.cat((feature_list[0],feature_list[1],feature_list[2],feature_list[3]),dim=1)

                aggregated_feature = self.fusion_layer(aggregated_features)
        
        return [aggregated_feature,features[1],features[2],features[3]]
                
        
# Segformer 
class SegformerV2(nn.Module):
    def __init__(self,patch_size=[7,3,3,3,3],overlap_stride=[4,2,2,2,2],
                embed_dims=[64,128,256,512,1024],num_heads=[1,2,4,8,8],
                mlp_ratios=[4,4,4,4,4],depth=[2,2,2,2,2],
                sr_ratios=[8,4,2,1,1],
                decoder_dim = 128,
                feature_fusion_type=None):
        super(SegformerV2,self).__init__()
        
        self.patch_size = patch_size
        self.overlap_stride = overlap_stride
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratios
        self.depth = depth
        self.sr_ratios = sr_ratios
        self.decoder_dim = decoder_dim

        self.feature_fusion_type = feature_fusion_type

        self.feature_extractor = MixVisionTransformerV2(
            patch_size=self.patch_size, 
            overlap_stride=self.overlap_stride,
            embed_dims=self.embed_dims, 
            num_heads=self.num_heads, 
            mlp_ratios=self.mlp_ratio,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=self.depth, 
            sr_ratios=self.sr_ratios,
            drop_rate=0.0, 
            drop_path_rate=0.1
        )
        # if feature_fusion_type is not None:
        #     self.fuse_list = nn.ModuleList()
        #     if self.feature_fusion_type =="conv":
        #         fuse1 = nn.Sequential(
        #             nn.Conv2d(in_channels=embed_dims[1],
        #                         out_channels=decoder_dim,
        #                         kernel_size=1,stride=1)
        #         )
        #         self.fuse_list.append(fuse1)
        #         fuse2 = nn.Sequential(
        #             nn.Conv2d(in_channels=embed_dims[2],
        #                     out_channels=decoder_dim,
        #                     kernel_size=1,stride=1)
        #         )
        #         self.fuse_list.append(fuse2)
        #         fuse3 = nn.Sequential(
        #             nn.Conv2d(in_channels=embed_dims[3],
        #                     out_channels=decoder_dim,
        #                     kernel_size=1,stride=1)
        #         )
        #         self.fuse_list.append(fuse3)
        #         self.fusion_layer = nn.Sequential(
        #             nn.Conv2d(in_channels=decoder_dim*4-64,out_channels=decoder_dim,
        #                 kernel_size=3,stride=1,padding=1),
        #                 nn.BatchNorm2d(decoder_dim),
        #                 nn.ReLU(True)
        #         )

        #     elif self.feature_fusion_type=='mlp':
        #         fuse1 = LinearMLP(input_dim=embed_dims[1],embed_dim=decoder_dim)
        #         self.fuse_list.append(fuse1)
        #         fuse2 = LinearMLP(input_dim=embed_dims[2],embed_dim=decoder_dim)
        #         self.fuse_list.append(fuse2)
        #         fuse3 = LinearMLP(input_dim=embed_dims[3],embed_dim=decoder_dim)
        #         self.fuse_list.append(fuse3)
        #         self.fusion_layer = nn.Sequential(
        #             nn.Conv2d(in_channels=decoder_dim*4-64,out_channels=decoder_dim,
        #                 kernel_size=3,stride=1,padding=1),
        #                 nn.BatchNorm2d(decoder_dim),
        #                 nn.ReLU(True))
        #     else:
        #         raise NotImplementedError
                
    def forward(self,x):
        '''
        Features 1/4
        Features 1/8
        Features 1/16
        '''
        features = self.feature_extractor(x)
        
        if self.feature_fusion_type is None:
            return features
        # else:
        #     feature_list = []
        #     if self.feature_fusion_type=="conv":
        #         target_H = features[0].size(-2)
        #         target_W = features[0].size(-1)
        #         feature_list.append(features[0])
        #         for idx,feature in enumerate(features):
        #             if feature.size(-2)!=target_H:
        #                 feature = self.fuse_list[idx-1](feature)
        #                 feature = F.interpolate(feature,size=[target_H,target_W],mode='bilinear',align_corners=False)
        #                 feature_list.append(feature)
        #         aggregated_features = torch.cat((feature_list[0],feature_list[1],feature_list[2],feature_list[3]),dim=1)
                
                
        #         aggregated_feature = self.fusion_layer(aggregated_features)
        #     elif self.feature_fusion_type=="mlp":
                
        #         target_batch = features[0].size(0)
        #         target_H = features[0].size(-2)
        #         target_W = features[0].size(-1)
        #         feature_list.append(features[0])
        #         for idx,feature in enumerate(features):
        #             if idx==0:
        #                 continue
        #             else:
        #                 feature = self.fuse_list[idx-1](feature).permute(0,2,1).reshape(target_batch, -1, feature.shape[2], feature.shape[3])
        #                 feature = F.interpolate(feature,size=[target_H,target_W],mode='bilinear',align_corners=False)
        #                 feature_list.append(feature)
        #         aggregated_features = torch.cat((feature_list[0],feature_list[1],feature_list[2],feature_list[3]),dim=1)

        #         aggregated_feature = self.fusion_layer(aggregated_features)
        
        # return [aggregated_feature,features[1],features[2],features[3]]



# Segformer 
class SegformerV3(nn.Module):
    def __init__(self,patch_size=[3,3],overlap_stride=[2,2],
                embed_dims=[128,256],num_heads=[1,2],
                mlp_ratios=[4,4],depth=[2,2],
                sr_ratios=[8,4],
                decoder_dim = 128,
                feature_fusion_type=None):
        super(SegformerV3,self).__init__()
        
        self.patch_size = patch_size
        self.overlap_stride = overlap_stride
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratios
        self.depth = depth
        self.sr_ratios = sr_ratios
        self.decoder_dim = decoder_dim

        self.feature_fusion_type = feature_fusion_type

        self.feature_extractor = MixVisionTransformerV3(
            patch_size=self.patch_size, 
            in_chans=64,
            overlap_stride=self.overlap_stride,
            embed_dims=self.embed_dims, 
            num_heads=self.num_heads, 
            mlp_ratios=self.mlp_ratio,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=self.depth, 
            sr_ratios=self.sr_ratios,
            drop_rate=0.0, 
            drop_path_rate=0.1
        )           
    def forward(self,x):
        '''
        Features 1/4
        Features 1/8
        Features 1/16
        '''
        features = self.feature_extractor(x)
        
        return features


# Segformer 
class SegformerV4(nn.Module):
    def __init__(self,patch_size=[3,3,3],overlap_stride=[2,2,2],
                embed_dims=[256,512,1024],num_heads=[4,8,8],
                mlp_ratios=[4,4,4],depth=[2,2,2],
                sr_ratios=[2,1,1],
                decoder_dim = 128,
                feature_fusion_type=None):
        super(SegformerV4,self).__init__()
        
        self.patch_size = patch_size
        self.overlap_stride = overlap_stride
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratios
        self.depth = depth
        self.sr_ratios = sr_ratios
        self.decoder_dim = decoder_dim

        self.feature_fusion_type = feature_fusion_type

        self.feature_extractor = MixVisionTransformerV4(
            patch_size=self.patch_size, 
            in_chans=256,
            overlap_stride=self.overlap_stride,
            embed_dims=self.embed_dims, 
            num_heads=self.num_heads, 
            mlp_ratios=self.mlp_ratio,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=self.depth, 
            sr_ratios=self.sr_ratios,
            drop_rate=0.0, 
            drop_path_rate=0.1
        )           
    def forward(self,x):
        '''
        Features 1/4
        Features 1/8
        Features 1/16
        '''
        features = self.feature_extractor(x)
        
        return features



if __name__=="__main__":
    input_1 = torch.randn(8,256,40,80).cuda()
    encoder = SegformerV4().cuda()
    features = encoder(input_1)
    print_tensor_shape(features)