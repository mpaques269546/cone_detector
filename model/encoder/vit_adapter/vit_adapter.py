# Copyright (c) Shanghai AI Lab. All rights reserved.
#import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .msda import DeformableHeadAttention
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
from functools import partial
import math

#from .vision_transformer import VisionTransformer
from .timm_vision_transformer import TIMMVisionTransformer
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs, InteractionBlockWithCls



"""
window_attn=[True, True, True, True, True, False,
True, True, True, True, True, False,
True, True, True, True, True, False,
True, True, True, True, True, False]

window_size=[14, 14, 14, 14, 14, None,
14, 14, 14, 14, 14, None,
14, 14, 14, 14, 14, None,
14, 14, 14, 14, 14, None],
"""


class ViTAdapter(TIMMVisionTransformer):
    def __init__(self, pretrain_size=512, num_heads=16, conv_inplane=64, n_points=4,
                 init_values=0., deform_num_heads=16,
                 patch_size= 16 , depth= 24, embed_dim= 1024, drop_rate=0,
                 interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]], # 12 layers split in 4 blocks
                 with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=0.5, add_vit_feature=True, pretrained=None,
                 use_extra_extractor=True, with_cp=False, with_cls_token=False, qkv_bias=True, window_attn=False, window_size=14,
                  *args, **kwargs):
    	
        """
        super().__init__(img_size=[pretrain_size], patch_size=patch_size, in_chans=3, num_classes=1, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=None, drop_rate=drop_rate, attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm)
        """
        super().__init__(img_size=pretrain_size, patch_size=patch_size, in_chans=3, residual_indices=[], embed_dim=embed_dim,
                 depth=depth, num_heads=num_heads, mlp_ratio=4., qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=0.,
                 drop_path_rate=0., layer_scale=True,  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU, window_attn=window_attn, window_size=window_size, with_cp=with_cp, pretrained=None, num_tokens=1)
       
        #self.cls_token = self.cls_token # [1,1,D] from TIMMVisionTransformer
        self.patch_size = patch_size
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        #embed_dim =  self.embed_dim #self.d_model #
        self.with_cls_token = with_cls_token
        
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=with_cp)
        
        self.interactions = nn.Sequential(*[
            InteractionBlockWithCls(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
             init_values=init_values, drop_path=self.drop_path_rate,
             norm_layer=nn.LayerNorm ,#self.norm, #partial(nn.LayerNorm, eps=1e-6),  #self.norm_layer
            with_cffn=with_cffn,cffn_ratio=cffn_ratio,  
            deform_ratio=deform_ratio,
            extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
            with_cp=with_cp,
            with_cls_token=with_cls_token) for i in range(len(interaction_indexes))])
       

        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        # orginal code: SyncBatchNorm, ours: GroupNorm to allow Gradient Accumulation
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.norm1 = nn.SyncBatchNorm(embed_dim) if device=='cuda' else nn.BatchNorm2d(embed_dim) #nn.Identity()
        self.norm2 = nn.SyncBatchNorm(embed_dim) if device=='cuda' else nn.BatchNorm2d(embed_dim) #nn.Identity()
        self.norm3 = nn.SyncBatchNorm(embed_dim) if device=='cuda' else nn.BatchNorm2d(embed_dim) #nn.Identity()
        self.norm4 = nn.SyncBatchNorm(embed_dim) if device=='cuda' else nn.BatchNorm2d(embed_dim) #nn.Identity()
   
        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)
        # added
        #self.patch_size = 8 # patch_size
        self.feat_dim = [embed_dim for i in range(4)]


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def _init_deform_weights(self, m):
        #if isinstance(m, MSDeformAttn):
        if isinstance(m, DeformableHeadAttention):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)
        [reference_points, spatial_shapes, level_start_index] = deform_inputs1
        # deform_inputs: [reference_points, spatial_shapes, level_start_index]
        
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)
        #print(f"c {c.shape}, c1 {c1.shape}, c2 {c2.shape}, c3 {c3.shape}, c4 {c4.shape}")
        # c [5, 1029, 384]), c1 [5, 384, 56, 56]), 
        # c2 [5, 784, 384]), c3 [5, 196, 384]),  c4 [5, 49, 384])
        c2_size, c3_size = c2.size(1), c3.size(1)
        del c2,c3,c4
       

        # prepare tokens (cat cls_token and add pos_embed)
        x, H, W = self.prepare_tokens(x)
        if not self.with_cls_token:
            x = x[:,1:,:] # remove cls_token
        bs, n, dim = x.shape # [batchsize, num_patches+num_tokens, embed_dim]

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c,  self.blocks[indexes[0]:indexes[-1] + 1],
                    deform_inputs1, deform_inputs2, H, W)
            if self.with_cls_token:
                outs.append(x[:, 1:, ].transpose(1, 2).view(bs, dim, H, W).contiguous())
                cls_token = x[:,:1,:]
            else:
                outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())
        del x
        #print('interaction ', outs[0].shape) # [B, D, H//patchsize=28, W//patchsize=28]
        
        # Split & Reshape
        c2 = c[:, 0:c2_size, :] #c[:, 0:c2.size(1), :]
        c3 = c[:, c2_size:c2_size + c3_size, :] #c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2_size + c3_size:, :] #c[:, c2.size(1) + c3.size(1):, :]
        #print('c',c1.shape,  c2.shape, c3.shape, c4.shape) [1,384,56,56], [1,784,384], [1,196,384], [1,49,384]

        ## original code#if self.patch_size==16:
        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            ## original code: paper: patch_size=16
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            #x3: same dim
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
        
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
            del x1, x2, x3, x4

        # Final Norm
        f1 = self.norm1(c1) #f1 (B, D, H//4, W//4)
        f2 = self.norm2(c2) #f2 (B, D, H//8, W//8)
        f3 = self.norm3(c3) #f3 (B, D, H//16, W//16)
        f4 = self.norm4(c4) #f4 (B, D, H//32, W//32)
        del c1,c2,c3,c4
        if self.with_cls_token:
            return [f1, f2, f3, f4, cls_token]
        else:
            return [f1,f2,f3,f4]



    def forward_cls_token(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)
        [reference_points, spatial_shapes, level_start_index] = deform_inputs1
        # deform_inputs: [reference_points, spatial_shapes, level_start_index]
        
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)
       

        # prepare tokens (cat cls_token and add pos_embed)
        x,H,W = self.prepare_tokens(x)
        

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c,  self.blocks[indexes[0]:indexes[-1] + 1],
                    deform_inputs1, deform_inputs2, H, W)
        cls_token, x = x[:, :1, ], x[:, 1:, ]
    
        del x, c
        return cls_token.squeeze(1) # [B,1,D] -> [B,D]
            
    def get_attention_map(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)
        [reference_points, spatial_shapes, level_start_index] = deform_inputs1
        # deform_inputs: [reference_points, spatial_shapes, level_start_index]
        
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)
       

        # prepare tokens (cat cls_token and add pos_embed)
        x,H,W = self.prepare_tokens(x)
        if not self.with_cls_token:
            x = x[:, 1:, ] # remove class token

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            if i<len(self.interactions)-1:
                x, c = layer(x, c,  self.blocks[indexes[0]:indexes[-1] + 1],
                        deform_inputs1, deform_inputs2, H, W, return_attention=False)
            else:
                attn = layer(x, c,  self.blocks[indexes[0]:indexes[-1] + 1],
                            deform_inputs1, deform_inputs2, H, W, return_attention=True)
        
        return attn 



"""
if __name__ == "__main__":
    h,w = 224, 224
    b, d  = 3, 384
    input = torch.rand(b,3,h,w)


    model =  ViTAdapter(pretrain_size=h, num_heads=6, conv_inplane=64, n_points=4,
                 init_values=0., 
                 patch_size= 8 , depth= 12, embed_dim= d, drop_rate=0,
                 interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], # 12 layers split in 4 blocks
                 with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True, pretrained=None,
                 use_extra_extractor=True, with_cp=False)
        
   
    output = model(input)
    print('output=', output.size())
"""