# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


from utils import batch_index_select,get_index
import numpy as np
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=to_2tuple(patch_size), stride=to_2tuple(patch_size))

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    """
        return the attention map
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x2, atten = self.attn(self.norm1(x))
        x = x + self.drop_path(x2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, atten


class CFVisionTransformer(nn.Module):
    """ 
    Vision Transformer with support for patch or hybrid CNN input stage
    
    """
    def __init__(self, img_size_list=[96, 192, 384], patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.informative_selection = False
        self.alpha = 0.5
        self.beta = 0.99
        self.target_index = [3,4,5,6,7,8,9,10,11]

        self.img_size_list = img_size_list
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.patch_size = patch_size

        num_patches_list = [(img_size // patch_size) ** 2 for img_size in img_size_list]
        self.num_patches_list = num_patches_list

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_list = [nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) for num_patches in num_patches_list]
        self.pos_embed_list = nn.ParameterList(self.pos_embed_list)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.reuse_block = nn.Sequential(
                norm_layer(embed_dim),
                Mlp(in_features=embed_dim, hidden_features=mlp_ratio*embed_dim,out_features=embed_dim,act_layer=nn.GELU,drop=drop_rate)
            ) 

        for pos_embed in self.pos_embed_list:
            trunc_normal_(pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed_list', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def get_vis_data(self):
        """
            For Visualization
        """
        return self.important_index

    def forward(self, xx, train=False, thresholds=[0.8, 0.5]): # thresholds should be descending
        if train:
            # let each level run on each sample (thus full no_exit indicies)
            # and collect all results for learning
            all_results = []
        else: # thresholds only used during evaluation; keep final_result only
            assert len(thresholds) == len(self.img_size_list) - 1

        global_attention = 0

        # coarse stage
        x = xx[0]
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed_list[0]
        x = self.pos_drop(x)
        embedding_x1 = x
        for index,blk in enumerate(self.blocks):
            x, atten = blk(x)
            if index in self.target_index:
                global_attention = self.beta*global_attention + (1-self.beta)*atten
        x = self.norm(x)
        self.first_stage_output = x

        coarse_result = self.head(x[:, 0])

        no_exit = {}

        if train:
            all_results.append(coarse_result)

            for level in range(1, len(self.img_size_list)):
                no_exit[level] = torch.ones_like(coarse_result) # full indicies

        else:
            final_result = coarse_result

            probs_temp = F.softmax(coarse_result, dim=1)
            max_preds, _ = probs_temp.max(dim=1)
            # example: 4 levels in img_size_list (Lv0, Lv1, Lv2, Lv3)
            # (not necessarily the img_size_list actually used!)
            
            # then 3 levels of thresholds:
            #   [0] Lv1/0 - higher than this = exit
            #   [1] Lv2/1
            #   [2] Lv3/2 - lowest
            # (note: the coarse stage above was Lv0)

            # range(1, len(thresholds)) is range (1, 3), or like [1, 2]
            # so here we're handling:
            #   infer with Lv1 again <- Lv2/1 < max_pred <= Lv1/0
            #          level=1   thresholds[1]       thresholds[0]
            #   infer with Lv2 again <- Lv3/2 < max_pred <= Lv2/1
            #          level=2   thresholds[2]       thresholds[1]
            for level in range(1, len(thresholds)):
                no_exit[level] = torch.logical_and(max_preds > thresholds[level+1], max_preds <= thresholds[level])
            # then we handle:
            #   infer with Lv3 again <- max_pred <= Lv3/2
            #          level=len(thresholds) thresholds[-1]
            no_exit[len(thresholds)] = max_preds <= thresholds[-1]

        # reuse (preparation)
        feature_temp = self.first_stage_output[:,1:,:][no_exit]
        feature_temp = self.reuse_block(feature_temp)
        B, new_HW, C = feature_temp.shape
        feature_temp = feature_temp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        feature_temp = F.interpolate(feature_temp, to_2tuple(self.img_size_list[level]//self.patch_size), mode='nearest')

        # fine stage
        fine_result = {}
        if self.informative_selection:
            self.important_index = {}

        # recall the example above, then range(1, len(self.img_size_list))
        # is range (1, 4), or like [1, 2, 3]
        for level in range(1, len(self.img_size_list)):
            x = xx[level][no_exit[level]]
            B = x.shape[0]
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

            # reuse (applying)
            feature_temp_this = feature_temp.view(B, C, x.size(1) - 1).transpose(1, 2)
            feature_temp_this = torch.cat((torch.zeros(B, 1, self.embed_dim).cuda(), feature_temp_this), dim=1)

            x = x+feature_temp_this      # shortcut
            embedding_x2 = x + self.pos_embed_list[level]

            if self.informative_selection:
                cls_attn = global_attention[no_exit].mean(dim=1)[:,0,1:] # not calculating cls_token itself
                import_token_num = math.ceil(self.alpha * self.num_patches_list[0])
                policy_index = torch.argsort(cls_attn, dim=1, descending=True)
                unimportant_index = policy_index[:, import_token_num:]
                important_index = policy_index[:, :import_token_num]
                unimportant_tokens = batch_index_select(embedding_x1[no_exit], unimportant_index+1)
                important_index = get_index(important_index,image_size=self.img_size_list[level])
                self.important_index[level] = important_index
                cls_index = torch.zeros((B,1)).cuda().long()
                important_index = torch.cat((cls_index, important_index+1), dim=1)
                important_tokens = batch_index_select(embedding_x2, important_index)
                x = torch.cat((important_tokens, unimportant_tokens), dim=1)

            x = self.pos_drop(x)
            for blk in self.blocks:
                x, _ = blk(x)
            x = self.norm(x)
            fine_result[level] = self.head(x[:, 0])

            if train:
                all_results.append(fine_result[level])
            else:
                final_result[no_exit[level]] = fine_result[level]

        return all_results if train else final_result


@register_model
def cf_deit_small(pretrained=False, **kwargs):
    model = CFVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

