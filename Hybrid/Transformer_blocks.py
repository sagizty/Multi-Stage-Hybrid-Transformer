"""
Transformer blocks script  ver： OCT 27th 20：00 official release

by the authors, check our github page:
https://github.com/sagizty/Multi-Stage-Hybrid-Transformer

based on：timm
https://www.freeaihub.com/post/94067.html

"""

import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_

from .attention_modules import simam_module, cbam_module, se_module


class FFN(nn.Module):  # Mlp from timm
    """
    FFN (from timm)

    :param in_features:
    :param hidden_features:
    :param out_features:
    :param act_layer:
    :param drop:
    """

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


class Attention(nn.Module):  # qkv Transform + MSA(MHSA) (Attention from timm)
    """
    qkv Transform + MSA(MHSA) (from timm)

    # input  x.shape = batch, patch_number, patch_dim
    # output  x.shape = batch, patch_number, patch_dim

    :param dim: dim=CNN feature dim, because the patch size is 1x1
    :param num_heads:
    :param qkv_bias:
    :param qk_scale: by default head_dim ** -0.5  (squre root)
    :param attn_drop: dropout rate after MHSA
    :param proj_drop:

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
        # input x.shape = batch, patch_number, patch_dim
        batch, patch_number, patch_dim = x.shape

        qkv = self.qkv(x).reshape(batch, patch_number, 3, self.num_heads, patch_dim //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)  # Dropout

        x = (attn @ v).transpose(1, 2).reshape(batch, patch_number, patch_dim)

        x = self.proj(x)
        x = self.proj_drop(x)  # mlp

        # output x.shape = batch, patch_number, patch_dim
        return x


class Encoder_Block(nn.Module):  # teansformer Block from timm

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        # input x.shape = batch, patch_number, patch_dim
        # output x.shape = batch, patch_number, patch_dim

        :param dim: dim
        :param num_heads:
        :param mlp_ratio: FFN
        :param qkv_bias:
        :param qk_scale: by default head_dim ** -0.5  (squre root)
        :param drop:
        :param attn_drop: dropout rate after Attention
        :param drop_path: dropout rate after sd
        :param act_layer: FFN act
        :param norm_layer: Pre Norm
        """
        super().__init__()
        # Pre Norm
        self.norm1 = norm_layer(dim)  # Transformer used the nn.LayerNorm
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        # NOTE from timm: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # stochastic depth

        # Add & Norm
        self.norm2 = norm_layer(dim)

        # FFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Guided_Attention(nn.Module):  # q1 k1 v0 Transform + MSA(MHSA) (based on timm Attention)
    """
    notice the q abd k is guided information from Focus module
    qkv Transform + MSA(MHSA) (from timm)

    # 3 input of x.shape = batch, patch_number, patch_dim
    # 1 output of x.shape = batch, patch_number, patch_dim

    :param dim: dim = CNN feature dim, because the patch size is 1x1
    :param num_heads:
    :param qkv_bias:
    :param qk_scale: by default head_dim ** -0.5  (squre root)
    :param attn_drop:
    :param proj_drop:

    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qT = nn.Linear(dim, dim, bias=qkv_bias)
        self.kT = nn.Linear(dim, dim, bias=qkv_bias)
        self.vT = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_encoder, k_encoder, v_input):
        # 3 input of x.shape = batch, patch_number, patch_dim
        batch, patch_number, patch_dim = v_input.shape

        q = self.qT(q_encoder).reshape(batch, patch_number, 1, self.num_heads,
                                       patch_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.kT(k_encoder).reshape(batch, patch_number, 1, self.num_heads,
                                       patch_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.vT(v_input).reshape(batch, patch_number, 1, self.num_heads,
                                     patch_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k = k[0]
        v = v[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)  # Dropout

        x = (attn @ v).transpose(1, 2).reshape(batch, patch_number, patch_dim)

        x = self.proj(x)
        x = self.proj_drop(x)  # mlp Dropout

        # output of x.shape = batch, patch_number, patch_dim
        return x


class Decoder_Block(nn.Module):
    # FGD Decoder (Transformer encoder + Guided Attention block block)
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        # input x.shape = batch, patch_number, patch_dim
        # output x.shape = batch, patch_number, patch_dim

        :param dim: dim=CNN feature dim, because the patch size is 1x1
        :param num_heads: multi-head
        :param mlp_ratio: FFN expand ratio
        :param qkv_bias: qkv MLP bias
        :param qk_scale: by default head_dim ** -0.5  (squre root)
        :param drop: the MLP after MHSA equipt a dropout rate
        :param attn_drop: dropout rate after attention block
        :param drop_path: dropout rate for stochastic depth
        :param act_layer: FFN act
        :param norm_layer: Pre Norm strategy with norm layer
        """
        super().__init__()
        # Pre Norm
        self.norm0 = norm_layer(dim)  # nn.LayerNorm
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        # stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Pre Norm
        self.norm1 = norm_layer(dim)

        # FFN1
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.FFN1 = FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Guided_Attention
        self.Guided_attn = Guided_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            attn_drop=attn_drop, proj_drop=drop)

        # Add & Norm
        self.norm2 = norm_layer(dim)
        # FFN2
        self.FFN2 = FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Add & Norm
        self.norm3 = norm_layer(dim)

    def forward(self, q_encoder, k_encoder, v_input):
        v_self = v_input + self.drop_path(self.attn(self.norm0(v_input)))

        v_self = v_self + self.drop_path(self.FFN1(self.norm1(v_self)))

        # norm layer for v only, the normalization of q and k is inside FGD Focus block
        v_self = v_self + self.drop_path(self.Guided_attn(q_encoder, k_encoder, self.norm2(v_self)))

        v_self = v_self + self.drop_path(self.FFN2(self.norm3(v_self)))

        return v_self


'''
# testing example

model=Decoder_Block(dim=768)
k = torch.randn(7, 49, 768)
q = torch.randn(7, 49, 768)
v = torch.randn(7, 49, 768)
x = model(k,q,v)
print(x.shape)
'''


class PatchEmbed(nn.Module):  # PatchEmbed from timm
    """
    Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)

        # x: (B, 14*14, 768)
        return x


class Hybrid_feature_map_Embed(nn.Module):  # HybridEmbed from timm
    """
    CNN Feature Map Embedding, required backbone which is just for referance here
    Extract feature map from CNN, flatten, project to embedding dim.

    # input x.shape = batch, feature_dim, feature_size[0], feature_size[1]
    # output x.shape = batch, patch_number, patch_dim
    """

    def __init__(self, backbone, img_size=224, patch_size=1, feature_size=None, feature_dim=None,
                 in_chans=3, embed_dim=768):
        super().__init__()

        assert isinstance(backbone, nn.Module)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone

        if feature_size is None or feature_dim is None:  # backbone output feature_size
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            '''
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
            '''

        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0

        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])  # patchlize

        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        x = self.proj(x).flatten(2).transpose(1, 2)  # shape = ( )
        """
        x.shape:  batch, feature_dim, feature_size[0], feature_size[1]
        proj(x).shape:  batch, embed_dim, patch_height_num, patch_width_num
        flatten(2).shape:  batch, embed_dim, patch_num
        .transpose(1, 2).shape:  batch feature_patch_number feature_patch_dim
        """
        # output: x.shape = batch, patch_number, patch_dim
        return x


class Last_feature_map_Embed(nn.Module):
    """
    use this block to connect last CNN stage to the first Transformer block
    Extract feature map from CNN, flatten, project to embedding dim.

    # input x.shape = batch, feature_dim, feature_size[0], feature_size[1]
    # output x.shape = batch, patch_number, patch_dim
    """

    def __init__(self, patch_size=1, feature_size=(7, 7), feature_dim=2048, embed_dim=768,
                 Attention_module=None):
        super().__init__()

        # Attention module
        if Attention_module is not None:
            if Attention_module == 'SimAM':
                self.Attention_module = simam_module(e_lambda=1e-4)
            elif Attention_module == 'CBAM':
                self.Attention_module = cbam_module(gate_channels=feature_dim)
            elif Attention_module == 'SE':
                self.Attention_module = se_module(channel=feature_dim)
        else:
            self.Attention_module = None

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        feature_size = to_2tuple(feature_size)

        # feature map should be matching the size
        assert feature_size[0] % self.patch_size[0] == 0 and feature_size[1] % self.patch_size[1] == 0

        self.grid_size = (feature_size[0] // self.patch_size[0], feature_size[1] // self.patch_size[1])  # patch

        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # use the conv to split the patch by the following design:
        self.proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim,
                              kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        if self.Attention_module is not None:
            x = self.Attention_module(x)

        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        x = self.proj(x).flatten(2).transpose(1, 2)
        """
        x.shape:  batch, feature_dim, feature_size[0], feature_size[1]
        proj(x).shape:  batch, embed_dim, patch_height_num, patch_width_num
        flatten(2).shape:  batch, embed_dim, patch_num
        .transpose(1, 2).shape:  batch feature_patch_number feature_patch_dim
        """
        # output 格式 x.shape = batch, patch_number, patch_dim
        return x


class Focus_Embed(nn.Module):  # Attention guided module for hybridzing the early stages CNN feature
    """
    FGD Focus module
    Extract feature map from CNN, flatten, project to embedding dim. and use them as attention guidance

    input: x.shape = batch, feature_dim, feature_size[0], feature_size[1]

    Firstly, an attention block will be used to stable the feature projecting process

    Secondly, for each feature map，the focus will be 2 path: gaze and glance
    in gaze path Max pool will be applied to get prominent information
    in glance path Avg pool will be applied to get general information

    after the dual pooling path 2 seperate CNNs will be used to project the dimension
    Finally, flattern and transpose will be applied

    output 2 attention guidance: gaze, glance
    x.shape = batch, patch_number, patch_dim


    ref:
    ResNet50's feature map from different stages (edge size of 224)
    stage 1 output feature map: torch.Size([b, 256, 56, 56])
    stage 2 output feature map: torch.Size([b, 512, 28, 28])
    stage 3 output feature map: torch.Size([b, 1024, 14, 14])
    stage 4 output feature map: torch.Size([b, 2048, 7, 7])
    """

    def __init__(self, patch_size=1, target_feature_size=(7, 7), feature_size=(56, 56), feature_dim=256, embed_dim=768,
                 Attention_module=None, norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        feature_size = to_2tuple(feature_size)  # patch size of the current feature map

        target_feature_size = to_2tuple(target_feature_size)  # patch size of the last feature map

        # cheak feature map can be patchlize to target_feature_size
        assert feature_size[0] % target_feature_size[0] == 0 and feature_size[1] % target_feature_size[1] == 0

        # cheak target_feature map can be patchlize to patch
        assert target_feature_size[0] % patch_size[0] == 0 and target_feature_size[1] % patch_size[1] == 0

        # Attention block
        if Attention_module is not None:
            if Attention_module == 'SimAM':
                self.Attention_module = simam_module(e_lambda=1e-4)
            elif Attention_module == 'CBAM':
                self.Attention_module = cbam_module(gate_channels=feature_dim)
            elif Attention_module == 'SE':
                self.Attention_module = se_module(channel=feature_dim)
        else:
            self.Attention_module = None

        # split focus ROI
        self.focus_size = (feature_size[0] // target_feature_size[0], feature_size[1] // target_feature_size[1])
        self.num_focus = self.focus_size[0] * self.focus_size[1]
        # by kernel_size=focus_size, stride=focus_size design
        # output_size=target_feature_size=7x7 so as to match the minist feature map

        self.gaze = nn.MaxPool2d(self.focus_size, stride=self.focus_size)
        self.glance = nn.AvgPool2d(self.focus_size, stride=self.focus_size)
        # x.shape:  batch, feature_dim, target_feature_size[0], target_feature_size[1]

        # split patch
        self.grid_size = (target_feature_size[0] // patch_size[0], target_feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # use CNN to project dim to patch_dim
        self.gaze_proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim,
                                   kernel_size=patch_size, stride=patch_size)
        self.glance_proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        self.norm_q = norm_layer(embed_dim)  # Transformer nn.LayerNorm
        self.norm_k = norm_layer(embed_dim)  # Transformer nn.LayerNorm

    def forward(self, x):
        if self.Attention_module is not None:
            x = self.Attention_module(x)

        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        q = self.norm_q(self.gaze_proj(self.gaze(x)).flatten(2).transpose(1, 2))
        k = self.norm_k(self.glance_proj(self.glance(x)).flatten(2).transpose(1, 2))
        """
        x.shape:  batch, feature_dim, feature_size[0], feature_size[1]
        gaze/glance(x).shape:  batch, feature_dim, target_feature_size[0], target_feature_size[1]
        proj(x).shape:  batch, embed_dim, patch_height_num, patch_width_num
        flatten(2).shape:  batch, embed_dim, patch_num
        .transpose(1, 2).shape:  batch feature_patch_number feature_patch_dim
        """
        # output x.shape = batch, patch_number, patch_dim
        return q, k


'''
# test sample
model = Focus_Embed()
x = torch.randn(4, 256, 56, 56)
y1,y2 = model(x)
print(y1.shape)
print(y2.shape)
'''


class Focus_SEmbed(nn.Module):  # Attention guided module for hybridzing the early stages CNN feature
    """

    self focus (q=k)  based on FGD Focus block

    Extract feature map from CNN, flatten, project to embedding dim. and use them as attention guidance

    input: x.shape = batch, feature_dim, feature_size[0], feature_size[1]

    Firstly, an attention block will be used to stable the feature projecting process

    Secondly, for each feature map，the focus will be 1 path: glance
    in glance path Avg pool will be applied to get general information

    after the pooling process 1 CNN will be used to project the dimension
    Finally, flattern and transpose will be applied

    output 2 attention guidance: glance, glance
    x.shape = batch, patch_number, patch_dim
    """

    def __init__(self, patch_size=1, target_feature_size=(7, 7), feature_size=(56, 56), feature_dim=256, embed_dim=768,
                 Attention_module=None, norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        feature_size = to_2tuple(feature_size)

        target_feature_size = to_2tuple(target_feature_size)

        assert feature_size[0] % target_feature_size[0] == 0 and feature_size[1] % target_feature_size[1] == 0

        assert target_feature_size[0] % patch_size[0] == 0 and target_feature_size[1] % patch_size[1] == 0

        if Attention_module is not None:
            if Attention_module == 'SimAM':
                self.Attention_module = simam_module(e_lambda=1e-4)
            elif Attention_module == 'CBAM':
                self.Attention_module = cbam_module(gate_channels=feature_dim)
            elif Attention_module == 'SE':
                self.Attention_module = se_module(channel=feature_dim)
        else:
            self.Attention_module = None

        self.focus_size = (feature_size[0] // target_feature_size[0], feature_size[1] // target_feature_size[1])
        self.num_focus = self.focus_size[0] * self.focus_size[1]

        self.gaze = nn.MaxPool2d(self.focus_size, stride=self.focus_size)

        self.grid_size = (target_feature_size[0] // patch_size[0], target_feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim, kernel_size=patch_size,
                              stride=patch_size)

        self.norm_f = norm_layer(embed_dim)

    def forward(self, x):
        if self.Attention_module is not None:
            x = self.Attention_module(x)

        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        q = self.norm_f(self.proj(self.gaze(x)).flatten(2).transpose(1, 2))
        k = q
        """
        x.shape:  batch, feature_dim, feature_size[0], feature_size[1]
        gaze/glance(x).shape:  batch, feature_dim, target_feature_size[0], target_feature_size[1]
        proj(x).shape:  batch, embed_dim, patch_height_num, patch_width_num
        flatten(2).shape:  batch, embed_dim, patch_num
        .transpose(1, 2).shape:  batch feature_patch_number feature_patch_dim
        """
        # output x.shape = batch, patch_number, patch_dim
        return q, k


class Focus_Aggressive(nn.Module):  # Attention guided module for hybridzing the early stages CNN feature
    """
    Aggressive CNN Focus based on FGD Focus block

    Extract feature map from CNN, flatten, project to embedding dim. and use them as attention guidance

    input: x.shape = batch, feature_dim, feature_size[0], feature_size[1]

    Firstly, an attention block will be used to stable the feature projecting process

    Secondly, 2 CNNs will be used to project the dimension

    Finally, flattern and transpose will be applied

    output 2 attention guidance: gaze, glance
    x.shape = batch, patch_number, patch_dim

    """

    def __init__(self, patch_size=1, target_feature_size=(7, 7), feature_size=(56, 56), feature_dim=256, embed_dim=768,
                 Attention_module=None, norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = to_2tuple(patch_size)  # patch size of the last feature map
        feature_size = to_2tuple(feature_size)

        target_feature_size = to_2tuple(target_feature_size)

        assert feature_size[0] % target_feature_size[0] == 0 and feature_size[1] % target_feature_size[1] == 0

        assert target_feature_size[0] % patch_size[0] == 0 and target_feature_size[1] % patch_size[1] == 0

        if Attention_module is not None:
            if Attention_module == 'SimAM':
                self.Attention_module = simam_module(e_lambda=1e-4)
            elif Attention_module == 'CBAM':
                self.Attention_module = cbam_module(gate_channels=feature_dim)
            elif Attention_module == 'SE':
                self.Attention_module = se_module(channel=feature_dim)
        else:
            self.Attention_module = None

        self.focus_size = (feature_size[0] // target_feature_size[0], feature_size[1] // target_feature_size[1])

        self.grid_size = (self.focus_size[0] * patch_size[0], self.focus_size[1] * patch_size[1])
        self.num_patches = (feature_size[0] // self.grid_size[0]) * (feature_size[1] // self.grid_size[1])

        self.gaze_proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim,
                                   kernel_size=self.grid_size, stride=self.grid_size)
        self.glance_proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim,
                                     kernel_size=self.grid_size, stride=self.grid_size)

        self.norm_q = norm_layer(embed_dim)
        self.norm_k = norm_layer(embed_dim)

    def forward(self, x):
        if self.Attention_module is not None:
            x = self.Attention_module(x)

        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        q = self.norm_q(self.gaze_proj(x).flatten(2).transpose(1, 2))
        k = self.norm_k(self.glance_proj(x).flatten(2).transpose(1, 2))
        """
        x.shape:  batch, feature_dim, feature_size[0], feature_size[1]
        proj(x).shape:  batch, embed_dim, patch_height_num, patch_width_num
        flatten(2).shape:  batch, embed_dim, patch_num
        .transpose(1, 2).shape:  batch feature_patch_number feature_patch_dim
        """
        # output x.shape = batch, patch_number, patch_dim
        return q, k


class Focus_SAggressive(nn.Module):  # Attention guided module for hybridzing the early stages CNN feature
    """
    Aggressive CNN self Focus
    Extract feature map from CNN, flatten, project to embedding dim. and use them as attention guidance

    input: x.shape = batch, feature_dim, feature_size[0], feature_size[1]

    Firstly, an attention block will be used to stable the feature projecting process

    Secondly, 1 CNN will be used to project the dimension

    Finally, flattern and transpose will be applied

    output 2 attention guidance: glance, glance
    x.shape = batch, patch_number, patch_dim
    """

    def __init__(self, patch_size=1, target_feature_size=(7, 7), feature_size=(56, 56), feature_dim=256, embed_dim=768,
                 Attention_module=None, norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        feature_size = to_2tuple(feature_size)

        target_feature_size = to_2tuple(target_feature_size)

        assert feature_size[0] % target_feature_size[0] == 0 and feature_size[1] % target_feature_size[1] == 0

        assert target_feature_size[0] % patch_size[0] == 0 and target_feature_size[1] % patch_size[1] == 0

        if Attention_module is not None:
            if Attention_module == 'SimAM':
                self.Attention_module = simam_module(e_lambda=1e-4)
            elif Attention_module == 'CBAM':
                self.Attention_module = cbam_module(gate_channels=feature_dim)
            elif Attention_module == 'SE':
                self.Attention_module = se_module(channel=feature_dim)
        else:
            self.Attention_module = None

        self.focus_size = (feature_size[0] // target_feature_size[0], feature_size[1] // target_feature_size[1])

        self.grid_size = (self.focus_size[0] * patch_size[0], self.focus_size[1] * patch_size[1])
        self.num_patches = (feature_size[0] // self.grid_size[0]) * (feature_size[1] // self.grid_size[1])

        self.proj = nn.Conv2d(in_channels=feature_dim, out_channels=embed_dim,
                              kernel_size=self.grid_size, stride=self.grid_size)

        self.norm_f = norm_layer(embed_dim)

    def forward(self, x):
        if self.Attention_module is not None:
            x = self.Attention_module(x)

        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        q = self.norm_f(self.proj(x).flatten(2).transpose(1, 2))
        k = q
        """
        x.shape:  batch, feature_dim, feature_size[0], feature_size[1]
        proj(x).shape:  batch, embed_dim, patch_height_num, patch_width_num
        flatten(2).shape:  batch, embed_dim, patch_num
        .transpose(1, 2).shape:  batch feature_patch_number feature_patch_dim
        """
        # output x.shape = batch, patch_number, patch_dim
        return q, k


class VisionTransformer(nn.Module):  # From timm to review the ViT and ViT_resn5
    """
    Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Encoder_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                          attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.head_dist = None

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])  # use cls token for cls head

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Stage_wise_hybrid_Transformer(nn.Module):
    """
    MSHT: Multi Stage Hybrid Transformer
    stem+4个ResNet stages（Backbone）is used as backbone
    then, last feature map patch embedding is used to connect the CNN output to the decoder1 input

    horizonally, 4 ResNet Stage has its feature map connecting to the Focus module
    which we be use as attention guidance into the FGD decoder
    """

    def __init__(self, backbone, num_classes=1000, patch_size=1, embed_dim=768, depth=4, num_heads=8, mlp_ratio=4.,
                 qkv_bias=True, representation_size=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 use_cls_token=True, use_pos_embedding=True, use_att_module='SimAM', stage_size=(56, 28, 14, 7),
                 stage_dim=(256, 512, 1024, 2048), norm_layer=None, act_layer=None):
        """
        Args:
            backbone (nn.Module): input backbone = stem + 4 ResNet stages
            num_classes (int): number of classes for classification head
            patch_size (int, tuple): patch size
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate

            use_cls_token(bool): classification token
            use_pos_embedding(bool): use positional embedding
            use_att_module(str or None): use which attention module in embedding

            stage_size (int, tuple): the stage feature map size of ResNet stages
            stage_dim (int, tuple): the stage feature map dimension of ResNet stages
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        if len(stage_dim) != len(stage_size):
            raise TypeError('stage_dim and stage_size mismatch!')
        else:
            self.stage_num = len(stage_dim)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.cls_token_num = 1 if use_cls_token else 0
        self.use_pos_embedding = use_pos_embedding

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # backbone CNN
        self.backbone = backbone

        # Attention module
        if use_att_module is not None:
            if use_att_module in ['SimAM', 'CBAM', 'SE']:
                Attention_module = use_att_module
            else:
                Attention_module = None
        else:
            Attention_module = None

        self.patch_embed = Last_feature_map_Embed(patch_size=patch_size, feature_size=stage_size[-1],
                                                  feature_dim=stage_dim[-1], embed_dim=self.embed_dim,
                                                  Attention_module=Attention_module)
        num_patches = self.patch_embed.num_patches

        # blobal sharing cls token and positional embedding
        self.cls_token_0 = nn.Parameter(torch.zeros(1, 1, embed_dim))  # like message token
        if self.use_pos_embedding:
            self.pos_embed_0 = nn.Parameter(torch.zeros(1, num_patches + self.cls_token_num, embed_dim))

        '''
        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.cls_token_3 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.cls_token_4 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        '''

        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.dec1 = Decoder_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
                                  act_layer=act_layer)
        self.Fo1 = Focus_Embed(patch_size=patch_size, target_feature_size=stage_size[-1], feature_size=stage_size[0],
                               feature_dim=stage_dim[0], embed_dim=embed_dim, Attention_module=Attention_module,
                               norm_layer=norm_layer)

        self.dec2 = Decoder_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
                                  act_layer=act_layer)
        self.Fo2 = Focus_Embed(patch_size=patch_size, target_feature_size=stage_size[-1], feature_size=stage_size[1],
                               feature_dim=stage_dim[1], embed_dim=embed_dim, Attention_module=Attention_module,
                               norm_layer=norm_layer)

        self.dec3 = Decoder_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[2], norm_layer=norm_layer,
                                  act_layer=act_layer)
        self.Fo3 = Focus_Embed(patch_size=patch_size, target_feature_size=stage_size[-1], feature_size=stage_size[2],
                               feature_dim=stage_dim[2], embed_dim=embed_dim, Attention_module=Attention_module,
                               norm_layer=norm_layer)

        if self.stage_num == 4:
            self.dec4 = Decoder_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[3], norm_layer=norm_layer,
                                      act_layer=act_layer)
            self.Fo4 = Focus_Embed(patch_size=patch_size, target_feature_size=stage_size[-1],
                                   feature_size=stage_size[-1],
                                   feature_dim=stage_dim[-1], embed_dim=embed_dim, Attention_module=Attention_module,
                                   norm_layer=norm_layer)

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.head_dist = None

    def forward_features(self, x):
        if self.stage_num == 3:
            stage1_out, stage2_out, stage3_out = self.backbone(x)
            # embedding the last feature map
            x = self.patch_embed(stage3_out)

        elif self.stage_num == 4:
            stage1_out, stage2_out, stage3_out, stage4_out = self.backbone(x)
            # embedding the last feature map
            x = self.patch_embed(stage4_out)
        else:
            raise TypeError('stage_dim is not legal !')

        # get guidance info
        s1_q, s1_k = self.Fo1(stage1_out)
        s2_q, s2_k = self.Fo2(stage2_out)
        s3_q, s3_k = self.Fo3(stage3_out)
        if self.stage_num == 4:
            s4_q, s4_k = self.Fo4(stage4_out)

        if self.cls_token_num != 0:  # concat cls token
            # process the（cls token / message token）
            cls_token_0 = self.cls_token_0.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token_0, x), dim=1)  # 增加classification head patch

            s1_q = torch.cat((cls_token_0, s1_q), dim=1)
            s1_k = torch.cat((cls_token_0, s1_k), dim=1)
            s2_q = torch.cat((cls_token_0, s2_q), dim=1)
            s2_k = torch.cat((cls_token_0, s2_k), dim=1)
            s3_q = torch.cat((cls_token_0, s3_q), dim=1)
            s3_k = torch.cat((cls_token_0, s3_k), dim=1)
            if self.stage_num == 4:
                s4_q = torch.cat((cls_token_0, s4_q), dim=1)
                s4_k = torch.cat((cls_token_0, s4_k), dim=1)

        if self.use_pos_embedding:

            s1_q = self.pos_drop(s1_q + self.pos_embed_0)
            s1_k = self.pos_drop(s1_k + self.pos_embed_0)
            s2_q = self.pos_drop(s2_q + self.pos_embed_0)
            s2_k = self.pos_drop(s2_k + self.pos_embed_0)
            s3_q = self.pos_drop(s3_q + self.pos_embed_0)
            s3_k = self.pos_drop(s3_k + self.pos_embed_0)
            if self.stage_num == 4:
                s4_q = self.pos_drop(s4_q + self.pos_embed_0)
                s4_k = self.pos_drop(s4_k + self.pos_embed_0)

            # plus to encoding positional infor
            x = self.pos_drop(x + self.pos_embed_0)

        else:

            s1_q = self.pos_drop(s1_q)
            s1_k = self.pos_drop(s1_k)
            s2_q = self.pos_drop(s2_q)
            s2_k = self.pos_drop(s2_k)
            s3_q = self.pos_drop(s3_q)
            s3_k = self.pos_drop(s3_k)
            if self.stage_num == 4:
                s4_q = self.pos_drop(s4_q)
                s4_k = self.pos_drop(s4_k)

            # stem's feature map
            x = self.pos_drop(x)

        # Decoder module use the guidance to help global modeling process

        x = self.dec1(s1_q, s1_k, x)

        x = self.dec2(s2_q, s2_k, x)

        x = self.dec3(s3_q, s3_k, x)

        if self.stage_num == 4:
            x = self.dec4(s4_q, s4_k, x)

        x = self.norm(x)
        return self.pre_logits(x[:, 0])  # take the first cls token

    def forward(self, x):
        x = self.forward_features(x)  # connect the cls token to the cls head
        x = self.head(x)
        return x
