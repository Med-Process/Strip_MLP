# --------------------------------------------------------
# Copyright (c) 2023 CVIP of SUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Guiping Cao
# --------------------------------------------------------


from ast import Pass
from email.quoprimime import body_check

# from turtle import color, forward
from matplotlib.pyplot import axis
from numpy import append

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from timm.models.layers import to_2tuple, trunc_normal_, DropPath



class BN_Activ_Conv(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride=(1, 1), dilation=(1, 1), groups=1):
        super(BN_Activ_Conv, self).__init__()
        self.BN = nn.BatchNorm2d(out_channels)
        self.Activation = activation
        padding = [int((dilation[j] * (kernel_size[j] - 1) - stride[j] + 1) / 2) for j in range(2)]  # Same padding
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=False)

    def forward(self, img):
        img = self.BN(img)
        img = self.Activation(img)
        img = self.Conv(img)
        return img


class DepthWise_Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_merge = BN_Activ_Conv(channels, nn.GELU(), channels, (3, 3), groups=channels)

    def forward(self, img):
        img = self.conv_merge(img)
        return img


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)
        trunc_normal_(self.embeddings_table, std=.02)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class StripMLP_Block(nn.Module):
    def __init__(self, channels, H, W):
        super().__init__()
        assert W == H
        self.channels = channels
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(channels//2)

        if channels % 80 == 0:
            patch = 2
        else:
            patch = 4

        self.ratio = 1; self.C = int(channels *0.5/ patch); self.chan = self.ratio * self.C

        self.proj_h = nn.Conv2d(H*self.C, self.chan*H, (1, 3), stride=1, padding=(0, 1), groups=self.C,bias=True)
        self.proj_w = nn.Conv2d(self.C*W, self.chan*W, (1, 3), stride=1, padding=(0, 1), groups=self.C, bias=True)

        self.fuse_h = nn.Conv2d(channels, channels//2, (1,1), (1,1), bias=False)
        self.fuse_w = nn.Conv2d(channels, channels//2, (1,1), (1,1), bias=False)

        self.mlp=nn.Sequential(nn.Conv2d(channels, channels, 1, 1,bias=True),nn.BatchNorm2d(channels),nn.GELU())

        dim = channels // 2

        self.fc_h = nn.Conv2d(dim, dim, (3,7), stride=1, padding=(1,7//2), groups=dim, bias=False) 
        self.fc_w = nn.Conv2d(dim, dim, (7,3), stride=1, padding=(7//2,1), groups=dim, bias=False)

        self.reweight = Mlp(dim, dim // 2, dim * 3)

        self.fuse = nn.Conv2d(channels, channels, (1,1), (1,1), bias=False)

        self.relate_pos_h = RelativePosition(channels//2, H)
        self.relate_pos_w = RelativePosition(channels//2, W)

    def forward(self, x):
        N, C, H, W = x.shape

        x = self.mlp(x)

        x_1 = x[:, :C//2, :, :]
        x_2 = x[:, C//2:, :, :]
        
        x_1 = self.strip_mlp(x_1)

        # for x_2
        x_w = self.fc_h(x_2)
        x_h = self.fc_w(x_2)
        att = F.adaptive_avg_pool2d(x_h + x_w + x_2, output_size=1)
        att = self.reweight(att).reshape(N, C//2, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x_2 = x_h * att[0] + x_w * att[1] + x_2 * att[2]

        x = self.fuse(torch.cat([x_1, x_2], dim=1))

        return x

    def strip_mlp(self, x):
        N, C, H, W = x.shape

        pos_h = self.relate_pos_h(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        pos_w = self.relate_pos_w(H, W).unsqueeze(0).permute(0, 3, 1, 2)

        C1 = int(C/self.C)
        
        x_h = x + pos_h
        x_h = x_h.view(N, C1, self.C, H, W)     # N C1 C2 H W

        x_h = x_h.permute(0, 1, 3, 2, 4).contiguous().view(N, C1, H, self.C*W)  # N C1 H WC2   

        x_h = self.proj_h(x_h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_h = x_h.view(N, C1, H, self.C, W).permute(0, 1, 3, 2, 4).contiguous().view(N, C, H, W) # N C1 C2 H W

        x_h = self.fuse_h(torch.cat([x_h, x], dim=1))
        x_h = self.activation(self.BN(x_h)) + pos_w

        x_w = self.proj_w(x_h.view(N, C1, H*self.C, W).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_w = x_w.contiguous().view(N, C1, self.C, H, W).view(N, C, H, W)

        x = self.fuse_w(torch.cat([x, x_w], dim=1))

        return x


class TokenMixing(nn.Module):
    r""" Token mixing of Strip MLP

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, C, H, W):
        super().__init__()
        self.smlp_block = StripMLP_Block(C, H, W)
        self.dwsc = DepthWise_Conv(C)
    
    def forward(self, x):
        x = self.dwsc(x)
        x = self.smlp_block(x)

        return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ChannelMixing(nn.Module):

    def __init__(self, in_channel, alpha, use_dropout=False, drop_rate=0):
        super().__init__()

        self.use_dropout = use_dropout

        self.conv_77 = nn.Conv2d(in_channel, in_channel, 7, 1, 3, groups=in_channel, bias=False)
        self.layer_norm = nn.LayerNorm(in_channel)
        self.fc1 = nn.Linear(in_channel, alpha * in_channel)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(alpha * in_channel, in_channel)

        self.grn = GRN(3*in_channel)

    
    def forward(self, x):
        N, C, H, W = x.shape

        x = self.conv_77(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.grn(x)

        x = self.fc2(x)

        x = x.permute(0, 3, 1, 2)

        return x



class BasicBlock(nn.Module):
    def __init__(self, in_channel, H, W, alpha, use_dropout=False, drop_rate=0):
        super().__init__()

        self.token_mixing = TokenMixing(in_channel, H, W)
        self.channel_mixing = ChannelMixing(in_channel, alpha, use_dropout, drop_rate)
        
        drop_rate = 0.1

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.token_mixing(x))
        x = x + self.drop_path(self.channel_mixing(x))

        return x



class StripMLPNet(nn.Module):
    """
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        layers (tuple(int)): Depth of each Swin Transformer layer.
        drop_rate (float): Dropout rate. Default: 0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=80, layers=[2, 8, 14, 2], drop_rate=0.5,
                 norm_layer=nn.BatchNorm2d, alpha=3, use_dropout=False, patch_norm=True, **kwargs):
        super(StripMLPNet, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(layers)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_norm = patch_norm
        self.drop_rate = drop_rate

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size, bias=False)
        
        patches_resolution = [img_size // patch_size, img_size // patch_size]

        self.patches_resolution = patches_resolution

        self.avgpool = nn.AvgPool2d(2,2)

        self.blocks1 = nn.ModuleList()
        for i in range(layers[0]):
            basic = BasicBlock(embed_dim, self.patches_resolution[0], self.patches_resolution[1], alpha, use_dropout=use_dropout, drop_rate=drop_rate)
            self.blocks1.append(basic)

        self.blocks2 = nn.ModuleList()
        for i in range(layers[1]):
            basic = BasicBlock(embed_dim*2, int(self.patches_resolution[0]/2), int(self.patches_resolution[1]/2), alpha, use_dropout=use_dropout, drop_rate=drop_rate)
            self.blocks2.append(basic)
        
        self.blocks3 = nn.ModuleList()
        for i in range(layers[2]):
            basic = BasicBlock(embed_dim*4, int(self.patches_resolution[0]/4), int(self.patches_resolution[1]/4), alpha, use_dropout=use_dropout, drop_rate=drop_rate)
            self.blocks3.append(basic)

        self.blocks4 = nn.ModuleList()
        for i in range(layers[3]):
            basic = BasicBlock(embed_dim*8, int(self.patches_resolution[0]/8), int(self.patches_resolution[1]/8), alpha, use_dropout=use_dropout, drop_rate=drop_rate)
            self.blocks4.append(basic)

        self.merging1 = nn.Conv2d(embed_dim, embed_dim*2, 2, 2, bias=False)
        self.merging2 = nn.Conv2d(embed_dim*2, embed_dim*4, 2, 2, bias=False)
        self.merging3 = nn.Conv2d(embed_dim*4, embed_dim*8, 2, 2, bias=False)

        self.conv_s1_28 = nn.Conv2d(embed_dim*2, embed_dim*4, (2,2), 2, 0, groups=embed_dim*2, bias=False)
        self.conv_s1_14 = nn.Conv2d(embed_dim*4, embed_dim*8, (2,2), 2, 0, groups=embed_dim*4, bias=False)
        self.conv_s2_14 = nn.Conv2d(embed_dim*4, embed_dim*8, (2,2), 2, 0, groups=embed_dim*4, bias=False)

        self.head = nn.Linear(int(self.num_features), num_classes)
        
        self.norm = nn.BatchNorm2d(self.num_features)

    def forward_features(self, x):
        x = self.patch_embed(x)

        x = self.blocks(self.blocks1, x)

        x = self.merging1(x)

        x_s1_14 = self.conv_s1_28(x)
        x_s1_7 = self.conv_s1_14(x_s1_14)

        x = self.blocks(self.blocks2, x)

        x = self.merging2(x)

        x_s2_7 = self.conv_s2_14(x)

        x = self.blocks(self.blocks3, x + x_s1_14)
        
        x = self.merging3(x)

        x = self.blocks(self.blocks4, x + x_s1_7 + x_s2_7)

        x = self.norm(x)  # N C H W

        x = x.mean(dim=[2,3]).flatten(1)

        x = torch.flatten(x, 1)

        return x

    def blocks(self, blocks, x):
        for b in blocks:
            x = b(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

if __name__ == "__main__":
    import os
    data = torch.rand((1, 3, 224, 224))

    num_classes = 1000
    # smlp = StripMLPNet(in_chans=3, embed_dim=80, num_classes=num_classes, layers=[2, 2, 6, 2])      # Light Tiny
    # smlp = StripMLPNet(in_chans=3, embed_dim=80, num_classes=num_classes, layers=[2, 2, 12, 2])   # Tiny
    # smlp = StripMLPNet(in_chans=3, embed_dim=96, num_classes=num_classes, layers=[2, 2, 18, 2])   # Small
    smlp = StripMLPNet(in_chans=3, embed_dim=112, num_classes=num_classes, layers=[2, 2, 18, 2])  # Base

    out = smlp(data)

    save_state = {'model': smlp.state_dict()}
    # torch.save(save_state, "model.pth")

    from ptflops import get_model_complexity_info
    ops, params = get_model_complexity_info(smlp, (3, 224, 224), as_strings=True,
    print_per_layer_stat=True, verbose=True)
    print("The model paramater:", params)
    print("The model flops:", ops)
    print("Get output successed!...")