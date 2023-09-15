# --------------------------------------------------------
# Copyright (c) 2023 CVIP of SUST
# Licensed under The MIT License [see LICENSE for details]
# Written by Guiping Cao
# --------------------------------------------------------

import torch

def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x


def shift_x(x):
    # shift every column: 每列 左移两个像素
    shift_x = torch.cat([x[:, :, 2:, :], x[:, :, :2, :]], dim=2)

    # shift every row: 每行 上移两个像素
    shift_x = torch.cat([shift_x[:, :, :, 2:], shift_x[:, :, :, :2]], dim=3)
    return shift_x

def restore_shift_x(x):
    # restore the row: 每行下移一个像素
    restore_x = torch.cat([x[:, :, :, -1:], x[:, :, :, :-1]], dim=3)

    # restore the column: 每列右移一个像素
    restore_x = torch.cat([restore_x[:, :, -1:, :], restore_x[:, :, :-1, :]], dim=2)
    return restore_x
