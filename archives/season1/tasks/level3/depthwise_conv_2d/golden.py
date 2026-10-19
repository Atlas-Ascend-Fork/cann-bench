#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch

"""
DepthwiseConv2D算子Torch Golden参考实现

二维深度卷积运算（每输入通道独立卷积，groups = C_in = C_out）
y[n,c,h,w] = bias[c] + Σ_{kh,kw} x[n,c,h·s+kh·d-p,w·s+kw·d-p] · weight[c,kh,kw]
"""
def depthwise_conv_2d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, kernelSize: list, stride: list, padding: list, dilation: list, groups: int
) -> torch.Tensor:
    """
    二维深度卷积运算

    每个输入通道独立卷积，无跨通道求和（depthwise 约束：groups = C_in = C_out）。

    Args:
        x: 输入特征图，shape [N, C, H, W]
        weight: 卷积核，shape [C, K_h, K_w]（每通道一个独立 K_h×K_w 核）
        bias: 偏置，shape [C]
        kernelSize: 卷积核大小 [K_h, K_w]
        stride: 步长 [s_h, s_w]
        padding: 填充 [p_h, p_w]
        dilation: 膨胀率 [d_h, d_w]
        groups: 分组数，必须等于 C

    Returns:
        输出特征图，shape [N, C, H_out, W_out]
    """

    stride_val = (stride[0], stride[1])
    padding_val = (padding[0], padding[1])
    dilation_val = (dilation[0], dilation[1])

    # spec 的 weight shape 是 [C, K_h, K_w]；torch.conv2d 要 [C_out, C_in/groups, K_h, K_w]，
    # depthwise 下 C_in/groups = 1，unsqueeze 这个冗余维供 PyTorch API 用。
    weight = weight.unsqueeze(1)

    y = torch.nn.functional.conv2d(x, weight, bias, stride=stride_val, padding=padding_val, dilation=dilation_val, groups=groups)
    return y
