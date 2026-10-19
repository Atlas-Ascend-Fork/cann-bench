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
import torch.nn.functional as F
from typing import Optional


def aggregate_hidden(input: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    对 hidden 层的 token 进行一维分组因果卷积。

    数学公式:
        output[i,j] = mask[j,i] * sum(k=0..W-1) input[i-k,j] * weight[W-1-k]
        其中无效位置 padding 为 0，W 固定为 3。

    参数:
        input: [S, B, H] 待计算数据, bfloat16/float16
        weight: [W, H] 卷积权重, W=3, 数据类型与 input 一致
        mask: [B, S] 可选掩码, bool, 默认 None (等价全 True)

    返回:
        output: [S, B, H] 卷积输出, 数据类型与 input 一致
    """
    S, B, H = input.shape
    W = weight.shape[0]

    # 转 float64 提高精度
    input_fp64 = input.double()
    weight_fp64 = weight.double()

    # weight [W, H] -> [H, 1, W] for grouped Conv1d
    conv_weight = weight_fp64.t().unsqueeze(1)  # [H, 1, W]

    # input [S, B, H] -> [B, H, S]
    conv_input = input_fp64.permute(1, 2, 0)

    # causal padding: prepend W-1 zeros
    conv_input = torch.cat([
        torch.zeros((B, H, W - 1), device=input.device, dtype=torch.float64),
        conv_input
    ], dim=-1)  # [B, H, S + W - 1]

    # grouped 1D convolution
    conv_output = F.conv1d(conv_input, conv_weight, groups=H)  # [B, H, S]

    # [B, H, S] -> [S, B, H]
    output = conv_output.permute(2, 0, 1)

    # apply mask
    if mask is not None:
        # mask [B, S] -> [S, B]
        mask_sb = mask.t()
        output[~mask_sb] = 0

    return output.to(input.dtype)