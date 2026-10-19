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
MaskedScale算子Torch Golden参考实现

对输入张量进行掩码缩放，支持x和mask的不同数据类型组合
公式: y = x * mask * scale
"""
def masked_scale(
    x: torch.Tensor, mask: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """
    对输入张量进行掩码缩放，支持x和mask的不同数据类型组合
    
    公式: y = x * mask * scale
    
    Args:
        x: 输入张量
        mask: 掩码张量
        scale: 缩放因子
    
    Returns:
        掩码缩放结果
    """

    y = x * mask * scale
    return y.to(x.dtype)
