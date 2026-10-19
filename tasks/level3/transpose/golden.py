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
Transpose算子Torch Golden参考实现

对tensor的任意维度进行调换
公式: y[i0,...,in-1] = x[i_perm[0],...,i_perm[n-1]]
"""
def transpose(
    x: torch.Tensor, perm: list
) -> torch.Tensor:
    """
    对tensor的任意维度进行调换
    
    公式: y[i0,...,in-1] = x[i_perm[0],...,i_perm[n-1]]
    
    Args:
        x: 输入张量
        perm: 维度置换顺序
    
    Returns:
        输出张量，转置后的结果
    """

    y = torch.permute(x, perm)
    return y
