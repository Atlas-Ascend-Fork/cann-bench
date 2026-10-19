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
TopK算子Torch Golden参考实现

返回k个最大或最小的元素及其索引
公式: y, idx = topk(x, k, dim)
"""
def top_k(
    x: torch.Tensor, k: int, dim: int, largest: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回k个最大或最小的元素及其索引
    
    公式: y, idx = topk(x, k, dim)
    
    Args:
        x: 输入张量
        k: 返回的topk数量 (取值范围: 1 <= k <= dim_size)
        dim: 排序维度 (取值范围: -ndim ~ ndim-1)
        largest: 是否返回最大值 (false时返回最小值)
    
    Returns:
        y, idx
    """

    values, indices = torch.topk(x, k=k, dim=dim, largest=largest)
    return values, indices
