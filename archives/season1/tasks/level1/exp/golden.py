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
Exp算子Torch Golden参考实现

计算输入张量的指数函数
- base <= 0: y = exp(scale * x + shift)
- base > 0: y = exp((shift + scale * x) * ln(base))
"""
def exp(
    x: torch.Tensor, base: float = -1.0, scale: float = 1.0, shift: float = 0.0
) -> torch.Tensor:
    """
    计算输入张量的指数函数

    - base <= 0: y = exp(scale * x + shift)
    - base > 0: y = exp((shift + scale * x) * ln(base))

    Args:
        x: 输入张量
        base: 指数底数，base <= 0 表示使用自然底数 e
        scale: 输入缩放因子
        shift: 输入偏移量

    Returns:
        指数计算结果
    """
    # 检测输入 dtype
    input_dtype = x.dtype

    # FP16/BF16 输入需要升到 FP32 计算以保证精度
    # FP32/FP64 输入保持原样计算
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    # 转换到计算精度
    x_compute = x.to(compute_dtype)

    temp = scale * x_compute + shift
    if base > 0:
        temp = temp * torch.log(torch.tensor(base, dtype=temp.dtype, device=temp.device))
    y = torch.exp(temp)

    # 转回原始 dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        return y.to(input_dtype)
    return y