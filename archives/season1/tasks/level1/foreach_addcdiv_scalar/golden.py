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
from typing import List

"""
ForeachAddcdivScalar 算子 Torch Golden 参考实现

对多个张量进行逐元素加、乘、除操作
公式：y_i = x1_i + (x2_i / x3_i) * scalar
"""
def foreach_addcdiv_scalar(
    x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor], scalar: float
) -> List[torch.Tensor]:
    """
    对多个张量进行逐元素加、乘、除操作

    公式：y_i = x1_i + (x2_i / x3_i) * scalar

    Args:
        x1: 第 1 个输入张量列表 (TensorList)
        x2: 第 2 个输入张量列表 (TensorList)
        x3: 第 3 个输入张量列表 (TensorList)
        scalar: 缩放因子

    Returns:
        输出张量列表
    """
    # 检测输入 dtype
    input_dtype = x1[0].dtype if x1 else torch.float32

    # FP16/BF16 输入需要升到 FP32 计算以保证精度
    # FP32/FP64 输入保持原样计算
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    # 转换到计算精度
    x1_compute = [t.to(compute_dtype) for t in x1]
    x2_compute = [t.to(compute_dtype) for t in x2]
    x3_compute = [t.to(compute_dtype) for t in x3]

    # 计算
    y = [x1_i + (x2_i / x3_i) * scalar for x1_i, x2_i, x3_i in zip(x1_compute, x2_compute, x3_compute)]

    # 转回原始 dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        return [t.to(input_dtype) for t in y]
    return y
