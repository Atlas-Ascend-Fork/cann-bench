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
ForeachNorm 算子 Torch Golden 参考实现

对输入张量列表的每个张量进行范数运算
公式：y = (sum |x_i|^p)^(1/p)
"""
def foreach_norm(
    x: List[torch.Tensor], scalar: float
) -> List[torch.Tensor]:
    """
    对输入张量列表的每个张量进行范数运算

    公式：y = (sum |x_i|^p)^(1/p)

    Args:
        x: 输入张量列表 (TensorList)
        scalar: 范数阶数

    Returns:
        输出张量列表，每个张量的范数结果
    """
    # 检测输入 dtype
    input_dtype = x[0].dtype if x else torch.float32

    # FP16/BF16 输入需要升到 FP32 计算以保证精度
    # FP32/FP64 输入保持原样计算
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    # 转换到计算精度
    x_compute = [t.to(compute_dtype) for t in x]

    y = [torch.norm(tensor, p=scalar) for tensor in x_compute]

    # 转回原始 dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        return [t.to(input_dtype) for t in y]
    return y
