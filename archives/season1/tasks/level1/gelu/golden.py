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
Gelu 算子 Torch Golden 参考实现

高斯误差线性单元激活函数
公式：y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
"""
def gelu(
    x: torch.Tensor,
    approximate: str = "none"
) -> torch.Tensor:
    """
    高斯误差线性单元激活函数

    公式：y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x: 输入张量
        approximate: GELU 近似计算算法，可选值：'none'(精确计算) 或 'tanh'(tanh 近似)

    Returns:
        输出张量，GELU 激活结果
    """

    y = torch.nn.functional.gelu(x, approximate=approximate)
    return y
