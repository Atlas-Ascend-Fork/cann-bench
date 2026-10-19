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
Mish算子Torch Golden参考实现

自正则化的非单调神经网络激活函数
公式: y = x * tanh(softplus(x))
"""
def mish(
    x: torch.Tensor
) -> torch.Tensor:
    """
    自正则化的非单调神经网络激活函数

    公式: y = x * tanh(softplus(x))

    Args:
        x: 输入张量

    Returns:
        输出张量，Mish激活结果
    """
    return torch.nn.functional.mish(x)
