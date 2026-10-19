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
AdaptiveAvgPool3D算子Torch Golden参考实现

完成输入张量的3D自适应平均池化计算
公式: y = adaptive_avg_pool3d(x, output_size)
"""
def adaptive_avg_pool_3d(
    x: torch.Tensor, output_size: tuple[int, int, int]
) -> torch.Tensor:
    """
    完成输入张量的3D自适应平均池化计算

    公式: y = adaptive_avg_pool3d(x, output_size)

    Args:
        x: 输入张量，shape 为 [N, C, D, H, W]
        output_size: 输出尺寸，格式为 (output_d, output_h, output_w)

    Returns:
        输出张量，池化结果
    """

    y = torch.nn.functional.adaptive_avg_pool3d(x, output_size)
    return y
