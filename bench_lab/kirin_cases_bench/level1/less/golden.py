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
"""Less Golden 实现（numpy 逐元素比较）。

逐元素小于比较：y = (x1 < x2)，输出 bool（uint8_t 0x00/0x01）。
Kernel 将输入展平为 1D 处理，golden 同样展平比较后还原 shape。
支持 dtype: float32, float16, int32, int16, int8, uint16, uint32, uint8。
统一接口：
    golden(x1, x2, **kw) -> np.ndarray (uint8)
    gen_tiling(shape, attrs, dtype_name) -> bytes
    output_shape(shape, attrs) -> tuple
"""

from __future__ import annotations

import struct

import numpy as np


def golden(x1: np.ndarray, x2: np.ndarray, **kw) -> np.ndarray:
    """逐元素小于比较 golden。

    直接用 numpy `<` 计算，IEEE754 语义与 ONNX Less 一致：
      NaN < x  -> False
      x < NaN  -> False
      +Inf < +Inf -> False, -Inf < -Inf -> False
      +0 < -0  -> False（IEEE754: +0 == -0）
    对 int/uint 类型无精度损失（不做 float32 中转）。

    Args:
        x1, x2: 同 shape 的 tensor，dtype ∈ {float32, float16, int32, int16,
                                       int8, uint16, uint32, uint8}

    Returns:
        uint8 tensor（同 shape），1 表示 x1 < x2 为真，0 否则
    """
    assert x1.shape == x2.shape, f"shape mismatch: {x1.shape} vs {x2.shape}"
    return (x1 < x2).astype(np.uint8)


def gen_tiling(shape, attrs, dtype_name: str) -> bytes:
    """生成 TilingData 二进制（1 × uint32 = 4 bytes）。

    Layout 与 less_kernel.cpp 的 LessTilingData 一致。
    """
    totalLen = 1
    for d in shape:
        totalLen *= d
    return struct.pack("<I", totalLen)


def output_shape(shape, attrs):
    """Less 输出 shape 与输入一致。"""
    return tuple(shape)


def less(x1, x2, **kw):
    """schema 同名入口别名（与 golden() 等价，满足 tasks 交付契约）。"""
    return golden(x1, x2, **kw)
