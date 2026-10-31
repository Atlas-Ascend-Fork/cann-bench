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
"""Yuv420ToRgb Golden 实现（numpy，逐字节复刻 OpenCV 标量参考算法）。

标杆：OpenCV cvtColor(src, COLOR_YUV2RGB_I420) 的算法
（modules/imgproc/src/color_yuv.simd.hpp 的 cvtYuv42xxp2RGB8，BT.601 limited range，
常数 ITUR_BT_601_*；已与 cv2 4.10/5.0 在多尺寸下逐字节交叉验证一致）。

语义：
    输入 x: [3H/2, W] uint8 紧凑缓冲（I420/YU12 平面排布）
        Y 平面: 字节偏移 [0, H*W)
        U 平面: 字节偏移 [H*W, H*W + H*W/4)
        V 平面: 字节偏移 [H*W + H*W/4, 3*H*W/2)
    输出 y: [H, W, 3] uint8 RGB 交错

统一接口：
    golden(x, **attrs) -> np.ndarray
    physical_shapes(dims, attrs) -> dict[str, tuple]    # 逻辑 H/W -> 输入物理 shape
    output_shape(shape, attrs) -> tuple                 # 输入物理 shape -> 输出 shape
    gen_tiling(shape, attrs, dtype_name) -> bytes
"""

from __future__ import annotations

import struct

import numpy as np

# OpenCV color_yuv.simd.hpp 的 BT.601 定点系数（ITUR_BT_601_*，<<20 定标）
# 注意 CUG = -409993（-0.391*2^20 的精确取整；旧版本曾误用 -408994，
# 4.10/4.12/5.0 均为 -409993，已与 cv2 逐字节交叉验证）
_SHIFT = 20
_HALF = 1 << (_SHIFT - 1)  # 2^19 舍入常数
_CY = 1220542  # 1.164    Y 项
_CVR = 1673527  # 1.596   V -> R
_CVG = -852492  # -0.813  V -> G
_CUG = -409993  # -0.391  U -> G
_CUB = 2116026  # 2.018   U -> B

_DTYPE_MAP = {
    "uint8": (np.uint8, 0),  # (numpy dtype, tiling dtype id)
}


def _plane_views(x: np.ndarray):
    """按平坦字节偏移切出 Y/U/V 三个平面视图。

    注意：U/V 平面边界是字节偏移，不是行边界（H%4!=0 时 U/V 起始
    不落在整行上），必须从扁平缓冲上切。
    """
    ph, w = x.shape
    if ph % 3 != 0 or w % 2 != 0:
        raise ValueError(f"input shape {x.shape} not valid I420 (3H/2 x W, W even)")
    h = ph * 2 // 3
    flat = x.reshape(-1)
    hw = h * w
    y = flat[:hw].reshape(h, w)
    u = flat[hw : hw + hw // 4].reshape(h // 2, w // 2)
    v = flat[hw + hw // 4 : hw * 3 // 2].reshape(h // 2, w // 2)
    return y, u, v


def golden(x: np.ndarray, **kw) -> np.ndarray:
    """I420 -> RGB（OpenCV 标量定点算法，逐字节一致）。

    Args:
        x: [3H/2, W] uint8，I420 平面排布（Y + U + V，平坦字节偏移）

    Returns:
        [H, W, 3] uint8，RGB 交错
    """
    if x.dtype != np.uint8:
        raise TypeError(f"expect uint8 input, got {x.dtype}")
    y, u, v = _plane_views(x)

    # 2x2 最近邻上采样（每色度样本覆盖对应 2x2 luma 块）
    u2 = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
    v2 = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

    uu = u2.astype(np.int32) - 128
    vv = v2.astype(np.int32) - 128
    ruv = _HALF + _CVR * vv
    guv = _HALF + _CVG * vv + _CUG * uu
    buv = _HALF + _CUB * uu
    y00 = np.maximum(y.astype(np.int32) - 16, 0) * _CY

    # 算术右移（numpy int32 >> 与 C++ 有符号 >> 一致），再饱和到 0..255
    r = np.clip((y00 + ruv) >> _SHIFT, 0, 255).astype(np.uint8)
    g = np.clip((y00 + guv) >> _SHIFT, 0, 255).astype(np.uint8)
    b = np.clip((y00 + buv) >> _SHIFT, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def physical_shapes(dims: dict, attrs: dict) -> dict:
    """全局命名维度值 -> 输入物理 shape：H/W 是输出图像维度，输入是 [3H/2, W]。"""
    h, w = dims["H"], dims["W"]
    return {"x": (h * 3 // 2, w)}


def output_shape(shape: tuple, attrs: dict) -> tuple:
    """输入物理 shape [3H/2, W] -> 输出 shape [H, W, 3]。"""
    ph, w = shape[0], shape[1]
    h = ph * 2 // 3
    return (h, w, 3)


def gen_tiling(shape: tuple, attrs: dict, dtype_name: str) -> bytes:
    """生成 TilingData 二进制（5 x uint32 = 20 bytes）。

    Layout 与 desc.md 的 Yuv420ToRgbTilingData 一致。
    shape 是输入物理 shape [3H/2, W]。
    """
    ph, w = shape[0], shape[1]
    h = ph * 2 // 3
    dtype_id = _DTYPE_MAP[dtype_name][1]
    return struct.pack("<IIIII", h, w, h * w, h * w // 4, dtype_id)


def yuv420_to_rgb(x, **kw):
    """schema 同名入口别名（与 golden() 等价，满足 tasks 交付契约）。"""
    return golden(x, **kw)
