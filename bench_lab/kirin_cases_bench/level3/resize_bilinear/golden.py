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
"""ResizeBilinear Golden 实现（numpy，逐字节复刻 OpenCV cv2.resize INTER_LINEAR 8U 单通道）。

标杆：cv2.resize(x, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
（modules/imgproc/src/resize.cpp 的 hal::resize CV_8U 路径，4.x/5.x 同码，十年未变；
已与本地 cv2 5.0.0 二进制在数百组随机尺寸下逐字节交叉验证一致）。

语义：
    输入 x: [H, W] uint8（单通道 2 维）
    输出 y: [out_h, out_w] uint8

分派逻辑（必须完全复刻，两条路径舍入规则不同）：
    1. 恒等（out == in）→ 直接拷贝
    2. 精确 2x 双维下采样（inH == 2*outH 且 inW == 2*outW）→ 2x2 盒式平均
       （is_area_fast && iscale==2x2 时 INTER_LINEAR 被提升为 INTER_AREA 的
       resizeAreaFast 8U 快速路径）：
           y = (a + b + c + d + 2) >> 2        （round-half-up）
    3. 其他 → 通用定点双线性（半像素中心 + 11bit 定点系数 + 两段式融合舍入）

通用路径算法（与 resize.cpp 逐句对应）：
    坐标（水平，垂直同）：scale = in/out（double）
        fx = float32((d + 0.5)*scale - 0.5)     # double 算完再转 f32
        sx = floor(fx);  fx -= sx              # f32 运算，fx ∈ [0, 1)
    边缘：sx < 0 → fx=0, sx=0（左缘 replicate）
          sx >= in-1 → fx=0, sx=in-1（右缘 replicate；该像素起为"单抽头"后缀）
    权重：a0 = rint((1.f - fx) * 2048)（f32 乘，round-half-even）
          a1 = rint(fx * 2048)
    水平（int32 精确，无舍入）：
        H[dx] = S[sx]*a0 + S[sx+1]*a1         # 单抽头列等价于 a=(2048, 0)
    垂直行：r0 = clip(sy, 0, inH-1)，r1 = clip(sy+1, 0, inH-1)（sy 本身未裁剪）
    融合舍入（>>4 先截断低 4 位，是 INTER_LINEAR 与 INTER_LINEAR_EXACT 差异的根源）：
        y = ((b0*(H0 >> 4)) >> 16 + (b1*(H1 >> 4)) >> 16 + 2) >> 2

统一接口：
    golden(x, out_h=, out_w=) -> np.ndarray
    output_shape(shape, attrs) -> tuple                 # 输入物理 shape + attrs -> 输出 shape
    gen_tiling(shape, attrs, dtype_name) -> bytes
"""

from __future__ import annotations

import struct

import numpy as np

# resize.cpp: INTER_RESIZE_COEF_BITS = 11
_COEF_SCALE = 2048

_DTYPE_MAP = {
    "uint8": (np.uint8, 0),  # (numpy dtype, tiling dtype id)
}


def _axis_tab(in_n: int, out_n: int, clip_edges: bool):
    """单轴坐标表：OpenCV resize.cpp 半像素中心 + 11bit 定点权重。

    Args:
        clip_edges: True = 水平轴（init 时左缘 sx<0 → fx=0,sx=0；右缘 sx>=in-1
            → fx=0,sx=in-1，边缘权重收敛为 (2048,0)）；
            False = 垂直轴（init 完全不裁剪，sy 可为 -1，beta 用原始 fy ——
            行选择阶段才 clip 行号，权重拆分影响 >>4 截断，不能清零）。

    Returns:
        (s, a0, a1): 每个输出位置 d 的源索引与两个 int 权重（和为 2048）。
    """
    scale = in_n / out_n  # python float = double，与 C++ double 一致
    # (d+0.5)*scale - 0.5 在 double 里算，然后 (float) 截断到 f32（C++ 逐位一致）
    fx = np.array(
        [(d + 0.5) * scale - 0.5 for d in range(out_n)], dtype=np.float64
    ).astype(np.float32)
    s = np.floor(fx).astype(np.int64)
    fx = (fx - s.astype(np.float32)).astype(np.float32)  # f32 - f32 = f32

    if clip_edges:
        # 边缘 replicate（resize.cpp 水平 init: sx<0 → fx=0,sx=0；sx>=in-1 → fx=0,sx=in-1）
        left = s < 0
        right = s >= in_n - 1
        s = np.where(left, 0, s)
        fx = np.where(left, np.float32(0.0), fx)
        s = np.where(right, in_n - 1, s)
        fx = np.where(right, np.float32(0.0), fx)

    # saturate_cast<short>(cbuf*2048)：f32 乘法 + cvRound（round-half-even）
    a0 = np.rint((np.float32(1.0) - fx) * np.float32(_COEF_SCALE)).astype(np.int64)
    a1 = np.rint(fx * np.float32(_COEF_SCALE)).astype(np.int64)
    return s, a0, a1


def golden(x: np.ndarray, out_h: int = 0, out_w: int = 0, **kw) -> np.ndarray:
    """cv2.resize INTER_LINEAR 8U 单通道（逐字节一致）。

    Args:
        x: [H, W] uint8
        out_h, out_w: 输出高宽

    Returns:
        [out_h, out_w] uint8
    """
    if x.dtype != np.uint8:
        raise TypeError(f"expect uint8 input, got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(f"expect 2D [H, W] input, got shape {x.shape}")
    ih, iw = x.shape
    oh, ow = int(out_h), int(out_w)
    if oh <= 0 or ow <= 0:
        raise ValueError(f"invalid output size ({oh}, {ow})")

    # 1) 恒等 → 拷贝（cv::resize 的 dsize == ssize 分支）
    if oh == ih and ow == iw:
        return x.copy()

    # 2) 精确 2x 双维下采样 → resizeAreaFast 8U 盒式平均
    #    (INTER_LINEAR && is_area_fast && iscale_x==2 && iscale_y==2 → INTER_AREA)
    if ih == 2 * oh and iw == 2 * ow:
        s = x.astype(np.int32)
        y = (s[0::2, 0::2] + s[0::2, 1::2] + s[1::2, 0::2] + s[1::2, 1::2] + 2) >> 2
        return y.astype(np.uint8)

    # 3) 通用定点双线性（水平 init 裁剪边缘，垂直 init 不裁剪）
    sx, a0, a1 = _axis_tab(iw, ow, clip_edges=True)
    sy, b0, b1 = _axis_tab(ih, oh, clip_edges=False)

    # 水平第二采样点（右缘列 a1=0，clip 仅防越界，无数值影响）
    sx1 = np.minimum(sx + 1, iw - 1)
    # 垂直行（sy 未裁剪，可能为 -1；sy+1 可能为 ih）→ replicate
    r0 = np.clip(sy, 0, ih - 1)
    r1 = np.clip(sy + 1, 0, ih - 1)

    # 水平 pass（int32 精确；对 r0/r1 涉及的行 gather）
    s0 = x[r0].astype(np.int32)  # [oh, iw]
    s1 = x[r1].astype(np.int32)
    h0 = s0[:, sx] * a0 + s0[:, sx1] * a1  # [oh, ow]
    h1 = s1[:, sx] * a0 + s1[:, sx1] * a1

    # 垂直 pass（两段式融合舍入，先 >>4 截断）
    bb0 = b0.astype(np.int32)[:, None]
    bb1 = b1.astype(np.int32)[:, None]
    v = ((bb0 * (h0 >> 4)) >> 16) + ((bb1 * (h1 >> 4)) >> 16) + 2
    return (v >> 2).astype(np.uint8)


def output_shape(shape: tuple, attrs: dict) -> tuple:
    """输入物理 shape [H, W] + attrs -> 输出 shape [out_h, out_w]。"""
    return (int(attrs["out_h"]), int(attrs["out_w"]))


def gen_tiling(shape: tuple, attrs: dict, dtype_name: str) -> bytes:
    """生成 TilingData 二进制（5 x uint32 = 20 bytes）。

    Layout 与 desc.md 的 ResizeBilinearTilingData 一致。
    shape 是输入物理 shape [H, W]。
    """
    ih, iw = shape[0], shape[1]
    oh, ow = int(attrs["out_h"]), int(attrs["out_w"])
    dtype_id = _DTYPE_MAP[dtype_name][1]
    return struct.pack("<IIIII", ih, iw, oh, ow, dtype_id)


def resize_bilinear(x, out_h=0, out_w=0, **kw):
    """schema 同名入口别名（与 golden() 等价，满足 tasks 交付契约）。"""
    return golden(x, out_h=out_h, out_w=out_w, **kw)
