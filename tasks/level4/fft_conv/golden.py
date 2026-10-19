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
FftConv 算子 Torch Golden 参考实现

Hyena 长卷积：y = irfft(rfft(u, 2L) ⊙ rfft(k, 2L))[..., :L] + u ⊙ bias
FFT 长度取 2L 零填充，使循环卷积等价于因果线性卷积（无混叠）。
plain golden 以 fp32 计算（bench 语义）；fft_conv_oracle 跟随输入精度
（golden_precision=fp64_cpu 下 rfft/irfft 以 fp64/complex128 计算，即为 fp64 真值）。
"""


def _fft_conv_core(u, k_filter, bias, compute_dtype):
    """核心计算：以 compute_dtype 精度执行 rfft → 频域逐点乘 → irfft → 残差门控。"""
    L = u.shape[-1]
    n = 2 * L  # 零填充到 2L，避免循环混叠

    u_f = u.to(compute_dtype)
    k_f = k_filter.to(compute_dtype)

    u_freq = torch.fft.rfft(u_f, n=n)                     # [B, D, L+1] complex
    k_freq = torch.fft.rfft(k_f, n=n)                     # [D, L+1] complex
    y = torch.fft.irfft(u_freq * k_freq, n=n)[..., :L]    # 频域逐点复乘 + iFFT，截取前 L 点

    y = y + u_f * bias.to(compute_dtype).view(1, -1, 1)   # 残差门控
    return y


def fft_conv(
    u: torch.Tensor,
    k_filter: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Hyena 长卷积 golden reference（plain golden = bench：fp32 计算）

    Args:
        u: [B, D, L] 输入序列（通道优先布局）
        k_filter: [D, L] 时域全局滤波器（每通道一条长度 L 的隐式卷积核）
        bias: [D] 残差门控系数（逐通道）

    Returns:
        y: [B, D, L] 因果全局卷积输出，dtype 与 u 一致（float32）
    """
    y = _fft_conv_core(u, k_filter, bias, torch.float32)
    return y.to(u.dtype)


def fft_conv_oracle(
    u: torch.Tensor,
    k_filter: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _fft_conv_core(u, k_filter, bias, u.dtype)
