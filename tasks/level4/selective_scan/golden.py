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
SelectiveScan算子Torch Golden参考实现

Mamba S6 选择性状态空间扫描，融合离散化、时间维递归扫描、输出投影、
跳跃连接与 SiLU 门控。

公式: dA_t = exp(delta_t ⊗ A), dBu_t = delta_t ⊗ B_t ⊗ u_t
      h_t = dA_t ⊙ h_{t-1} + dBu_t     (h_0 = 0)
      y_t = C_t · h_t + D_skip ⊙ u_t
      y   = y ⊙ SiLU(z)
输入约定: delta > 0 且 A < 0 (由 value_range 保证)，使 dA ∈ (0, 1)、递归收缩稳定。
plain golden 内部计算全程 fp32（bench 语义），输出转回输入 dtype；
selective_scan_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""


def _selective_scan_core(u, delta, A, B_mat, C_mat, D_skip, z, compute_dtype):
    """核心计算：以 compute_dtype 精度执行离散化 + 递归扫描 + 门控。"""
    u_f = u.to(compute_dtype)
    delta_f = delta.to(compute_dtype)
    A_f = A.to(compute_dtype)
    B_f = B_mat.to(compute_dtype)
    C_f = C_mat.to(compute_dtype)
    D_f = D_skip.to(compute_dtype)
    z_f = z.to(compute_dtype)

    Bsz, L, D = u_f.shape
    N = A_f.shape[1]

    # 时间维递归扫描：逐步计算，避免物化 [B, L, D, N] 的 dA/dBu 大张量
    h = torch.zeros(Bsz, D, N, dtype=compute_dtype, device=u.device)
    ys = []
    for t in range(L):
        # 离散化: dA_t = exp(delta_t ⊗ A), dBu_t = delta_t ⊗ B_t ⊗ u_t
        dA_t = torch.exp(delta_f[:, t].unsqueeze(-1) * A_f)                          # [B, D, N]
        dBu_t = (delta_f[:, t] * u_f[:, t]).unsqueeze(-1) * B_f[:, t].unsqueeze(1)   # [B, D, N]
        # 递归: h_t = dA_t ⊙ h_{t-1} + dBu_t
        h = dA_t * h + dBu_t
        # 输出投影: y_t = C_t · h_t（状态维 N 缩约）
        ys.append(torch.einsum('bdn,bn->bd', h, C_f[:, t]))                          # [B, D]
    y = torch.stack(ys, dim=1)                                                       # [B, L, D]

    # 跳跃连接 + SiLU 门控
    y = y + D_f * u_f
    y = y * (z_f * torch.sigmoid(z_f))
    return y


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    D_skip: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """
    Mamba S6 选择性状态空间扫描（plain golden = bench：内部 fp32 计算）

    Args:
        u: [B, L, D] 输入序列特征
        delta: [B, L, D] 离散化步长 Δ（严格为正）
        A: [D, N] 连续域状态转移矩阵（严格为负）
        B_mat: [B, L, N] 输入投影矩阵（输入依赖）
        C_mat: [B, L, N] 输出投影矩阵（输入依赖）
        D_skip: [D] 跳跃连接缩放系数
        z: [B, L, D] SiLU 门控分支输入

    Returns:
        y: [B, L, D] 选择性扫描输出，dtype 与输入一致
    """
    y = _selective_scan_core(u, delta, A, B_mat, C_mat, D_skip, z, torch.float32)
    return y.to(u.dtype)


def selective_scan_oracle(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    D_skip: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _selective_scan_core(u, delta, A, B_mat, C_mat, D_skip, z, u.dtype)
