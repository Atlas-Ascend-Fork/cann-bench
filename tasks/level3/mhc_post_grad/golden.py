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
from typing import Optional


def get_input(grad_output: torch.Tensor, x: torch.Tensor,
              h_res: torch.Tensor, h_out: torch.Tensor,
              h_post: torch.Tensor, **kwargs):
    """
    预处理输入，确保满足算子的语义约束：
    - h_res: 使用 Sinkhorn 算法生成双随机矩阵（行和=列和=1）
    """
    h_res = _generate_dsmat_sinkhorn(h_res)
    return (grad_output, x, h_res, h_out, h_post)


def _generate_dsmat_sinkhorn(orig_tsr: torch.Tensor,
                              max_iter: int = 100,
                              tol: float = 1e-8):
    """使用 Sinkhorn 算法生成双随机矩阵（行和=列和=1）"""
    shape_list = list(orig_tsr.shape)
    orig_device = orig_tsr.device
    orig_dtype = orig_tsr.dtype

    # 固定种子：确保精度阶段与性能阶段生成相同双随机矩阵，跨 eval 可复现
    # （CPU generator 生成后再 .to(device)，避免非 CPU 设备不支持 Generator）
    g = torch.Generator().manual_seed(0)
    random_tensor = torch.rand(shape_list, dtype=torch.float32, generator=g).to(orig_device) + 1e-10

    for _ in range(max_iter):
        row_sum = random_tensor.sum(dim=-1, keepdim=True)
        random_tensor = random_tensor / (row_sum + 1e-10)
        col_sum = random_tensor.sum(dim=-2, keepdim=True)
        random_tensor = random_tensor / (col_sum + 1e-10)
        row_check = torch.abs(random_tensor.sum(dim=-1) - 1).max()
        col_check = torch.abs(random_tensor.sum(dim=-2) - 1).max()
        if row_check < tol and col_check < tol:
            break

    return random_tensor.to(orig_dtype).to(orig_device)


def mhc_post_grad(
        grad_output: torch.Tensor, x: torch.Tensor,
        h_res: torch.Tensor, h_out: torch.Tensor,
        h_post: torch.Tensor):
    """
    Manifold Constrained Hyper Connection Post 反向梯度算子（小算子拼接实现）。
    使用 float64 高精度计算，避免 float32 大 reduction 数值相消导致的精度抖动。

    前向: y[i,j] = h_post[i] * h_out[j] + sum_k h_res[k,i] * x[k,j]

    反向梯度推导:
        grad_h_post[i] = sum_j dy[i,j] * h_out[j]      → elementwise mul + sum
        grad_h_out[j]  = sum_i dy[i,j] * h_post[i]     → elementwise mul + sum
        grad_h_res     = x @ dy^T                      → matmul
        grad_x         = h_res @ dy                    → matmul

    参数:
        grad_output: [..., S, n, D] float16/bfloat16 — 上游梯度
        x: [..., S, n, D] float16/bfloat16 — 前向输入
        h_res: [..., S, n, n] float32 — 双随机矩阵
        h_out: [..., S, D] float16/bfloat16 — 前向输出权重
        h_post: [..., S, n] float32 — 前向后处理权重

    返回:
        (grad_x, grad_h_res, grad_h_out, grad_h_post)
    """
    orig_dtype = grad_output.dtype
    orig_h_res_dtype = h_res.dtype

    # 使用 float64 高精度计算 golden
    dy = grad_output.double()
    x_fp64 = x.double()
    h_res_fp64 = h_res.double()
    h_out_fp64 = h_out.double()
    h_post_fp64 = h_post.double()

    # grad_h_post[i] = sum_j dy[i,j] * h_out[j]
    grad_h_post = (dy * h_out_fp64.unsqueeze(-2)).sum(dim=-1)

    # grad_h_out[j] = sum_i dy[i,j] * h_post[i]
    grad_h_out = (dy * h_post_fp64.unsqueeze(-1)).sum(dim=-2)

    # grad_h_res[k,i] = sum_j x[k,j] * dy[i,j] = x @ dy^T
    grad_h_res = torch.matmul(x_fp64, dy.transpose(-2, -1))

    # grad_x[k,j] = sum_i h_res[k,i] * dy[i,j] = h_res @ dy
    grad_x = torch.matmul(h_res_fp64, dy)

    return (grad_x.to(orig_dtype), grad_h_res.to(orig_h_res_dtype),
            grad_h_out.to(orig_dtype), grad_h_post.to(orig_h_res_dtype))