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
import random
import numpy as np


def get_input(x: torch.Tensor, h_res: torch.Tensor,
              h_out: torch.Tensor, h_post: torch.Tensor, **kwargs):
    """
    预处理输入，确保满足算子的语义约束：
    - h_res: 使用 Sinkhorn 算法生成双随机矩阵（行和=列和=1）
    - h_post: 应用 sigmoid 归一化到 (0, 1) 范围
    """
    # h_res → doubly stochastic matrix via Sinkhorn
    h_res = _generate_dsmat_sinkhorn(h_res)

    # h_post → sigmoid normalization
    h_post = torch.sigmoid(h_post)

    return (x, h_res, h_out, h_post)


def _generate_dsmat_sinkhorn(orig_tsr: torch.Tensor,
                              max_iter: int = 100,
                              tol: float = 1e-8):
    """使用 Sinkhorn 算法生成双随机矩阵（行和=列和=1）"""
    shape_list = list(orig_tsr.shape)
    orig_device = orig_tsr.device
    orig_dtype = orig_tsr.dtype

    # 创建随机正矩阵
    # 固定种子：确保精度阶段与性能阶段生成相同双随机矩阵，跨 eval 可复现
    # （CPU generator 生成后再 .to(device)，避免非 CPU 设备不支持 Generator）
    g = torch.Generator().manual_seed(0)
    random_tensor = torch.rand(shape_list, dtype=torch.float32, generator=g).to(orig_device) + 1e-10

    # Sinkhorn 迭代归一化
    for _ in range(max_iter):
        # 行归一化
        row_sum = random_tensor.sum(dim=-1, keepdim=True)
        random_tensor = random_tensor / (row_sum + 1e-10)

        # 列归一化
        col_sum = random_tensor.sum(dim=-2, keepdim=True)
        random_tensor = random_tensor / (col_sum + 1e-10)

        # 检查收敛
        row_check = torch.abs(random_tensor.sum(dim=-1) - 1).max()
        col_check = torch.abs(random_tensor.sum(dim=-2) - 1).max()
        if row_check < tol and col_check < tol:
            break

    return random_tensor.to(orig_dtype).to(orig_device)


def manifold_constrained_hyper_connection_post(
        x: torch.Tensor, h_res: torch.Tensor,
        h_out: torch.Tensor, h_post: torch.Tensor):
    """
    Manifold Constrained Hyper Connection 后处理算子。

    公式:
        y = h_post.unsqueeze(-1) * h_out.unsqueeze(-2)
            + sum(h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=-3)

    参数:
        x: [..., S, n, D] float16/bfloat16
        h_res: [..., S, n, n] float32 — 双随机矩阵
        h_out: [..., S, D] float16/bfloat16, dtype 与 x 一致
        h_post: [..., S, n] float32 — 已 sigmoid 归一化

    返回:
        output: [..., S, n, D], dtype 与 x 一致
    """
    orig_dtype = x.dtype

    x_fp32 = x.float()
    h_out_fp32 = h_out.float()
    h_res_fp32 = h_res.float()
    h_post_fp32 = h_post.float()

    y = (h_post_fp32.unsqueeze(-1) * h_out_fp32.unsqueeze(-2) +
         torch.sum(h_res_fp32.unsqueeze(-1) * x_fp32.unsqueeze(-2), dim=-3))

    return y.to(orig_dtype)