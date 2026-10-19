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
from torch import Tensor


def aggregate_hidden_grad(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    AggregateHiddenGrad 算子的 Torch Golden 参考实现。

    对 hidden states 聚合操作（宽度为 3 的因果卷积）进行反向梯度计算。

    Args:
        grad_output: 反向梯度输入，形状为 [T, B, D]。
        input: 前向输入，形状为 [T, B, D]。
        weight: 卷积权重，形状为 [3, D]。
        mask: 可选的掩码张量，形状为 [B, T]，数据类型为 bool。

    Returns:
        grad_input: 输入梯度，形状为 [T, B, D]。
        grad_weight: 权重梯度，形状为 [3, D]。
    """
    dtype = grad_output.dtype
    grad_output = grad_output.to(torch.float32)
    input = input.to(torch.float32)
    weight = weight.to(torch.float32)

    if mask is not None:
        grad_output = grad_output.clone()
        grad_output[~mask.transpose(0, 1)] = 0

    grad_input0 = grad_output * weight[0].unsqueeze(0).unsqueeze(0)
    grad_input1 = grad_output * weight[1].unsqueeze(0).unsqueeze(0)
    grad_input2 = grad_output * weight[2].unsqueeze(0).unsqueeze(0)

    grad_input2[:-1, :, :] += grad_input1[1:, :, :]
    grad_input2[:-2, :, :] += grad_input0[2:, :, :]

    grad_weight = torch.stack(
        [
            (grad_output[2:, :, :] * input[:-2, :, :]).sum(dim=0).sum(dim=0),
            (grad_output[1:, :, :] * input[:-1, :, :]).sum(dim=0).sum(dim=0),
            (grad_output[:, :, :] * input[:, :, :]).sum(dim=0).sum(dim=0),
        ],
        dim=0,
    )

    grad_input2 = grad_input2.to(dtype)
    grad_weight = grad_weight.to(dtype)

    return grad_input2, grad_weight