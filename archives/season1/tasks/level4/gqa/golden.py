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
GQA算子Torch Golden参考实现

分组查询注意力 (Grouped Query Attention)，多个 query head 共享一组 KV head
公式: 扩展 KV heads 匹配 Q heads，y = softmax(Q @ K^T * scaleValue) @ V
"""


def gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaleValue: float = -1.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    分组查询注意力 (Grouped Query Attention)

    Args:
        query: 查询张量 [B, S, N_q, D]（已分头）
        key: 键张量 [B, S_kv, N_kv, D]（已分头）
        value: 值张量 [B, S_kv, N_kv, D]（已分头）
        scaleValue: 缩放因子，<=0 时自动使用 1/sqrt(D)
        is_causal: 是否启用因果掩码（右下角对齐），True 时 scores[..., i, j] 满足
            j > i + (S_kv - S) 的位置在 softmax 前置为 -inf。要求 S <= S_kv。

    Returns:
        输出张量 [B, S, N_q, D]
    """
    B, S, N_q, D = query.shape
    S_kv = key.shape[1]
    N_kv = key.shape[2]

    if scaleValue <= 0:
        scaleValue = 1.0 / (D ** 0.5)

    # GQA: G = N_q // N_kv 个 Q head 共享一组 KV head。不把 K/V 物化到 N_q 头
    # (fp64 oracle 下大 batch 用例会物化上百 GiB → OOM)，而是把 group 维折叠进
    # matmul 的 M 维: Q -> [B, N_kv, G*S, D]，K/V 保持 N_kv 头随 batched matmul 复用。
    # 数值上与展开完全等价，峰值内存由 scores 决定而非物化后的 K/V。
    G = N_q // N_kv
    q = query.reshape(B, S, N_kv, G, D).permute(0, 2, 3, 1, 4).reshape(B, N_kv, G * S, D)
    k = key.permute(0, 2, 1, 3)    # [B, N_kv, S_kv, D]
    v = value.permute(0, 2, 1, 3)  # [B, N_kv, S_kv, D]

    # 缩放点积注意力: scores [B, N_kv, G*S, S_kv]，还原 G/S 两维以便掩码与 softmax
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    scores = scores.reshape(B, N_kv, G, S, S_kv)
    if is_causal:
        i = torch.arange(S, device=scores.device).unsqueeze(-1)
        j = torch.arange(S_kv, device=scores.device).unsqueeze(0)
        causal_mask = j > (i + (S_kv - S))  # 右下角对齐：上三角置 -inf；[S, S_kv] 广播到各 group
        scores = scores.masked_fill(causal_mask, float('-inf'))
    # F217: 全 mask 行 (整行 = -inf) 在 softmax 时得 0/0 = NaN，对齐
    # sparse_flash_attention 加显式保护 → 全 mask 行权重置 0。
    scores_max = scores.max(dim=-1, keepdim=True).values
    all_masked = torch.isinf(scores_max) & (scores_max < 0)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    attn_weights = torch.where(all_masked, torch.zeros_like(attn_weights), attn_weights)
    attn_weights = attn_weights.reshape(B, N_kv, G * S, S_kv)
    attn_output = torch.matmul(attn_weights, v)  # [B, N_kv, G*S, D]

    # 转回 [B, S, N_q, D]（N_q 头序 = n_kv * G + g，与展开路径一致）
    # 末尾 reshape 在 permute 后的布局上仍是 view, 而输出契约要求 contiguous (issue #146)
    return attn_output.reshape(B, N_kv, G, S, D).permute(0, 3, 1, 2, 4).reshape(B, S, N_q, D).contiguous()
