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
import numpy as np
from typing import Optional

"""
MoeReRouting 算子 Torch/Numpy Golden 参考实现

MoE 网络中，将 token 按照专家顺序重新排列
公式：SrcOffset/DstOffset 双重求和计算位置映射
"""

def moe_re_routing(
    tokens: torch.Tensor,
    expert_token_num_per_rank: torch.Tensor,
    per_token_scales: Optional[torch.Tensor] = None,
    expert_token_num_type: int = 1,
    idx_type: int = 0
):
    """
    MoeReRouting 算子 Torch/Numpy Golden 参考实现

    MoE 网络中，将 token 按照专家顺序重新排列

    Args:
        tokens: 待重新排布的 token，shape (A, H)
        expert_token_num_per_rank: 每张卡上各个专家处理的 token 数，shape (N, E)
        per_token_scales: 每个 token 对应的 scale，shape (A)，可选
        expert_token_num_type: 输出 expert_token_num 的模式，0=cumsum, 1=count，当前只支持 1
        idx_type: 输出 permute_token_idx 的索引类型，0=gather, 1=scatter，当前只支持 0

    Returns:
        (permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num)
    """
    # 判断输入类型
    is_torch = isinstance(tokens, torch.Tensor)

    # 获取参数
    if is_torch:
        N, E = expert_token_num_per_rank.shape
        A, H = tokens.shape
        dtype = tokens.dtype
        device = tokens.device
        int_dtype = expert_token_num_per_rank.dtype
    else:
        N, E = expert_token_num_per_rank.shape
        A, H = tokens.shape
        dtype = tokens.dtype
        int_dtype = expert_token_num_per_rank.dtype

    # 确保总和匹配
    if is_torch:
        total_tokens = expert_token_num_per_rank.sum().item()
    else:
        total_tokens = expert_token_num_per_rank.sum()
    assert total_tokens == A, f"Sum of expert_token_num_per_rank ({total_tokens}) must equal A ({A})"

    # Vectorised offset + gather-index construction. The old form built src/dst
    # offsets and src->dst maps with N*E + A nested Python loops, each step doing a
    # `.item()` (device->host sync) — on NPU that serialised thousands of syncs and
    # timed out. Here the index math runs on CPU over the tiny (N,E) counts; the
    # heavy token gather below stays on the input device. Output is bit-identical.
    #   src order = (rank, expert) row-major; dst order = (expert, rank).
    if is_torch:
        c = expert_token_num_per_rank.to(torch.int64).cpu()
        cnt_src = c.reshape(-1)        # block sizes in src (rank, expert) order
        cnt_dst = c.t().reshape(-1)    # block sizes in dst (expert, rank) order
        _excl = lambda t: torch.cat([t.new_zeros(1), t.cumsum(0)[:-1]])
        src_start = _excl(cnt_src)
        dst_start = _excl(cnt_dst).reshape(E, N).t().reshape(-1)  # back to (rank, expert)
        block = torch.repeat_interleave(torch.arange(N * E), cnt_src)  # block id per src pos
        dst_pos = dst_start[block] + (torch.arange(A) - src_start[block])  # src -> dst
        permute_token_idx = torch.empty(A, dtype=torch.int32)
        permute_token_idx[dst_pos] = torch.arange(A, dtype=torch.int32)   # invert -> gather idx
        permute_token_idx = permute_token_idx.to(device)
    else:
        c = expert_token_num_per_rank.astype(np.int64)
        cnt_src = c.reshape(-1)
        cnt_dst = c.T.reshape(-1)
        _excl = lambda a: np.concatenate([[0], np.cumsum(a)[:-1]])
        src_start = _excl(cnt_src)
        dst_start = _excl(cnt_dst).reshape(E, N).T.reshape(-1)
        block = np.repeat(np.arange(N * E), cnt_src)
        dst_pos = dst_start[block] + (np.arange(A) - src_start[block])
        permute_token_idx = np.empty(A, dtype=np.int32)
        permute_token_idx[dst_pos] = np.arange(A, dtype=np.int32)

    # 重排 tokens
    if is_torch:
        permute_tokens = tokens[permute_token_idx]
    else:
        permute_tokens = tokens[permute_token_idx]

    # 重排 per_token_scales（如果存在）
    if per_token_scales is not None:
        if is_torch:
            permute_per_token_scales = per_token_scales[permute_token_idx]
        else:
            permute_per_token_scales = per_token_scales[permute_token_idx]
    else:
        if is_torch:
            permute_per_token_scales = torch.zeros(A, dtype=torch.float32, device=device)
        else:
            permute_per_token_scales = np.zeros(A, dtype=np.float32)

    # 计算 expert_token_num (count 模式)
    if expert_token_num_type == 1:
        if is_torch:
            expert_token_num = expert_token_num_per_rank.sum(dim=0)
        else:
            expert_token_num = expert_token_num_per_rank.sum(axis=0)
    else:
        # cumsum 模式（暂不支持）
        if is_torch:
            expert_token_num = torch.zeros(E, dtype=int_dtype, device=device)
        else:
            expert_token_num = np.zeros(E, dtype=int_dtype)

    return permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num


def get_input(
    tokens: torch.Tensor,
    expert_token_num_per_rank: torch.Tensor,
    per_token_scales: Optional[torch.Tensor] = None,
    expert_token_num_type: int = 1,
    idx_type: int = 0,
    **kwargs
):
    """
    输入数据预处理函数

    调整 expert_token_num_per_rank 使其总和等于 tokens 数量 (A)

    Args:
        tokens: 待重新排布的 token，shape (A, H)
        expert_token_num_per_rank: 每张卡上各个专家处理的 token 数，shape (N, E)
        per_token_scales: 每个 token 对应的 scale，shape (A)，可选

    Returns:
        处理后的输入数据列表 [tokens, expert_token_num_per_rank, per_token_scales]
    """
    A = tokens.shape[0]
    N, E = expert_token_num_per_rank.shape

    # 计算每个位置的基础值，确保总和等于 A
    total_cells = N * E
    base_value = A // total_cells
    remainder = A % total_cells

    # proto.yaml 声明 expert_token_num_per_rank 的元素**必须大于 0**。A < N*E 时
    # base_value 为 0，除最后一格外全是 0 —— 静默产出违反契约的输入，候选 kernel
    # 在"某张卡的某个专家分到 0 个 token"上的行为是未定义的，且失败会表现为莫名的
    # 精度不符而非配置错误。当前用例集最小 base_value = 4，不会走到这里；这是为
    # 后续新增小 A 用例设的护栏，宁可显式报错也不静默失效。
    if base_value < 1:
        raise ValueError(
            f"moe_re_routing get_input: tokens 数 A={A} 少于 expert_token_num_per_rank "
            f"的格子数 N*E={N}*{E}={total_cells}，无法让每格都 >0（proto 要求元素必须"
            f"大于 0）。请调整该用例的 shape 使 A >= N*E。"
        )

    # 生成新的 expert_token_num_per_rank
    if isinstance(expert_token_num_per_rank, torch.Tensor):
        new_expert_token_num = torch.full((N, E), base_value, dtype=expert_token_num_per_rank.dtype)
        new_expert_token_num[-1, -1] += remainder
    else:
        new_expert_token_num = np.full((N, E), base_value, dtype=expert_token_num_per_rank.dtype)
        new_expert_token_num[-1, -1] += remainder

    return [tokens, new_expert_token_num, per_token_scales]