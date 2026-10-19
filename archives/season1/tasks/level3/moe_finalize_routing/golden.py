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
from copy import deepcopy

from typing import Optional

"""
MoeFinalizeRouting 算子 Torch Golden 参考实现

在 MoE 计算的最后，合并 MoE FFN 的输出结果
公式：out = skip1 + skip2 + Σ(scales * (expanded_permuted_rows + bias))
"""

def moe_finalize_routing(
    expanded_permuted_rows: torch.Tensor,
    expanded_src_to_dst_row: torch.Tensor,
    skip1: Optional[torch.Tensor] = None,
    skip2: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_for_source_row: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 0,
):
    """
    MoE Finalize Routing 算子 Torch Golden 参考实现

    在 MoE 计算的最后，合并 MoE FFN 的输出结果

    Args:
        expanded_permuted_rows: MoE FFN 输出，shape 为 (NUM_ROWS * K, H) 或 (E, C, H)
        expanded_src_to_dst_row: 行索引映射，shape 为 (NUM_ROWS * K)
        skip1: 共享专家1，shape 为 (NUM_ROWS, H)
        skip2: 共享专家2，shape 为 (NUM_ROWS, H)
        bias: 专家偏置，shape 为 (E, H)
        scales: 路由权重，shape 为 (NUM_ROWS, K)
        expert_for_source_row: 专家索引，shape 为 (NUM_ROWS, K)
        drop_pad_mode: 模式选择，取值范围 [0, 3]
            0: drop less, 按列排列
            1: drop pad, 按列排列
            2: drop less, 按行排列
            3: drop pad, 按行排列

    Returns:
        输出张量，shape 为 (NUM_ROWS, H)
    """
    # 确定输入类型（numpy 或 torch）
    is_torch = isinstance(expanded_permuted_rows, torch.Tensor)

    # 低精度类型升到 fp32，避免累积循环中的舍入误差与 fp64 参考值产生偏差
    original_dtype = None
    if is_torch:
        original_dtype = expanded_permuted_rows.dtype
        _low_prec = original_dtype in (torch.float16, torch.bfloat16)
        if _low_prec:
            expanded_permuted_rows = expanded_permuted_rows.float()
            if skip1 is not None:
                skip1 = skip1.float()
            if skip2 is not None:
                skip2 = skip2.float()
            if bias is not None:
                bias = bias.float()
            if scales is not None:
                scales = scales.float()

    # 确定 K 和 num_rows
    NK = expanded_src_to_dst_row.shape[0]
    K = scales.shape[1] if scales is not None else 1
    num_rows = NK // K
    H = expanded_permuted_rows.shape[-1]

    # 将 expanded_permuted_rows reshape 为 2D
    if is_torch:
        expanded_permuted_rows = expanded_permuted_rows.reshape(-1, H)
        dtype = expanded_permuted_rows.dtype
        device = expanded_permuted_rows.device
    else:
        expanded_permuted_rows = expanded_permuted_rows.reshape(-1, H)
        dtype = expanded_permuted_rows.dtype

    # 初始化输出：skip1 + skip2
    if (skip1 is not None) and (skip2 is not None):
        if is_torch:
            out = skip1.clone() + skip2
        else:
            out = skip1.copy() + skip2
    elif (skip2 is not None) and (skip1 is None):
        if is_torch:
            out = skip2.clone()
        else:
            out = deepcopy(skip2)
    elif (skip2 is None) and (skip1 is not None):
        if is_torch:
            out = skip1.clone()
        else:
            out = deepcopy(skip1)
    else:
        if is_torch:
            out = torch.zeros(num_rows, H, dtype=dtype, device=device)
        else:
            out = np.zeros([num_rows, H], dtype=dtype)

    # Vectorised over rows; only the small K (~topk) loop remains. The old form
    # looped num_rows*K times doing a per-step `.item()` (device->host sync) and
    # timed out on NPU. Keeping the k-loop preserves the per-row accumulation order,
    # so the fp32 result is bit-identical to the original; the heavy H-dim gather /
    # scaled-add run vectorised on the input device.
    rows = torch.arange(num_rows) if is_torch else np.arange(num_rows)
    num_experts = bias.shape[0] if bias is not None else 0
    for k in range(K):
        # row index into expanded_src_to_dst_row for this k (per drop_pad_mode layout)
        index_pos = (k * num_rows + rows) if drop_pad_mode in (0, 1) else (rows * K + k)
        value = expanded_src_to_dst_row[index_pos]  # (num_rows,)

        # value == -1 (drop pad) contributes a zero row; otherwise gather it.
        if is_torch:
            valid = value != -1
            dst_row = expanded_permuted_rows[value.clamp(min=0).long()].clone()
            dst_row[~valid] = 0
        else:
            valid = value != -1
            dst_row = expanded_permuted_rows[np.clip(value, 0, None).astype(np.int64)].copy()
            dst_row[~valid] = 0
        term = dst_row

        # F510: bound-check expert_id before indexing bias. The upstream
        # `moe_gating_top_k_softmax` golden uses num_expert (== E, out-of-range) as a
        # sentinel for finished tokens; such rows contribute the unbiased value.
        if bias is not None and expert_for_source_row is not None:
            expert_id = expert_for_source_row[:, k]
            valid_e = (expert_id >= 0) & (expert_id < num_experts)
            if is_torch:
                bias_row = bias[expert_id.clamp(0, num_experts - 1).long()].clone()
                bias_row[~valid_e] = 0
            else:
                bias_row = bias[np.clip(expert_id, 0, num_experts - 1).astype(np.int64)].copy()
                bias_row[~valid_e] = 0
            term = dst_row + bias_row

        # A dropped (-1) token must contribute NOTHING — not even bias. The real
        # MoeFinalizeRoutingV2 kernel skips the whole term for a -1 row; the previous
        # golden zeroed only dst_row but still added scale*bias here, so every
        # drop_pad case with an occasional -1 index diverged from the kernel (q7
        # probe: golden with this line == V2 kernel bit-exact, max_diff=0). valid is
        # (num_rows,), so this also zeros the bias part for dropped rows.
        term[~valid] = 0

        if scales is not None:
            out = out + scales[:, k][:, None] * term
        else:
            out = out + 1.0 * term

    if is_torch and original_dtype is not None and _low_prec:
        out = out.to(original_dtype)

    return out


def generate_moe_finalize_routing_inputs(
    expert_num=16,
    hidden_dim=512,
    topk=8,
    num_rows=1024,
    dtype="float16",
    use_skip2=True,
    use_bias=True,
    use_scales=True,
    drop_pad_mode=0,
    expert_capacity=None,
    seed=42
):
    """
    生成 MoeFinalizeRouting 算子的测试输入数据

    Args:
        expert_num: 专家数量 E
        hidden_dim: 隐藏层维度 H
        topk: 每个token选择的专家数 K
        num_rows: token 数量 NUM_ROWS
        dtype: 数据类型，支持 float16, float32, bfloat16
        use_skip2: 是否使用 skip2
        use_bias: 是否使用 bias
        use_scales: 是否使用 scales
        drop_pad_mode: 模式选择 [0, 3]
        expert_capacity: drop pad 模式下的专家容量 C
        seed: 随机种子

    Returns:
        包含所有输入参数的字典
    """
    # 设置随机种子
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 数据类型映射
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)
    np_dtype = {
        "float16": np.float16,
        "float32": np.float32,
        "bfloat16": np.float32  # numpy不支持bfloat16，用float32代替
    }.get(dtype, np.float16)

    # 根据 drop_pad_mode 生成不同 shape 的 expanded_permuted_rows
    if drop_pad_mode == 0 or drop_pad_mode == 2:
        # drop less 模式：2D shape
        expanded_permuted_rows = torch.randn(num_rows * topk, hidden_dim, dtype=torch_dtype)
        expanded_permuted_rows_np = np.random.randn(num_rows * topk, hidden_dim).astype(np_dtype)
        # 索引范围 [0, NUM_ROWS * K - 1]
        expanded_src_to_dst_row = torch.randint(0, num_rows * topk, (num_rows * topk,), dtype=torch.int32)
        expanded_src_to_dst_row_np = np.arange(num_rows * topk).astype(np.int32)
        np.random.shuffle(expanded_src_to_dst_row_np)
    else:
        # drop pad 模式：3D shape
        if expert_capacity is None:
            expert_capacity = num_rows // expert_num + 10
        expanded_permuted_rows = torch.randn(expert_num, expert_capacity, hidden_dim, dtype=torch_dtype)
        expanded_permuted_rows_np = np.random.randn(expert_num, expert_capacity, hidden_dim).astype(np_dtype)
        # 索引范围 [-1, E * C - 1]
        expanded_src_to_dst_row = torch.randint(-1, expert_num * expert_capacity - 1, (num_rows * topk,), dtype=torch.int32)
        expanded_src_to_dst_row_np = np.random.randint(-1, expert_num * expert_capacity - 1, num_rows * topk).astype(np.int32)

    # skip1
    skip1 = torch.randn(num_rows, hidden_dim, dtype=torch_dtype)
    skip1_np = np.random.randn(num_rows, hidden_dim).astype(np_dtype)

    # skip2
    if use_skip2:
        skip2 = torch.randn(num_rows, hidden_dim, dtype=torch_dtype)
        skip2_np = np.random.randn(num_rows, hidden_dim).astype(np_dtype)
    else:
        skip2 = None
        skip2_np = None

    # bias
    if use_bias:
        bias = torch.randn(expert_num, hidden_dim, dtype=torch_dtype)
        bias_np = np.random.randn(expert_num, hidden_dim).astype(np_dtype)
        # expert_for_source_row 必须存在
        expert_for_source_row = torch.randint(0, expert_num, (num_rows, topk), dtype=torch.int32)
        expert_for_source_row_np = np.random.randint(0, expert_num, size=(num_rows, topk)).astype(np.int32)
    else:
        bias = None
        bias_np = None
        expert_for_source_row = None
        expert_for_source_row_np = None

    # scales
    if use_scales:
        scales = torch.randn(num_rows, topk, dtype=torch_dtype)
        scales_np = np.random.randn(num_rows, topk).astype(np_dtype)
    else:
        scales = None
        scales_np = None

    return {
        "torch": {
            "expanded_permuted_rows": expanded_permuted_rows,
            "skip1": skip1,
            "skip2": skip2,
            "bias": bias,
            "scales": scales,
            "expanded_src_to_dst_row": expanded_src_to_dst_row,
            "expert_for_source_row": expert_for_source_row,
            "drop_pad_mode": drop_pad_mode
        },
        "numpy": {
            "expanded_permuted_rows": expanded_permuted_rows_np,
            "skip1": skip1_np,
            "skip2": skip2_np,
            "bias": bias_np,
            "scales": scales_np,
            "expanded_src_to_dst_row": expanded_src_to_dst_row_np,
            "expert_for_source_row": expert_for_source_row_np,
            "drop_pad_mode": drop_pad_mode
        }
    }


if __name__ == "__main__":
    # 测试示例
    inputs = generate_moe_finalize_routing_inputs(
        expert_num=16, hidden_dim=512, topk=8, num_rows=1024,
        dtype="float16", drop_pad_mode=0
    )

    # 使用 numpy 计算 golden
    golden_np = moe_finalize_routing(**inputs["numpy"])
    print(f"Golden output shape: {golden_np.shape}")
    print(f"Golden output dtype: {golden_np.dtype}")
    print(f"Golden output sample: {golden_np[0, :5]}")

    # 使用 torch 计算 golden
    golden_torch = moe_finalize_routing(**inputs["torch"])
    print(f"Torch golden output shape: {golden_torch.shape}")
    print(f"Torch golden output dtype: {golden_torch.dtype}")
    print(f"Torch golden output sample: {golden_torch[0, :5]}")

def get_input(
    expanded_permuted_rows: torch.Tensor,
    expanded_src_to_dst_row: torch.Tensor,
    skip1: Optional[torch.Tensor] = None,
    skip2: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_for_source_row: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 0,
    **kwargs,
):
    """把 expanded_src_to_dst_row 重建为满足单射契约的行映射。

    该输入是 MoeInitRouting 产出的"源行 -> 展开缓冲区行"映射：每个源行 (token, k)
    占据展开缓冲区中**互不相同**的一行。drop_less (mode 0/2) 下它是 [0, NK) 的完整
    置换；drop_pad (mode 1/3) 下是到 [0, E*C) 的单射，容量溢出的源行取 -1 表示丢弃。

    而单区间 value_range 表达不了"互异"，通用生成器按 randint 独立有放回采样，导致
    部分展开行被读多次、另一部分从未被读。golden 与真实 MoeFinalizeRoutingV2 都做
    gather，重复索引下数值仍然可比（现网用例即如此通过），但：
      1. 访存模式与真实场景不符——真实场景每行恰好读一次，重复采样把它变成随机重复
         读，L2 命中率虚高，perf 数据不具代表性；
      2. 任何采用 scatter 方向实现的候选（遍历展开行写回目的行，与 gather 等价当且
         仅当映射单射）都会与 golden 分叉，而契约本身是站在候选一边的。

    这里按 case 的实际形状推导目的空间大小 num_dst = expanded_permuted_rows 展平成
    (num_dst, H) 后的行数（drop_less 下等于 NK，drop_pad 下等于 E*C），为非 -1 的位置
    分配 [0, num_dst) 的互异值。**-1 的位置原样保留**，以维持 drop_pad 用例既有的
    丢弃覆盖与随种子可复现的行为。

    注：drop_pad 的完整契约还要求目的行落在该源行所属专家的容量块 [e*C, (e+1)*C) 内
    （e 取自 expert_for_source_row）。golden 不校验这一点，且按专家分配会让丢弃率降到
    近乎 0（当前用例 E*C 远大于 NK），反而削弱 -1 路径覆盖，故此处只做单射重建；如需
    专家对齐应由用例设计一并调整容量。

    kernel_eval 用输入名 + attrs 作为关键字调用本函数，并用返回值（按 golden 签名的
    Tensor 顺序）同时替换 golden 与候选的输入，故比较公平。

    Returns:
        [expanded_permuted_rows, expanded_src_to_dst_row, skip1, skip2, bias, scales,
         expert_for_source_row]，顺序与 moe_finalize_routing 签名一致。
    """
    unchanged = [expanded_permuted_rows, expanded_src_to_dst_row, skip1, skip2,
                 bias, scales, expert_for_source_row]
    esdr = expanded_src_to_dst_row
    if not isinstance(esdr, torch.Tensor) or esdr.numel() == 0:
        return unchanged

    H = int(expanded_permuted_rows.shape[-1])
    num_dst = expanded_permuted_rows.numel() // H

    flat = esdr.reshape(-1)
    keep = flat != -1
    n_keep = int(keep.sum())
    if n_keep > num_dst:
        # 抽屉原理：单射不存在（当前用例集不会走到这里），保持原样
        return unchanged

    g = torch.Generator().manual_seed(0)  # 固定种子：跨 eval 运行必须可复现
    # argsort(rand) 即随机置换；取前 n_keep 项得到 [0, num_dst) 内互异的目的行
    dst = torch.rand(num_dst, generator=g).argsort()[:n_keep]

    new_flat = flat.clone()
    new_flat[keep] = dst.to(dtype=flat.dtype, device=flat.device)
    new_esdr = new_flat.reshape(esdr.shape)

    return [expanded_permuted_rows, new_esdr, skip1, skip2, bias, scales,
            expert_for_source_row]
