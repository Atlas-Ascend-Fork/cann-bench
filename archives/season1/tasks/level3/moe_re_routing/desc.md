# MoeReRouting 算子 API 描述

## 1. 算子简介

MoE 网络中，进行 AlltoAll 操作从其他卡上拿到需要算的 token 后，将 token 按照专家顺序重新排列。

**主要应用场景**：
- Mixture of Experts (MoE) 模型中 token 到专家的数据分发
- 分布式 MoE 训练中跨 rank 的 token 重新路由
- MoE 前馈网络前的 token 重排，使同一专家处理的 token 连续排列

**算子特征**：
- 难度等级：L3（LayoutTransform）
- 多输入多输出，根据每个 rank 的专家 token 数量对输入 token 进行重新排列
- 支持可选的 per_token_scales 同步重排

## 2. 算子定义

### 数学公式

通过双重求和计算当前 token 在源位置的偏移量：

$$
\text{SrcOffset} = \sum_{i=0}^{\text{cur\_rank}} \left( \sum_{j=0}^{\text{cur\_expert}} \text{expert\_token\_num\_per\_rank}(i,j) \right)
$$

通过双重求和计算当前 token 在目标位置的偏移量：

$$
\text{DstOffset} = \sum_{j=0}^{\text{cur\_expert}} \left( \sum_{i=0}^{\text{cur\_rank}} \text{expert\_token\_num\_per\_rank}(i,j) \right)
$$

- **SrcOffset**：当前需要移动的 token 源偏移，根据输入 `expert_token_num_per_rank` 的值进行计算
- **DstOffset**：当前需要移动的 token 目的偏移
- **cur_rank**：`expert_token_num_per_rank` 的纵轴索引，表示该 token 原本在的卡
- **cur_expert**：`expert_token_num_per_rank` 的横轴索引，表示该 token 由卡上专家 cur_expert 计算

### 处理流程

1. 根据 `expert_token_num_per_rank` 矩阵计算每个 token 的源位置和目标位置
2. 将 token 从源位置移动到目标位置，实现按专家顺序排列
3. 若提供 `per_token_scales`，同步进行重排
4. 输出重排后的 token、scales、索引及每个专家的 token 数量

## 3. 接口规范

### 算子原型

```python
cann_bench.moe_re_routing(
    Tensor tokens,
    Tensor expert_token_num_per_rank,
    Tensor? per_token_scales = None,
    int expert_token_num_type = 1,
    int idx_type = 0
) -> (Tensor permute_tokens, Tensor permute_per_token_scales, Tensor permute_token_idx, Tensor expert_token_num)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 | Shape |
|------|------|--------|------|-------|
| tokens | Tensor | 必选 | 待重新排布的 token | (A, H) |
| expert_token_num_per_rank | Tensor | 必选 | 每张卡上各个专家处理的 token 数，矩阵元素 [i,j] 表示从卡 i 获取的专家 j 处理的 token 数 | (N, E) |
| per_token_scales | Tensor | None | 每个 token 对应的 scale，需要随 token 同样进行重新排布 | (A) |
| expert_token_num_type | int | 1 | 输出 expert_token_num 的模式：0=cumsum，1=count。当前只支持为 1 | - |
| idx_type | int | 0 | 输出 permute_token_idx 的索引类型：0=gather索引，1=scatter索引。当前只支持为 0 | - |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| permute_tokens | (A, H) | 与 tokens 相同 | 重新排布后的 token |
| permute_per_token_scales | (A) | float32 | 重新排布后的 per_token_scales（输入不携带时无效） |
| permute_token_idx | (A) | int32 | 每个 token 在原排布方式的索引 |
| expert_token_num | (E) | 与 expert_token_num_per_rank 相同 | 每个专家处理的 token 数 |

### 数据类型

| tokens dtype | expert_token_num_per_rank dtype | per_token_scales dtype |
|-------------|--------------------------------|----------------------|
| float16 | int32 / int64 | float32 |
| bfloat16 | int32 / int64 | float32 |
| int8 | int32 / int64 | float32 |

### Shape 变量说明

- **A**：token 个数，取值要求 Sum(expert_token_num_per_rank) = A
- **H**：token 长度（hidden_dim），取值要求 0 < H < 16384
- **N**：卡数（rank 数），取值无限制
- **E**：卡上的专家数，取值无限制

### 规则与约束

1. `tokens` 的形状为 2D (A, H)
2. `expert_token_num_per_rank` 的形状为 2D (N, E)，元素必须大于 0
3. 所有元素之和必须等于 A：`Sum(expert_token_num_per_rank) = A`
4. `expert_token_num_type` 当前只支持为 1（count 模式）
5. `idx_type` 当前只支持为 0（gather 索引模式）
6. `per_token_scales` 为可选参数，存在时 shape 必须为 (A)

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `A`（token 总数，`tokens` 第 0 维 / `per_token_scales` 长度） | 1 ~ 1048576 | 必须满足 `Sum(expert_token_num_per_rank) = A`；cases.csv 实测 64 ~ 524288 |
| `H`（token 长度，`tokens` 第 1 维） | 1 ~ 16383 | 来自 §3 约束 `0 < H < 16384`；cases.csv 实测 32 ~ 2048 |
| `N`（rank 数，`expert_token_num_per_rank` 第 0 维） | 1 ~ 128 | §3 标注「取值无限制」，受 A 总量约束；cases.csv 实测 1 ~ 64 |
| `E`（专家数，`expert_token_num_per_rank` 第 1 维 / `expert_token_num` 长度） | 1 ~ 128 | §3 标注「取值无限制」，受 A 总量约束；cases.csv 实测 1 ~ 64 |
| `tokens` dtype | float16 / bfloat16 / int8 | cases.csv 实测三种 dtype 均覆盖 |
| `expert_token_num_per_rank` dtype | int32 / int64 | cases.csv 实测两种 dtype 均覆盖；元素必须 > 0 |
| `per_token_scales` dtype | float32（或 None） | cases.csv 实测 float32 与 None 两种 |
| `expert_token_num_type`（attr） | {0, 1} | cases.csv 实测仅 1；当前算子只支持 1（count 模式） |
| `idx_type`（attr） | {0, 1} | cases.csv 实测仅 0；当前算子只支持 0（gather 索引） |

约束：
- `tokens.shape == (A, H)`、`expert_token_num_per_rank.shape == (N, E)`，且 `Sum(expert_token_num_per_rank) == A`（双重求和必须严格匹配 token 总数）。
- `expert_token_num_per_rank` 元素必须满足 `> 0`（每个 (rank, expert) 组合至少有 1 个 token），且全部元素之和恰好等于 A；元素 dtype 为 int32 或 int64。
- `per_token_scales` 为可选；提供时 `shape == (A,)`，dtype 必须为 float32。
- dtype 一致性：`tokens.dtype == permute_tokens.dtype`，`expert_token_num_per_rank.dtype == expert_token_num.dtype`，`permute_token_idx.dtype == int32`，`permute_per_token_scales.dtype == float32`。
- attr 限制：当前实现仅支持 `expert_token_num_type == 1`（count 模式）与 `idx_type == 0`（gather 索引）；其余取值未启用。
- 输出索引 `permute_token_idx ∈ [0, A)`，构成长度 A 的全排列（双射）。

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。


## 5. 标准 Golden 代码

```python
import torch
import numpy as np

def moe_re_routing(
    tokens: torch.Tensor,
    expert_token_num_per_rank: torch.Tensor,
    per_token_scales: torch.Tensor = None,
    expert_token_num_type: int = 1,
    idx_type: int = 0
) -> tuple:
    """
    MoeReRouting 算子 Torch Golden 参考实现

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
    # 获取参数
    N, E = expert_token_num_per_rank.shape
    A, H = tokens.shape
    
    # 确保总和匹配
    total_tokens = expert_token_num_per_rank.sum().item()
    assert total_tokens == A, f"Sum of expert_token_num_per_rank ({total_tokens}) must equal A ({A})"
    
    # 构建 src_offset 和 dst_offset 映射
    # 计算每个 (rank, expert) 位置的源偏移和目标偏移
    src_offsets = {}  # (rank, expert) -> src_offset
    dst_offsets = {}  # (rank, expert) -> dst_offset
    
    # 计算 SrcOffset：按 rank 和 expert 的顺序累加
    src_acc = 0
    for i in range(N):  # cur_rank
        for j in range(E):  # cur_expert
            src_offsets[(i, j)] = src_acc
            src_acc += expert_token_num_per_rank[i, j].item()
    
    # 计算 DstOffset：按 expert 和 rank 的顺序累加
    dst_acc = 0
    for j in range(E):  # cur_expert
        for i in range(N):  # cur_rank
            dst_offsets[(i, j)] = dst_acc
            dst_acc += expert_token_num_per_rank[i, j].item()
    
    # 构建重排映射：src_pos -> dst_pos
    src_to_dst = {}
    for i in range(N):
        for j in range(E):
            num_tokens = expert_token_num_per_rank[i, j].item()
            src_start = src_offsets[(i, j)]
            dst_start = dst_offsets[(i, j)]
            for k in range(num_tokens):
                src_to_dst[src_start + k] = dst_start + k
    
    # 构建反向映射用于 gather 索引
    dst_to_src = {v: k for k, v in src_to_dst.items()}
    
    # 生成 permute_token_idx (gather 索引)
    permute_token_idx = torch.zeros(A, dtype=torch.int32)
    for dst_pos in range(A):
        permute_token_idx[dst_pos] = dst_to_src[dst_pos]
    
    # 重排 tokens
    permute_tokens = tokens[permute_token_idx]
    
    # 重排 per_token_scales（如果存在）
    if per_token_scales is not None:
        permute_per_token_scales = per_token_scales[permute_token_idx]
    else:
        permute_per_token_scales = torch.zeros(A, dtype=torch.float32)
    
    # 计算 expert_token_num (count 模式)
    if expert_token_num_type == 1:
        expert_token_num = expert_token_num_per_rank.sum(dim=0)  # 每个专家的总 token 数
    else:
        # cumsum 模式（暂不支持）
        expert_token_num = torch.zeros(E, dtype=expert_token_num_per_rank.dtype)
    
    return permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench
import math
import random

# 基础示例
tokens_num = 16384
tokens_length = 7168
rank_num = 16
expert_num = 16

tokens = torch.randint(low=-10, high=20, size=(tokens_num, tokens_length), dtype=torch.int8)
expert_token_num_per_rank = torch.ones(rank_num, expert_num, dtype=torch.int32)

# 设置每个位置的 token 数量，确保总和等于 tokens_num
tokens_sum = 0
for i in range(rank_num):
    for j in range(expert_num):
        if i == rank_num - 1 and j == expert_num - 1:
            expert_token_num_per_rank[i][j] = tokens_num - tokens_sum
            break
        rand_num = 1
        expert_token_num_per_rank[i][j] = rand_num
        tokens_sum += rand_num

per_token_scales = torch.randn(tokens_num, dtype=torch.float32)

# 调用算子
permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num = cann_bench.moe_re_routing(
    tokens, expert_token_num_per_rank, per_token_scales=per_token_scales
)
```
