# MoeGatingTopKSoftmax 算子 API 描述

## 1. 算子简介

MoE 门控网络中 Softmax 和 TopK 的融合算子，先对输入执行 Softmax 归一化，再取 TopK 个最大值，同时输出专家索引和行位置索引。

**主要应用场景**：
- Mixture of Experts (MoE) 模型中的门控路由选择
- 稀疏激活 Transformer 中的专家选择策略
- MoE 层中 token 到专家的路由分配

**算子特征**：
- 难度等级：L3（LayoutTransform）
- 支持 2D 或 3D 输入，输出 TopK 结果、专家索引、行索引三个输出

## 2. 算子定义

### 数学公式

$$
softmaxOut = softmax(x, axis=-1)
$$

$$
yOut, expertIdxOut = topK(softmaxOut, k)
$$

$$
rowIdxRange = arange(expertIdxOut.shape[0] \times expertIdxOut.shape[1])
$$

$$
rowIdxOut = rowIdxRange.reshape([expertIdxOut.shape[1], expertIdxOut.shape[0]]).transpose(1, 0)
$$

### 处理流程

1. 对输入 $x$ 沿最后一维执行 Softmax：$\text{softmaxOut}[i] = \frac{e^{x_i}}{\sum_j e^{x_j}}$
2. 从 Softmax 结果中选取前 $k$ 个最大值作为输出 $yOut$，对应索引为 $expertIdxOut$
3. 计算 $rowIdxOut$，表示展平后的全局位置索引（用于指示每个输出位置在展平数组中的位置）
4. 如果 `finished` 中对应行为 True，则 $expertIdxOut$ 中填入 $num\_expert$（即 $x$ 的最后一维大小）

## 3. 接口规范

### 算子原型

```python
cann_bench.moe_gating_top_k_softmax(Tensor x, Tensor? finished=None, int k=1) -> (Tensor y, Tensor expert_idx, Tensor row_idx)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，支持 2D 或 3D，shape 为 (..., E)，E 为专家数 |
| finished | Tensor? | None | 可选，标记哪些行参与计算，dtype 为 bool，shape 为 x_shape[:-1] |
| k | int | 1 | topK 数量，要求 0 < k <= x最后一维大小，且 k <= 1024 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | (..., k) | 与输入 x 相同 | 输出张量，topK 结果值 |
| expert_idx | (..., k) | int32 | topK 对应的专家索引 |
| row_idx | (..., k) | int32 | 展平后的全局位置索引 |

### 数据类型

| 输入 x dtype | 输出 y dtype | 输出 expert_idx dtype | 输出 row_idx dtype |
|-------------|-------------|---------------------|-------------------|
| float16 | float16 | int32 | int32 |
| bfloat16 | bfloat16 | int32 | int32 |
| float32 | float32 | int32 | int32 |

### 规则与约束

- 输入 `x` 支持 2D 或 3D 张量
- `k` 必须为正整数，且满足 0 < k <= x最后一维大小，k <= 1024
- `finished` 为可选参数，若提供则 shape 必须为 x_shape[:-1]，dtype 必须为 bool
- Softmax 和 TopK 均沿最后一维（dim=-1）计算
- 当 `finished` 中某行为 True 时，该行对应的 `expert_idx` 值会被设为 `num_expert`

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch，3D 输入时） | 1 ~ 64 | x 为 3D `(B, N, E)` 时使用；cases.csv 实测 4 ~ 8 |
| `N`（tokens） | 1 ~ 262144 | cases.csv 实测 64 ~ 131072 |
| `E`（专家数，x 最后一维） | 1 ~ 2048 | cases.csv 实测 4 ~ 1024 |
| `k`（topK） | 1 ~ 1024 | 必须满足 0 < k ≤ E；cases.csv 实测 1 ~ 128 |
| `finished` | shape = x_shape[:-1] | 可选；dtype 必须为 bool；存在时对应行的 `expert_idx` 置为 E |

约束：
- `x` rank ∈ {2, 3}；2D 时 shape `(N, E)`，3D 时 shape `(B, N, E)`；Softmax 与 TopK 沿最后一维 (axis=-1) 计算。
- `0 < k ≤ E` 且 `k ≤ 1024`；`y`、`expert_idx`、`row_idx` 的最后一维大小均为 `k`。
- `finished.shape == x_shape[:-1]` 且 `finished.dtype == bool`；当 `finished[i] == True` 时对应行的 `expert_idx` 填入 `E`（专家总数），表示该 token 不参与路由。
- 输出 dtype：`y` 与 `x` 一致；`expert_idx` 与 `row_idx` 均为 int32。
- `x` 元素必须为非 NaN 有限值（NaN 会破坏 Softmax 归一化，导致 TopK 排序未定义；本基准的 case 集不引入 NaN）。

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

"""
MoeGatingTopKSoftmax算子Torch Golden参考实现

公式:
  softmaxOut = softmax(x, axis=-1)
  yOut, expertIdxOut = topK(softmaxOut, k)
  rowIdxRange = arange(expertIdxOut.shape[0] * expertIdxOut.shape[1])
  rowIdxOut = rowIdxRange.reshape([expertIdxOut.shape[1], expertIdxOut.shape[0]]).transpose(1, 0)

注意: row_idx是展平后的全局位置索引，而非行号
"""
def moe_gating_top_k_softmax(
    x: torch.Tensor,
    finished: torch.Tensor = None,
    k: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MoE门控网络中Softmax和TopK的融合

    Args:
        x: 输入张量，shape (..., E)
        finished: 可选，标记哪些行参与计算，bool类型，shape x_shape[:-1]
        k: topK数量

    Returns:
        y: topK值，shape (..., k)
        expert_idx: topK索引（专家序号），shape (..., k)，int32
        row_idx: 行位置索引（展平后的全局位置），shape (..., k)，int32
    """
    # Softmax沿最后一维
    softmax_out = torch.nn.functional.softmax(x, dim=-1)

    # TopK沿最后一维
    values, indices = torch.topk(softmax_out, k, dim=-1)

    # 计算row_idx，严格对标op-plugin测试代码
    # 公式: rowIdxRange = arange(shape[0] * shape[1])
    #       rowIdxOut = rowIdxRange.reshape([shape[1], shape[0]]).transpose(1, 0)
    # 注意: row_idx是展平后的全局位置索引
    output_shape = indices.shape

    if len(output_shape) == 2:
        # 2D: (N, k)
        # row_idx = arange(N*k).reshape(k, N).transpose(1, 0) -> (N, k)
        row_idx_range = torch.arange(output_shape[0] * output_shape[1], dtype=torch.int32)
        row_idx = row_idx_range.reshape(output_shape[1], output_shape[0]).transpose(0, 1)
    else:
        # 3D: (B, N, k)
        # 先把(B, N)看作整体，计算展平后的索引
        # row_idx_range = arange(B*N*k)
        # reshape成(k, B*N)，transpose成(B*N, k)
        # 再reshape成(B, N, k)
        row_idx_range = torch.arange(output_shape[0] * output_shape[1] * output_shape[2], dtype=torch.int32)
        row_idx = row_idx_range.reshape(output_shape[2], output_shape[0] * output_shape[1]).transpose(0, 1)
        row_idx = row_idx.reshape(output_shape)

    # 处理finished参数
    # F510: `num_expert` (== E) is an out-of-range SENTINEL marking finished
    # tokens — downstream consumers MUST treat `expert_id == num_expert` as
    # "skip routing", NOT as a valid expert index. `moe_finalize_routing`
    # golden bounds-checks before `bias[expert_id, :]` to honor this contract;
    # any non-golden consumer of `indices` must do the same.
    if finished is not None:
        num_expert = x.shape[-1]
        finished_expanded = finished.unsqueeze(-1).expand_as(indices)
        indices = torch.where(finished_expanded, num_expert, indices)

    # row_idx 由 transpose 构造, 留下的是 view, 而输出契约要求 contiguous (issue #146)
    return values, indices.to(torch.int32), row_idx.to(torch.int32).contiguous()
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# 2D 输入
x = torch.randn(1024, 64, dtype=torch.float16, device="npu")
y, expert_idx, row_idx = cann_bench.moe_gating_top_k_softmax(x, k=8)

# 带 finished 参数
finished = torch.randint(0, 2, (1024,), dtype=torch.bool, device="npu")
y, expert_idx, row_idx = cann_bench.moe_gating_top_k_softmax(x, finished=finished, k=8)

# 3D 输入
x_3d = torch.randn(4, 1024, 64, dtype=torch.float32, device="npu")
y, expert_idx, row_idx = cann_bench.moe_gating_top_k_softmax(x_3d, k=4)
```
