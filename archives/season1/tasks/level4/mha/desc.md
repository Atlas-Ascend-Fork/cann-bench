# MHA 算子 API 描述

## 1. 算子简介

多头注意力 (Multi-Head Attention) 算子，对已分头的 Q/K/V 执行缩放点积注意力计算，广泛应用于 Transformer 架构。

**主要应用场景**：
- Transformer 编码器和解码器中的自注意力与交叉注意力
- 大语言模型和视觉 Transformer 中的核心注意力模块
- 多模态模型中的跨模态注意力融合

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（query, key, value）单输出，执行缩放点积注意力
- 输入为已分头的张量，不包含 QKV 投影和输出投影步骤
- 支持可配置的缩放因子
- 支持 `is_causal` 因果掩码：仅计算 attention 矩阵 [S, S_kv] 中从右下角向左上方延伸 45° 对角线及其下方部分（其余位置在 softmax 前置 -inf）

## 2. 算子定义

### 数学公式

$$
y = \text{softmax}\left(Q \times K^T \times \text{scaleValue}\right) \times V
$$

其中：
- $Q$、$K$、$V$ 为已分头的查询、键、值张量
- $\text{scaleValue}$ 为缩放因子（<=0 时自动使用 $1/\sqrt{D}$，$D$ 为每头维度）
- softmax 沿最后一维计算

具体子步骤：
1. **缩放点积**：$\text{scores} = Q \times K^T \times \text{scaleValue}$
2. **因果掩码（可选）**：当 `is_causal=True` 时，对 `scores[..., i, j]` 满足 $j > i + (S_{kv} - S)$ 的位置置为 $-\infty$（即仅保留从右下角向左上方 45° 延伸的对角线及其下方部分），$S=S_{kv}$ 时退化为标准下三角掩码
3. **Softmax 归一化**：$\text{attn\_weights} = \text{softmax}(\text{scores}, \text{dim}=-1)$
4. **加权求和**：$y = \text{attn\_weights} \times V$

## 3. 接口规范

### 算子原型

```python
cann_bench.mha(Tensor query, Tensor key, Tensor value, float scaleValue=-1.0, bool is_causal=False) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| query | Tensor | 必选 | 查询张量（已分头），shape 为 [B, S, N, D] |
| key | Tensor | 必选 | 键张量（已分头），shape 为 [B, S_kv, N, D] |
| value | Tensor | 必选 | 值张量（已分头），shape 为 [B, S_kv, N, D] |
| scaleValue | float | -1.0 | 缩放因子，<=0 时自动使用 1/sqrt(D) |
| is_causal | bool | False | 是否启用因果掩码。False 时全计算；True 时仅计算 [S, S_kv] attention 矩阵中从右下角向左上方 45° 延伸的对角线及其下方部分（即满足 $j \le i + (S_{kv} - S)$ 的位置），上方部分在 softmax 前置 -inf |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [B, S, N, D] | 与输入 query 相同 | 多头注意力输出张量 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有输入 Tensor（query, key, value）的 dtype 必须一致
- `query` 的 shape 为 [B, S, N, D]，`key` 和 `value` 的 shape 为 [B, S_kv, N, D]
- N 为注意力头数，D 为每头维度，均从输入 shape 中推断
- `scaleValue` 通常设置为 $1/\sqrt{D}$，当 <= 0 时自动使用该值
- `is_causal=True` 时要求 $S \le S_{kv}$（否则 mask 会将部分 query 行全部屏蔽，导致 softmax 出现 NaN）

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 128 | cases.csv 实测 1 ~ 128 |
| `S`（query 序列长度） | 1 ~ 2048 | cases.csv 实测 1 ~ 1024；decode 场景 S=1 或 2，prefill 场景 S 与 S_kv 同量级 |
| `S_kv`（key/value 序列长度） | 1 ~ 4096 | cases.csv 实测 128 ~ 2048；`is_causal=True` 时要求 S ≤ S_kv |
| `N`（注意力头数） | 1 ~ 64 | cases.csv 实测 8 ~ 32 |
| `D`（每头维度） | 64 ~ 256，64 对齐 | cases.csv 实测 64 / 128 / 256 |
| `scaleValue` | 任意 float | cases.csv 实测 -1.0（auto = 1/sqrt(D)）和 0.08838（显式 ≈ 1/sqrt(128)）；<=0 时回退到 1/sqrt(D) |
| `is_causal` | {False, True} | cases.csv 实测两值均覆盖（True 走右下角对齐因果掩码，False 全计算） |
| 输入 value range | 任意有限实数 | cases.csv 实测 [-1, 1]（常态高斯采样）和 [0, 0]（全零退化输入） |
| 输入 dtype | float16, bfloat16 | Q/K/V 三个 tensor dtype 必须一致 |

约束：`query.shape = [B, S, N, D]`，`key.shape = value.shape = [B, S_kv, N, D]`，四个共享维度 B/N/D 必须严格相等；`is_causal=True` 时要求 `S ≤ S_kv`，否则部分 query 行会被全部屏蔽导致 softmax 出现 NaN。

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
MHA算子Torch Golden参考实现

多头注意力 (Multi-Head Attention)，对已分头的 Q/K/V 执行缩放点积注意力
公式: y = softmax(Q @ K^T * scaleValue) @ V
"""


def mha(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaleValue: float = -1.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    多头注意力 (Multi-Head Attention)

    Args:
        query: 查询张量 [B, S, N, D]（已分头）
        key: 键张量 [B, S_kv, N, D]（已分头）
        value: 值张量 [B, S_kv, N, D]（已分头）
        scaleValue: 缩放因子，<=0 时自动使用 1/sqrt(D)
        is_causal: 是否启用因果掩码（右下角对齐），True 时 scores[..., i, j] 满足
            j > i + (S_kv - S) 的位置在 softmax 前置为 -inf。要求 S <= S_kv。

    Returns:
        输出张量 [B, S, N, D]
    """
    B, S, N, D = query.shape
    S_kv = key.shape[1]

    if scaleValue <= 0:
        scaleValue = 1.0 / (D ** 0.5)

    # 转置为 [B, N, S, D]
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # 缩放点积注意力
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    if is_causal:
        i = torch.arange(S, device=scores.device).unsqueeze(-1)
        j = torch.arange(S_kv, device=scores.device).unsqueeze(0)
        causal_mask = j > (i + (S_kv - S))  # 右下角对齐：上三角置 -inf
        scores = scores.masked_fill(causal_mask, float('-inf'))
    # F217: 全 mask 行 (整行 = -inf) 在 softmax 时得 0/0 = NaN，对齐
    # sparse_flash_attention 加显式保护 → 全 mask 行权重置 0。
    scores_max = scores.max(dim=-1, keepdim=True).values
    all_masked = torch.isinf(scores_max) & (scores_max < 0)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    attn_weights = torch.where(all_masked, torch.zeros_like(attn_weights), attn_weights)
    attn_output = torch.matmul(attn_weights, v)

    # 转回 [B, S, N, D]；transpose 只改 stride, 而输出契约要求 contiguous (issue #146)
    return attn_output.transpose(1, 2).contiguous()
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, S, S_kv, N, D = 2, 128, 128, 8, 64
query = torch.randn(B, S, N, D, dtype=torch.float16, device="npu")
key = torch.randn(B, S_kv, N, D, dtype=torch.float16, device="npu")
value = torch.randn(B, S_kv, N, D, dtype=torch.float16, device="npu")
y = cann_bench.mha(query, key, value, scaleValue=-1.0, is_causal=False)
```
