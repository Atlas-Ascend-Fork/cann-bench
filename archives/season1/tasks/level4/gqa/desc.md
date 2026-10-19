# GQA 算子 API 描述

## 1. 算子简介

分组查询注意力 (Grouped Query Attention) 算子，多个 query head 共享一组 key/value head，对已分头的 Q/K/V 执行注意力计算，在保持模型质量的同时显著减少 KV cache 内存占用和推理计算量。

**主要应用场景**：
- 大语言模型推理中的高效注意力计算（如 LLaMA-2 70B、Mistral）
- 长序列推理场景中降低 KV cache 内存开销
- 需要在模型质量和推理效率之间平衡的 Transformer 架构

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（query, key, value）单输出，执行分组缩放点积注意力
- 输入为已分头的张量，不包含 QKV 投影和输出投影步骤
- N_q 必须能被 N_kv 整除，每个 KV head 被 N_q/N_kv 个 query head 共享
- 支持 `is_causal` 因果掩码：仅计算 attention 矩阵 [S, S_kv] 中从右下角向左上方延伸 45° 对角线及其下方部分（其余位置在 softmax 前置 -inf）

## 2. 算子定义

### 数学公式

对于第 $i$ 个 query head，使用第 $\lfloor i \times N_{kv} / N_q \rfloor$ 个 KV head：

$$
\text{head}_i = \text{softmax}\left(Q_i \times K_{g(i)}^T \times \text{scaleValue}\right) \times V_{g(i)}
$$

其中：
- $N_q$ 为 query 头数，$N_{kv}$ 为 KV 头数，$N_q$ 必须能被 $N_{kv}$ 整除
- $g(i) = \lfloor i \times N_{kv} / N_q \rfloor$ 为第 $i$ 个 query head 对应的 KV head 索引
- $D$ 为每个头的维度
- $\text{scaleValue}$ 为缩放因子（<=0 时自动使用 $1/\sqrt{D}$）
- 每个 KV head 被 $N_q / N_{kv}$ 个 query head 共享

计算步骤（对每个 batch $b$ 与 query head 索引 $h_q \in [0, N_q)$，令 $g = g(h_q)$）：

1. **缩放点积**：$\text{scores}[i, j] = (Q[b, i, h_q, :] \cdot K[b, j, g, :]) \times \text{scaleValue}$，形状 $[S, S_{kv}]$
2. **因果掩码（可选）**：当 `is_causal=True` 时，对 `scores[i, j]` 满足 $j > i + (S_{kv} - S)$ 的位置置为 $-\infty$（即仅保留 $[S, S_{kv}]$ 矩阵从右下角向左上方 45° 延伸的对角线及其下方部分），$S = S_{kv}$ 时退化为标准下三角掩码
3. **Softmax 归一化**：$\text{attn\_weights} = \text{softmax}(\text{scores}, \text{dim}=-1)$
4. **加权求和**：$y[b, :, h_q, :] = \text{attn\_weights} \times V[b, :, g, :]$，形状 $[S, D]$

### 实现建议

> 本节为参考建议，**不是算子语义约束**。任何与上文数学公式等价、满足 §4 精度要求的实现均符合 benchmark 要求；本节仅就常见性能陷阱与误读 §5 Golden 代码的风险给出提示。

- §5 Golden 代码把分组维 `G = N_q / N_kv` 折叠进 matmul 的 M 维（`Q` 变形为 `[B, N_kv, G*S, D]`），`K`/`V` 保持 `N_kv` 头随 batched matmul 复用、不物化到 `N_q` 头；该写法即下条建议的实现，可直接参考。
- 建议按头索引 `g(h_q)` 直接复用 N_kv 个 KV head；若在算子内部或调用前将 K/V 在头维度物化复制 G 份扩展到 N_q，会抹掉 GQA 相对 MHA 的 KV cache 内存收益（占用变为 $N_q \cdot S_{kv} \cdot D$ 而非 $N_{kv} \cdot S_{kv} \cdot D$），并引入冗余访存。
- CANN 原生 GQA 算子（如 `FusedInferAttentionScore`、`npu_fusion_attention`）通过 `num_key_value_heads` 属性告知 kernel 分组比、内部按索引复用 KV，可作为参考实现路径。

## 3. 接口规范

### 算子原型

```python
cann_bench.gqa(Tensor query, Tensor key, Tensor value, float scaleValue=-1.0, bool is_causal=False) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| query | Tensor | 必选 | 查询张量（已分头），shape 为 [B, S, N_q, D] |
| key | Tensor | 必选 | 键张量（已分头），shape 为 [B, S_kv, N_kv, D] |
| value | Tensor | 必选 | 值张量（已分头），shape 为 [B, S_kv, N_kv, D] |
| scaleValue | float | -1.0 | 缩放因子，<=0 时自动使用 1/sqrt(D) |
| is_causal | bool | False | 是否启用因果掩码。False 时全计算；True 时仅计算 [S, S_kv] attention 矩阵中从右下角向左上方 45° 延伸的对角线及其下方部分（即满足 $j \le i + (S_{kv} - S)$ 的位置），上方部分在 softmax 前置 -inf |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [B, S, N_q, D] | 与输入 query 相同 | 分组查询注意力输出张量 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有输入 Tensor（query, key, value）的 dtype 必须一致
- `query` 的 shape 为 [B, S, N_q, D]，`key` 和 `value` 的 shape 为 [B, S_kv, N_kv, D]
- N_q 必须能被 N_kv 整除，分组比 G = N_q / N_kv
- 当 N_kv == N_q 时退化为标准多头注意力 (MHA)
- 当 N_kv == 1 时退化为多查询注意力 (MQA)
- `scaleValue` 通常设置为 $1/\sqrt{D}$，当 <= 0 时自动使用该值
- `is_causal=True` 时要求 $S \le S_{kv}$（否则 mask 会将部分 query 行全部屏蔽，导致 softmax 出现 NaN）

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 256 | cases.csv 实测 2 ~ 128 |
| `S`（query 序列长度） | 1 ~ 4096 | cases.csv 实测 1 ~ 1024；S=1 / 2 / 4 对应 decode / MTP 场景 |
| `S_kv`（key/value 序列长度） | 1 ~ 8192 | cases.csv 实测 128 ~ 2048；`is_causal=True` 要求 $S \le S_{kv}$ |
| `N_q`（query 头数） | 1 ~ 256 | cases.csv 实测 32 ~ 128；必须满足 `N_q % N_kv == 0` |
| `N_kv`（KV 头数） | 1 ~ 256 | cases.csv 实测 1 / 4 / 8 / 32；`N_kv==N_q` 退化为 MHA，`N_kv==1` 退化为 MQA |
| `D`（每头维度） | 64 ~ 512，64 对齐 | cases.csv 实测 128 / 256 |
| `scaleValue` | 任意 float | cases.csv 实测 -1.0（自动 $1/\sqrt{D}$）和 0.08838（显式 $1/\sqrt{128}$）；<=0 时自动使用 $1/\sqrt{D}$ |
| `is_causal` | {False, True} | cases.csv 两者皆覆盖；True 时按右下角对齐生成 [S, S_kv] 因果掩码 |
| 输入 dtype | float16 / bfloat16 | cases.csv 两者皆覆盖；query / key / value 三者 dtype 必须一致 |

约束：
- `N_q % N_kv == 0`，分组比 `G = N_q / N_kv`
- `is_causal=True` 时要求 `S <= S_kv`，否则部分 query 行被全屏蔽产生 NaN
- query / key / value 三者 dtype 必须一致，且最后一维 `D` 三者相同

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
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, S, S_kv, D = 2, 128, 128, 128
N_q, N_kv = 32, 8
query = torch.randn(B, S, N_q, D, dtype=torch.float16, device="npu")
key = torch.randn(B, S_kv, N_kv, D, dtype=torch.float16, device="npu")
value = torch.randn(B, S_kv, N_kv, D, dtype=torch.float16, device="npu")
y = cann_bench.gqa(query, key, value, scaleValue=-1.0, is_causal=False)
```
