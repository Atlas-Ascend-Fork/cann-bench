# MLA 算子 API 描述

## 1. 算子简介

多头潜在注意力 (Multi-Head Latent Attention) 算子，仅包含注意力计算部分（不含 KV 解压缩），是 DeepSeek-V2/V3 等模型的核心注意力机制。

**主要应用场景**：
- 大语言模型推理中的 MLA 注意力计算（如 DeepSeek-V2、DeepSeek-V3）
- 超长序列推理场景
- 需要在推理效率和模型质量之间取得平衡的大规模 Transformer 架构

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（q_nope, q_rope, k_nope, k_rope, v）单输出
- Q 和 K 均分为 nope 和 rope 两部分传入，内部拼接后计算注意力
- V 与 K_nope 不仅 shape 相同 (`[B, S_kv, N_kv, d_nope]`)，**数值上也完全相同**——生产环境中 V 就是从 latent KV cache 读出的 K_nope 那段内存，共享同一份数据；算子接口为兼容通用 attention API 保留为独立入参
- 输出 head dim 为 d_nope
- 支持 GQA 模式（N_q 个 query head 共享 N_kv 个 KV head），MLA 典型配置 N_kv=1
- 支持 BSND 和 BNSD 两种输入输出 layout
- 支持 `is_causal` 因果掩码：仅计算 attention 矩阵 [S, S_kv] 中从右下角向左上方延伸 45° 对角线及其下方部分（其余位置在 softmax 前置 -inf）

## 2. 算子定义

### 数学公式

$$
Q = \text{concat}(Q_{nope}, Q_{rope}) \quad \text{dim: } d_{nope} + d_{rope}
$$

$$
K = \text{concat}(K_{nope}, K_{rope}) \quad \text{dim: } d_{nope} + d_{rope}
$$

$$
y = \text{softmax}\left(Q \times K^T \times \text{scaleValue}\right) \times V
$$

其中：
- $Q_{nope}$、$K_{nope}$ 为 nope 部分，维度 $d_{nope}$
- $Q_{rope}$、$K_{rope}$ 为 rope 部分（经过 RoPE 编码），维度 $d_{rope}$
- $V$ 的 head dim 为 $d_{nope}$，因此输出的 head dim 也为 $d_{nope}$
- $\text{scaleValue}$ 为缩放因子（<=0 时自动使用 $1/\sqrt{d_{nope} + d_{rope}}$）
- 当 $N_q > N_{kv}$ 时进行 GQA 扩展，每个 KV head 被 $N_q / N_{kv}$ 个 query head 共享

具体子步骤：
1. **Q 拼接**：$Q = \text{concat}(Q_{nope}, Q_{rope})$
2. **K 拼接**：$K = \text{concat}(K_{nope}, K_{rope})$
3. **GQA 扩展**：将 KV head 复制 $N_q / N_{kv}$ 次匹配 query head 数
4. **缩放点积**：$\text{scores} = Q \times K^T \times \text{scaleValue}$
5. **因果掩码（可选）**：当 `is_causal=True` 时，对 `scores[..., i, j]` 满足 $j > i + (S_{kv} - S)$ 的位置置为 $-\infty$（即仅保留从右下角向左上方 45° 延伸的对角线及其下方部分），$S=S_{kv}$ 时退化为标准下三角掩码
6. **Softmax 归一化**：$\text{attn\_weights} = \text{softmax}(\text{scores}, \text{dim}=-1)$
7. **加权求和**：$y = \text{attn\_weights} \times V$

## 3. 接口规范

### 算子原型

```python
cann_bench.mla(Tensor q_nope, Tensor q_rope, Tensor k_nope, Tensor k_rope, Tensor v, int numKVHeads=1, float scaleValue=-1.0, str inputLayout="BSND", bool is_causal=False) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| q_nope | Tensor | 必选 | query 的 nope 部分，BSND: [B, S, N_q, d_nope]，BNSD: [B, N_q, S, d_nope] |
| q_rope | Tensor | 必选 | query 的 rope 部分，BSND: [B, S, N_q, d_rope]，BNSD: [B, N_q, S, d_rope] |
| k_nope | Tensor | 必选 | key 的 nope 部分，BSND: [B, S_kv, N_kv, d_nope]，BNSD: [B, N_kv, S_kv, d_nope] |
| k_rope | Tensor | 必选 | key 的 rope 部分，BSND: [B, S_kv, N_kv, d_rope]，BNSD: [B, N_kv, S_kv, d_rope] |
| v | Tensor | 必选 | 值张量，BSND: [B, S_kv, N_kv, d_nope]，BNSD: [B, N_kv, S_kv, d_nope]。语义上要求 `v == k_nope`（数值完全相同，共享 latent KV cache） |
| numKVHeads | int | 1 | KV 头数 |
| scaleValue | float | -1.0 | 缩放因子，<=0 时自动使用 1/sqrt(d_nope + d_rope) |
| inputLayout | str | "BSND" | 输入输出 layout，"BSND" 或 "BNSD" |
| is_causal | bool | False | 是否启用因果掩码。False 时全计算；True 时仅计算 [S, S_kv] attention 矩阵中从右下角向左上方 45° 延伸的对角线及其下方部分（即满足 $j \le i + (S_{kv} - S)$ 的位置），上方部分在 softmax 前置 -inf |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | BSND: [B, S, N_q, d_nope]；BNSD: [B, N_q, S, d_nope] | 与输入相同 | 注意力输出张量，head dim 为 d_nope |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有输入 Tensor 的 dtype 必须一致
- N_q 必须能被 N_kv 整除
- q_nope 和 k_nope 的 head dim 一致（d_nope），q_rope 和 k_rope 的 head dim 一致（d_rope）
- v 的 head dim = d_nope，输出的 head dim 也为 d_nope
- **语义约束 `v == k_nope`**：v 与 k_nope 在数值上必须完全相同（共享同一份 latent KV cache）。算子接口保留独立入参以兼容标准 attention API，但调用方有责任保证两者一致；本规范的 Golden 实现不强制检查
- 所有输入和输出遵循相同的 layout（BSND 或 BNSD）
- `is_causal=True` 时要求 $S \le S_{kv}$（否则 mask 会将部分 query 行全部屏蔽，导致 softmax 出现 NaN）

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 128 | cases.csv 实测 1 ~ 128 |
| `S`（query 序列长度） | 1 ~ 512 | cases.csv 实测 BSND: 1 / 2 / 128 / 256；BNSD: 1 / 2 / 128 / 512 |
| `S_kv`（KV 序列长度） | 128 ~ 2048 | cases.csv 实测 128 / 256 / 512 / 1024 / 2048；`is_causal=True` 时要求 $S \le S_{kv}$ |
| `N_q`（query head 数） | 32 ~ 128 | cases.csv 实测 64 / 128；必须能被 `N_kv` 整除 |
| `N_kv`（KV head 数 = `numKVHeads`） | 1 | MLA 典型配置，固定为 1 |
| `d_nope`（nope head dim） | 128 ~ 512，64 对齐 | cases.csv 实测 448 / 512；与 `v`、输出 head dim 一致 |
| `d_rope`（rope head dim） | 32 ~ 128，64 对齐 | cases.csv 实测均为 64 |
| `scaleValue` | float (`<=0` 自动取 $1/\sqrt{d_{nope}+d_{rope}}$) | cases.csv 实测均为 -1.0（走默认缩放） |
| `inputLayout` | `"BSND"` / `"BNSD"` | cases.csv 实测两种 layout 均覆盖 |
| `is_causal` | `True` / `False` | cases.csv 实测两种取值均覆盖 |

约束：`N_q` 必须能被 `N_kv` 整除；`q_nope`、`k_nope`、`v` 的最后一维必须相同（均为 `d_nope`），`q_rope`、`k_rope` 的最后一维必须相同（均为 `d_rope`）；所有输入 dtype 必须一致；`is_causal=True` 时必须满足 $S \le S_{kv}$。

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
MLA算子Torch Golden参考实现

多头潜在注意力 (Multi-Head Latent Attention)，仅包含注意力计算部分
Q 和 K 均分为 nope 和 rope 两部分传入，内部拼接后计算注意力
支持 BSND 和 BNSD 两种输入 layout

语义约束:
    v 与 k_nope 在数值上完全相同 (共享同一份 latent KV cache)。
    算子接口为兼容通用 attention API 保留独立入参；调用方需保证两者
    一致，本 Golden 实现不做强制检查。

公式:
    Q = concat(Q_nope, Q_rope)   dim: d_nope + d_rope
    K = concat(K_nope, K_rope)   dim: d_nope + d_rope
    V = K_nope                    dim: d_nope     (语义上等价)
    y = softmax(Q @ K^T * scaleValue) @ V
"""


def mla(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    numKVHeads: int = 1,
    scaleValue: float = -1.0,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """
    多头潜在注意力 (Multi-Head Latent Attention)

    Args:
        q_nope: query 的 nope 部分，BSND: [B, S, N_q, d_nope]，BNSD: [B, N_q, S, d_nope]
        q_rope: query 的 rope 部分，BSND: [B, S, N_q, d_rope]，BNSD: [B, N_q, S, d_rope]
        k_nope: key 的 nope 部分，BSND: [B, S_kv, N_kv, d_nope]，BNSD: [B, N_kv, S_kv, d_nope]
        k_rope: key 的 rope 部分，BSND: [B, S_kv, N_kv, d_rope]，BNSD: [B, N_kv, S_kv, d_rope]
        v: 值张量，BSND: [B, S_kv, N_kv, d_nope]，BNSD: [B, N_kv, S_kv, d_nope]
        numKVHeads: KV 头数
        scaleValue: 缩放因子，<=0 时自动使用 1/sqrt(d_nope + d_rope)
        inputLayout: 输入 layout，"BSND" 或 "BNSD"
        is_causal: 是否启用因果掩码（右下角对齐），True 时 scores[..., i, j] 满足
            j > i + (S_kv - S) 的位置在 softmax 前置为 -inf。要求 S <= S_kv。

    Returns:
        输出张量，与输入 layout 一致，head dim 为 d_nope
    """
    # 统一转为 BSND 内部计算
    if inputLayout == "BNSD":
        q_nope = q_nope.permute(0, 2, 1, 3)
        q_rope = q_rope.permute(0, 2, 1, 3)
        k_nope = k_nope.permute(0, 2, 1, 3)
        k_rope = k_rope.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

    B, S, N_q, d_nope = q_nope.shape
    d_rope = q_rope.shape[-1]
    D_qk = d_nope + d_rope
    S_kv = k_nope.shape[1]
    N_kv = numKVHeads

    if scaleValue <= 0:
        scaleValue = 1.0 / (D_qk ** 0.5)

    # 拼接 Q = [Q_nope, Q_rope]: [B, S, N_q, d_nope + d_rope]
    q = torch.cat([q_nope, q_rope], dim=-1)

    # 拼接 K = [K_nope, K_rope]: [B, S_kv, N_kv, d_nope + d_rope]
    k = torch.cat([k_nope, k_rope], dim=-1)

    # GQA: 每个 KV head 被 G = N_q // N_kv 个 Q head 共享。不把 K/V 物化到 N_q 头
    # (fp64 oracle 下大 batch 用例会物化上百 GiB → OOM)，而是把 group 维折叠进
    # matmul 的 M 维: Q -> [B, N_kv, G*S, D]，K/V 保持 N_kv 头随 batched matmul 复用。
    # 数值上与展开完全等价，峰值内存由 scores 决定而非物化后的 K/V。
    G = N_q // N_kv
    q = q.reshape(B, S, N_kv, G, D_qk).permute(0, 2, 3, 1, 4).reshape(B, N_kv, G * S, D_qk)
    k = k.permute(0, 2, 1, 3)  # [B, N_kv, S_kv, D_qk]
    v = v.permute(0, 2, 1, 3)  # [B, N_kv, S_kv, d_nope]

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
    out = torch.matmul(attn_weights, v)  # [B, N_kv, G*S, d_nope]

    # 还原为 BSND: [B, S, N_q, d_nope]（N_q 头序 = n_kv * G + g，与展开路径一致）
    out = out.reshape(B, N_kv, G, S, d_nope).permute(0, 3, 1, 2, 4).reshape(B, S, N_q, d_nope)

    # 按输入 layout 输出
    if inputLayout == "BNSD":
        out = out.permute(0, 2, 1, 3)

    # 上面的 reshape/permute 链留下的是 view, 而输出契约要求 contiguous (issue #146)
    return out.contiguous()
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, S, S_kv = 2, 128, 256
N_q, N_kv = 128, 1
d_nope, d_rope = 512, 64

# BSND layout
q_nope = torch.randn(B, S, N_q, d_nope, dtype=torch.float16, device="npu")
q_rope = torch.randn(B, S, N_q, d_rope, dtype=torch.float16, device="npu")
k_nope = torch.randn(B, S_kv, N_kv, d_nope, dtype=torch.float16, device="npu")
k_rope = torch.randn(B, S_kv, N_kv, d_rope, dtype=torch.float16, device="npu")
v = torch.randn(B, S_kv, N_kv, d_nope, dtype=torch.float16, device="npu")
y = cann_bench.mla(q_nope, q_rope, k_nope, k_rope, v,
                      numKVHeads=N_kv, scaleValue=-1.0, inputLayout="BSND", is_causal=False)
# y shape: [B, S, N_q, d_nope] = [2, 128, 128, 512]
```
