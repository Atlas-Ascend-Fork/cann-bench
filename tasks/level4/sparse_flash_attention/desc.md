# SparseFlashAttention 算子 API 描述

## 1. 算子简介

大序列长度推理场景的高效稀疏注意力计算算子。与标准 FlashAttention 对全部 KV 位置计算注意力不同，SparseFlashAttention 通过 `sparseIndices` 指定每个 query 位置需要关注的 KV 子集，仅对选中的 KV 位置执行缩放点积注意力，从而降低计算和内存开销。

支持 Grouped Query Attention（GQA）：query 头数 N1 可以大于 KV 头数 N2（要求 N1 整除 N2），每 N1/N2 个 query 头共享同一组 KV 头及其稀疏索引。同时支持 query/key 和 value 使用不同的 head dim（Dk 和 Dv）。

**主要应用场景**：
- 大语言模型长序列推理中的高效注意力计算
- 长文本理解与生成任务中降低注意力计算复杂度（从 O(S1×S2) 降至 O(S1×topK)）
- 需要稀疏注意力模式的 Transformer 推理加速
- GQA / MLA 架构下的稀疏注意力

**算子特征**：
- 难度等级：L4（FusedComposite）
- 四输入（query, key, value, sparseIndices）单输出
- 融合稀疏索引 gather、缩放点积注意力与 softmax 计算
- 支持 GQA（N1 ≥ N2，N1 % N2 == 0）和不同 head dim（Dk、Dv）
- 支持 BSND 和 BNSD 两种张量布局
- 支持 `is_causal` 因果掩码：在稀疏 gather 之上再叠加 [S1, S2] attention 矩阵的右下角对角线掩码，sparseIndices 选中但落在对角线之上（即 KV 序列位置满足 idx > s1 + (S2 - S1)）的条目在 softmax 前置 -inf
- **value 与 key 在数值上同源**（MLA 推理 latent KV cache 语义）：
  - 当 `Dk == Dv` 时，`value == key`（K 与 V 完全相同）
  - 当 `Dk > Dv`（典型 MLA: Dk=576=512_nope+64_rope, Dv=512），`value == key[..., :Dv]`（V 是 K 的前 Dv 维前缀切片）
  - 接口保留 value 为独立入参以兼容通用 attention API，调用方需保证此关系

## 2. 算子定义

### 稀疏机制

对于每个 KV 头组 `(b, n2, s1)`，`sparseIndices` 中对应位置给出该 query 需要关注的 topK 个 KV 序列位置索引（值域 `[0, S2)`）。该组内的 N1/N2 个 query 头共享相同的稀疏索引，attention 计算仅在这 topK 个选中的 KV 位置上进行。

`sparseIndices` 的前三维布局与 `inputLayout` 一致：
- BSND 布局：sparseIndices shape 为 `[B, S1, N2, topK]`
- BNSD 布局：sparseIndices shape 为 `[B, N2, S1, topK]`

### 数学公式

$$
K_{sel} = \text{gather}(K, \text{sparseIndices}) \quad V_{sel} = \text{gather}(V, \text{sparseIndices})
$$

$$
y = \text{softmax}\left(Q \times K_{sel}^T \times \text{scaleValue}\right) \times V_{sel}
$$

具体子步骤：
1. **稀疏 Gather**：根据 `sparseIndices` 从 key/value 中提取选中的 KV 子集
   - 以 BNSD 为例：$K_{sel}[b, n2, s1, i, :] = K[b, n2, \text{sparseIndices}[b, n2, s1, i], :]$，shape `[B, N2, S1, topK, Dk]`
   - $V_{sel}$ 同理，shape `[B, N2, S1, topK, Dv]`
2. **GQA 扩展**：将 `K_sel` 和 `V_sel` 沿 head 维度复制 G=N1/N2 次，扩展到 `[B, N1, S1, topK, ...]`
3. **缩放点积**：$\text{scores} = Q \cdot K_{sel}^T \times \text{scaleValue}$，shape `[B, N1, S1, topK]`
4. **因果掩码（可选）**：当 `is_causal=True` 时，对每个 query 位置 `s1` 和每个 topK 槽位 `i`，若 `sparseIndices[..., s1, i] > s1 + (S2 - S1)`（即所选 KV 位置落在 [S1, S2] attention 矩阵右下角对角线之上），则该位置 scores 置为 $-\infty$
5. **Softmax 归一化**：$\text{attn\_weights} = \text{softmax}(\text{scores}, \text{dim}=-1)$，在 topK 维上归一化
6. **加权求和**：$y = \text{attn\_weights} \times V_{sel}$

### 布局说明

| inputLayout | query | key | value | sparseIndices | output |
|-------------|-------|-----|-------|---------------|--------|
| BSND | [B, S1, N1, Dk] | [B, S2, N2, Dk] | [B, S2, N2, Dv] | [B, S1, N2, topK] | [B, S1, N1, Dv] |
| BNSD | [B, N1, S1, Dk] | [B, N2, S2, Dk] | [B, N2, S2, Dv] | [B, N2, S1, topK] | [B, N1, S1, Dv] |

### 与标准 FlashAttention 的区别

| 项目 | FlashAttention | SparseFlashAttention |
|------|---------------|---------------------|
| 注意力范围 | 全部 S2 个 KV 位置 | sparseIndices 指定的 topK 个 KV 位置 |
| 计算复杂度 | O(S1 × S2 × D) | O(S1 × topK × D)，topK << S2 |
| 额外输入 | 无 | sparseIndices |
| softmax 范围 | S2 维 | topK 维（仅选中的 KV 位置） |
| GQA 支持 | 是 | 是（N1 % N2 == 0） |
| head dim | Q/K 共享 Dk，V 可用不同 Dv | 同左 |

## 3. 接口规范

### 算子原型

```python
sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor sparseIndices, float scaleValue, str inputLayout="BSND", bool is_causal=False) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| query | Tensor | 是 | float16/bfloat16 | BSND: [B, S1, N1, Dk]；BNSD: [B, N1, S1, Dk] | 查询张量 |
| key | Tensor | 是 | float16/bfloat16 | BSND: [B, S2, N2, Dk]；BNSD: [B, N2, S2, Dk] | 键张量，head dim 与 query 一致 |
| value | Tensor | 是 | float16/bfloat16 | BSND: [B, S2, N2, Dv]；BNSD: [B, N2, S2, Dv] | 值张量，head dim 可与 key 不同。语义上要求 `value == key[..., :Dv]`（同一份 latent KV cache 的前缀视图，Dk==Dv 时退化为 `value == key`） |
| sparseIndices | Tensor | 是 | int32 | BSND: [B, S1, N2, topK]；BNSD: [B, N2, S1, topK] | 按 KV 头分组的稀疏索引，前三维布局与 inputLayout 一致 |
| scaleValue | float | 是 | - | 标量 | 缩放因子，通常为 1/sqrt(Dk) |
| inputLayout | str | 否 | - | - | 张量布局格式，"BSND"（默认）或 "BNSD" |
| is_causal | bool | 否 | - | - | 是否启用因果掩码（右下角对齐），默认 False。True 时在稀疏 gather 之上额外屏蔽 `sparseIndices[..., s1, i] > s1 + (S2 - S1)` 的位置 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| y | 与 query 一致 | BSND: [B, S1, N1, Dv]；BNSD: [B, N1, S1, Dv] | 稀疏注意力输出，布局与输入一致 |

### 数据类型

| query/key/value dtype | sparseIndices dtype | 输出 dtype |
|----------------------|--------------------|-----------| 
| bfloat16 | int32 | bfloat16 |
| float16 | int32 | float16 |

### 规则与约束

- query、key、value 的 dtype 必须一致
- query 和 key 的 head dim 必须一致（Dk），value 的 head dim（Dv）可以不同
- N1 必须整除 N2（GQA 分组约束），N1 == N2 时退化为 MHA
- sparseIndices 的值域为 `[0, S2)`，即 KV 序列长度范围内的有效索引
- topK（每个 query 关注的 KV 数量）可任意取值，1 ≤ topK ≤ S2
- scaleValue 通常设置为 $1/\sqrt{Dk}$
- inputLayout 必须为 "BSND" 或 "BNSD"，所有张量（含 sparseIndices 和输出）的布局保持一致
- `is_causal=True` 时要求 $S1 \le S2$；若某个 query 行的全部 topK 槽位都被因果掩码屏蔽（极端情况），该行 softmax 输出按全 0 处理
- **语义约束 `value == key[..., :Dv]`**：value 必须等于 key 的前 Dv 维前缀（同一份 latent KV cache 的不同视图）。当 `Dk == Dv` 时退化为 `value == key`；当 `Dk == Dv + 64`（MLA 典型 Dk=576/Dv=512）时 value 即 key_nope。算子接口保留 value 为独立入参以兼容通用 attention API，调用方有责任保证此关系；本规范的 Golden 实现不强制检查

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 64 | cases.csv 实测 1 ~ 32 |
| `S1`（query 序列长度） | 1 ~ 2048 | cases.csv 实测 1 ~ 1024；S1=1 对应 decode，S1=2/4 对应 MTP，S1 较大对应 prefill |
| `S2`（KV 序列长度） | 1 ~ 16384 | cases.csv 实测 1024 ~ 8192；`is_causal=True` 时要求 `S1 ≤ S2` |
| `N1`（query 头数） | 1 ~ 256 | cases.csv 实测 32 ~ 128 |
| `N2`（KV 头数） | 1 ~ 64 | cases.csv 实测 1 / 8；必须满足 `N1 % N2 == 0`（GQA 分组约束），`N1 == N2` 时退化为 MHA |
| `Dk`（query/key head dim） | 64 对齐 | cases.csv 实测 128 / 192 / 512 / 576；典型 MHA/GQA Dk=128 或 192，典型 MLA Dk=512（nope）或 Dk=576（nope=512+rope=64） |
| `Dv`（value head dim） | 64 对齐，Dv ≤ Dk | cases.csv 实测 128 / 512；语义上 `value == key[..., :Dv]` |
| `topK`（稀疏选中数量） | 1 ~ 2048 | cases.csv 实测 512 / 1024；要求 `1 ≤ topK ≤ S2` |
| `sparseIndices` 取值 | [0, S2) | int32，必须为 KV 序列长度范围内的有效索引 |
| `scaleValue` | 任意正浮点 | cases.csv 实测 0.04167 ~ 0.08838，约 `1/sqrt(Dk)` |
| `inputLayout` | {"BSND", "BNSD"} | cases.csv 实测两种均覆盖；所有张量（含 sparseIndices 和输出）的布局保持一致 |
| `is_causal` | {True, False} | cases.csv 实测两种均覆盖；True 时屏蔽 `sparseIndices[..., s1, i] > s1 + (S2 - S1)` 的位置 |
| dtype | float16 / bfloat16 | cases.csv 实测两种均覆盖；query/key/value dtype 必须一致；sparseIndices 固定 int32 |

约束：`N1 % N2 == 0`；`Dv ≤ Dk` 且语义上要求 `value == key[..., :Dv]`；`is_causal=True` 时要求 `S1 ≤ S2`；若某 query 行的全部 topK 槽位都被因果掩码屏蔽，该行 softmax 输出按全 0 处理。

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


def sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """
    稀疏 FlashAttention，支持 GQA、不同 head dim 和 BSND/BNSD 布局

    语义约束: value 与 key 在数值上同源 (MLA latent KV cache):
        - Dk == Dv 时: value == key
        - Dk >  Dv 时: value == key[..., :Dv]   (典型 MLA: Dk=576, Dv=512)
        本 Golden 不做强制检查，调用方需保证此关系。

    Args:
        query: 查询张量，BSND: [B, S1, N1, Dk]，BNSD: [B, N1, S1, Dk]
        key: 键张量，BSND: [B, S2, N2, Dk]，BNSD: [B, N2, S2, Dk]
        value: 值张量，BSND: [B, S2, N2, Dv]，BNSD: [B, N2, S2, Dv]；语义上等于 key[..., :Dv]
        sparseIndices: 稀疏索引（int32），BSND: [B, S1, N2, topK]，BNSD: [B, N2, S1, topK]
        scaleValue: 缩放因子
        inputLayout: 张量布局，"BSND" 或 "BNSD"
        is_causal: 是否启用因果掩码（右下角对齐），True 时在稀疏 gather 之上额外屏蔽
            sparseIndices[..., s1, i] > s1 + (S2 - S1) 的位置

    Returns:
        注意力输出，布局与输入一致
    """
    # 统一转为 BNSD 内部计算
    if inputLayout == "BSND":
        q = query.permute(0, 2, 1, 3)              # [B, N1, S1, Dk]
        k = key.permute(0, 2, 1, 3)                 # [B, N2, S2, Dk]
        v = value.permute(0, 2, 1, 3)               # [B, N2, S2, Dv]
        si = sparseIndices.permute(0, 2, 1, 3)      # [B, N2, S1, topK]
    else:  # BNSD
        q, k, v, si = query, key, value, sparseIndices

    B, N1, S1, Dk = q.shape
    N2 = k.shape[1]
    S2 = k.shape[2]
    Dv = v.shape[-1]
    topK = si.shape[-1]
    G = N1 // N2

    # sparseIndices 为 int32，转为 long 用于 gather
    si = si.long()

    # Gather 选中的 KV: si [B, N2, S1, topK]
    idx_k = si.reshape(B, N2, S1 * topK).unsqueeze(-1).expand(-1, -1, -1, Dk)
    idx_v = si.reshape(B, N2, S1 * topK).unsqueeze(-1).expand(-1, -1, -1, Dv)
    k_sel = k.gather(2, idx_k).reshape(B, N2, S1, topK, Dk)  # [B, N2, S1, topK, Dk]
    v_sel = v.gather(2, idx_v).reshape(B, N2, S1, topK, Dv)  # [B, N2, S1, topK, Dv]

    # GQA: 将 KV 头扩展到 N1 个 query 头
    k_sel = k_sel.unsqueeze(2).expand(-1, -1, G, -1, -1, -1).reshape(B, N1, S1, topK, Dk)
    v_sel = v_sel.unsqueeze(2).expand(-1, -1, G, -1, -1, -1).reshape(B, N1, S1, topK, Dv)

    # Attention: Q @ K_sel^T -> softmax -> @ V_sel
    scores = torch.einsum('bnsd,bnskd->bnsk', q, k_sel) * scaleValue
    if is_causal:
        # 右下角对齐：sparseIndices 选中的 KV 位置 idx > s1 + (S2 - S1) 时需屏蔽
        s1_idx = torch.arange(S1, device=scores.device).view(1, 1, S1, 1)  # [1,1,S1,1]
        # si 已为 [B, N2, S1, topK]；需扩展到 N1
        si_n1 = si.unsqueeze(2).expand(-1, -1, G, -1, -1).reshape(B, N1, S1, topK)
        causal_mask = si_n1 > (s1_idx + (S2 - S1))
        scores = scores.masked_fill(causal_mask, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    # 处理整行全 -inf 导致的 NaN：置 0
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    out = torch.einsum('bnsk,bnskd->bnsd', attn_weights, v_sel)  # [B, N1, S1, Dv]

    # 转回原始布局
    if inputLayout == "BSND":
        return out.permute(0, 2, 1, 3)   # [B, S1, N1, Dv]
    else:
        return out                        # [B, N1, S1, Dv]
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, S1, S2, N1, N2, Dk, Dv, topK = 2, 1024, 8192, 32, 8, 128, 128, 512

# BSND 布局
query = torch.randn(B, S1, N1, Dk, dtype=torch.float16, device="npu")
key = torch.randn(B, S2, N2, Dk, dtype=torch.float16, device="npu")
value = torch.randn(B, S2, N2, Dv, dtype=torch.float16, device="npu")
sparseIndices = torch.stack([
    torch.randperm(S2)[:topK] for _ in range(B * N2 * S1)
]).reshape(B, S1, N2, topK).to(dtype=torch.int32, device="npu")
y = sparse_flash_attention(query, key, value, sparseIndices,
                            scaleValue=1.0 / (Dk ** 0.5),
                            inputLayout="BSND",
                            is_causal=False)
# y.shape: [B, S1, N1, Dv]
```
