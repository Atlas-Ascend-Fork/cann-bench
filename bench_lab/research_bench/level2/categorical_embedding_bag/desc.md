# categorical_embedding_bag 算子 API 描述

## 1. 算子简介

`CategoricalEmbeddingBag` 对 tabular 类别特征的 bag 维度执行 embedding lookup 与 sum/mean pooling，用于补充 research benchmark 中的模型前后处理与表格特征路径覆盖。

**算子特征**：

- 难度等级：L2（Index）
- indices: [B,F,K], embedding_table: [V,D], bias: [F,D] or null, tokens: [B,F,D]
- 输出 shape 固定，便于精度比较

## 2. 算子定义

```text
tokens[b,f] = sum_or_mean(embedding_table[indices[b,f,:]]) + optional bias[f]
```

## 3. 接口规范

```python
cann_bench.categorical_embedding_bag(Tensor indices, Tensor embedding_table, Tensor? bias=None, bool mean_pool=False, int padding_idx=-1) -> Tensor tokens
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `indices` | `[B,F,K]` | int32 / int64 | 类别 id bag |
| `embedding_table` | `[V,D]` | float16 / bfloat16 / float32 | embedding 表 |
| `bias` | `[F,D] or null` | float16 / bfloat16 / float32 / None | 可选 per-feature bias |
| `mean_pool` | 标量 | bool | true 时对 bag 做 mean pooling，否则做 sum pooling |
| `padding_idx` | 标量 | int | 等于该 id 的位置不参与 pooling |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `tokens` | `[B,F,D]` | 与主浮点输入相同 | 类别 token 输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|---|---|
| float16 | float16 |
| bfloat16 | bfloat16 |
| float32 | float32 |

### 规则与约束

- 输入 shape 需满足 `proto.yaml` 中声明的维度关系。
- 所有整数属性需处于有效范围。
- 参考实现仅依赖 PyTorch，输出 dtype 与算子定义保持一致。

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：MERE / MARE；bool 输出按精确匹配处理。

## 5. 标准 Golden 代码

```python
import torch


def categorical_embedding_bag(
    indices: torch.Tensor,
    embedding_table: torch.Tensor,
    bias: torch.Tensor = None,
    mean_pool: bool = False,
    padding_idx: int = -1,
) -> torch.Tensor:
    if indices.dim() != 3 or embedding_table.dim() != 2:
        raise ValueError("indices must be [B,F,K], embedding_table must be [V,D]")
    _, features, _ = indices.shape
    vocab, dim = embedding_table.shape
    if bias is not None and bias.shape != (features, dim):
        raise ValueError("bias must be [F,D]")
    idx = indices.to(torch.long)
    valid = idx != int(padding_idx)
    idx = torch.remainder(idx, vocab)
    emb = embedding_table[idx].float().masked_fill(~valid.unsqueeze(-1), 0.0)
    pooled = emb.sum(dim=2)
    if mean_pool:
        denom = valid.sum(dim=2, keepdim=True).clamp_min(1).float()
        pooled = pooled / denom
    if bias is not None:
        pooled = pooled + bias.float().unsqueeze(0)
    return pooled.to(embedding_table.dtype)
```
