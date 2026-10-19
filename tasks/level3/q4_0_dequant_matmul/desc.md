# Q4_0DequantMatmul 算子 API 描述

## 1. 算子简介

`Q4_0DequantMatmul` 面向 GGUF 量化模型推理，严格按照 Q4_0 的 32 元素 block 格式完成权重反量化和矩阵乘。接口接收从 GGUF 文件中提取出的 packed codes 与 block scales，不负责解析 GGUF 容器。

**主要应用场景**：
- GGUF LLM linear 层推理
- Q4_0 权重格式在线反量化
- dequant + matmul 融合优化

**算子特征**：
- 难度等级：L3（Contraction）
- 首版只支持 `quant_type="q4_0"`
- 每个 block 使用 16 个 `uint8` 表示 32 个 code；第 `j` 个字节的低 4 位对应元素 `j`，高 4 位对应元素 `j+16`
- Q4_0 code 解码为 `code - 8`，再乘 block scale
- 每个 block 的 scale 使用 GGUF Q4_0 定义的 FP16 存储

## 2. 算子定义

### 数学公式

```text
W = dequant_q4_0(qweight, scales, block_size)
y = x @ W.T
```

其中 `x.shape[-1] = K`，`W.shape = [N, K]`。

## 3. 接口规范

### 算子原型

```python
cann_bench.q4_0_dequant_matmul(
    Tensor x,
    Tensor qweight,
    Tensor scales,
    int out_features,
    int in_features,
    int block_size=32,
    str quant_type="q4_0",
) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | Shape | dtype | 描述 |
|------|------|-------|-------|------|
| `x` | Tensor | `[..., K]` | float16 / bfloat16 / float32 | activation 输入 |
| `qweight` | Tensor | `[N*K/2]` | uint8 | 按 `[N, K/32, 16]` 展平的 Q4_0 packed codes |
| `scales` | Tensor | `[N, K/32]` | float16 | 从 Q4_0 block 提取的 scale |
| `out_features` | int | - | - | 输出特征 `N` |
| `in_features` | int | - | - | 输入特征 `K` |
| `block_size` | int | - | - | Q4_0 block size |
| `quant_type` | str | - | - | 首版仅支持 `"q4_0"` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| `y` | `[..., N]` | 与 `x` 相同 | matmul 输出 |

### 规则与约束

- `quant_type` 必须为 `"q4_0"`。
- `x.shape[-1] == in_features`。
- `in_features` 必须能被 32 整除。
- `qweight` 必须包含 `out_features * in_features / 2` 个 uint8。
- `scales.dtype == float16` 且 `scales.shape == [out_features, in_features / 32]`。
- `block_size` 必须为 32；该属性仅用于显式声明固定 Q4_0 block 规格。

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `M` | 1 ~ 128 | `x` 可为 1D/2D/3D，末维为 K |
| `K` | 32 ~ 512 | 必须为 32 的倍数 |
| `N` | 16 ~ 512 | 输出特征 |
| `block_size` | 32 | GGUF Q4_0 固定值 |

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
#!/usr/bin/python3
# coding=utf-8

import torch


def _unpack_q4_0(
    qweight: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    blocks_per_row = in_features // 32
    packed = qweight.to(torch.uint8).reshape(out_features, blocks_per_row, 16)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.cat((low, high), dim=-1).reshape(out_features, in_features).to(torch.int16)
    return (codes - 8).to(torch.float32)


def q4_0_dequant_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    out_features: int,
    in_features: int,
    block_size: int = 32,
    quant_type: str = "q4_0",
) -> torch.Tensor:
    if quant_type != "q4_0":
        raise ValueError("first version only supports q4_0")
    if x.shape[-1] != in_features:
        raise ValueError("x.shape[-1] must equal in_features")
    if block_size != 32:
        raise ValueError("GGUF Q4_0 requires block_size=32")
    if in_features % block_size != 0:
        raise ValueError("in_features must be divisible by 32 for GGUF Q4_0")
    expected_packed = out_features * in_features // 2
    if qweight.numel() != expected_packed:
        raise ValueError("qweight must contain exactly out_features * in_features / 2 bytes")
    expected_scale_shape = (out_features, in_features // block_size)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError("scales must have shape [out_features, in_features / 32]")

    q = _unpack_q4_0(qweight, out_features, in_features).to(x.device)
    expanded_scales = scales.float().repeat_interleave(block_size, dim=1)
    weight = q * expanded_scales.to(q.device)
    out = torch.matmul(x.float(), weight.t())
    return out.to(x.dtype)
```
