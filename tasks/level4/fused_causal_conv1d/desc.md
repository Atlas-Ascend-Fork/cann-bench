# FusedCausalConv1d 算子 API 描述

## 1. 算子简介

FusedCausalConv1d 算子对序列执行**因果一维卷积**，并融合 APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接、原地更新、可选 silu 激活等特性。相较于标准 causal_conv1d，本算子新增了 APC 缓存复用、PD 混部、残差连接可选等功能。

**主要应用场景**：
- 大模型推理中的因果一维卷积（如 Mamba / State Space Model）
- 变长序列 / 投机解码 / APC 缓存场景下的融合卷积计算
- Transformer 推理加速中的融合卷积模块

**算子特征**：
- 难度等级：L4（Convolution）
- 10 个张量输入（含可选输入），2 个输出，8 个属性参数
- 支持 ND 格式输入
- 支持原地更新 `x`（`inplace=True`）

## 2. 算子定义

### 数学公式

设 $K$ 为卷积核宽度，$L$ 为当前 batch 的序列长度，$dim$ 为特征维度，$batchId$ 为当前处理的变长序列索引，$C$ 为 `conv_states` 的缓存空间长度（`conv_states.size(1)`），$B$ 为 APC 块大小（`block_size`）。对输入 $x$ 中的每一个 batch 执行如下操作：

**1）缓存读取**

缓存行索引：

$$
readCacheLine = \begin{cases}
cache\_indices[batchId, \; initial\_state\_idx[batchId]], & \text{APC 模式} \\
cache\_indices[batchId], & \text{非 APC 且 cache\_indices 存在} \\
batchId, & \text{其他}
\end{cases}
$$

- 首次计算（`num_computed_tokens[batchId] == 0`）：
  $$
  cachedState[i, dim] = 0, \quad 0 \leq i < K-1
  $$
  $$
  offset = 0
  $$

- 投机解码模式（`num_accepted_tokens` 存在）：
  $$
  offset = numAcceptedTokens[batchId] - 1
  $$
  $$
  cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i < offset + K - 1
  $$

- 默认模式：
  $$
  offset = C - (K - 1)
  $$
  $$
  cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i < offset + K - 1
  $$

**2）缓存拼接**

$$
paddedInput[i, dim] =
\begin{cases}
cachedState[i, dim], & 0 \leq i < offset + K - 1 \\
x[i - (offset + K - 1), dim], & offset + K - 1 \leq i < offset + K - 1 + L
\end{cases}
$$

**3）缓存更新**

$$
Len = offset + K - 1 + L
$$
$$
M = \min(C, \; Len)
$$

$$
writeCacheLine = \begin{cases}
cache\_indices[batchId, \; idxLast], & \text{APC 模式} \\
cache\_indices[batchId], & \text{非 APC 且 cache\_indices 存在} \\
batchId, & \text{其他}
\end{cases}
$$

$$
convStates[writeCacheLine][C - M + i, dim] = paddedInput[Len - M + i, dim], \quad i = 0, 1, \dots, M-1
$$

**4）Offset 裁剪**

$$
x'[i, dim] = paddedInput[i + offset, dim], \quad 0 \leq i < K - 1 + L
$$

**5）APC 缓存填充（APC 模式下）**

$$
seqCompletedOffsetToken = numComputedTokens[batchId] \mod B
$$
$$
seqCompletedOffset = B - seqCompletedOffsetToken
$$
$$
seqEndOffset = (L - seqCompletedOffset) \mod B
$$

$$
lastFullBlockTokenIndex = \begin{cases}
L - seqEndOffset - B, & seqEndOffset = 0 \\
L - seqEndOffset, & \text{otherwise}
\end{cases}
$$

$$
nBlockToFill = idxLast - idxFirst
$$

对每个 $chunk = 0, 1, \dots, nBlockToFill - 1$：

$$
boundaryIdx = lastFullBlockTokenIndex - (nBlockToFill - chunk - 1) \times B
$$

$$
convStates[cacheIndices[batchId, \; idxFirst + chunk]][C-(K-1)+j, \; dim] = x'[boundaryIdx + j, \; dim], \quad j = 0, \dots, K-2
$$

**6）因果 1 维卷积**

$$
y[i, dim] = \sum_{k=0}^{K-1} w[k, dim] \cdot x'[i + k, dim], \quad i = 0, 1, \dots, L-1
$$

**7）零填充重置（`conv_mode == 1` 且 `num_computed_tokens` 不为空）**

$$
resetIdx = \min\!\Big(\max\!\big(K - 1 - numComputedTokens[batchId], \; 0\big), \; L\Big)
$$

$$
y[i, dim] = 0, \quad 0 \leq i < resetIdx
$$

**8）残差连接（可选）**

$$
y[i, dim] = x[i, dim] + y[i, dim]
$$

**9）激活函数（仅支持 silu，可选）**

$$
y[i, dim] = silu(y[i, dim])
$$

## 3. 接口规范

### 算子原型

```python
cann_bench.fused_causal_conv1d(
    Tensor x,
    Tensor weight,
    Tensor conv_states,
    Tensor query_start_loc,
    Tensor cache_indices,
    Tensor num_accepted_tokens,
    Tensor num_computed_tokens,
    Tensor block_idx_first_scheduled_token,
    Tensor block_idx_last_scheduled_token,
    Tensor initial_state_idx,
    int64 pad_slot_id=-1,
    int64 max_query_len=-1,
    int64 residual_connection=1,
    int64 block_size=128,
    int64 conv_mode=1,
    bool inplace=False,
    string activation="none",
    int64 run_mode=0
) -> (Tensor y, Tensor conv_states)
```

### 输入参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| x | 输入/输出 | 输入序列；`inplace=True` 时原地更新 | 2D `[cuSeqLen, dim]` 或 3D `[batch, seqLen, dim]`；`dim` 为 16 的倍数 | BFLOAT16/FLOAT16 | ND | 2D/3D | 支持（dim 维 stride=1） |
| weight | 输入 | 因果 1D 卷积核 | `K ∈ {3,4,5,6}`，不支持空 Tensor | 同 x | ND | `[K, dim]` | 支持 |
| conv_states | 输入/输出 | 历史 token 缓存，原地更新 | 不支持空 Tensor；`stateLen = K-1+m` | 同 x | ND | `[..., stateLen, dim]` | 支持（dim 维 stride=1） |
| query_start_loc | 可选输入 | 2D 场景下各序列起始偏移 | 2D 时不可省略；`query_start_loc[0]=0`，末项为 `cuSeqLen`；支持尾部空 batch | INT32 | ND | `[batch+1]`，batch ∈ [1, 4096] | 支持 |
| cache_indices | 可选输入 | 缓存 slot 索引 | 1D 表示非 APC；2D 表示开启 APC；APC 开启时不可省略 | INT32 | ND | `[batch]` 或 `[batch, maxNumBlocks]` | 支持 |
| initial_state_mode | 可选输入 | 每个序列的 padding 策略 | **暂不支持此字段** | INT32 | ND | `[batch]` | 支持 |
| bias | 可选输入 | 卷积偏置 | **暂不支持此字段** | 同 x | ND | `[dim]` | 支持 |
| num_accepted_tokens | 可选输入 | 投机解码每 batch 接受 token 数 | 值范围 `[1, seqlen]`；为空表示非投机模式 | INT32 | ND | `[batch]` | 支持 |
| num_computed_tokens | 可选输入 | 当前 batch 已处理 token 总数 | Pangu V2 首 token 时不能为空 | INT32 | ND | `[batch]` | 支持 |
| block_idx_first_scheduled_token | 可选输入 | APC 第一个调度 block 索引 | APC 开启时不能为空 | INT32 | ND | `[batch]` | 支持 |
| block_idx_last_scheduled_token | 可选输入 | APC 最后一个调度 block 索引 | APC 开启时不能为空 | INT32 | ND | `[batch]` | 支持 |
| initial_state_idx | 可选输入 | APC 初始 state 读取索引 | APC 开启时不能为空 | INT32 | ND | `[batch]` | 支持 |
| activation | 输入 | 激活函数类型 | 支持 `"none"` / `"silu"` / `"swish"` | STR | - | - | - |
| pad_slot_id | 输入 | 无效 slot ID，跳过对应 batch | 仅支持不参与计算的序列在 `x` 的开头或结尾 | INT64 | - | - | - |
| run_mode | 输入 | 0=prefill，1=decode | 历史遗留字段，暂不支持 | INT64 | - | - | - |
| max_query_len | 输入 | 最大序列长度 | -1 表示自动推导；≥9 视为 prefill，[1,8] 视为 decode | INT64 | - | - | - |
| residual_connection | 输入 | 是否残差连接 | 0=否，1=输出 y 与输入 x 相加 | INT64 | - | - | - |
| block_size | 输入 | APC block 大小 | 典型值 128/256 | INT64 | - | - | - |
| conv_mode | 输入 | 卷积实现模式 | 0=Qwen3-Next 社区版本；1=Pangu V2 | INT64 | - | - | - |
| inplace | 输入 | 是否原地更新 x | `True` 时结果写回 x；`False` 时通过 y 输出 | BOOL | - | - | - |

### 输出

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| y | 输出 | 卷积计算结果 | 与输入 `x` shape 一致 | 同 x | ND | 同输入 | 支持 |
| conv_states | 输出 | 更新后的缓存状态 | 与输入 `conv_states` shape 一致 | 同 x | ND | 同输入 | 支持 |

### 数据类型

| x / weight / conv_states dtype | index tensors dtype | y / conv_states dtype |
|-------------------------------|---------------------|-----------------------|
| bfloat16 | int32 | bfloat16 |
| float16 | int32 | float16 |

### 规则与约束

- **确定性计算**
  - 本算子默认为确定性实现，暂不支持非确定性实现。

- **输入值域范围**
  - 算子入参输入值域范围为 `inf`/`-inf`/`nan`，或输入值域/中间计算结果出现超过 float16/bfloat16 数据类型范围时，无法确定输出结果有效性。

- **输入 shape 限制**
  - `x`、`weight`、`conv_states` 不支持空 Tensor。
  - `dim` 必须是 16 的整数倍，范围 `[64, 16384]`。
  - `weight` 的第 0 维（卷积核宽度 $K$）仅支持 `{3, 4, 5, 6}`。
  - `conv_states` 的 `stateLen == K - 1 + m`：
    - prefill 场景下 `m = 0`；
    - decode / PD 混部场景下 `m ∈ [0, 7]`，对应 `num_accepted_tokens` 的值。
  - `cache_indices` 中的元素值不保证升序排列，值不重复（除非等于 `pad_slot_id`），需保证最大值小于等于 `conv_states.shape[0]`。
  - `query_start_loc` 的元素值为升序，取值范围为 `0 ~ cuSeqLen`，首项必须为 0，末项必须等于 `cuSeqLen`。
  - `num_accepted_tokens` 的元素值取值范围为 `1 ~ seqlen`。
  - `x` 和 `conv_states` 在非连续情况下，最后一维（dim 维）的 stride 必须为 1。

- **其他输入参数限制**
  - `block_size` 必须大于等于 2。
  - APC 开启时（`cache_indices` 为 2D），必须提供 `block_idx_first_scheduled_token`、`block_idx_last_scheduled_token` 及 `initial_state_idx`，且对每个 batch $i$：
    - `initial_state_idx[i] <= block_idx_first_scheduled_token[i] + 1`
    - `initial_state_idx[i] <= block_idx_last_scheduled_token[i]`
    - `block_idx_first_scheduled_token[i] <= block_idx_last_scheduled_token[i]`
  - `num_accepted_tokens` 不为 None 时，值需满足：`num_accepted_tokens[i] >= 1`。
  - Pangu V2 模式（`conv_mode = 1`）下，首次运行 `num_computed_tokens` 不能为 None。

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `cuSeqLen`（2D 总 token 数） | 1 ~ 2097152 | 文档约束 |
| `batch`（batch 数） | 1 ~ 4096 | 文档约束 |
| `seqLen`（3D 单 batch 长度） | 1 ~ 2097152 | cases 实测范围 |
| `dim`（特征维度） | 64 ~ 16384 | 必须为 16 的倍数 |
| `K`（卷积核宽度） | 3 / 4 / 5 / 6 | - |
| `stateLen` | K-1 ~ K-1+7 | prefill 取 K-1，decode/PD 取 K-1+m |
| `block_size` | 128 / 256 | 典型值 |
| `conv_mode` | 0 / 1 | 0=Qwen3-Next，1=Pangu V2 |
| `residual_connection` | 0 / 1 | - |
| `inplace` | False / True | - |
| `activation` | "none" / "silu" / "swish" | 实际仅 silu 生效 |
| `max_query_len` | -1 或 1 ~ 1048576 | -1 表示自动推导 |
| `pad_slot_id` | -1 或有效 slot 外值 | 用于跳过无效 batch |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}\left(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}}\right)
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max\left(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}}\right)
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

```python
import random
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int):
    """设置全局随机种子，保证跨机器确定性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fused_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    pad_slot_id: int = -1,
    run_mode: int = 0,
    max_query_len: int = -1,
    residual_connection: int = 1,
    block_size: int = 128,
    conv_mode: int = 1,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FusedCausalConv1d 的 Torch Golden 参考实现入口。"""
    act = None
    if activation is not None and activation.lower() not in ("none", ""):
        if activation.lower() in ("silu", "swish"):
            act = "silu"
        else:
            raise NotImplementedError(f"activation {activation} not supported")

    return causal_conv1d_golden(
        x=x,
        weight=weight,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        max_query_len=max_query_len,
        pad_slot_id=pad_slot_id,
        num_accepted_tokens=num_accepted_tokens,
        num_computed_tokens=num_computed_tokens,
        block_idx_first_scheduled_token=block_idx_first_scheduled_token,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        B_size=block_size,
        conv_mode=conv_mode,
        inplace=inplace,
        residual=bool(residual_connection),
        activation=act,
    )


def causal_conv1d_golden(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    pad_slot_id: int = -1,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    B_size: int = 128,
    conv_mode: int = 1,
    inplace: bool = False,
    residual: bool = True,
    activation: Optional[str] = None,
) -> tuple:
    """FusedCausalConv1d 的 CPU Golden 参考实现。"""
    if x.ndim == 3:
        flattened = True
        bsz, seq_len_3d, dim = x.shape
        x = x.view(-1, dim)
        if query_start_loc is None:
            query_start_loc = torch.arange(
                0, (bsz + 1) * seq_len_3d, step=seq_len_3d,
                dtype=torch.int32, device=x.device
            )
    else:
        flattened = False

    cu_seq_len, dim = x.shape
    batch_size = query_start_loc.shape[0] - 1
    width = weight.size(0)
    assert conv_states.size(1) >= width - 1

    apc_enabled = block_idx_last_scheduled_token is not None
    out = torch.zeros_like(x)

    for batch_idx in range(batch_size):
        start_idx = query_start_loc[batch_idx].item()
        end_idx = query_start_loc[batch_idx + 1].item()
        seq_len = end_idx - start_idx
        seq_x = x[start_idx:end_idx]

        if seq_len == 0:
            continue

        if apc_enabled:
            seq_completed_offset_token = num_computed_tokens[batch_idx].item() % B_size
            seq_completed_offset = B_size - seq_completed_offset_token
            seq_end_offset = (seq_len - seq_completed_offset) % B_size
            last_full_block_token_index = seq_len - seq_end_offset
            if seq_end_offset == 0:
                last_full_block_token_index -= B_size
            idx_first = block_idx_first_scheduled_token[batch_idx].item()
            idx_last = block_idx_last_scheduled_token[batch_idx].item()
            n_block_to_fill = idx_last - idx_first

            assert cache_indices is not None and cache_indices.ndim == 2
            read_cache_line = cache_indices[batch_idx, initial_state_idx[batch_idx]].item()
            write_cache_line = cache_indices[batch_idx, idx_last].item()
        else:
            if cache_indices is not None:
                read_cache_line = cache_indices[batch_idx].item()
                write_cache_line = cache_indices[batch_idx].item()
            else:
                read_cache_line = batch_idx
                write_cache_line = batch_idx

        if read_cache_line == pad_slot_id:
            continue

        # Step 1: 读取历史 cache
        if num_computed_tokens is not None and num_computed_tokens[batch_idx] == 0:
            cached_state = torch.zeros((width - 1, dim), device=x.device, dtype=x.dtype)
            offset = 0
        else:
            if num_accepted_tokens is not None:
                accepted_tokens = num_accepted_tokens[batch_idx].item()
                assert 1 <= accepted_tokens <= seq_len
                offset = accepted_tokens - 1
            else:
                offset = conv_states.size(1) - (width - 1)
            cached_state = conv_states[read_cache_line][:offset + width - 1]

        padded_input = torch.cat([cached_state, seq_x], dim=0)

        # Step 2: 写入 running cache
        cache_len = min(conv_states.size(1), padded_input.size(0))
        conv_states[write_cache_line][-cache_len:] = padded_input[-cache_len:]

        padded_input = padded_input[offset:]

        # Step 2b: 写入 prefix cache（APC 模式）
        if apc_enabled:
            for chunk in range(n_block_to_fill):
                boundary_idx = (
                    last_full_block_token_index - (n_block_to_fill - chunk - 1) * B_size
                )
                assert boundary_idx > 0
                wc = cache_indices[batch_idx, idx_first + chunk]
                conv_states[wc][-(width - 1):] = padded_input[boundary_idx : boundary_idx + width - 1]

        # Step 3: 因果卷积
        result = F.conv1d(
            padded_input.transpose(0, 1).unsqueeze(0),
            weight.transpose(0, 1).unsqueeze(1),
            bias=None, stride=1, padding=0, groups=dim,
        ).squeeze(0).transpose(0, 1)

        # Pangu v2：将初始填充段置零
        if conv_mode == 1:
            assert num_computed_tokens is not None
            last_reset_idx = width - 1 - num_computed_tokens[batch_idx].item()
            last_reset_idx = min(max(last_reset_idx, 0), seq_len)
            result[:last_reset_idx] = 0

        result = result + seq_x if residual else result

        if activation is not None:
            if activation not in [None, "silu"]:
                raise NotImplementedError("activation only supports None or 'silu'")
            result = F.silu(result)

        out[start_idx:end_idx] = result
        if inplace:
            x[start_idx:end_idx] = out[start_idx:end_idx]

    if inplace:
        return (x if not flattened else x.view(bsz, -1, dim), conv_states)
    return (out if not flattened else out.view(bsz, -1, dim), conv_states)


def FusedCausalConv1d_Golden(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    pad_slot_id: int = -1,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    B_size: int = 128,
    conv_mode: int = 1,
    inplace: bool = False,
    residual: bool = True,
    activation: Optional[str] = None,
) -> tuple:
    """FusedCausalConv1d Golden 统一入口。"""
    return causal_conv1d_golden(
        x=x, weight=weight, conv_states=conv_states,
        query_start_loc=query_start_loc, cache_indices=cache_indices,
        max_query_len=max_query_len, pad_slot_id=pad_slot_id,
        num_accepted_tokens=num_accepted_tokens,
        num_computed_tokens=num_computed_tokens,
        block_idx_first_scheduled_token=block_idx_first_scheduled_token,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        B_size=B_size, conv_mode=conv_mode, inplace=inplace,
        residual=residual, activation=activation,
    )
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

dim = 512
width = 3
batch = 4
seq_len = 16
cu_seq_len = batch * seq_len
block_size = 128

dtype = torch.bfloat16
device = "npu"

x = torch.randn(cu_seq_len, dim, dtype=dtype, device=device)
weight = torch.randn(width, dim, dtype=dtype, device=device)
state_len = width - 1
conv_states = torch.randn(100, state_len, dim, dtype=dtype, device=device)

query_start_loc = torch.arange(0, cu_seq_len + 1, seq_len, dtype=torch.int32, device=device)
cache_indices = torch.arange(batch, dtype=torch.int32, device=device)
num_computed_tokens = torch.zeros(batch, dtype=torch.int32, device=device)

y, conv_states = cann_bench.fused_causal_conv1d(
    x, weight, conv_states,
    query_start_loc=query_start_loc,
    cache_indices=cache_indices,
    num_computed_tokens=num_computed_tokens,
    activation="silu",
    pad_slot_id=-1,
    max_query_len=-1,
    residual_connection=1,
    block_size=block_size,
    conv_mode=1,
    inplace=False,
)
```
