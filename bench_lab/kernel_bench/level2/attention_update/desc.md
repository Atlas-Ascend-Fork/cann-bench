# AttentionUpdate 算子 API 描述

## 1. 算子简介

将各 SP（Sequence Parallel）域上 PA（Prompt Attention）/IFA（Incremental Flash Attention）算子输出的中间结果 lse 和 localOut 两个局部变量结果更新为全局结果。通过 Log-Sum-Exp 归约合并多个 SP 分区的局部 softmax 归一化因子与局部 attention 输出，得到全局 attention 输出。

**主要应用场景**：
- 序列并行（Sequence Parallel, SP）场景下的 attention 结果合并
- 大模型推理/训练中跨 SP 域的 softmax 归约与 attention output 加权求和
- DeepSeek 等 MoE 模型中的 MLA（Multi-head Latent Attention）后处理

**算子特征**：
- 难度等级：L2（Attention）
- 动态输入（tensorList），支持 2 输出，2 个属性参数
- 支持 ND 格式输入
- 属性：update_type（控制是否输出 lseOut）、sp（序列并行度）
- 输入 localOut 支持 FLOAT16 和 BFLOAT16，lse 固定为 FLOAT32

## 2. 算子定义

### 数学公式

输入 $lse_i$（各 SP 域的局部 log-sum-exp）和 $O_i$（各 SP 域的局部 attention output），合并得到全局结果 $O$ 和 $lse_m$：

$$
lse_{max} = \max_i lse_i
$$

$$
lse = \sum_i \exp(lse_i - lse_{max})
$$

$$
lse_m = lse_{max} + \log(lse)
$$

$$
O = \sum_i O_i \cdot \exp(lse_i - lse_m)
$$

其中 $i \in [0, sp-1]$ 表示不同 SP 域的索引。

### 功能说明

- 当 `update_type = 0` 时：仅输出全局 attention output `out`，不输出 `lseOut`（lseOut 为空 tensor）
- 当 `update_type = 1` 时：同时输出全局 attention output `out` 和全局 lse `lseOut`
- 支持空 tensor 输入（bsh = 0 的场景）

## 3. 接口规范

### 算子原型

```python
cann_bench.attention_update(Tensor lse, Tensor local_out, int update_type=0) -> (Tensor out, Tensor lse_out)
```

### 输入参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
|--------|----------|------|---------|---------|------------|
| lse | 输入 | 各 SP 域的局部 lse（stacked），每个 SP 域提供 shape [bsh] 的 FLOAT32 张量，共 sp 个，stack 后为 [sp, bsh] | FLOAT32 | ND | 2D |
| local_out | 输入 | 各 SP 域的局部 attention output（stacked），每个 SP 域提供 shape [bsh, headDim] 的张量，共 sp 个，stack 后为 [sp, bsh, headDim] | FLOAT16 / BFLOAT16 | ND | 3D |
| update_type | 属性 | 控制 lseOut 是否输出，0 表示不输出 lseOut，1 表示输出 lseOut | INT64 | - | - |

### 输出

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
|--------|----------|------|---------|---------|------------|
| out | 输出 | 更新后的全局 attention output | 与 local_out 一致（FLOAT16 / BFLOAT16） | ND | [bsh, headDim] |
| lse_out | 输出 | 更新后的全局 lse（update_type=0 时为空 tensor） | FLOAT32 | ND | [bsh] |

### 数据类型

| lse dtype | local_out dtype | out dtype | lse_out dtype |
|-----------|----------------|-----------|--------------|
| float32 | float16 | float16 | float32 |
| float32 | bfloat16 | bfloat16 | float32 |

### 规则与约束

- lse 仅支持 float32 类型，local_out / out 支持 float16 和 bfloat16 类型
- local_out 的 headDim（最后一维）取值范围为 [8, 512] 且必须是 8 的倍数
- sp（stack 后的第一维）表示序列并行度，Atlas A2/A3 产品支持 [1, 128]，Ascend 950 系列支持 [1, 16]
- 所有张量的 bsh（batch × seqLen × headNum）维度必须一致
- 不支持非连续 Tensor
- 支持空 Tensor（bsh = 0）

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| sp（序列并行度） | 1 ~ 16（950）/ 1 ~ 128（A2/A3） | lse 和 local_out 的第 0 维 |
| bsh（batch×seqLen×headNum） | 1 ~ 512000 | lse 的第 1 维，local_out 的第 1 维 |
| headDim | 8 ~ 512，且被 8 整除 | local_out 的第 2 维 |
| lse dtype | float32 | 仅支持 float32 |
| local_out dtype | float16, bfloat16 | 仅支持 float16 和 bfloat16 |
| update_type | 0 或 1 | 0=不输出 lseOut, 1=输出 lseOut |

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

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from typing import List

import torch


def attention_update(
    lse: List[torch.Tensor], local_out: List[torch.Tensor], update_type: int = 0
):
    """
    将各 SP 域的局部 lse 和 localOut 更新为全局结果。

    公式:
        lse_max = max_i(lse_i)
        lse = sum_i(exp(lse_i - lse_max))
        lse_m = lse_max + log(lse)
        O = sum_i(O_i * exp(lse_i - lse_m))

    参数:
        lse: TensorList，每个 tensor shape 为 [bsh]，dtype 为 float32
        local_out: TensorList，每个 tensor shape 为 [bsh, head_dim]，
                   数据类型为 float16 或 bfloat16
        update_type: 0 表示不输出 lse_out（返回空 tensor），
                     1 表示输出 lse_out

    返回:
        (out, lse_out) 的元组:
        - out: [bsh, head_dim]，全局 attention output，数据类型与 local_out 一致
        - lse_out: [bsh] float32，全局 lse（update_type=0 时为空 tensor）
    """
    dtype = local_out[0].dtype

    # Stack tensor list → [sp, bsh] / [sp, bsh, head_dim]
    lse_stacked = torch.stack(lse, dim=0)  # [sp, bsh]
    local_out_stacked = torch.stack(local_out, dim=0).float()  # [sp, bsh, head_dim]

    # Step 1: lse_max = max_i(lse_i)
    lse_max, _ = torch.max(lse_stacked, dim=0)  # [bsh]

    # Step 2: lse = sum_i(exp(lse_i - lse_max))
    lse_sub = lse_stacked - lse_max.unsqueeze(0)  # [sp, bsh]
    lse_sub_exp = torch.exp(lse_sub)  # [sp, bsh]
    lse_sum = torch.sum(lse_sub_exp, dim=0)  # [bsh]

    # Step 3: lse_m = lse_max + log(lse)
    lse_out = lse_max + torch.log(lse_sum)  # [bsh]

    # Step 4: O = sum_i(O_i * exp(lse_i - lse_m))
    # Normalize shifted exponentials directly to avoid cancellation in lse_i - lse_out.
    lse_weight = (lse_sub_exp / lse_sum.unsqueeze(0)).unsqueeze(2)  # [sp, bsh, 1]
    out = torch.sum(local_out_stacked * lse_weight, dim=0)  # [bsh, head_dim]

    if update_type == 0:
        lse_out = torch.zeros(0)

    return out.to(dtype), lse_out
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

sp = 2
bsh = 256
head_dim = 128

lse = [torch.randn(bsh, dtype=torch.float32, device="npu") for _ in range(sp)]
local_out = [torch.randn(bsh, head_dim, dtype=torch.float16, device="npu") for _ in range(sp)]

out, lse_out = cann_bench.attention_update(lse, local_out, update_type=0)
# out shape: [256, 128]
# lse_out shape: torch.Size([0]) (update_type=0 时为空 tensor)
```
