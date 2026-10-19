# AggregateHiddenGrad 算子 API 描述

## 1. 算子简介

AggregateHiddenGrad 算子实现 hidden states 聚合操作的反向梯度计算。

**主要应用场景**：
- 深度学习自定义算子
- 模型训练与推理
- 特定领域计算加速

**算子特征**：
- 难度等级：L3（Convolution）
- 4 输入，2 输出，0 个属性参数
- 支持 ND 格式输入

## 2. 算子定义

### 数学公式

- 假定卷积输入input、卷积输出的梯度grad_output和卷积输入的梯度grad_input的shape是[S, B, H]，weight的shape是[W, H]，i和j分别表示S/B轴的索引，k为卷积窗口W内的索引，那么计算将被表示为：

  $$
  grad\_output\_masked[i,j] = mask[j,i] * grad\_output[i,j]
  $$

  $$
  grad\_input[i,j] = \sum_{k=0}^{W-1} grad\_output\_masked[i+k,j] * weight[W-1-k]
  $$

  $$
  grad\_weight[k] = \sum_{j=0}^{B-1}\sum_{i=0}^{S-1} grad\_output\_masked[i+W-1-k,j] * input[i,j]
  $$

  其中，无效位置的padding为0填充；当前W仅支持3。

### 特殊情况

| 输入 | 输出 |
|------|------|
| 各维度为 1 的退化 shape | 正常计算，输出 shape 与输入一致 |
| 空张量（某维度为 0） | 未定义行为，需避免 |

## 3. 接口规范

### 算子原型

```python
cann_bench.aggregate_hidden_grad(Tensor grad_output, Tensor input, Tensor weight, Tensor? mask=None) -> (Tensor grad_input, Tensor grad_weight)
```

### 输入参数说明

<table style="undefined;table-layout: fixed; width: 1565px">
  <colgroup>
    <col style="width: 146px">
    <col style="width: 135px">
    <col style="width: 326px">
    <col style="width: 246px">
    <col style="width: 275px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>grad_output</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示分组卷积输出output的梯度，对应公式中的grad_output。</td>
      <td><ul><li>不支持空Tensor。<li>shape为[S, B, H]。</ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示分组卷积输入，对应公式中的input。</td>
      <td><ul><li>不支持空Tensor。<li>shape和数据类型与grad_output一致。</ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示卷积权重，对应公式中的weight。</td>
      <td><ul><li>不支持空Tensor。<li>shape为[W, H]。<li>W目前只支持3。<li>数据类型需与grad_output一致。</ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>maskOptional</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示卷积操作的输出掩码，对应公式中的mask。</td>
      <td><ul><li>shape为[B, S]。<li>可选输入，默认值是None。</ul></td>
      <td>BOOL</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
  </tbody>
</table>


### 输出

<table style="undefined;table-layout: fixed; width: 1565px">
  <colgroup>
    <col style="width: 146px">
    <col style="width: 135px">
    <col style="width: 326px">
    <col style="width: 246px">
    <col style="width: 275px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>Device侧的aclTensor，表示分组卷积输入input的梯度，对应公式中的grad_input。</td>
      <td><ul><li>不支持空Tensor。<li>shape和数据类型与grad_output一致。</ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grad_weight</td>
      <td>输出</td>
      <td>Device侧的aclTensor，表示分组卷积输入weight的梯度，对应公式中的grad_weight。</td>
      <td><ul><li>不支持空Tensor。<li>shape为[W, H]。<li>W目前只支持3。<li>数据类型需与grad_output一致。</ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
  </tbody>
</table>


### 数据类型

| grad_output dtype | input dtype | weight dtype | mask dtype | grad_input dtype | grad_weight dtype |
|-------------------|-------------|--------------|------------|------------------|-------------------|
| float16 | float16 | float16 | bool | float16 | float16 |
| bfloat16 | bfloat16 | bfloat16 | bool | bfloat16 | bfloat16 |

### 规则与约束

- 输入 `grad_output`、`input` 形状为 `[T, B, D]`。
- 输入 `weight` 形状为 `[3, D]`。
- 可选输入 `mask` 形状为 `[B, T]`，数据类型为 bool。
- 输出 `grad_input` 形状与 `grad_output` 相同。
- 输出 `grad_weight` 形状与 `weight` 相同。

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `ndim`（输入维度数） | 2 ~ 3 | cases 实测范围 |
| `dim_0`（第0维大小） | 1 ~ 16384 | cases 实测范围 |
| `dim_1`（第1维大小） | 1 ~ 24576 | cases 实测范围 |
| `dim_2`（第2维大小） | 384 ~ 24576 | cases 实测范围 |
| `dtype` | bfloat16, bool, float16 | cases 实测覆盖 |

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
from torch import Tensor


def aggregate_hidden_grad(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    AggregateHiddenGrad 算子的 Torch Golden 参考实现。

    对 hidden states 聚合操作（宽度为 3 的因果卷积）进行反向梯度计算。

    Args:
        grad_output: 反向梯度输入，形状为 [T, B, D]。
        input: 前向输入，形状为 [T, B, D]。
        weight: 卷积权重，形状为 [3, D]。
        mask: 可选的掩码张量，形状为 [B, T]，数据类型为 bool。

    Returns:
        grad_input: 输入梯度，形状为 [T, B, D]。
        grad_weight: 权重梯度，形状为 [3, D]。
    """
    dtype = grad_output.dtype
    grad_output = grad_output.to(torch.float32)
    input = input.to(torch.float32)
    weight = weight.to(torch.float32)

    if mask is not None:
        grad_output = grad_output.clone()
        grad_output[~mask.transpose(0, 1)] = 0

    grad_input0 = grad_output * weight[0].unsqueeze(0).unsqueeze(0)
    grad_input1 = grad_output * weight[1].unsqueeze(0).unsqueeze(0)
    grad_input2 = grad_output * weight[2].unsqueeze(0).unsqueeze(0)

    grad_input2[:-1, :, :] += grad_input1[1:, :, :]
    grad_input2[:-2, :, :] += grad_input0[2:, :, :]

    grad_weight = torch.stack(
        [
            (grad_output[2:, :, :] * input[:-2, :, :]).sum(dim=0).sum(dim=0),
            (grad_output[1:, :, :] * input[:-1, :, :]).sum(dim=0).sum(dim=0),
            (grad_output[:, :, :] * input[:, :, :]).sum(dim=0).sum(dim=0),
        ],
        dim=0,
    )

    grad_input2 = grad_input2.to(dtype)
    grad_weight = grad_weight.to(dtype)

    return grad_input2, grad_weight
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

grad_output = torch.randn(4096, 4, 768, dtype=torch.float16, device="npu")
input = torch.randn(4096, 4, 768, dtype=torch.float16, device="npu")
weight = torch.randn(3, 768, dtype=torch.float16, device="npu")
mask = torch.ones(4, 4096, dtype=torch.bool, device="npu")
grad_input, grad_weight = cann_bench.aggregate_hidden_grad(grad_output, input, weight, mask=mask)
```
