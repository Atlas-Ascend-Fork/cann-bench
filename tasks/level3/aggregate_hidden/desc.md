# AggregateHidden 算子 API 描述

## 1. 算子简介

对hidden层的token之间进行一维分组因果卷积操作，卷积窗口大小固定为3。该算子主要用于大语言模型推理中的Mamba/状态空间模型场景，对隐藏状态进行滑动窗口聚合。

**主要应用场景**：
- LLM 推理中的 Mamba/状态空间模型
- 隐藏层 token 间的一维分组因果卷积
- 支持可选的掩码操作

**算子特征**：
- 难度等级：L3（Convolution）
- 3 输入（其中 mask 可选），1 输出，无属性参数
- 支持 ND 格式输入
- 确定性计算

## 2. 算子定义

### 数学公式

假设输入 input 和输出 output 的 shape 是 [S, B, H]，卷积权重 weight 的 shape 是 [W, H]，i 和 j 分别表示 S 和 B 轴的索引，那么输出将被表示为：

$$
output[i,j] = mask[j,i] \times \sum_{k=0}^{W-1} input[i-k,j] \times weight[W-1-k]
$$

其中，无效位置的 padding 为 0 填充；当前 W 仅支持 3。

当 mask 为 None 时，mask[j,i] 视为 1（即不进行掩码操作）。

## 3. 接口规范

### 算子原型

```python
cann_bench.aggregate_hidden(Tensor input, Tensor weight, Tensor? mask=None) -> Tensor output
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
      <td>input</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示待计算的数据，对应公式中的input。</td>
      <td><ul><li>不支持空Tensor。<li>shape为[S, B, H]。</ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示卷积权重，对应公式中的weight。</td>
      <td><ul><li>不支持空Tensor。<li>shape为[W, H]。<li>W目前只支持3。<li>数据类型需与input一致。</ul></td>
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


### 输出说明

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
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

### 数据类型

| input dtype | weight dtype | mask dtype | output dtype |
|------------|-------------|-----------|-------------|
| bfloat16 | bfloat16 | bool | bfloat16 |
| float16 | float16 | bool | float16 |

### 规则与约束

- input、weight 和 output 的数据类型必须一致
- B（BatchSize）：取值范围为 1~8
- S（SeqLength）：取值范围为 1~32K
- H（HiddenSize）：取值范围为 384（192×2）~24576（192×128）
- W：当前仅支持 3
- 确定性计算

### 支持范围表

| 参数 | 最小值 | 最大值 | 说明 |
|------|-------|-------|------|
| S | 1 | 32768 | 序列长度 |
| B | 1 | 8 | 批大小 |
| H | 384 | 24576 | 隐藏维度(192的倍数) |
| W | 3 | 3 | 卷积窗口(固定) |

## 4. 精度要求

| 输入类型 | MERE 阈值 | MARE 阈值 |
|---------|----------|----------|
| bfloat16 | ≤ 2.0 | ≤ 1.2 |
| float16 | ≤ 2.0 | ≤ 1.2 |

精度对比方法：cv_fused_double_benchmark（以 float64 精度为基准）。

## 5. 标准 Golden 代码

```python
import torch
import torch.nn.functional as F

def aggregate_hidden(input: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    对 hidden 层的 token 进行一维分组因果卷积。

    output[i,j] = mask[j,i] * sum(k=0..W-1) input[i-k,j] * weight[W-1-k]

    参数:
        input: [S, B, H] 待计算数据, bfloat16/float16
        weight: [W, H] 卷积权重, W=3, 数据类型与input一致
        mask: [B, S] 可选掩码, bool, 默认None

    返回:
        output: [S, B, H] 卷积输出, 数据类型与input一致
    """
    S, B, H = input.shape
    W = weight.shape[0]

    input_fp64 = input.double()
    weight_fp64 = weight.double()

    # weight [W, H] -> [H, 1, W] for grouped Conv1d
    conv_weight = weight_fp64.t().unsqueeze(1)  # [H, 1, W]

    # input [S, B, H] -> [B, H, S]
    conv_input = input_fp64.permute(1, 2, 0)

    # causal padding: prepend W-1 zeros
    conv_input = torch.cat([
        torch.zeros((B, H, W - 1), device=input.device, dtype=torch.float64),
        conv_input
    ], dim=-1)  # [B, H, S + W - 1]

    # grouped 1D convolution
    conv_output = F.conv1d(conv_input, conv_weight, groups=H)  # [B, H, S]

    # [B, H, S] -> [S, B, H]
    output = conv_output.permute(2, 0, 1)

    # apply mask
    if mask is not None:
        # mask [B, S] -> [S, B]
        mask_sb = mask.t()
        output[~mask_sb] = 0

    return output.to(input.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench
S, B, H, W = 4096, 4, 768, 3
input = torch.randn(S, B, H, dtype=torch.bfloat16, device='npu')
weight = torch.randn(W, H, dtype=torch.bfloat16, device='npu')
mask = torch.ones(B, S, dtype=torch.bool, device='npu')

output = cann_bench.aggregate_hidden(input, weight, mask=mask)
# output shape: [S, B, H] = [4096, 4, 768]
```
