# Conv3DBackpropFilter 算子 API 描述

## 1. 算子简介

Conv3D的filter梯度。

**主要应用场景**：
- 3D 卷积神经网络训练中的反向传播
- 视频理解模型中 Conv3D 层的权重梯度计算
- 医学影像 3D 分割模型的训练过程

**算子特征**：
- 难度等级：L3（Contraction）
- 双输入（输入特征图和输出梯度）单输出（filter 梯度）
- 输入 x 为 [N, C_in, D, H, W] 5维张量
- 输入 grad 为 [N, C_out, D_out, H_out, W_out] 5维张量

## 2. 算子定义

### 数学公式

$$
y = \text{conv3d\_filter\_grad}(x, \text{grad}, \text{filter\_size})
$$

计算 Conv3D 操作中卷积核（filter）的梯度。给定前向传播的输入特征图 $x$ 和来自下游的输出梯度 $\text{grad}$，通过反向传播计算得到 filter 的梯度 $y$。

### 输出 shape 计算

输出 filter 梯度的 shape 由 `filter_size` 参数指定：

$$
\text{shape}(y) = [C_{out}, C_{in}/groups, K_d, K_h, K_w]
$$

其中 grad 的 spatial 维度需满足：

$$
D_{out} = \frac{D_{in} + 2 \cdot \text{pad}_d - \text{dilation}_d \cdot (K_d - 1) - 1}{\text{stride}_d} + 1
$$

## 3. 接口规范

### 算子原型

```python
cann_bench.conv_3d_backprop_filter(Tensor x, Tensor grad, int[] strides, int[] pads, int[] dilations, int groups, int[] filter_size) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图，shape 为 [N, C_in, D, H, W] |
| grad | Tensor | 必选 | 输出梯度，shape 为 [N, C_out, D_out, H_out, W_out] |
| strides | int[] | 必选 | 步长，3元素 [stride_d, stride_h, stride_w] |
| pads | int[] | 必选 | 填充，6元素格式 [D_front, D_back, H_top, H_bottom, W_left, W_right] |
| dilations | int[] | 必选 | 膨胀率，3元素 [dilation_d, dilation_h, dilation_w] |
| groups | int | 1 | 分组数 |
| filter_size | int[] | 必选 | filter的shape [C_out, C_in/groups, K_d, K_h, K_w] |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [C_out, C_in/groups, K_d, K_h, K_w] | 与输入 x 相同 | filter梯度 |

### 数据类型

| 输入 (x, grad) dtype | 输出 dtype |
|---------------------|-----------|
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- x 的 shape 格式为 [N, C_in, D, H, W]
- grad 的 shape 格式为 [N, C_out, D_out, H_out, W_out]
- x 和 grad 的 dtype 须一致
- strides 指定 3D 卷积的步长，为 3 元素列表
- pads 指定填充值，为 6 元素列表 [D_front, D_back, H_top, H_bottom, W_left, W_right]
- dilations 指定膨胀率，为 3 元素列表
- groups 指定分组数，C_in 和 C_out 都须能被 groups 整除
- filter_size 指定输出 filter 梯度的 shape
- grad 的 spatial 维度必须与 x、filter_size、strides、pads、dilations 计算的输出维度一致

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `N`（batch） | 1 ~ 16 | cases.csv 实测 2 ~ 5 |
| `C_in`（输入通道） | 1 ~ 256 | cases.csv 实测 7 ~ 128；须能被 `groups` 整除 |
| `C_out`（输出通道） | 1 ~ 512 | cases.csv 实测 14 ~ 256；须能被 `groups` 整除 |
| `D`（depth） | 4 ~ 32 | cases.csv 实测 4 ~ 17 |
| `H`, `W`（空间） | 8 ~ 256 | cases.csv 实测 13 ~ 128 |
| `K_d`（depth 卷积核） | 1 ~ 8 | cases.csv 实测 1 / 3 / 5 |
| `K_h`, `K_w`（空间卷积核） | 1 ~ 16 | cases.csv 实测 1 / 3 / 5 |
| `strides[i]` | 1 ~ 4 | cases.csv 实测 (1,1,1) 和 (2,2,2) |
| `pads[i]` | 0 ~ 8 | cases.csv 实测 0 ~ 2（对称） |
| `dilations[i]` | 1 ~ 16 | cases.csv 实测 1 / 2 |
| `groups` | 1 ~ 64 | cases.csv 实测 1 / 2 / 64；须同时整除 `C_in` 和 `C_out` |

约束：`grad` 的 spatial 维度 `(D_out, H_out, W_out)` 必须与正向 conv3d 由 `(x.shape, filter_size, strides, pads, dilations)` 计算出的输出维度一致（见 §2 公式）。

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
import torch.nn.functional as F

"""
Conv3DBackpropFilter算子Torch Golden参考实现

Conv3D的filter梯度
公式: y = conv3d_filter_grad(x, grad, filter_size)
"""
def conv_3d_backprop_filter(
    x: torch.Tensor, grad: torch.Tensor, strides: list, pads: list, dilations: list, groups: int = 1, filter_size: list = None
) -> torch.Tensor:
    """
    Conv3D的filter梯度

    公式: y = conv3d_filter_grad(x, grad, filter_size)

    Args:
        x: 输入特征图，shape为[N, C_in, D, H, W]
        grad: 输出梯度，shape为[N, C_out, D_out, H_out, W_out]
        strides: 步长，3元素 [stride_d, stride_h, stride_w]
        pads: 填充，6元素 [D_front, D_back, H_top, H_bottom, W_left, W_right]，对称时取front/top/left
        dilations: 膨胀率，3元素 [dilation_d, dilation_h, dilation_w]
        groups: 分组数
        filter_size: filter的shape [C_out, C_in/groups, K_d, K_h, K_w]

    Returns:
        filter梯度，shape与filter_size相同
    """

    # pads 是 6 元素格式，对称 padding 时取 (D_front, H_top, W_left)
    # 即 pads[0], pads[2], pads[4]
    padding = (pads[0], pads[2], pads[4])
    stride = (strides[0], strides[1], strides[2])
    dilation = (dilations[0], dilations[1], dilations[2])

    # 使用 torch.nn.grad.conv3d_weight 计算 filter 梯度
    y = F.grad.conv3d_weight(x, tuple(filter_size), grad, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(2, 64, 8, 16, 16, dtype=torch.float16, device="npu")
grad = torch.randn(2, 128, 6, 14, 14, dtype=torch.float16, device="npu")

# filter_size: [C_out, C_in/groups, K_d, K_h, K_w]
y = cann_bench.conv_3d_backprop_filter(x, grad, strides=[1, 1, 1], pads=[1, 1, 1, 1, 1, 1], dilations=[1, 1, 1], groups=1, filter_size=[128, 64, 3, 3, 3])
```
