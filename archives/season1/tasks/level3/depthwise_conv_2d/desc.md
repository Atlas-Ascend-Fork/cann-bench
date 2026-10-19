# DepthwiseConv2D 算子 API 描述

## 1. 算子简介

二维深度卷积（Depthwise Convolution）运算。

**主要应用场景**：
- MobileNet、EfficientNet 等轻量级网络中的深度可分离卷积
- 边缘设备上的高效特征提取
- 图像处理中的逐通道滤波操作

**算子特征**：
- 难度等级：L3（Contraction）
- 三输入（特征图、卷积核、偏置）单输出，支持膨胀卷积
- 输入 x 为 [N, C, H, W]，卷积核 weight 为 [C, K_h, K_w]
- 是分组卷积的特例：约束 groups = C_in，且 C_out = C_in（channel_multiplier = 1），各通道独立卷积、不做跨通道求和

## 2. 算子定义

### 数学公式

对每个 batch n、通道 c、输出空间位置 (h, w)：

$$
y[n,c,h,w] = \text{bias}[c] + \sum_{k_h=0}^{K_h-1}\sum_{k_w=0}^{K_w-1} x[n,\,c,\,h\cdot s_h + k_h\cdot d_h - p_h,\,w\cdot s_w + k_w\cdot d_w - p_w]\cdot \text{weight}[c,0,k_h,k_w]
$$

其中 $s_h, s_w$ 为 stride、$p_h, p_w$ 为 padding、$d_h, d_w$ 为 dilation。
与标准 2D 卷积的关键区别：求和号中**没有跨输入通道的累加**，每个输出通道 c 只由对应的输入通道 c 与该通道独立的核 weight[c] 卷积得到（即 PyTorch / cuDNN 中 `groups = C_in = C_out` 的分组卷积特例）。

## 3. 接口规范

### 算子原型

```python
cann_bench.depthwise_conv_2d(Tensor x, Tensor weight, Tensor bias, int[] kernelSize, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图，shape 为 [N, C, H, W] |
| weight | Tensor | 必选 | 卷积核，shape 为 [C, K_h, K_w]（每个通道一个独立的 K_h×K_w 核） |
| bias | Tensor | 必选 | 偏置，shape 为 [C] |
| kernelSize | int[] | 必选 | 卷积核大小 [K_h, K_w]，须与 weight.shape[2:] 一致 |
| stride | int[] | 必选 | 步长 [s_h, s_w] |
| padding | int[] | 必选 | 填充 [p_h, p_w] |
| dilation | int[] | 必选 | 膨胀率 [d_h, d_w] |
| groups | int | 必选 | 分组数，**必须等于 C**（depthwise 定义性约束） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [N, C, H_out, W_out] | 与输入 x 相同 | 输出特征图 |

输出空间维度：
$$
H_{out} = \left\lfloor \frac{H + 2 p_h - d_h(K_h - 1) - 1}{s_h} \right\rfloor + 1, \quad
W_{out} = \left\lfloor \frac{W + 2 p_w - d_w(K_w - 1) - 1}{s_w} \right\rfloor + 1
$$

### 数据类型

| 输入 (x, weight, bias) dtype | 输出 dtype |
|-----------------------------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- x 的 shape 格式为 [N, C, H, W]
- weight 的 shape 格式为 [C, K_h, K_w]（每输入通道一个独立的 K_h×K_w 核，不跨通道累加）
- bias 的 shape 为 [C]
- **groups 必须等于 C（输入通道数）**；同时输出通道数 C_out = C_in = C（channel_multiplier = 1），这是 depthwise convolution 的定义性约束，违反则退化为普通分组卷积
- x、weight、bias 的 dtype 须一致
- kernelSize 须与 weight.shape[1:] = [K_h, K_w] 一致
- stride 指定卷积步长 [s_h, s_w]
- padding 指定对称填充 [p_h, p_w]
- dilation 指定膨胀率 [d_h, d_w]

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `N`（batch） | 1 ~ 256 | cases.csv 实测 2 ~ 8 |
| `C`（通道；=`C_in`=`C_out`=`groups`） | 1 ~ 2048 | cases.csv 实测 7 ~ 2048；depthwise 定义性约束 |
| `H`, `W`（空间） | 8 ~ 256 | cases.csv 实测 13 ~ 128 |
| `K_h`, `K_w`（卷积核） | 1 ~ 16 | cases.csv 实测 1 / 3 / 5 |
| `stride[i]` | 1 ~ 4 | cases.csv 实测 (1,1) 和 (2,2) |
| `padding[i]` | 0 ~ 8 | cases.csv 实测 0 ~ 2 |
| `dilation[i]` | 1 ~ 16 | cases.csv 实测 1 / 2 / 3 |
| `groups` | 1 ~ 2048 | **必须等于 `C`**（depthwise 定义性约束） |

约束：输出 spatial 维度 `(H_out, W_out)` 必须 ≥ 1（按 §3 输出公式计算）。

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
DepthwiseConv2D算子Torch Golden参考实现

二维深度卷积运算（每个输入通道独立卷积，groups = C_in = C_out）
y[n,c,h,w] = bias[c] + Σ_{kh,kw} x[n,c,h·s+kh·d-p,w·s+kw·d-p] · weight[c,kh,kw]
"""
def depthwise_conv_2d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, kernelSize: list, stride: list, padding: list, dilation: list, groups: int
) -> torch.Tensor:
    """
    二维深度卷积运算
    
    公式: y = bias + weight * x
    
    Args:
        x: 输入特征图
        weight: 卷积核
        bias: 偏置
        kernelSize: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        groups: 分组数
    
    Returns:
        输出特征图
    """

    stride_val = (stride[0], stride[1])
    padding_val = (padding[0], padding[1])
    dilation_val = (dilation[0], dilation[1])

    # spec 的 weight shape 是 [C, K_h, K_w]；torch.conv2d 要 [C_out, C_in/groups, K_h, K_w]，
    # depthwise 下 C_in/groups = 1，unsqueeze 这个冗余维供 PyTorch API 用。
    weight = weight.unsqueeze(1)

    y = torch.nn.functional.conv2d(x, weight, bias, stride=stride_val, padding=padding_val, dilation=dilation_val, groups=groups)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1, 64, 56, 56, dtype=torch.float16, device="npu")
weight = torch.randn(64, 3, 3, dtype=torch.float16, device="npu")
bias = torch.randn(64, dtype=torch.float16, device="npu")

y = cann_bench.depthwise_conv_2d(x, weight, bias, kernelSize=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64)
```
