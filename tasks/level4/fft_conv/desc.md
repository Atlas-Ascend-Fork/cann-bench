# FftConv 算子 API 描述

## 1. 算子简介

Hyena 长卷积算子：对每个通道以一条与序列等长的隐式滤波器做**因果全局线性卷积**，并叠加逐通道残差门控。这是 Hyena / H3 / 长卷积语言模型（Long Conv LM）中替代 attention 的核心算子——直接时域卷积复杂度为 O(D·L²)，实用实现必须走 FFT 路径：`y = irfft(rfft(u, 2L) ⊙ rfft(k, 2L))[..., :L] + u ⊙ bias`，复杂度降为 O(D·L·logL)。FFT 长度取 2L（零填充）保证循环卷积的前 L 点与因果线性卷积严格一致（避免循环混叠）。

**主要应用场景**：
- Hyena / StripedHyena、H3 等长卷积语言模型的 token mixing 层
- 超长序列（8k ~ 100k+）建模中以次二次复杂度替代 attention
- 基因组（HyenaDNA）、音频等长序列任务的全局卷积骨干

**算子特征**：
- 难度等级：L4（FusedComposite）
- 三输入（u, k_filter, bias）单输出（y）
- 融合 rfft × 2、频域复数逐点乘、irfft、截断与残差门控
- 仅支持 float32：FFT 蝶形运算对精度敏感，bf16/fp16 下误差随 logL 级联放大，无实用意义；kernel 内部为混合基 fp32 蝶形数据流

**为何是 L5**（约为 L4 FusedComposite 的 2 倍难度）：
- **NPU 无原生 FFT**：CANN/AscendC 没有 FFT 原语，kernel 需从零手写混合基（radix-2/radix-4，非 2 幂长度还需 radix-3/5 或 Bluestein）蝶形数据流——L4 融合算子的每个子步骤都有现成 CUBE/VEC 原语可用，本算子的核心计算本身就要重新发明
- **实数 FFT 的 pack/unpack**：利用实序列共轭对称性把实数 FFT 折半成复数 FFT（r2c pack、c2r unpack），涉及镜像索引与共轭组合的精细变换，写错任一项都只在部分频点出错、难以定位
- **位逆序重排与访存布局**：蝶形级间的位逆序（bit-reversal）/数位逆序重排是纯访存置换，需与 UB bank、连续搬运粒度协同设计，否则访存完全散乱
- **多级流水融合**：正变换（u 与 k_filter 两路）→ 复数逐点乘 → 逆变换 → 截断 + 残差门控，全链路在片上衔接，中间频域张量（复数、2L 长度）的驻留与切块策略复杂
- **迭代级数深**：L=16384 时 log2(2L) = 15 级蝶形，舍入误差逐级累积，精度控制（fp32 全程、twiddle 因子精度）本身就是设计约束

## 2. 算子定义

### 数学公式

因果线性卷积 + 残差门控（时域定义，即算子语义）：

$$
y[b, d, t] = \sum_{\tau=0}^{t} u[b, d, t-\tau] \cdot k[d, \tau] + u[b, d, t] \cdot \text{bias}[d]
$$

FFT 等价形式（Golden 与 kernel 的实际计算路径）：

$$
y = \text{irfft}\big(\text{rfft}(u, 2L) \odot \text{rfft}(k, 2L)\big)[\ldots, :L] + u \odot \text{bias}
$$

### 计算子步骤

1. **零填充正变换**：$U = \text{rfft}(u, n=2L)$，$K = \text{rfft}(k, n=2L)$，输出复数频谱 `[..., L+1]`（实数 FFT 只需半谱）
2. **频域逐点复乘**：$Y_{freq}[b, d, f] = U[b, d, f] \cdot K[d, f]$（K 沿 batch 广播）
3. **逆变换与截断**：$y_{conv} = \text{irfft}(Y_{freq}, n=2L)[\ldots, :L]$——零填充到 2L 保证前 L 点等于因果线性卷积（无循环混叠）
4. **残差门控**：$y = y_{conv} + u \odot \text{bias}$（bias 逐通道广播）

### 为何 FFT 长度必须为 2L

长度 n 的循环卷积在 $t < L$ 处混入 $\tau > t$ 的"回绕"项；取 $n \ge 2L - 1$ 后回绕项全部落在尾部 L 点，截取前 L 点即严格等于因果线性卷积。本算子固定 $n = 2L$（非 2 的幂 L 同样取 $n = 2L$，由混合基分解处理）。

## 3. 接口规范

### 算子原型

```python
fft_conv(Tensor u, Tensor k_filter, Tensor bias) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| u | Tensor | 是 | float32 | [B, D, L] | 输入序列（通道优先布局） |
| k_filter | Tensor | 是 | float32 | [D, L] | 时域全局滤波器（每通道一条长度 L 的隐式卷积核，沿 batch 广播） |
| bias | Tensor | 是 | float32 | [D] | 残差门控系数（逐通道） |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| y | float32 | [B, D, L] | 因果全局卷积输出 |

### 数据类型

| u/k_filter/bias dtype | 输出 dtype | 内部计算 |
|-----------------------|-----------|----------|
| float32 | float32 | fp32（含复数频域，等价 complex64） |

仅支持 float32：FFT 蝶形级联对舍入误差敏感（L=16384 时 15 级蝶形），bf16/fp16 输入无实用意义。

### 规则与约束

- u 与 k_filter 的通道维 D、序列维 L 必须一致；bias 长度必须等于 D
- FFT 长度固定为 n = 2L；L 为 2 的幂时可用纯 radix-2/4，非 2 的幂 L（如 1500，2L = 3000 = 2³·3·5³）需混合基分解或 Bluestein 算法
- 卷积语义为**因果**线性卷积（输出 t 只依赖输入 ≤ t 的位置），等价于 `conv1d(padding=L-1)` 后取前 L 点
- 三个输入相互独立，任意随机输入均合法（无隐式契约）

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 32 | cases.csv 实测 1 ~ 8 |
| `D`（通道数 / d_model） | 64 ~ 4096 | cases.csv 实测 768（Hyena-small）/ 2048（Hyena-1.3B） |
| `L`（序列长度） | 512 ~ 32768 | cases.csv 实测 1024 / 2048 / 8192 / 16384（2 的幂）及 1500 / 3000（非 2 幂，混合基） |
| dtype | float32 | 唯一支持 |
| 输入数值范围 | [-1, 1] 典型 | 鲁棒性 case 使用有限变体：幅值低至 [-1e-4, 1e-4]、高至 [-1e4, 1e4]，含逐输入尺度对比（如 u 正常 / k 微幅）；所有取值均为有限实数 |

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

**通过标准**（本算子放宽阈值）：

kernel 的混合基蝶形与 Golden（torch.fft，pocketfft 后端）的求和顺序差异较大，且输出每点为 L 项随机乘积之和（元素级相消不可避免），相对生态标准适度放宽：

| 数据类型 | FLOAT32 |
|----------|---------|
| **通过阈值(Threshold)** | 0.001 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。输出存在零穿越点，小值域与相消场景由评测框架的兜底标准处理。

## 5. 标准 Golden 代码

```python
import torch


def _fft_conv_core(u, k_filter, bias, compute_dtype):
    """核心计算：以 compute_dtype 精度执行 rfft → 频域逐点乘 → irfft → 残差门控。"""
    L = u.shape[-1]
    n = 2 * L  # 零填充到 2L，避免循环混叠

    u_f = u.to(compute_dtype)
    k_f = k_filter.to(compute_dtype)

    u_freq = torch.fft.rfft(u_f, n=n)                     # [B, D, L+1] complex
    k_freq = torch.fft.rfft(k_f, n=n)                     # [D, L+1] complex
    y = torch.fft.irfft(u_freq * k_freq, n=n)[..., :L]    # 频域逐点复乘 + iFFT，截取前 L 点

    y = y + u_f * bias.to(compute_dtype).view(1, -1, 1)   # 残差门控
    return y


def fft_conv(
    u: torch.Tensor,
    k_filter: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Hyena 长卷积 golden reference（plain golden = bench：fp32 计算）

    Args:
        u: [B, D, L] 输入序列（通道优先布局）
        k_filter: [D, L] 时域全局滤波器（每通道一条长度 L 的隐式卷积核）
        bias: [D] 残差门控系数（逐通道）

    Returns:
        y: [B, D, L] 因果全局卷积输出，dtype 与 u 一致（float32）
    """
    y = _fft_conv_core(u, k_filter, bias, torch.float32)
    return y.to(u.dtype)


def fft_conv_oracle(
    u: torch.Tensor,
    k_filter: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _fft_conv_core(u, k_filter, bias, u.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, D, L = 4, 768, 2048

u = torch.randn(B, D, L, dtype=torch.float32, device="npu")
k_filter = torch.randn(D, L, dtype=torch.float32, device="npu")
bias = torch.randn(D, dtype=torch.float32, device="npu")

y = fft_conv(u, k_filter, bias)
# y.shape: [B, D, L]
```

### 与时域直接卷积的等价性

小规模下可用 `torch.nn.functional.conv1d` 交叉验证（因果卷积 = 翻转核 + 左填充 L-1）：

```python
import torch.nn.functional as F

y_ref = F.conv1d(F.pad(u, (L - 1, 0)).reshape(1, B * D, -1),
                 k_filter.flip(-1).repeat(B, 1).unsqueeze(1),
                 groups=B * D).reshape(B, D, L) + u * bias.view(1, -1, 1)
# y_ref ≈ fft_conv(u, k_filter, bias)，fp32 容差内一致
```

### 参考文献

- Poli, M. et al. (2023). "Hyena Hierarchy: Towards Larger Convolutional Language Models". ICML 2023
- Fu, D. et al. (2023). "Hungry Hungry Hippos: Towards Language Modeling with State Space Models". ICLR 2023（H3，FFT 卷积路径）
