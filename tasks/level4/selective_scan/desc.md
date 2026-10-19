# SelectiveScan 算子 API 描述

## 1. 算子简介

Mamba 架构（S6，Selective State Space Model）的核心算子。状态空间模型以线性时间复杂度完成序列混合，是长序列场景下注意力机制的高效替代；S6 在经典 SSM 之上引入"选择性"机制——离散化步长 delta、输入投影 B 和输出投影 C 均为输入依赖的张量，使模型能够按内容选择记忆或遗忘。SelectiveScan 将离散化、时间维递归扫描、输出投影、跳跃连接与 SiLU 门控融合为单一算子。

**主要应用场景**：
- Mamba-130M/1.4B/2.8B 等状态空间语言模型的核心计算（每层一次）
- 长序列建模（基因组、音频、时序）中替代注意力的线性复杂度序列混合
- Mamba/Transformer 混合架构（如 Jamba）的 SSM 分支

**算子特征**：
- 难度等级：L4（FusedComposite）
- 七输入单输出，融合逐元素离散化（exp）、时间维一阶线性递归、状态维缩约与 SiLU 门控
- 时间维 L 存在真实数据依赖（h_t 依赖 h_{t-1}），kernel 侧需用分块扫描（chunk-wise scan）或并行前缀扫描（Blelloch scan）打破串行瓶颈，这是本算子的主要难度来源
- 状态维 N 很小（典型 16），单步计算强度低，递归调度与访存组织是优化关键
- delta 严格为正、A 严格为负（由 value_range 保证），使 dA = exp(delta·A) ∈ (0, 1)，递归收缩、数值稳定

## 2. 算子定义

### 数学公式

**离散化**（ZOH 简化形式，逐元素）：

$$
\overline{A}[b,l,d,n] = \exp(\Delta[b,l,d] \cdot A[d,n]) \qquad
\overline{B}u[b,l,d,n] = \Delta[b,l,d] \cdot B[b,l,n] \cdot u[b,l,d]
$$

**时间维递归扫描**（$h_0 = 0$，沿 $l = 1 \dots L$）：

$$
h_l = \overline{A}_l \odot h_{l-1} + \overline{B}u_l
$$

**输出投影 + 跳跃连接**（状态维 N 缩约）：

$$
y[b,l,d] = \sum_{n=1}^{N} C[b,l,n] \cdot h[b,l,d,n] + D_{skip}[d] \cdot u[b,l,d]
$$

**SiLU 门控**：

$$
y = y \odot \text{SiLU}(z), \qquad \text{SiLU}(z) = z \cdot \sigma(z)
$$

### 计算子步骤

1. **离散化 dA**：`dA = exp(einsum('bld,dn->bldn', delta, A))`，将连续域状态转移矩阵 A 按输入依赖步长 delta 离散化
2. **离散化 dBu**：`dBu[b,l,d,n] = delta[b,l,d] * B_mat[b,l,n] * u[b,l,d]`，输入注入项
3. **递归扫描**：沿时间维 `h_t = dA_t ⊙ h_{t-1} + dBu_t`，隐状态 shape [B, D, N]
4. **输出投影**：`y_t = einsum('bdn,bn->bd', h_t, C_t)`，状态维缩约回特征空间
5. **跳跃连接与门控**：`y = (y + D_skip * u) * SiLU(z)`

### 并行化特点

- 递归为**一阶线性递归**：$h_l = a_l h_{l-1} + b_l$ 满足结合律（$(a_1,b_1)\circ(a_2,b_2)=(a_1 a_2,\ a_2 b_1 + b_2)$），可用并行前缀扫描或分块串行 + 块间状态传递实现
- dA/dBu 为 [B, L, D, N] 的四维中间量，融合实现应逐块生成、避免落盘
- delta > 0 且 A < 0 保证 dA ∈ (0, 1)，任意扫描顺序（串行 / 分块 / 树形）均数值稳定

## 3. 接口规范

### 算子原型

```python
selective_scan(Tensor u, Tensor delta, Tensor A, Tensor B_mat, Tensor C_mat, Tensor D_skip, Tensor z) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| u | Tensor | 是 | bfloat16/float16/float32 | [B, L, D] | 输入序列特征 |
| delta | Tensor | 是 | 与 u 一致 | [B, L, D] | 输入依赖的离散化步长 Δ，**严格为正**（评测 value_range [0.001, 0.1]） |
| A | Tensor | 是 | 与 u 一致 | [D, N] | 连续域状态转移矩阵，**严格为负**（评测 value_range [-1, -0.01]） |
| B_mat | Tensor | 是 | 与 u 一致 | [B, L, N] | 输入依赖的输入投影矩阵 B |
| C_mat | Tensor | 是 | 与 u 一致 | [B, L, N] | 输入依赖的输出投影矩阵 C |
| D_skip | Tensor | 是 | 与 u 一致 | [D] | 跳跃连接缩放系数 |
| z | Tensor | 是 | 与 u 一致 | [B, L, D] | SiLU 门控分支输入 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| y | 与 u 一致 | [B, L, D] | 选择性扫描输出序列 |

### 数据类型

| u/delta/A/B_mat/C_mat/D_skip/z dtype | 输出 dtype | 内部计算 |
|--------------------------------------|-----------|---------|
| bfloat16 | bfloat16 | fp32 |
| float16 | float16 | fp32 |
| float32 | float32 | fp32 |

### 规则与约束

- 七个输入的 dtype 必须一致，输出 dtype 与输入一致
- delta 必须严格为正、A 必须严格为负：二者共同保证 dA = exp(delta·A) ∈ (0, 1)，递归收缩、任意扫描顺序数值稳定（评测框架按 value_range 独立随机生成输入，上述符号约束由 value_range 保证）
- u、delta、z 的 shape 必须完全一致（[B, L, D]）；B_mat、C_mat 的 shape 必须完全一致（[B, L, N]）
- A 的第一维、D_skip 的长度必须等于 D；A 的第二维为状态维 N
- 隐状态初值 h_0 = 0，不作为输入
- 递归沿时间维 L 顺序执行，输出 y[:, l] 只依赖 l' ≤ l 的输入（因果性）
- 内部计算（exp、递归累加、缩约、门控）全程 fp32，低精度输入升精度计算后输出转回原 dtype

### 支持范围

| 维度 / 参数 | 支持值 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 32 | 热点区间 1 ~ 8 |
| `L`（序列长度） | 1 ~ 8192 | 热点枚举 512 / 1024 / 2048 / 4096；含非对齐值 |
| `D`（特征维，模型 d_inner） | 1 ~ 8192（含非对齐 / 质数值） | 热点枚举 {768, 2048, 5120}：Mamba-130M / Mamba-1.4B / Mamba-2.8B |
| `N`（状态维 d_state） | 1 ~ 256 | 热点值 16（Mamba 系列默认）；kernel 不得硬编码 N=16 |
| `delta` 取值 | [0.001, 0.1] | 必须严格为正 |
| `A` 取值 | [-1, -0.01] | 必须严格为负 |
| 其余输入取值 | [-1, 1] 典型 | u / B_mat / C_mat / D_skip / z；鲁棒性 case 使用有限变体（低至 ±1e-4、高至 ±1e4） |
| dtype | bfloat16 / float16 / float32 | cases.csv 三种均覆盖 |

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


def _selective_scan_core(u, delta, A, B_mat, C_mat, D_skip, z, compute_dtype):
    """核心计算：以 compute_dtype 精度执行离散化 + 递归扫描 + 门控。"""
    u_f = u.to(compute_dtype)
    delta_f = delta.to(compute_dtype)
    A_f = A.to(compute_dtype)
    B_f = B_mat.to(compute_dtype)
    C_f = C_mat.to(compute_dtype)
    D_f = D_skip.to(compute_dtype)
    z_f = z.to(compute_dtype)

    Bsz, L, D = u_f.shape
    N = A_f.shape[1]

    # 时间维递归扫描：逐步计算，避免物化 [B, L, D, N] 的 dA/dBu 大张量
    h = torch.zeros(Bsz, D, N, dtype=compute_dtype, device=u.device)
    ys = []
    for t in range(L):
        # 离散化: dA_t = exp(delta_t ⊗ A), dBu_t = delta_t ⊗ B_t ⊗ u_t
        dA_t = torch.exp(delta_f[:, t].unsqueeze(-1) * A_f)                          # [B, D, N]
        dBu_t = (delta_f[:, t] * u_f[:, t]).unsqueeze(-1) * B_f[:, t].unsqueeze(1)   # [B, D, N]
        # 递归: h_t = dA_t ⊙ h_{t-1} + dBu_t
        h = dA_t * h + dBu_t
        # 输出投影: y_t = C_t · h_t（状态维 N 缩约）
        ys.append(torch.einsum('bdn,bn->bd', h, C_f[:, t]))                          # [B, D]
    y = torch.stack(ys, dim=1)                                                       # [B, L, D]

    # 跳跃连接 + SiLU 门控
    y = y + D_f * u_f
    y = y * (z_f * torch.sigmoid(z_f))
    return y


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    D_skip: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """
    Mamba S6 选择性状态空间扫描（plain golden = bench：内部 fp32 计算）

    Args:
        u: [B, L, D] 输入序列特征
        delta: [B, L, D] 离散化步长 Δ（严格为正）
        A: [D, N] 连续域状态转移矩阵（严格为负）
        B_mat: [B, L, N] 输入投影矩阵（输入依赖）
        C_mat: [B, L, N] 输出投影矩阵（输入依赖）
        D_skip: [D] 跳跃连接缩放系数
        z: [B, L, D] SiLU 门控分支输入

    Returns:
        y: [B, L, D] 选择性扫描输出，dtype 与输入一致
    """
    y = _selective_scan_core(u, delta, A, B_mat, C_mat, D_skip, z, torch.float32)
    return y.to(u.dtype)


def selective_scan_oracle(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    D_skip: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _selective_scan_core(u, delta, A, B_mat, C_mat, D_skip, z, u.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, L, D, N = 4, 1024, 768, 16  # Mamba-130M 配置

u = torch.randn(B, L, D, dtype=torch.bfloat16, device="npu")
delta = torch.empty(B, L, D, dtype=torch.bfloat16, device="npu").uniform_(0.001, 0.1)
A = torch.empty(D, N, dtype=torch.bfloat16, device="npu").uniform_(-1.0, -0.01)
B_mat = torch.randn(B, L, N, dtype=torch.bfloat16, device="npu")
C_mat = torch.randn(B, L, N, dtype=torch.bfloat16, device="npu")
D_skip = torch.randn(D, dtype=torch.bfloat16, device="npu")
z = torch.randn(B, L, D, dtype=torch.bfloat16, device="npu")

y = selective_scan(u, delta, A, B_mat, C_mat, D_skip, z)
# y.shape: [B, L, D]
```
