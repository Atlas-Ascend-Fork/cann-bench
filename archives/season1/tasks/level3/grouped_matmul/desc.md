# GroupedMatmul 算子 API 描述

## 1. 算子简介

分组矩阵乘法算子。激活 `x` 沿 `M` 轴合并为单 tensor `[M, K]`；权重 `weight` 按 expert 维堆叠为 `[E, K, N]`（或 `[E, N, K]`）；通过 `group_list`（cumsum 语义）描述每组 token 在 `M` 轴上的边界。

**主要应用场景**：
- MoE（Mixture of Experts）模型中多专家的并行矩阵运算
- 多头注意力机制中的分组线性变换
- 批量处理同一 K、N 维度、按 M 切分多组的矩阵乘法

**算子特征**：
- 难度等级：L3（Contraction）
- 输入 / 输出容器与 `level4/grouped_matmul_swiglu_quant` 对齐：`x` 单 tensor + `weight` 堆叠 + `group_list` 切组
- 所有组的 `K` 与 `N` 在 expert 维上一致，仅 `M` 各组可不同

## 2. 算子定义

### 数学公式

$$
\begin{aligned}
&\text{对每个专家 } g \in [0, E),\ \text{根据 } group\_list\ (\text{cumsum}) \text{ 取属于该组的 token 行 } rows_g: \\
&y[rows_g] = x[rows_g] \cdot weight[g] \ (+\ bias[g])
\end{aligned}
$$

其中：
- `x` shape 为 `[M, K]`，所有组沿 `M` 轴合并
- `weight` shape 为 `[E, K, N]`（`transpose_weight=false`）或 `[E, N, K]`（`transpose_weight=true`）
- `bias` shape 为 `[E, N]`（可选）
- `group_list` 长度 `E`，cumsum 语义，最后一个值等于 `M`
- `y` shape 为 `[M, N]`

### group_list 语义

`group_list = [c_0, c_1, ..., c_{E-1}]`，表示前 g+1 个 expert 累计接收 `c_g` 个 token。每组的行范围为 `[c_{g-1}, c_g)`（约定 `c_{-1} = 0`）。允许 `c_g == c_{g-1}`，表示该 expert 为空组（跳过计算）。

## 3. 接口规范

### 算子原型

```python
cann_bench.grouped_matmul(
    Tensor x,
    Tensor weight,
    Tensor? bias,
    int[] group_list,
    int split_item = 0,
    bool transpose_weight = False,
) -> Tensor[] y
```

### 输入参数

| 参数 | 类型 | Shape | 描述 |
|------|------|-------|------|
| x | Tensor | `[M, K]` | 激活矩阵，所有组沿 `M` 合并 |
| weight | Tensor | `[E, K, N]` 或 `[E, N, K]` | 专家权重，按 expert 维堆叠 |
| bias | Tensor?（可选） | `[E, N]` | 偏置 |
| group_list | List[int] | 长度 `E` | 累计 token 数（cumsum），最后值等于 `M` |
| split_item | int | — | 输出切分模式（详见下） |
| transpose_weight | bool | — | 是否转置权重 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | Tensor[] | 与 x 相同 | split_item=0/1 时长度 E（每组 `[m_i, N]`），split_item=2/3 时长度 1（`[M, N]`） |

### 数据类型

| 输入 (x) dtype | 输入 (weight) dtype | 输入 (bias) dtype | 输出 (y) dtype |
|---------------|-------------------|-----------------|---------------|
| float16 | float16 | float16 | float16 |
| bfloat16 | bfloat16 | bfloat16 | bfloat16 |
| float32 | float32 | float32 | float32 |

### split_item 取值

| split_item | 输出形式 | 说明 |
|------------|---------|------|
| 0 / 1 | `Tensor[]` 长度 E | 按 `group_list` 把 `y` 切回每组 `[m_i, N]`；空组返回长度 0 的 `[0, N]` 张量 |
| 2 / 3 | `Tensor[]` 长度 1 | 返回沿 M 轴拼接好的 `[M, N]` 单元素列表 |

### 规则与约束

- **K 一致性**：`x.shape[1]` 必须等于 `weight` 中 K 维（`weight.shape[1]` 或 `weight.shape[2]`，由 `transpose_weight` 决定）
- **N 一致性**：所有 expert 共享同一 `N`
- **group_list**：长度等于 `E`，严格非递减，最后一个值等于 `M`
- **空组**：允许 `group_list[i] == group_list[i-1]`，对应 expert `i` 不参与计算
- **transpose_weight**：
  - false：weight 形状 `[E, K, N]`，每片 `weight[g]` 直接参与 matmul
  - true：weight 形状 `[E, N, K]`，每片 `weight[g]` 需 transpose 最后两维后参与 matmul
- **维度限制**：每维大小在 32 字节对齐后应小于 int32 最大值

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `E`（expert / 组数） | 1 ~ 128 | cases.csv 实测 1 ~ 8；等于 `weight.shape[0]` 与 `group_list` 长度 |
| `M`（token 总数 / x 行数） | 1 ~ 16384 | cases.csv 实测 64 ~ 7001（含非 2 幂奇数 1009 / 1023 / 1031 / 2047 / 7001）；等于 `group_list[-1]` |
| `K`（contraction 维） | 1 ~ 8192 | cases.csv 实测 32 ~ 1040（含非 2 幂奇数 127 / 511、非 2 幂 1023 / 1040） |
| `N`（输出维） | 1 ~ 8192 | cases.csv 实测 32 ~ 2048（含非 2 幂奇数 255 / 1023、非 2 幂 1056） |
| `m_i`（每组 token 数） | 0 ~ `M` | cases.csv 实测 0 ~ 1751；允许 `m_i = 0` 表示空组（case_7 含 idx=2,5 两个空组） |
| `group_list` | 长度 `E`，单调非递减，末值 = `M` | cumsum 语义；相邻相等表示空组；元素类型 int64 |
| `split_item` | {0, 1, 2, 3} | cases.csv 实测 0 / 2 / 3；0/1 → `Tensor[]` 长度 `E`，2/3 → `Tensor[]` 长度 1（`[M, N]`） |
| `transpose_weight` | {false, true} | cases.csv 实测均覆盖；false → weight `[E, K, N]`，true → weight `[E, N, K]` 需对最后两维 transpose |
| `bias` | 可选（None 或 `[E, N]`） | cases.csv 实测含 bias / 无 bias 均覆盖；提供时 dtype 与 x 一致或 fp32 |

约束：
- `x.shape[1] == K`，且 `weight` 中对应维（`weight.shape[1]` 若 tw=false，否则 `weight.shape[2]`）必须等于 `K`
- `weight.shape[0] == E`，所有 expert 共享同一 `N`
- `group_list[-1] == M`，`group_list[i] >= group_list[i-1]`
- 每维大小在 32 字节对齐后应小于 int32 最大值

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
from typing import List, Optional

"""
GroupedMatmul 算子 Torch Golden 参考实现

分组矩阵乘法算子，x 沿 M 轴合并、weight 按 expert 维堆叠。
公式：对每个专家 g ∈ [0, E)，根据 group_list（cumsum）取属于该组的 token 行 rows_g：
        y[rows_g] = x[rows_g] @ weight[g] (+ bias[g])
"""


def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_list=None,
    split_item: int = 0,
    transpose_weight: bool = False,
) -> List[torch.Tensor]:
    M, K = x.shape
    E = weight.shape[0]
    if transpose_weight:
        N = weight.shape[1]
    else:
        N = weight.shape[2]

    if isinstance(group_list, torch.Tensor):
        ends = group_list.to(torch.int64).tolist()
    else:
        ends = list(group_list)
    starts = [0] + ends[:-1]

    y = torch.zeros((M, N), dtype=x.dtype, device=x.device)
    x_f = x.float()
    for g in range(E):
        s, e = starts[g], ends[g]
        if s == e:
            continue
        w_g = weight[g].float()
        if transpose_weight:
            mm = torch.matmul(x_f[s:e], w_g.transpose(-2, -1))
        else:
            mm = torch.matmul(x_f[s:e], w_g)
        if bias is not None:
            mm = mm + bias[g].float().unsqueeze(0)
        y[s:e] = mm.to(x.dtype)

    if split_item in (0, 1):
        return [y[starts[g]:ends[g]] for g in range(E)]
    return [y]
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

M, K, N, E = 128, 256, 512, 4

x = torch.randn(M, K, dtype=torch.float16, device="npu")
weight = torch.randn(E, K, N, dtype=torch.float16, device="npu")
bias = torch.randn(E, N, dtype=torch.float16, device="npu")
group_list = [32, 64, 96, 128]   # cumsum 语义，最后值 = M

# split_item=2: 输出 Tensor[] 长度 1，[M, N]
y_list = cann_bench.grouped_matmul(x, weight, bias, group_list, split_item=2, transpose_weight=False)
# y_list = [Tensor[128,512]]
y = y_list[0]  # 取单 tensor

# split_item=0: 输出 Tensor[] 长度 E
y_list = cann_bench.grouped_matmul(x, weight, bias, group_list, split_item=0, transpose_weight=False)
# y_list = [Tensor[32,512], Tensor[32,512], Tensor[32,512], Tensor[32,512]]

# transpose_weight=True: weight 形状 [E, N, K]，需 transpose 后 matmul
weight_t = torch.randn(E, N, K, dtype=torch.float16, device="npu")
y_list = cann_bench.grouped_matmul(x, weight_t, bias, group_list, split_item=2, transpose_weight=True)
```
