# MhcPostGrad 算子 API 描述

## 1. 算子简介

Manifold Constrained Hyper Connection Post 的反向梯度算子。根据上游梯度 grad_output 和前向保存的中间结果（x, h_res, h_out, h_post），计算各输入的梯度。

**主要应用场景**：
- DeepSeek 等 MoE 模型的超连接后处理反向传播
- 流形约束下多流信息融合的梯度计算

**算子特征**：
- 难度等级：L3（Attention）
- 5 输入，4 输出，无属性参数
- 支持 ND 格式输入
- x / grad_output / h_out 支持 FLOAT16 / BFLOAT16，h_res / h_post 固定 FLOAT32
- 反向算子，golden 需用小算子拼接实现

## 2. 算子定义

### 数学公式

前向公式：
$$
y_{i,j} = h\_post_i \cdot h\_out_j + \sum_k h\_res_{k,i} \cdot x_{k,j}
$$

反向梯度推导（小算子拼接）：

**grad_h_post** — dy 与 h_out 逐元素乘后沿 D 维求和：
$$
grad\_h\_post_i = \sum_j dy_{i,j} \cdot h\_out_j
$$

**grad_h_out** — dy 与 h_post 逐元素乘后沿 n 维求和：
$$
grad\_h\_out_j = \sum_i dy_{i,j} \cdot h\_post_i
$$

**grad_h_res** — x 与 dy^T 的矩阵乘：
$$
grad\_h\_res_{k,i} = \sum_j x_{k,j} \cdot dy_{i,j} = (x \cdot dy^T)_{k,i}
$$

**grad_x** — h_res 与 dy 的矩阵乘：
$$
grad\_x_{k,j} = \sum_i h\_res_{k,i} \cdot dy_{i,j} = (h\_res \cdot dy)_{k,j}
$$

## 3. 接口规范

### 算子原型

```python
cann_bench.mhc_post_grad(
    Tensor grad_output, Tensor x, Tensor h_res, Tensor h_out, Tensor h_post
) -> (Tensor grad_x, Tensor grad_h_res, Tensor grad_h_out, Tensor grad_h_post)
```

### 输入参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|----------|------|---------|---------|
| grad_output | 输入 | 上游梯度, [..., S, n, D] | FLOAT16 / BFLOAT16 | ND |
| x | 输入 | 前向输入 x, [..., S, n, D] | FLOAT16 / BFLOAT16 | ND |
| h_res | 输入 | 前向残差权重, [..., S, n, n] | FLOAT32 | ND |
| h_out | 输入 | 前向输出权重, [..., S, D] | FLOAT16 / BFLOAT16 | ND |
| h_post | 输入 | 前向后处理权重, [..., S, n] | FLOAT32 | ND |

### 输出

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|----------|------|---------|---------|
| grad_x | 输出 | x 的梯度, [..., S, n, D] | 与 grad_output 一致 | ND |
| grad_h_res | 输出 | h_res 的梯度, [..., S, n, n] | FLOAT32 | ND |
| grad_h_out | 输出 | h_out 的梯度, [..., S, D] | 与 grad_output 一致 | ND |
| grad_h_post | 输出 | h_post 的梯度, [..., S, n] | FLOAT32 | ND |

### 规则与约束

- grad_output、x、h_out 的 dtype 必须一致
- h_res 和 h_post 固定为 FLOAT32
- 所有输入的前缀维度必须一致
- n 的取值范围为 4~8
- D 支持 384, 2560, 3264, 4368, 5120, 6544

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| grad_output / x dtype | float16 / bfloat16 | 与 h_out 一致 |
| h_res / h_post dtype | float32 | 固定 |
| n | 4 ~ 8 | 超连接流数 |
| D | 384 ~ 6544 | head_dim 维度 |
| S | 1 ~ 2048 | 序列维度 |

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
from typing import Optional


def get_input(grad_output: torch.Tensor, x: torch.Tensor,
              h_res: torch.Tensor, h_out: torch.Tensor,
              h_post: torch.Tensor, **kwargs):
    """
    预处理输入，确保满足算子的语义约束：
    - h_res: 使用 Sinkhorn 算法生成双随机矩阵（行和=列和=1）
    """
    h_res = _generate_dsmat_sinkhorn(h_res)
    return (grad_output, x, h_res, h_out, h_post)


def _generate_dsmat_sinkhorn(orig_tsr: torch.Tensor,
                              max_iter: int = 100,
                              tol: float = 1e-8):
    """使用 Sinkhorn 算法生成双随机矩阵（行和=列和=1）"""
    shape_list = list(orig_tsr.shape)
    orig_device = orig_tsr.device
    orig_dtype = orig_tsr.dtype

    random_tensor = torch.rand(shape_list, dtype=torch.float32, device=orig_device) + 1e-10

    for _ in range(max_iter):
        row_sum = random_tensor.sum(dim=-1, keepdim=True)
        random_tensor = random_tensor / (row_sum + 1e-10)
        col_sum = random_tensor.sum(dim=-2, keepdim=True)
        random_tensor = random_tensor / (col_sum + 1e-10)
        row_check = torch.abs(random_tensor.sum(dim=-1) - 1).max()
        col_check = torch.abs(random_tensor.sum(dim=-2) - 1).max()
        if row_check < tol and col_check < tol:
            break

    return random_tensor.to(orig_dtype).to(orig_device)


def mhc_post_grad(
        grad_output: torch.Tensor, x: torch.Tensor,
        h_res: torch.Tensor, h_out: torch.Tensor,
        h_post: torch.Tensor):
    """
    Manifold Constrained Hyper Connection Post 反向梯度算子（小算子拼接实现）。
    使用 float64 高精度计算，避免 float32 大 reduction 数值相消导致的精度抖动。

    前向: y[i,j] = h_post[i] * h_out[j] + sum_k h_res[k,i] * x[k,j]

    反向梯度推导:
        grad_h_post[i] = sum_j dy[i,j] * h_out[j]      → elementwise mul + sum
        grad_h_out[j]  = sum_i dy[i,j] * h_post[i]     → elementwise mul + sum
        grad_h_res     = x @ dy^T                      → matmul
        grad_x         = h_res @ dy                    → matmul

    参数:
        grad_output: [..., S, n, D] float16/bfloat16 — 上游梯度
        x: [..., S, n, D] float16/bfloat16 — 前向输入
        h_res: [..., S, n, n] float32 — 双随机矩阵
        h_out: [..., S, D] float16/bfloat16 — 前向输出权重
        h_post: [..., S, n] float32 — 前向后处理权重

    返回:
        (grad_x, grad_h_res, grad_h_out, grad_h_post)
    """
    orig_dtype = grad_output.dtype
    orig_h_res_dtype = h_res.dtype

    # 使用 float64 高精度计算 golden
    dy = grad_output.double()
    x_fp64 = x.double()
    h_res_fp64 = h_res.double()
    h_out_fp64 = h_out.double()
    h_post_fp64 = h_post.double()

    # grad_h_post[i] = sum_j dy[i,j] * h_out[j]
    grad_h_post = (dy * h_out_fp64.unsqueeze(-2)).sum(dim=-1)

    # grad_h_out[j] = sum_i dy[i,j] * h_post[i]
    grad_h_out = (dy * h_post_fp64.unsqueeze(-1)).sum(dim=-2)

    # grad_h_res[k,i] = sum_j x[k,j] * dy[i,j] = x @ dy^T
    grad_h_res = torch.matmul(x_fp64, dy.transpose(-2, -1))

    # grad_x[k,j] = sum_i h_res[k,i] * dy[i,j] = h_res @ dy
    grad_x = torch.matmul(h_res_fp64, dy)

    return (grad_x.to(orig_dtype), grad_h_res.to(orig_h_res_dtype),
            grad_h_out.to(orig_dtype), grad_h_post.to(orig_h_res_dtype))
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# 示例: batch=1, S=1024, n=4, D=5120
grad_output = torch.randn(1, 1024, 4, 5120, dtype=torch.float16, device="npu")
x = torch.randn(1, 1024, 4, 5120, dtype=torch.float16, device="npu")
h_res = torch.randn(1, 1024, 4, 4, dtype=torch.float32, device="npu")
h_out = torch.randn(1, 1024, 5120, dtype=torch.float16, device="npu")
h_post = torch.randn(1, 1024, 4, dtype=torch.float32, device="npu")

grad_x, grad_h_res, grad_h_out, grad_h_post = cann_bench.mhc_post_grad(
    grad_output, x, h_res, h_out, h_post
)
```