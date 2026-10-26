# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch

"""
Tbmv 算子 Torch Golden 参考实现

计算三角带状矩阵-向量乘 y := op(A) * x（arch22 legacy 语义: x/y 分离输出，
见 ops-blas/blas/tbmv/README.md 数学表达式 "y = A * x（arch22，输入输出分离）"）。

语义与 ops-blas/blas/tbmv（arch22）实现对齐：
- A 为带状打包缓冲，"维度 lda x n"（README），列主序带状存储（BLAS 标准，
  以 README 调用示例 n=4/k=1/lda=2 解码验证: A(i,j) = buf[(i-j) + j*lda]（下三角）,
  A(i,j) = buf[(k+i-j) + j*lda]（上三角））
- k 为半带宽: LOWER 时 A(j+b, j) 位于打包第 b 行第 j 列（b=0 主对角线）;
  UPPER 时 A(j, j+d) 位于打包第 k-d 行第 j 列
  （与 arch35 fallback kernel 及官方 cblas ColMajor golden 一致; 注意 arch22 优化
  kernel 的 a[bandIdx*lda+col] 为对角主序, 与本公式是转置关系, 勿照抄其索引）
- trans: N -> op(A)=A; T -> op(A)=A^T; C -> op(A)=A^H（实数下与 T 等价）
- diag: NON_UNIT 对角元从 A 读取; UNIT 对角元固定为 1（乘法因子, 无除法,
  存储对角值不影响合法性）
- incx != 0: x 按步长访问, 元素 i 位于 x[i*incx]; y 的 incx 仅为存储布局,
  数学结果与 incx 无关, golden 返回长度 n 的稠密结果向量
- 输出与输入分离: golden 返回新张量 y, 不覆写 x
- n <= 0 时空操作直接返回（README 约束 n >= 0）
"""


def _norm_uplo(v):
    m = {"LOWER": "LOWER", "UPPER": "UPPER", 121: "UPPER", 122: "LOWER"}
    if v not in m:
        raise ValueError(f"uplo 必须为 UPPER/LOWER(121/122)，got {v!r}")
    return m[v]


def _norm_trans(v):
    m = {"N": "N", "T": "T", "C": "C", 111: "N", 112: "T", 113: "C"}
    if v not in m:
        raise ValueError(f"trans 必须为 N/T/C(111/112/113)，got {v!r}")
    return m[v]


def _norm_diag(v):
    m = {"NON_UNIT": "NON_UNIT", "UNIT": "UNIT", 131: "NON_UNIT", 132: "UNIT"}
    if v not in m:
        raise ValueError(f"diag 必须为 NON_UNIT/UNIT(131/132)，got {v!r}")
    return m[v]


def tbmv(
    A: torch.Tensor,
    x: torch.Tensor,
    uplo="LOWER",
    trans="N",
    diag="NON_UNIT",
    k: int = 0,
    incx: int = 1,
    **kwargs,
) -> torch.Tensor:
    """计算 y = op(A) * vec，返回稠密结果向量（与输入 x 分离）。

    Args:
        A: [lda, n] 带状打包缓冲（README: 维度 lda x n），lda >= k+1
        x: 输入向量缓冲区，长度 (n-1)*|incx|+1，按 incx 步长访问
        uplo: UPPER/LOWER，指定 A 的三角部分
        trans: N/T/C，指定 op(A)
        diag: NON_UNIT/UNIT
        k: 半带宽
        incx: x 的存储增量，非零
    """
    incx = int(incx)
    k = int(k)
    if incx == 0:
        raise ValueError("incx 不能为 0")
    if k < 0:
        raise ValueError("k 不能为负")

    n = int(A.shape[1])
    lda = int(A.shape[0])
    if lda < k + 1:
        raise ValueError(f"lda({lda}) 必须 >= k+1({k + 1})")

    uplo = _norm_uplo(uplo)
    trans = _norm_trans(trans)
    diag = _norm_diag(diag)

    if incx > 0:
        vec = x[::incx]
    else:
        vec = x.flip(0)[::(-incx)]
    if vec.numel() < n:
        raise ValueError(f"x 缓冲区长度 {x.numel()} 不足以容纳 n={n} 个元素")

    # 打包解读: 列主序带状, P[b, j] = A_bl(j+b, j)（LOWER）/ A_bl(j, j+b)（UPPER 取 flip）
    # torch [lda, n] 行主序缓冲的 flat 重解释为 (k+1, n)、步长 (1, lda)
    P = A.reshape(-1).as_strided((k + 1, n), (1, lda))

    # 展开为稠密逻辑矩阵（升精度, 供参考计算）
    M = torch.zeros(n, n, dtype=torch.float64)
    nband = min(k, n - 1)
    idx_base = torch.arange(n)
    for b in range(nband + 1):
        L = n - b
        if L <= 0:
            break
        if uplo == "LOWER":
            M[idx_base[b:], idx_base[:L]] = P[b, :L].to(torch.float64)
        else:  # UPPER: A(i, i+b) = buf[(k-b) + (i+b)*lda]（打包第 k-b 行、第 i+b 列）
            M[idx_base[:L], idx_base[b:]] = P[k - b, b:].to(torch.float64)

    if diag == "UNIT":
        M[idx_base, idx_base] = 1

    if trans == "T":
        M = M.t()
    elif trans == "C":
        M = M.t().conj()

    y = torch.matmul(M, vec[:n].to(torch.float64))
    return y.to(x.dtype)
