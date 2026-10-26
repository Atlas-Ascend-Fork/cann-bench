# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tbmv packed-storage regressions, including bandwidths larger than n."""

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = (
    Path(__file__).resolve().parents[2] / "bench_lab/blas-ops_bench/blas_tbmv/golden.py"
)
_SPEC = importlib.util.spec_from_file_location("tbmv_golden", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize("n,k", [(1, 0), (1, 2), (2, 3), (4, 1)])
@pytest.mark.parametrize("uplo", ["UPPER", "LOWER"])
@pytest.mark.parametrize("trans", ["N", "T", "C"])
@pytest.mark.parametrize("diag", ["NON_UNIT", "UNIT"])
@pytest.mark.parametrize("incx", [1, -2])
def test_packed_band_matches_dense_reference(n, k, uplo, trans, diag, incx):
    lda = k + 2
    packed = torch.arange(1, lda * n + 1, dtype=torch.float32).reshape(lda, n)
    flat = packed.flatten()
    matrix = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            if uplo == "UPPER" and 0 <= j - i <= k:
                matrix[i, j] = flat[k + i - j + j * lda]
            elif uplo == "LOWER" and 0 <= i - j <= k:
                matrix[i, j] = flat[i - j + j * lda]
    if diag == "UNIT":
        matrix.fill_diagonal_(1)
    if trans in ("T", "C"):
        matrix = matrix.T
    x = torch.arange(1, (n - 1) * abs(incx) + 2, dtype=torch.float32)
    original = x.clone()
    vector = x[::incx] if incx > 0 else x.flip(0)[:: abs(incx)]
    expected = (matrix @ vector.double()).float()
    actual = _MODULE.tbmv(packed, x, uplo=uplo, trans=trans, diag=diag, k=k, incx=incx)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x, original)


def test_empty_matrix_returns_empty_vector():
    actual = _MODULE.tbmv(torch.empty(4, 0), torch.empty(0), uplo="UPPER", k=3)
    assert actual.shape == (0,)
