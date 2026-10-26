# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Padding rows must not contaminate categorical embedding pooling."""

import importlib.util
from pathlib import Path

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "bench_lab/research_bench/level2/categorical_embedding_bag/golden.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "categorical_embedding_golden", _MODULE_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mean_pool", [False, True])
@pytest.mark.parametrize("padding_idx", [-1, 2])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_padding_rows_do_not_participate_in_pooling(
    dtype, mean_pool, padding_idx, nonfinite
):
    table = torch.tensor([[2.0, 4.0], [6.0, 8.0], [nonfinite, nonfinite]], dtype=dtype)
    indices = torch.tensor(
        [[[0, padding_idx, 1], [padding_idx, padding_idx, padding_idx]]]
    )
    bias = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    actual = _MODULE.categorical_embedding_bag(
        indices, table, bias, mean_pool, padding_idx
    )
    pooled = (table[0].float() + table[1].float()) / (2 if mean_pool else 1)
    expected = (
        torch.stack([pooled + bias[0].float(), bias[1].float()]).unsqueeze(0).to(dtype)
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mean_pool", [False, True])
def test_nonpadding_nonfinite_values_still_propagate(mean_pool):
    table = torch.tensor([[float("nan"), float("inf")], [2.0, 3.0]])
    indices = torch.tensor([[[0, 1]]])
    actual = _MODULE.categorical_embedding_bag(indices, table, mean_pool=mean_pool)
    expected = torch.tensor([[[float("nan"), float("inf")]]])
    torch.testing.assert_close(actual, expected, equal_nan=True)


@pytest.mark.parametrize("mean_pool", [False, True])
def test_finite_padding_and_wrapped_ids_keep_existing_semantics(mean_pool):
    table = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    indices = torch.tensor([[[3, -2, -1]]])
    actual = _MODULE.categorical_embedding_bag(indices, table, mean_pool=mean_pool)
    expected = ((table[0] + table[1]) / (2 if mean_pool else 1)).reshape(1, 1, 2)
    torch.testing.assert_close(actual, expected)
