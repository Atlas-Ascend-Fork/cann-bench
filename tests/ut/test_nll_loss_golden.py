#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""CPU regression for NLLLoss hard-label dtypes and weighted reductions."""

import importlib.util
from itertools import product
from pathlib import Path

import pytest
import torch

TASK = Path(__file__).resolve().parents[2] / "bench_lab/kernel_bench/level2/nll_loss"
SPEC = importlib.util.spec_from_file_location("nll_loss_golden", TASK / "golden.py")
GOLDEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GOLDEN)


def _inputs(layout, dtype, target_dtype, all_ignored=False):
    if layout == "scalar":
        scores = torch.tensor([0.0, 0.5, 1.0, -0.5])
        target = torch.tensor(2, dtype=target_dtype)
        axis = 0
    elif layout == "batch":
        scores = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 4
        target = torch.tensor([1, -100, 3], dtype=target_dtype)
        axis = 1
    else:
        scores = torch.arange(48, dtype=torch.float32).reshape(2, 4, 2, 3) / 16
        target = torch.arange(12).reshape(2, 2, 3).remainder(4).to(target_dtype)
        target[0, 0, 0] = -100
        axis = 1
    if all_ignored:
        target = torch.full_like(target, -100)
    inputs = torch.log_softmax(scores, dim=axis).to(dtype)
    weight = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=dtype)
    return inputs, target, weight


def _reference(inputs, target, weight, reduction):
    losses = torch.zeros(target.shape, dtype=torch.float64)
    total_weight = 0.0
    indices = product(*(range(size) for size in target.shape))
    for idx in indices:
        label = int(target[idx])
        if label == -100:
            continue
        input_idx = (label,) if target.ndim == 0 else (idx[0], label, *idx[1:])
        w = float(weight[label])
        losses[idx] = -float(inputs[input_idx]) * w
        total_weight += w
    if reduction == "none":
        out = losses
        total_weight = 0.0
    elif reduction == "sum":
        out = losses.sum()
    else:
        out = losses.sum() / total_weight
    return out.to(inputs.dtype), torch.tensor(total_weight, dtype=inputs.dtype)


@pytest.mark.parametrize("layout", ["scalar", "batch", "spatial"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("target_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_declared_label_dtypes(layout, dtype, target_dtype, reduction):
    inputs, target, weight = _inputs(layout, dtype, target_dtype)
    original_target = target.clone()
    expected = _reference(inputs, target, weight, reduction)
    actual = GOLDEN.nll_loss(inputs, target, weight, reduction)
    assert actual[0].shape == expected[0].shape
    assert actual[1].shape == torch.Size([])
    assert all(out.dtype == dtype for out in actual)
    tolerance = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 1e-6}[dtype]
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(target, original_target, rtol=0, atol=0)
    assert target.dtype == target_dtype


@pytest.mark.parametrize("target_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_all_ignored_labels(target_dtype, reduction):
    inputs, target, weight = _inputs(
        "batch", torch.float32, target_dtype, all_ignored=True
    )
    out, total_weight = GOLDEN.nll_loss(inputs, target, weight, reduction)
    assert total_weight.item() == 0.0
    if reduction == "mean":
        assert torch.isnan(out).item()
    else:
        assert torch.count_nonzero(out).item() == 0
