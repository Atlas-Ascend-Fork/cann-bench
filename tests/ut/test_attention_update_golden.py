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

"""CPU regression for stable partition weights in AttentionUpdate."""

import importlib.util
from pathlib import Path

import pytest
import torch

TASK = (
    Path(__file__).resolve().parents[2]
    / "bench_lab/kernel_bench/level2/attention_update"
)
SPEC = importlib.util.spec_from_file_location(
    "attention_update_golden", TASK / "golden.py"
)
GOLDEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GOLDEN)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sp", [1, 2, 7, 16])
@pytest.mark.parametrize("offset", [0.0, 1e6, 1e8, -1e8])
@pytest.mark.parametrize("update_type", [0, 1])
def test_large_common_lse_offset(dtype, sp, offset, update_type):
    lse = [
        (torch.tensor([i / 4, -i / 4], dtype=torch.float32) + offset).contiguous()
        for i in range(sp)
    ]
    local_out = [
        (torch.arange(16, dtype=torch.float32).reshape(2, 8) / 32 + (i + 1) / 32).to(
            dtype
        )
        for i in range(sp)
    ]
    original_lse = [x.clone() for x in lse]
    original_out = [x.clone() for x in local_out]
    logits = torch.stack(lse).double()
    weights = torch.softmax(logits, dim=0)
    expected = (
        (torch.stack(local_out).double() * weights.unsqueeze(-1)).sum(0).to(dtype)
    )
    out, lse_out = GOLDEN.attention_update(lse, local_out, update_type)
    tolerance = 2e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(out, expected, rtol=tolerance, atol=tolerance / 10)
    assert out.dtype == dtype
    assert lse_out.dtype == torch.float32
    if update_type == 1:
        torch.testing.assert_close(lse_out, torch.logsumexp(logits, dim=0).float())
    else:
        assert lse_out.shape == (0,)
    for got, original in zip(lse + local_out, original_lse + original_out):
        torch.testing.assert_close(got, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_equal_partitions_preserve_arithmetic_mean(dtype):
    lse = [torch.full((1,), 1e8, dtype=torch.float32) for _ in range(2)]
    local_out = [torch.full((1, 8), v, dtype=dtype) for v in (1.0, 3.0)]
    out, _ = GOLDEN.attention_update(lse, local_out, 1)
    torch.testing.assert_close(out, torch.full_like(out, 2.0), rtol=0, atol=0)
