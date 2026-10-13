#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torch_npu
import cann_bench
import pytest


def test_sqrt_interface_exist():
    """
    Test that the 'cann_bench.sqrt' operator is present in torch.ops.
    """
    print(torch.ops.cann_bench.sqrt)
    assert hasattr(torch.ops.cann_bench, "sqrt"), "The 'sqrt' operator is not registered in the 'torch.ops.cann_bench' namespace."


SHAPES = [
    (1,),
    (3,),
    (10,),
    (100,),
    (1024,),
    (10000,),
    # Regression cases at one and two UB tiles (tileSize is 12272 elements on
    # 910B3: UB / 16 B-per-element, depth included). These pin the exact tile
    # boundary and the tail-tile path at both scales; (1000, 1000) below keeps
    # the multi-tile-per-core loop covered.
    (12271,),
    (12272,),
    (12273,),
    (24544,),
    (24545,),
    (10, 10),
    (32, 32),
    (100, 100),
    (10, 100),
    (100, 10),
    (256, 512),
    (5, 10, 15),
    (16, 32, 64),
    (32, 64, 128),
    (1, 3, 32, 32),
    (4, 3, 64, 64),
    (8, 3, 128, 128),
    (1000, 1000),
]

DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
]

# The fp16/bf16 kernels round-trip through fp32 and cast back, so compare with
# a ~1-2 ulp tolerance of the storage dtype instead of the fp32-level one.
TOLERANCES = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
}


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sqrt_operator(shape, dtype):
    """
    Test the functionality of the sqrt operator.

    Parameters:
        shape: Tensor shape
        dtype: Data type
    """
    a = torch.randn(*shape, dtype=dtype)
    a = torch.abs(a)

    expected = torch.sqrt(a)
    a_npu = a.npu()
    result_npu = torch.ops.cann_bench.sqrt(a_npu)
    result = result_npu.cpu()

    rtol, atol = TOLERANCES[dtype]
    assert torch.allclose(result, expected, rtol=rtol, atol=atol), \
        f"Sqrt failed for shape {shape}, dtype {dtype}. " \
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"

    print(f"Test passed: shape={shape}, dtype={dtype}")
