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

"""SwiGLU references must preserve dimensions other than the split axis."""

import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.baseline.refs.level1 import swi_glu_ref
from tasks.level1.swi_glu.golden import swi_glu


@pytest.mark.parametrize(
    "shape,dim", [((4, 3), 0), ((2, 4, 3), 1), ((2, 4, 3), -2), ((3, 4), -1)]
)
def test_swiglu_reference_preserves_input(monkeypatch, shape, dim):
    x = (
        torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(
            shape
        )
        / 10
    )
    seen = []

    def npu_boundary(value, dim):
        seen.append(value)
        return swi_glu(value, dim=dim)

    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_swiglu=npu_boundary)
    )
    actual = swi_glu_ref([x], {"dim": dim})
    assert seen[0] is x
    torch.testing.assert_close(actual, swi_glu(x, dim=dim))
