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

"""CPU regression for the Exp baseline's logarithmic base multiplier."""

import pytest
import torch

from scripts.baseline.refs import level1
from tasks.level1.exp.golden import exp


@pytest.mark.parametrize("base", [1.0, -1.0, 0.0, 2.0, 10.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exp_baseline_matches_golden_on_cpu(monkeypatch, base, dtype):
    # Only bypass the NPU compiler boundary; run the real reference expression.
    monkeypatch.setattr(
        level1,
        "_get_compiled_exp",
        lambda _x, log_base, scale, shift: lambda x: level1._exp_composite(
            x, log_base, scale, shift
        ),
    )
    x = torch.tensor([-2.0, 0.0, 3.0], dtype=dtype)
    attrs = {"base": base, "scale": 2.0, "shift": 0.5}
    actual = level1.exp_ref([x], attrs)
    torch.testing.assert_close(actual, exp(x, **attrs))
