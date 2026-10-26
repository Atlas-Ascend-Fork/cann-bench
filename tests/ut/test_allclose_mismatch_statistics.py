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

"""AllClose mismatch statistics must use the verdict's elementwise rule."""

import pytest
import torch

from kernel_eval.checkers import AllCloseChecker


@pytest.mark.parametrize(
    "actual,golden,atol,expected",
    [
        (torch.tensor([True, False]), torch.tensor([True, False]), 0.0, 0),
        (torch.tensor([True, False]), torch.tensor([False, False]), 0.0, 1),
        (
            torch.tensor([float("nan"), float("inf")]),
            torch.tensor([1.0, float("inf")]),
            0.0,
            1,
        ),
        (torch.tensor([float("nan")]), torch.tensor([float("nan")]), 0.0, 1),
        (
            torch.tensor([-128], dtype=torch.int8),
            torch.tensor([127], dtype=torch.int8),
            1.0,
            1,
        ),
        (torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.01]), 0.02, 0),
    ],
)
def test_allclose_mismatch_count_matches_verdict(actual, golden, atol, expected):
    result = AllCloseChecker().check(
        actual,
        golden,
        str(actual.dtype),
        0.0,
        custom_thresholds={"rtol": 0.0, "atol": atol},
    )
    assert result.passed == (expected == 0)
    assert result.output_results[0].mismatch_count == expected
    assert result.get_metadata()["mismatch_count"] == expected
