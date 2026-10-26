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

"""CPU regression for the top/bottom/left/right Conv2D padding contract."""

import pytest
import torch

from scripts.baseline.refs.level3 import conv_2d_ref
from tasks.level3.conv_2d.golden import conv_2d


@pytest.mark.parametrize("path", ["golden", "reference"])
@pytest.mark.parametrize(
    "pads", [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 2, 2], [1, 0, 2, 3]]
)
def test_conv2d_four_value_padding_on_cpu(path, pads):
    x = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
    weight = torch.tensor([[[[1.0, 2.0], [-1.0, 3.0]]]])
    bias = torch.tensor([0.5])
    strides, dilations = [1, 2], [1, 1]
    padded = torch.nn.functional.pad(x, (pads[2], pads[3], pads[0], pads[1]))
    expected = torch.nn.functional.conv2d(padded, weight, bias, stride=strides)
    if path == "golden":
        actual = conv_2d(x, weight, bias, strides, pads, dilations)
    else:
        actual = conv_2d_ref(
            [x, weight, bias],
            {"strides": strides, "pads": pads, "dilations": dilations},
        )
    torch.testing.assert_close(actual, expected)
