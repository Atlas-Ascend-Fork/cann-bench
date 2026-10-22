#!/usr/bin/python3
# coding=utf-8

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

import torch.nn as nn

import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                position_bias: torch.Tensor | None = None):
        B_, H, N, D = q.shape
        scale = D ** -0.5

        q = q * scale
        attn = q @ k.transpose(-2, -1)

        if position_bias is not None:
            attn = attn + position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        output = attn @ v

        return output.to(torch.float16)


def window_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, position_bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Torch golden implementation aligned to the tile2asc reference Model.forward."""
    model = Model()
    return model(q, k, v, position_bias)
