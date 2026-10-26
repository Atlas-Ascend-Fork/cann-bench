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

from typing import List

import torch


def attention_update(
    lse: List[torch.Tensor], local_out: List[torch.Tensor], update_type: int = 0
):
    """
    将各 SP 域的局部 lse 和 localOut 更新为全局结果。

    公式:
        lse_max = max_i(lse_i)
        lse = sum_i(exp(lse_i - lse_max))
        lse_m = lse_max + log(lse)
        O = sum_i(O_i * exp(lse_i - lse_m))

    参数:
        lse: TensorList，每个 tensor shape 为 [bsh]，dtype 为 float32
        local_out: TensorList，每个 tensor shape 为 [bsh, head_dim]，
                   数据类型为 float16 或 bfloat16
        update_type: 0 表示不输出 lse_out（返回空 tensor），
                     1 表示输出 lse_out

    返回:
        (out, lse_out) 的元组:
        - out: [bsh, head_dim]，全局 attention output，数据类型与 local_out 一致
        - lse_out: [bsh] float32，全局 lse（update_type=0 时为空 tensor）
    """
    dtype = local_out[0].dtype

    # Stack tensor list → [sp, bsh] / [sp, bsh, head_dim]
    lse_stacked = torch.stack(lse, dim=0)  # [sp, bsh]
    local_out_stacked = torch.stack(local_out, dim=0).float()  # [sp, bsh, head_dim]

    # Step 1: lse_max = max_i(lse_i)
    lse_max, _ = torch.max(lse_stacked, dim=0)  # [bsh]

    # Step 2: lse = sum_i(exp(lse_i - lse_max))
    lse_sub = lse_stacked - lse_max.unsqueeze(0)  # [sp, bsh]
    lse_sub_exp = torch.exp(lse_sub)  # [sp, bsh]
    lse_sum = torch.sum(lse_sub_exp, dim=0)  # [bsh]

    # Step 3: lse_m = lse_max + log(lse)
    lse_out = lse_max + torch.log(lse_sum)  # [bsh]

    # Step 4: O = sum_i(O_i * exp(lse_i - lse_m))
    # Normalize shifted exponentials directly to avoid cancellation in lse_i - lse_out.
    lse_weight = (lse_sub_exp / lse_sum.unsqueeze(0)).unsqueeze(2)  # [sp, bsh, 1]
    out = torch.sum(local_out_stacked * lse_weight, dim=0)  # [bsh, head_dim]

    if update_type == 0:
        lse_out = torch.zeros(0)

    return out.to(dtype), lse_out
