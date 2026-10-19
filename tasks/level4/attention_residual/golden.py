# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""AttentionResidual golden：Block Attention Residuals 前向聚合参考实现。

语义以 desc.md 为唯一权威：RMSNorm(键路径) → 伪查询打分(/√D) →
块轴 softmax → 对原始 v_stack 加权求和 → 与 x 门控混合。
全部中间计算 float32，输出转回输入 dtype。纯 torch，无 numpy 依赖。
"""
import math

import torch


def attention_residual(v_stack: torch.Tensor, x: torch.Tensor, w: torch.Tensor,
              gamma: torch.Tensor, alpha: float,
              epsilon: float = 1e-6) -> torch.Tensor:
    assert v_stack.dim() == 4 and x.dim() == 3, (v_stack.shape, x.shape)
    n, b, t, d = v_stack.shape
    assert n >= 2, "N >= 2 为定义域硬约束（见 desc.md）"
    assert tuple(x.shape) == (b, t, d), (v_stack.shape, x.shape)
    assert w.shape == (d,) and gamma.shape == (d,), (w.shape, gamma.shape)
    assert 0.05 <= float(alpha) <= 0.95, alpha
    assert 1e-8 <= float(epsilon) <= 1e-3, epsilon

    v = v_stack.to(torch.float32)
    xf = x.to(torch.float32)
    wf = w.to(torch.float32)
    gf = gamma.to(torch.float32)

    # 步骤 1：键路径 RMSNorm（沿 D，float32 累加）
    rms = torch.sqrt(v.pow(2).mean(dim=-1, keepdim=True) + float(epsilon))
    k = v / rms * gf
    # 步骤 2：伪查询打分（/√D 温度）
    s = torch.einsum("nbtd,d->nbt", k, wf) / math.sqrt(d)
    # 步骤 3：深度 softmax（块轴 n；torch.softmax 自带 max-shift 数值稳定）
    p = torch.softmax(s, dim=0)
    # 步骤 4：对原始 v_stack 加权求和
    agg = torch.einsum("nbt,nbtd->btd", p, v)
    # 步骤 5：门控混合
    y = (1.0 - float(alpha)) * xf + float(alpha) * agg
    return y.to(x.dtype)
