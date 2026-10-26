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

"""level1 reference implementations on NPU."""

import math
import torch


# Cached torchair-compiled callables for exp(base, scale, shift) so the
# GE UB-fusion (AutomaticBufferFusionOp) bakes Mul+Add+Exp into a single
# AICore kernel. Keyed by (dtype, shape, base, scale, shift) — base/scale/shift
# are floats baked into the compiled graph as constants.
_EXP_TORCHAIR_CACHE = {}


def _exp_composite(x, log_base, scale, shift):
    """Inner callable that torchair compiles. log_base==0 ⇒ natural e."""
    t = scale * x + shift
    if log_base != 0.0:
        t = t * log_base
    return torch.exp(t)


def _get_compiled_exp(x, log_base, scale, shift):
    key = (x.dtype, tuple(x.shape), float(log_base), float(scale), float(shift))
    if key not in _EXP_TORCHAIR_CACHE:
        import torchair

        backend = torchair.get_npu_backend()

        # Bind constants into a closure so the compiled graph treats them as scalars
        def _fn(x):
            return _exp_composite(x, log_base, scale, shift)

        _EXP_TORCHAIR_CACHE[key] = torch.compile(_fn, backend=backend, dynamic=False)
    return _EXP_TORCHAIR_CACHE[key]


def exp_ref(inputs, attrs):
    """exp((scale*x + shift) * ln(base)) via torchair / GE UB fusion.

    Eager `scale*x + shift → (× ln(base) when base>0) → exp` would emit
    Mul + Add (+ Mul) + Exp as separate kernels. Compiling through torchair
    triggers GE's AutomaticBufferFusionOp which stages the entire chain
    through UB and emits a single fused AICore kernel — matching the spec's
    intent of `Exp(base, scale, shift)` as a single fused operator.
    """
    x = inputs[0]
    base = float(attrs.get("base", -1.0))
    scale = float(attrs.get("scale", 1.0))
    shift = float(attrs.get("shift", 0.0))
    log_base = math.log(base) if base > 0 else 0.0
    compiled = _get_compiled_exp(x, log_base, scale, shift)
    return compiled(x)


def gelu_ref(inputs, attrs):
    return torch.nn.functional.gelu(
        inputs[0], approximate=attrs.get("approximate", "none")
    )


def sigmoid_ref(inputs, attrs):
    return torch.sigmoid(inputs[0])


def mish_ref(inputs, attrs):
    return torch.nn.functional.mish(inputs[0])


def masked_scale_ref(inputs, attrs):
    """No torch_npu.npu_masked_scale; composite with native ops."""
    x, mask = inputs[0], inputs[1]
    scale = attrs.get("scale", 1.0)
    m = mask.to(x.dtype) if mask.dtype != x.dtype else mask
    return x * m * scale


def swi_glu_ref(inputs, attrs):
    """torch_npu.npu_swiglu 透传 attrs.dim (而非硬编码 -1)。

    Bug fix (refs-bug-4, 2026-05-15): 原版 dim=-1 硬编码。当前 cases.yaml 未含
    `dim` attr 时回落 -1 行为不变；将来若加 dim case 会被静默忽略。
    """
    import torch_npu

    x = inputs[0]
    return torch_npu.npu_swiglu(x, dim=attrs.get("dim", -1))


def foreach_norm_ref(inputs, attrs):
    """torch._foreach_norm with ord=scalar (golden uses ord=scalar)."""
    return list(torch._foreach_norm(inputs[0], ord=attrs.get("scalar", 1.0)))


def foreach_addcdiv_scalar_ref(inputs, attrs):
    return list(
        torch._foreach_addcdiv(
            inputs[0], inputs[1], inputs[2], value=float(attrs.get("scalar", 1.0))
        )
    )


REGISTRY = {
    "level1/exp": exp_ref,
    "level1/gelu": gelu_ref,
    "level1/sigmoid": sigmoid_ref,
    "level1/mish": mish_ref,
    "level1/masked_scale": masked_scale_ref,
    "level1/swi_glu": swi_glu_ref,
    "level1/foreach_norm": foreach_norm_ref,
    "level1/foreach_addcdiv_scalar": foreach_addcdiv_scalar_ref,
}
