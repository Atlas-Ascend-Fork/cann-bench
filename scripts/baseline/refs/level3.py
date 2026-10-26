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

"""level3 reference implementations on NPU.

Some level3 operators have no torch_npu fused API and either rely on a
torch composite or are deemed not directly comparable (output a special
NPU representation, e.g. NZ format). Those are omitted from REGISTRY so
the harness skips method 3 cleanly.
"""

import torch


def adaptive_avg_pool_3d_ref(inputs, attrs):
    return torch.nn.functional.adaptive_avg_pool3d(inputs[0], attrs.get("output_size"))


def add_rms_norm_dynamic_quant_ref(inputs, attrs):
    """torch_npu.npu_add_rms_norm_dynamic_quant(x1, x2, gamma, *, ..., output_mask=[bool,bool]).

    Schema declares `output_mask: bool[2]` but defaults to []; we must pass an
    explicit length-2 list, otherwise the binding raises
    `Tried to convert a List with 0 elements to a fixed-size array of size 2`.
    output_mask=[True, True] enables both the dequant scale and residual outputs.
    """
    import torch_npu

    if hasattr(torch_npu, "npu_add_rms_norm_dynamic_quant"):
        return torch_npu.npu_add_rms_norm_dynamic_quant(
            inputs[0],
            inputs[1],
            inputs[2],
            epsilon=attrs.get("epsilon", 1e-6),
            output_mask=[True, True],
        )
    return None


def conv_2d_ref(inputs, attrs):
    x = inputs[0]
    pads = attrs.get("pads", [0, 0, 0, 0])
    if len(pads) == 2:
        padding = tuple(pads)
    elif pads[0] == pads[1] and pads[2] == pads[3]:
        padding = (pads[0], pads[2])
    else:
        x = torch.nn.functional.pad(x, (pads[2], pads[3], pads[0], pads[1]))
        padding = 0
    return torch.nn.functional.conv2d(
        x,
        inputs[1],
        inputs[2] if len(inputs) > 2 else None,
        stride=tuple(attrs.get("strides", [1, 1])),
        padding=padding,
        dilation=tuple(attrs.get("dilations", [1, 1])),
    )


def depthwise_conv_2d_ref(inputs, attrs):
    """Depthwise conv = conv2d with groups=Cin. Spec weight shape is [C, Kh, Kw]
    (no redundant per-group in-channel dim). torch.conv2d needs the 4D form
    [C, 1, Kh, Kw], so unsqueeze when input is 3D."""
    x, w = inputs[0], inputs[1]
    bias = inputs[2] if len(inputs) > 2 else None
    if w.dim() == 3:
        w = w.unsqueeze(1)
    return torch.nn.functional.conv2d(
        x,
        w,
        bias,
        stride=tuple(attrs.get("stride", [1, 1])),
        padding=tuple(attrs.get("padding", [0, 0])[:2]),
        dilation=tuple(attrs.get("dilation", [1, 1])),
        groups=int(attrs.get("groups", x.shape[1])),
    )


def dequant_swiglu_quant_ref(inputs, attrs):
    """torch_npu.npu_dequant_swiglu_quant — Plan-A simplified signature
    (dynamic per-token quant only; static quant mode dropped):

      inputs[0] = x                 [TokensNum, 2H]  int32 | bfloat16 | float16
      inputs[1] = weight_scale      [1, 2H]          float32   (only if x=int32)
      inputs[2] = activation_scale  [TokensNum]      float32   (only if x=int32)
      inputs[3] = quant_scale       [1, H]           float32  (optional, last)

    The number of inputs is 1, 2, 3, or 4 depending on x dtype and whether
    quant_scale is provided. attrs has activate_left (bool) only;
    quant_mode is always pinned to 1 (dynamic per-token).
    """
    import torch_npu

    if not hasattr(torch_npu, "npu_dequant_swiglu_quant"):
        return None
    x = inputs[0]
    kwargs = {
        "activate_left": bool(attrs.get("activate_left", False)),
        "quant_mode": 1,  # always dynamic per-token in this benchmark
    }
    # Position 1/2 are weight_scale & activation_scale only when x is int32.
    if x.dtype == torch.int32 and len(inputs) >= 3:
        kwargs["weight_scale"] = inputs[1]
        kwargs["activation_scale"] = inputs[2]
        if len(inputs) >= 4:
            kwargs["quant_scale"] = inputs[3]
    elif x.dtype in (torch.bfloat16, torch.float16) and len(inputs) >= 2:
        # x=bf16/fp16: weight_scale must be None; quant_scale is the next slot.
        kwargs["quant_scale"] = inputs[1]
    return torch_npu.npu_dequant_swiglu_quant(x, **kwargs)


# Cached torchair-compiled callables for transpose_weight=True cases.
# Keyed by (dtype, x.shape, w.shape, has_bias) so each shape gets its own
# compiled .om once and is reused across the harness's 20 trial calls.
_GMM_TORCHAIR_CACHE = {}


def _gmm_eager_with_transpose(x, w_nk, bias_or_none, group_list):
    """Inner callable that torchair will compile. Transpose then call aclnn.
    GE applies GroupedMatmulTransFusionPass during graph compile, fusing
    Transpose + GroupedMatmul into a single GroupedMatmul kernel."""
    import torch_npu

    w_kn = w_nk.transpose(-2, -1).contiguous()
    out = torch_npu.npu_grouped_matmul(
        [x],
        [w_kn],
        bias=[bias_or_none] if bias_or_none is not None else None,
        group_list=group_list,
        split_item=2,
        group_type=0,
    )
    return out[0] if isinstance(out, list) else out


def _get_compiled_gmm(x, w_nk, has_bias):
    """Return a torchair-compiled callable for this (dtype, x.shape, w.shape, has_bias)."""
    import torch

    key = (x.dtype, tuple(x.shape), tuple(w_nk.shape), has_bias)
    if key not in _GMM_TORCHAIR_CACHE:
        import torchair

        backend = torchair.get_npu_backend()
        _GMM_TORCHAIR_CACHE[key] = torch.compile(
            _gmm_eager_with_transpose, backend=backend, dynamic=False
        )
    return _GMM_TORCHAIR_CACHE[key]


def grouped_matmul_ref(inputs, attrs):
    """torch_npu.npu_grouped_matmul with single-tensor + group_list convention.

    cases.yaml supplies (per new spec aligned with level4/grouped_matmul_swiglu_quant):
      inputs[0]: x      [M, K]                       - single 2D tensor
      inputs[1]: weight [E, K, N] or [E, N, K]       - single 3D stacked tensor
      inputs[2]: bias   [E, N]                       - optional, single 2D tensor
      attrs.group_list: List[int] cumsum, length E, last value == M
      attrs.split_item: 0/1 (List[Tensor] output) or 2/3 (single tensor output)
      attrs.transpose_weight: bool
        False → weight stored as [E, K, N], direct matmul
        True  → weight stored as [E, N, K], must transpose last two dims

    When transpose_weight=True we dispatch through torchair (GE graph mode)
    so GroupedMatmulTransFusionPass fuses Transpose+GroupedMatmul into a single
    kernel — this is the production path for graph-compiled MoE inference.
    When transpose_weight=False the eager aclnn path is already a single
    GroupedMatmul kernel, no graph compile needed.
    """
    import torch
    import torch_npu

    if not hasattr(torch_npu, "npu_grouped_matmul"):
        return None

    x = inputs[0]
    w = inputs[1]
    bias = inputs[2] if len(inputs) > 2 else None
    # aclnnGroupedMatmulV5 bias dtype rule (verified empirically on CANN 8.5):
    #   fp16 x → bias must be fp16
    #   bf16 x → bias must be fp32  (NOT bf16)
    #   fp32 x → bias must be fp32
    if bias is not None:
        required_bias_dtype = (
            torch.float16 if x.dtype == torch.float16 else torch.float32
        )
        if bias.dtype != required_bias_dtype:
            bias = bias.to(required_bias_dtype)

    gl_list = attrs.get("group_list")
    if gl_list is None:
        raise ValueError("grouped_matmul_ref: attrs.group_list missing")
    group_list = torch.tensor(gl_list, dtype=torch.int64, device=x.device)

    transpose_weight = bool(attrs.get("transpose_weight", False))

    if transpose_weight:
        # Graph-mode path: GE fuses the Transpose into GroupedMatmul.
        compiled = _get_compiled_gmm(x, w, has_bias=(bias is not None))
        return compiled(x, w, bias, group_list)

    # Eager path: aclnn directly. Already a single GroupedMatmul kernel.
    return torch_npu.npu_grouped_matmul(
        [x],
        [w],
        bias=[bias] if bias is not None else None,
        group_list=group_list,
        split_item=2,
        group_type=0,
    )


def moe_finalize_routing_ref(inputs, attrs):
    """Eager-mode torch_npu.npu_moe_finalize_routing routes unconditionally to
    aclnnMoeFinalizeRoutingV2 — msprof confirms MoeFinalizeRoutingV2 fires on all
    20 cases regardless of drop_pad_mode / which optional inputs are present. V2
    accepts drop_pad_mode in [0,1,2,3] and treats x1(skip1)/x2(skip2)/bias/scales/
    expert_idx as optional.

    cases.yaml positional order (matches V2 IR):
       [0]=expanded_permuted_rows  [1]=expanded_src_to_dst_row  [2]=skip1
       [3]=skip2  [4]=bias  [5]=scales  [6]=expert_for_source_row

    The torch_npu Python binding signature is V1-style positional:
       (expanded_permuted_rows, skip1, skip2, bias, scales,
        expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)

    so we remap input positions and pull drop_pad_mode from attrs.
    """
    import torch_npu

    if not hasattr(torch_npu, "npu_moe_finalize_routing"):
        return None
    perm = inputs[0]
    src_to_dst = inputs[1]
    skip1 = inputs[2] if len(inputs) > 2 else None
    skip2 = inputs[3] if len(inputs) > 3 else None
    bias = inputs[4] if len(inputs) > 4 else None
    scales = inputs[5] if len(inputs) > 5 else None
    expert = inputs[6] if len(inputs) > 6 else None
    drop_pad_mode = int(attrs.get("drop_pad_mode", 0))
    return torch_npu.npu_moe_finalize_routing(
        perm,
        skip1,
        skip2,
        bias,
        scales,
        src_to_dst,
        expert,
        drop_pad_mode,
    )


def moe_gating_top_k_softmax_ref(inputs, attrs):
    """torch_npu.npu_moe_gating_top_k_softmax(x, finished, k)."""
    import torch_npu

    x = inputs[0]
    finished = inputs[1] if len(inputs) > 1 else None
    k = int(attrs.get("k", 1))
    if hasattr(torch_npu, "npu_moe_gating_top_k_softmax"):
        return torch_npu.npu_moe_gating_top_k_softmax(x, finished, k)
    # composite fallback
    s = torch.softmax(x, dim=-1)
    if finished is not None:
        s = s.masked_fill(finished.unsqueeze(-1).bool(), 0.0)
    v, idx = torch.topk(s, k, dim=-1)
    return v, idx, idx.to(torch.int32)


def moe_re_routing_ref(inputs, attrs):
    """torch_npu.npu_moe_re_routing(tokens, expert_token_num_per_rank, *,
        per_token_scales=None, expert_token_num_type=1, idx_type=0).

    Only first 2 inputs are positional; 3rd (`per_token_scales`) is keyword-only.
    """
    import torch_npu

    if not hasattr(torch_npu, "npu_moe_re_routing"):
        return None
    tokens = inputs[0]
    expert_token_num_per_rank = inputs[1]
    per_token_scales = inputs[2] if len(inputs) > 2 else None
    return torch_npu.npu_moe_re_routing(
        tokens,
        expert_token_num_per_rank,
        per_token_scales=per_token_scales,
        expert_token_num_type=int(attrs.get("expert_token_num_type", 1)),
        idx_type=int(attrs.get("idx_type", 0)),
    )


def nms_ref(inputs, attrs):
    import torchvision

    return torchvision.ops.nms(
        inputs[0], inputs[1], float(attrs.get("iou_threshold", 0.5))
    )


def quant_matmul_ref(inputs, attrs):
    """torch_npu.npu_quant_matmul(x1, x2, scale, pertoken_scale=None, bias=None, output_dtype=...)."""
    import torch_npu

    if hasattr(torch_npu, "npu_quant_matmul"):
        x1, x2, scale = inputs[0], inputs[1], inputs[2]
        pertoken = inputs[3] if len(inputs) > 3 else None
        bias = inputs[4] if len(inputs) > 4 else None
        kwargs = {}
        if "output_dtype" in attrs:
            kwargs["output_dtype"] = getattr(torch, attrs["output_dtype"], None)
        return torch_npu.npu_quant_matmul(
            x1, x2, scale, pertoken_scale=pertoken, bias=bias, **kwargs
        )
    return None


def roi_align_ref(inputs, attrs):
    """torch_npu.npu_roi_align(features, rois, spatial_scale, pooled_height,
    pooled_width, sample_num, roi_end_mode). `aligned` is exposed by the task
    schema as a bool but the NPU binding takes int `roi_end_mode`; mapping
    follows torch_npu/contrib/module/roi_align.py: True→3, False→0."""
    import torch_npu

    features, rois = inputs[0], inputs[1]
    roi_end_mode = 3 if bool(attrs.get("aligned", False)) else 0
    return torch_npu.npu_roi_align(
        features,
        rois,
        float(attrs.get("spatial_scale", 1.0)),
        int(attrs.get("outputHeight", attrs.get("pooled_height", 7))),
        int(attrs.get("outputWidth", attrs.get("pooled_width", 7))),
        int(attrs.get("sampling_ratio", attrs.get("sample_num", 2))),
        roi_end_mode,
    )


def top_k_ref(inputs, attrs):
    return torch.topk(
        inputs[0],
        int(attrs.get("k", 1)),
        dim=int(attrs.get("dim", -1)),
        largest=bool(attrs.get("largest", True)),
    )


def transpose_ref(inputs, attrs):
    """torch.permute returns a view (no kernel); .contiguous() forces the actual
    Transpose kernel so msprof can measure it."""
    return torch.permute(inputs[0], tuple(attrs.get("perm"))).contiguous()


def unique_ref(inputs, attrs):
    res = torch.unique(
        inputs[0], return_inverse=bool(attrs.get("return_inverse", False))
    )
    return res


def weight_quant_batch_matmul_ref(inputs, attrs):
    """Use torch_npu.npu_weight_quant_batchmatmul (single fused kernel).

    API: (x, weight, antiquant_scale, antiquant_offset=None, quant_scale=None,
          quant_offset=None, bias=None, antiquant_group_size=0, inner_precise=0)
    cases.csv inputs order: [x, weight, antiquant_scale, antiquant_offset?, bias?]
    """
    import torch_npu

    x, w, antiscale = inputs[0], inputs[1], inputs[2]
    antioffset = inputs[3] if len(inputs) > 3 else None
    bias = inputs[4] if len(inputs) > 4 else None
    return torch_npu.npu_weight_quant_batchmatmul(
        x,
        w,
        antiscale,
        antiquant_offset=antioffset,
        quant_scale=None,
        quant_offset=None,
        bias=bias,
    )


def conv_3d_backprop_filter_ref(inputs, attrs):
    """Compute weight-gradient for Conv3D using torch_npu's fused backward op.

    Cases.csv supplies (x, grad). The current spec also lists `filter_size`
    in attrs; npu_conv3d_backward takes the weight tensor by SHAPE (an empty
    tensor of the same shape is enough — the kernel only reads the metadata).
    output_mask=[False, True, False] asks just for grad_weight, skipping
    grad_input / grad_bias and saving the corresponding compute.
    """
    import torch
    import torch_npu

    if not hasattr(torch_npu, "npu_conv3d_backward"):
        return None
    x, grad = inputs[0], inputs[1]
    filter_size = attrs.get("filter_size") or attrs.get("filterSize")
    if filter_size is None:
        raise ValueError("conv_3d_backprop_filter_ref: attrs.filter_size missing")
    strides = list(attrs.get("strides", [1, 1, 1]))
    pads = list(attrs.get("pads", [0, 0, 0, 0, 0, 0]))
    # spec uses 6-elem (front/back/top/bottom/left/right); aclnn takes 3 (sym front/top/left)
    padding = [pads[0], pads[2], pads[4]] if len(pads) == 6 else list(pads)
    dilations = list(attrs.get("dilations", [1, 1, 1]))
    groups = int(attrs.get("groups", 1))
    weight = torch.empty(filter_size, dtype=x.dtype, device=x.device)
    grad_in, grad_w, grad_b = torch_npu.npu_conv3d_backward(
        x,
        grad,
        weight,
        stride=strides,
        padding=padding,
        dilation=dilations,
        groups=groups,
        output_mask=[False, True, False],
    )
    return grad_w


def strided_slice_ref(inputs, attrs):
    """TF-style strided_slice via golden.py implementation.

    cases.yaml attrs: begin, end, strides (lists), plus optional masks
    (begin_mask, end_mask, ellipsis_mask, shrink_axis_mask, new_axis_mask).
    """
    import importlib.util
    from pathlib import Path

    x = inputs[0]
    _golden = None
    if _golden is None:
        spec = importlib.util.spec_from_file_location(
            "_golden_ss",
            Path(__file__).resolve().parents[3]
            / "tasks"
            / "level3"
            / "strided_slice"
            / "golden.py",
        )
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
        strided_slice_ref._golden_mod = _mod
    return strided_slice_ref._golden_mod.strided_slice(
        x,
        begin=list(attrs.get("begin", [])),
        end=list(attrs.get("end", [])),
        strides=list(attrs.get("strides", [])),
        begin_mask=int(attrs.get("begin_mask", 0)),
        end_mask=int(attrs.get("end_mask", 0)),
        ellipsis_mask=int(attrs.get("ellipsis_mask", 0)),
        shrink_axis_mask=int(attrs.get("shrink_axis_mask", 0)),
        new_axis_mask=int(attrs.get("new_axis_mask", 0)),
    )


# Operators registered. Omit (skip method 3) for ops with no torch / torch_npu /
# torchvision equivalent or whose semantic divergence makes the comparison
# meaningless (NZ output formats, custom CV-mix kernels):
#
# OMITTED: dilation_2d (no torch.nn equiv), engram (custom op), mhc_sinkhorn
#   (custom op).
REGISTRY = {
    "level3/adaptive_avg_pool_3d": adaptive_avg_pool_3d_ref,
    "level3/add_rms_norm_dynamic_quant": add_rms_norm_dynamic_quant_ref,
    "level3/conv_2d": conv_2d_ref,
    "level3/conv_3d_backprop_filter": conv_3d_backprop_filter_ref,
    "level3/depthwise_conv_2d": depthwise_conv_2d_ref,
    "level3/dequant_swiglu_quant": dequant_swiglu_quant_ref,
    "level3/grouped_matmul": grouped_matmul_ref,
    "level3/moe_finalize_routing": moe_finalize_routing_ref,
    "level3/moe_gating_top_k_softmax": moe_gating_top_k_softmax_ref,
    "level3/moe_re_routing": moe_re_routing_ref,
    "level3/nms": nms_ref,
    "level3/quant_matmul": quant_matmul_ref,
    "level3/roi_align": roi_align_ref,
    "level3/strided_slice": strided_slice_ref,
    "level3/top_k": top_k_ref,
    "level3/transpose": transpose_ref,
    "level3/unique": unique_ref,
    "level3/weight_quant_batch_matmul": weight_quant_batch_matmul_ref,
}
