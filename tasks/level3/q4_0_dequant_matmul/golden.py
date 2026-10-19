#!/usr/bin/python3
# coding=utf-8

import torch


def _unpack_q4_0(
    qweight: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    blocks_per_row = in_features // 32
    packed = qweight.to(torch.uint8).reshape(out_features, blocks_per_row, 16)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.cat((low, high), dim=-1).reshape(out_features, in_features).to(torch.int16)
    return (codes - 8).to(torch.float32)


def q4_0_dequant_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    out_features: int,
    in_features: int,
    block_size: int = 32,
    quant_type: str = "q4_0",
) -> torch.Tensor:
    if quant_type != "q4_0":
        raise ValueError("first version only supports q4_0")
    if x.shape[-1] != in_features:
        raise ValueError("x.shape[-1] must equal in_features")
    if block_size != 32:
        raise ValueError("GGUF Q4_0 requires block_size=32")
    if in_features % block_size != 0:
        raise ValueError("in_features must be divisible by 32 for GGUF Q4_0")
    expected_packed = out_features * in_features // 2
    if qweight.numel() != expected_packed:
        raise ValueError("qweight must contain exactly out_features * in_features / 2 bytes")
    expected_scale_shape = (out_features, in_features // block_size)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError("scales must have shape [out_features, in_features / 32]")

    q = _unpack_q4_0(qweight, out_features, in_features).to(x.device)
    expanded_scales = scales.float().repeat_interleave(block_size, dim=1)
    weight = q * expanded_scales.to(q.device)
    out = torch.matmul(x.float(), weight.t())
    return out.to(x.dtype)
