#!/usr/bin/python3
# coding=utf-8

import torch


def kv_cache_append(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("k_cache and v_cache must be 4D tensors")
    if new_k.dim() != 3 or new_v.dim() != 3:
        raise ValueError("new_k and new_v must be 3D tensors")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    if new_k.shape != new_v.shape:
        raise ValueError("new_k and new_v must have the same shape")
    if new_k.shape[1:] != k_cache.shape[2:]:
        raise ValueError("new token H/D dimensions must match cache")
    if slot_mapping.dim() != 1 or slot_mapping.shape[0] != new_k.shape[0]:
        raise ValueError("slot_mapping must be [T]")
    if not all(torch.isfinite(tensor).all() for tensor in (k_cache, v_cache, new_k, new_v)):
        raise ValueError("cache and new token tensors must contain finite values")

    k_out = k_cache.clone()
    v_out = v_cache.clone()
    total_slots = k_cache.shape[0] * k_cache.shape[1]
    slots = slot_mapping.to(torch.long)
    if torch.any(slots < 0) or torch.any(slots >= total_slots):
        raise ValueError("slot_mapping values must be in [0, num_blocks * page_block_size)")

    flat_k = k_out.reshape(total_slots, *k_cache.shape[2:])
    flat_v = v_out.reshape(total_slots, *v_cache.shape[2:])
    for i in range(new_k.shape[0]):
        slot = int(slots[i].item())
        flat_k[slot] = new_k[i]
        flat_v[slot] = new_v[i]

    return k_out, v_out
