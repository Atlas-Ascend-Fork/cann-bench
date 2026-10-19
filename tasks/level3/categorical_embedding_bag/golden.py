#!/usr/bin/python3
# coding=utf-8

import torch


def categorical_embedding_bag(
    indices: torch.Tensor,
    embedding_table: torch.Tensor,
    bias: torch.Tensor = None,
    mean_pool: bool = False,
    padding_idx: int = -1,
) -> torch.Tensor:
    if indices.dim() != 3 or embedding_table.dim() != 2:
        raise ValueError("indices must be [B,F,K], embedding_table must be [V,D]")
    batch, features, bag = indices.shape
    vocab, dim = embedding_table.shape
    if bias is not None and bias.shape != (features, dim):
        raise ValueError("bias must be [F,D]")
    idx = indices.to(torch.long)
    valid = idx != int(padding_idx)
    if valid.any() and ((idx[valid] < 0) | (idx[valid] >= vocab)).any():
        raise ValueError("indices must equal padding_idx or be in [0,V)")
    if not torch.isfinite(embedding_table).all() or (bias is not None and not torch.isfinite(bias).all()):
        raise ValueError("embedding_table and bias must contain finite values")
    safe_idx = idx.masked_fill(~valid, 0)
    emb = embedding_table[safe_idx].float() * valid.unsqueeze(-1).float()
    pooled = emb.sum(dim=2)
    if mean_pool:
        denom = valid.sum(dim=2, keepdim=True).clamp_min(1).float()
        pooled = pooled / denom
    if bias is not None:
        pooled = pooled + bias.float().unsqueeze(0)
    return pooled.to(embedding_table.dtype)
