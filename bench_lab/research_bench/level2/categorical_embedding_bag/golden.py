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
    _, features, _ = indices.shape
    vocab, dim = embedding_table.shape
    if bias is not None and bias.shape != (features, dim):
        raise ValueError("bias must be [F,D]")
    idx = indices.to(torch.long)
    valid = idx != int(padding_idx)
    idx = torch.remainder(idx, vocab)
    emb = embedding_table[idx].float().masked_fill(~valid.unsqueeze(-1), 0.0)
    pooled = emb.sum(dim=2)
    if mean_pool:
        denom = valid.sum(dim=2, keepdim=True).clamp_min(1).float()
        pooled = pooled / denom
    if bias is not None:
        pooled = pooled + bias.float().unsqueeze(0)
    return pooled.to(embedding_table.dtype)
