import torch

"""
EmbeddingHashTableLookupOrInsert 算子 Torch Golden 参考实现（season-2 重设计版）

直接映射哈希表的查找/插入（表状态为显式输入）：
- table_keys[slot] >= 0 表示 slot 被占用；< 0 表示空。
- 对每个查询 key k（按 C 序展平）：slot = k mod bucket_size。
  * 命中（slot 占用）：返回 table_values[slot]，flag=2。
  * 未命中（slot 空）：本次调用内映射到该 slot 的未过滤查询中，
    展平下标最小者执行插入（插入向量 = init_vector(k)，见下），其 flag=1；
    其余映射到同一 slot 的查询命中新插入项，返回同一向量，flag=2。
  * 过滤（filter_key_flag=True 且 k == filter_key）：返回全 0 行，flag=0，
    不探查也不插入。
- 插入初始化向量 init_vector(k)[j]（全部按 uint32 回绕算术，逐位精确）：
    s    = lo32(k * 2654435761)
    h_j  = s XOR lo32(j * 2654435769)          # j = 0 .. embedding_dim-1
    v[j] = float32(h_j & 0x7FF) * 2**-10 - 1.0  # ∈ [-1, 0.9990234375]，逐位精确
  实现用 int64 算术 + 0xFFFFFFFF 掩码模拟 uint32 回绕（k < 2^31 时精确）。
"""

_MASK32 = 0xFFFFFFFF
_KNUTH = 2654435761
_GOLD = 2654435769


def _init_vectors(keys_i64: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """init_vector(k) 批量计算：keys_i64 [M] int64 -> [M, D] float32（逐位精确）。"""
    s = (keys_i64.to(torch.int64) * _KNUTH) & _MASK32                     # lo32
    j = (torch.arange(embedding_dim, dtype=torch.int64) * _GOLD) & _MASK32
    h = torch.bitwise_xor(s.unsqueeze(1), j.unsqueeze(0))
    return (h & 0x7FF).to(torch.float32) * float(2.0 ** -10) - 1.0


def embedding_hash_table_lookup_or_insert(
    table_keys: torch.Tensor,
    table_values: torch.Tensor,
    keys: torch.Tensor,
    bucket_size: int,
    embedding_dim: int,
    filter_key_flag: bool = False,
    filter_key: int = -1,
):
    """
    Args:
        table_keys:   [bucket_size] int64，slot 占用标记（>=0 占用，<0 空）
        table_values: [bucket_size, embedding_dim] float32，已存 embedding
        keys:         任意形状 int64（非负），按 C 序展平处理
        bucket_size / embedding_dim: 必选属性，须与张量形状一致
        filter_key_flag / filter_key: 过滤开关与被过滤 key

    Returns:
        values: (N, embedding_dim) float32
        flags:  (N,) int32   0=被过滤 1=本查询执行插入 2=命中
    """
    tk = table_keys.detach().reshape(-1).to(torch.int64)
    tv = table_values.detach().reshape(int(bucket_size), -1).to(torch.float32)
    kq = keys.detach().reshape(-1).to(torch.int64)

    n = kq.numel()
    b = int(bucket_size)
    d = int(embedding_dim)

    values = torch.zeros((n, d), dtype=torch.float32)
    flags = torch.zeros(n, dtype=torch.int32)

    if filter_key_flag:
        filtered = kq == int(filter_key)
    else:
        filtered = torch.zeros(n, dtype=torch.bool)

    active = ~filtered
    slot = torch.where(active, kq % b, torch.zeros_like(kq))
    occupied0 = tk >= 0

    hit0 = active & occupied0[slot]
    values[hit0] = tv[slot[hit0]]
    flags[hit0] = 2

    # 未命中：每个空 slot 由展平下标最小的查询执行插入（first-insert-wins）
    miss = active & ~occupied0[slot]
    if bool(miss.any()):
        idx = torch.nonzero(miss, as_tuple=False).reshape(-1)
        winner = torch.full((b,), n, dtype=torch.int64)
        winner.scatter_reduce_(0, slot[idx], idx, reduce="amin", include_self=True)
        w = winner[slot[idx]]                      # 每个 miss 查询对应的赢家下标
        values[idx] = _init_vectors(kq[w], d)      # init_vector(赢家的 key)
        flags[idx] = torch.where(w == idx, 1, 2).to(torch.int32)

    return values, flags
