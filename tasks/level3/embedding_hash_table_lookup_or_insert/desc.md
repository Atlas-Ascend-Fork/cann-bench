# EmbeddingHashTableLookupOrInsert 算子 API 描述

## 1. 算子简介

直接映射哈希表的动态查找/插入算子（season-2 重设计版）。哈希表状态（每个
slot 的占用标记与已存 embedding）作为**显式张量输入**传入：查询 key 命中
已占用 slot 时返回表中实际存储的向量；未命中时按确定性规则"插入"一个由
key 派生的初始化向量并返回。输出同时给出每个查询的命中/插入/过滤状态。

**主要应用场景**：
- 推荐系统中的动态特征嵌入（在线 serving 的一个 batch：表已有历史内容，
  新特征首次出现时现场初始化）
- 大规模稀疏特征的在线学习
- 哈希表形式的 embedding 管理

**算子特征**：
- 难度等级：L3（HashTable）
- 直接映射（direct-mapped）哈希、slot 级占用判定、批内 first-insert-wins
- 仅支持 Ascend 950PR/Ascend 950DT 产品

## 2. 算子定义

### 功能说明

对 `keys` 按 C 序展平后的每个查询 key `k`（下标 `i = 0..N-1`）：

1. **过滤**：若 `filter_key_flag=True` 且 `k == filter_key`，输出行全 0，
   `flags[i]=0`，**不探查、不插入**（该查询对哈希表不可见）。
2. 否则计算 `slot = k mod bucket_size`：
   - **命中**：`table_keys[slot] >= 0`（slot 被占用）时，
     `values[i] = table_values[slot]`，`flags[i]=2`。
   - **未命中**：`table_keys[slot] < 0`（slot 为空）时，本次调用内映射到
     该 slot 的全部未过滤查询中，**展平下标最小者 w 执行插入**：插入向量
     为 `init_vector(keys[w])`（定义见下），`flags[w]=1`；该 slot 的其余
     查询（下标 > w）命中这条新插入项，返回同一向量，`flags`=2。

语义说明（设计约定，须按此实现）：
- 占用判定是 **slot 级**的：不比较 key 本身是否相等（direct-mapped 语义，
  碰撞即命中该 slot 现存内容）。
- 批内插入的可见性由 first-insert-wins 规则**完全确定**，与并行执行顺序
  无关（赢家 = 展平下标最小者，可用 scatter-min 并行求得）。
- 表的持久化更新不在输出中返回（无状态评测只校验 `values` 与 `flags`）。

### 数学公式

$$
\text{slot}_i = k_i \bmod \text{bucket\_size}
$$

$$
\text{values}[i] = \begin{cases}
  \mathbf{0} & k_i \text{ 被过滤} \\
  \text{table\_values}[\text{slot}_i] & \text{table\_keys}[\text{slot}_i] \ge 0 \\
  \text{init\_vector}(k_{w(\text{slot}_i)}) & \text{否则（} w(s) = \min\{\,i : \text{slot}_i = s,\ i \text{ 未过滤}\,\}\text{）}
\end{cases}
$$

### init_vector 定义（逐位精确，必须按位一致）

对 key `k`（`0 <= k < 2^31`）与分量下标 `j`（`0 <= j < embedding_dim`），
全部中间量按 **uint32 回绕算术**（即对 2^32 取模）：

```
s    = lo32(k * 2654435761)            # Knuth 乘法散列常数
h_j  = s XOR lo32(j * 2654435769)      # 黄金分割常数
v[j] = float32(h_j & 0x7FF) * 2^-10 - 1.0      # 取低 11 位，映射到 [-1, 0.9990234375]
```

`h_j & 0x7FF < 2048`，乘 `2^-10`（精确的 2 的幂）后减 1.0，每一步在
float32 中**精确**，因此不同实现必须逐位一致。

**算例**（`k = 7, embedding_dim = 4`）：`s = 1401181143`；
`h = [1401181143, 3451071086, 1877647525, 2300722940]`；
`h & 0x7FF = [983, 622, 165, 1788]`；
`v = [-0.0400390625, -0.392578125, -0.8388671875, 0.74609375]`。

## 3. 接口规范

### 算子原型

```python
cann_bench.embedding_hash_table_lookup_or_insert(
    Tensor table_keys,
    Tensor table_values,
    Tensor keys,
    int bucket_size,
    int embedding_dim,
    bool filter_key_flag=False,
    int filter_key=-1
) -> (Tensor values, Tensor flags)
```

### 输入参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| table_keys | Tensor | shape `[bucket_size]`，int64。`>= 0` 表示该 slot 被占用，`< 0` 表示空 slot |
| table_values | Tensor | shape `[bucket_size, embedding_dim]`，float32。各 slot 已存的 embedding（空 slot 的行为无效数据，不得被读出到输出） |
| keys | Tensor | 任意形状，int64，**取值非负**。按 C 序展平为 N 个查询 |

### 必选属性

| 参数 | 类型 | 描述 |
|------|------|------|
| bucket_size | int | 哈希表容量，必须等于 `table_keys.shape[0]` 与 `table_values.shape[0]` |
| embedding_dim | int | value 维度，必须等于 `table_values.shape[1]` |

### 可选属性

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| filter_key_flag | bool | False | True 时启用 filter_key 过滤 |
| filter_key | int | -1 | 被过滤的 key 值（仅 flag=True 时生效；生效时为非负值） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| values | (N, embedding_dim) | float32 | 命中值 / 插入的初始化向量 / 过滤的全 0 行 |
| flags | (N,) | int32 | 0=被过滤，1=本查询执行了插入，2=命中（含批内新插入项） |

### 数据类型

| table_keys | table_values | keys | values | flags |
|------------|--------------|------|--------|-------|
| int64 | float32 | int64 | float32 | int32 |

### 支持范围（硬性约束，case 不会超出）

- `bucket_size ∈ [64, 1048576]`；`embedding_dim ∈ [8, 1024]`
- `N = keys.numel ∈ [16, 1048576]`；`N·embedding_dim ≤ 2^28`；
  `bucket_size·embedding_dim ≤ 2^28`
- `keys` 取值 `∈ [0, 2^31 - 1]`；`table_keys` 取值 `∈ [-2^31 + 1, 2^31 - 1]`
- `table_values` 为有限 float32（本套 case 取值 `∈ [-10, 10]`，无 NaN/Inf）
- 仅支持 Ascend 950PR/Ascend 950DT 产品

### 规则与约束

- 输出 `values` 的行序与 `keys` 的 C 序展平一一对应
- 空 slot 的 `table_values` 行不得泄漏到任何输出行
- 被过滤查询对表零影响（不占用、不插入、不参与 first-insert-wins）
- 批内 first-insert-wins 的赢家判定与实现的并行方式无关，结果必须确定

## 4. 精度要求

计算结果与参考实现对比，需满足以下阈值：

| 输出 | 验证方式 | rtol | atol |
|------|---------|------|------|
| values (float32) | 相对误差 | 1e-4 | 1e-4 |
| flags (int32) | 逐元素精确相等 | — | — |

说明：golden 的三条取值路径（表值 gather、init_vector 精确公式、全 0 行）
在 float32 中均无舍入自由度，正确实现可逐位复现，阈值轻松可达。

## 5. 标准 Golden 代码

```python
import torch

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
        w = winner[slot[idx]]
        values[idx] = _init_vectors(kq[w], d)
        flags[idx] = torch.where(w == idx, 1, 2).to(torch.int32)

    return values, flags
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

bucket_size, embedding_dim = 1024, 128
# 表状态为显式输入：~75% slot 占用
table_keys = torch.randint(-256, 768, (bucket_size,), dtype=torch.int64, device="npu")
table_values = torch.randn(bucket_size, embedding_dim, device="npu")
keys = torch.randint(0, 65536, (4096,), dtype=torch.int64, device="npu")

values, flags = cann_bench.embedding_hash_table_lookup_or_insert(
    table_keys, table_values, keys,
    bucket_size=bucket_size, embedding_dim=embedding_dim)
# values: (4096, 128) float32;  flags: (4096,) int32
```

### 与 season-1 版本的差异（重设计说明）

season-1 版本的表内容不可传入（`table_handle` 仅为句柄占位），golden 每次
从空表开始，输出恒为 `default_value` 常数填充，与所有张量输入解耦。本版
将表状态改为显式输入、插入值改为 key 派生的确定性初始化向量，输出依赖全部
输入；`filter_mode/counter、default_key/default_value` 等依赖跨调用状态或
产生常数输出的属性一并移除。

### 相关算子

- **EmbeddingHashTableImport**: 哈希表导入算子
- **EmbeddingHashTableApplyAdamW**: 哈希表 AdamW 优化器更新算子
