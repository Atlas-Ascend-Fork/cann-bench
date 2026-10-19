# DynamicMxQuant 算子 API 描述

## 1. 算子简介

对输入张量执行 Microscaling (MX) 动态量化。在给定的轴 axis 上，按 blocksize 个数分组，计算每组的量化尺度 mxscale（E8M0），然后将每组元素除以 mxscale 并转换到目标低精度类型（FP4/FP8），输出量化结果 y 和对应的量化 scale。

**主要应用场景**：
- LLM 推理中的权重/激活量化压缩（FP4/FP8 低精度部署）
- MX 格式量化训练（OCP Microscaling 标准兼容）
- 大模型 KV Cache 压缩存储
- 混合精度推理中的动态量化前处理

**算子特征**：
- 难度等级：L4（Quantization）
- 1 输入，2 输出，4 个属性参数
- 支持 ND 格式输入，1-7 维
- 仅支持标准 OCP MX 量化算法（shared_exp = floor(log2(amax)) - emax），4 种目标量化格式，3 种舍入模式
- 分组量化：按 blocksize 将指定轴切分为若干 block，每个 block 独立计算 scale
- 支持硬件：Ascend 950PR / 950DT

## 2. 算子定义

### 数学公式

仅支持标准 OCP MX 算法。将输入 x 在 axis 维度上按 $k = \text{blocksize}$ 个数分组，一组 k 个数 $\{V_i\}_{i=1}^{k}$ 动态量化为 $\{mxscale, \{P_i\}_{i=1}^{k}\}$，k = blocksize：

$$
shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
mxscale = 2^{shared\_exp}\\
P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
$$

- 量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。
- emax: 对应数据类型的最大正则数的指数位。
- 全零块 $shared\_exp = -\infty$（mxscale 编码为 0x00）；shared_exp 超出 E8M0 范围时：大于 127 置为 NaN，小于 -127 截断到 -127。
- 仅支持此标准算法，不提供其他 scale 计算方法。

emax 对照：

|   DataType    | emax |
| :-----------: | :--: |
|  FLOAT4_E2M1  |  2   |
|  FLOAT4_E1M2  |  0   |
| FLOAT8_E4M3FN |  8   |
|  FLOAT8_E5M2  |  15  |

### 舍入模式说明

- **rint**: 银行家舍入（tie to even），0.5 取偶数
- **round**: 标准四舍五入（tie away from zero）
- **floor**: 向负无穷方向舍入

注：当dstType为FP8（35/36）时，仅支持 "rint" 模式；当dstType为FP4（40/41）时，支持 "rint"、"floor"、"round" 三种模式。

## 3. 接口规范

### 算子原型

```python
cann_bench.dynamic_mx_quant(Tensor x, int axis=-1, str round_mode="rint", int dst_type=40, int blocksize=32) -> (Tensor y, Tensor mxscale)
```

### 输入参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| x | 输入 | 待量化的输入张量 | 目的类型为FP4时最后一维必须是偶数，不支持空Tensor | FLOAT16, BFLOAT16 | ND | 1-7 | 支持 |
| axis | 输入 | 量化发生的轴 | 取值范围 [-D, D-1]，D 为 x 的维数 | INT64 | - | - | - |
| round_mode | 输入 | 舍入模式 | FP4 支持 {"rint","floor","round"}；FP8 仅支持 {"rint"} | STRING | - | - | - |
| dst_type | 输入 | 目标量化数据类型 | {35:FP8_E5M2, 36:FP8_E4M3FN, 40:FP4_E2M1, 41:FP4_E1M2} | INT64 | - | - | - |
| blocksize | 输入 | 每组量化的元素个数 | 32 的倍数，不能为 0，不超过 1024 | INT64 | - | - | - |

### 输出

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| y (yOut) | 输出 | 量化结果 | shape 与输入 x 一致（FP4时最后一维减半，两元素打包为一个 uint8） | FP4_E2M1, FP4_E1M2, FP8_E4M3FN, FP8_E5M2 | ND | 1-7 | 支持 |
| mxscale (mxscaleOut) | 输出 | 每个分组对应的量化尺度 | rank = rank(x)，详见 shape 规则 | FLOAT8_E8M0 (uint8) | ND | 1-7 | 支持 |

### mxscale 输出 shape 规则

- `rank(mxscaleOut) = rank(x)`
- `axis_norm = axis if axis >= 0 else axis + rank(x)`
- `mxscaleOut.shape[axis_norm] = ceil(x.shape[axis_norm] / blocksize)`
- 其他维度与输入 x 一致
- 第 b 块的尺度直接存于量化轴维下标 b 处；不做偶数补齐、不引入尾维 2、不涉及交织处理

### 数据类型

| x dtype | y dtype | mxscale dtype |
|---------|---------|---------------|
| float16 | FP4_E2M1 / FP4_E1M2 / FP8_E4M3FN / FP8_E5M2 | FP8_E8M0 (uint8) |
| bfloat16 | FP4_E2M1 / FP4_E1M2 / FP8_E4M3FN / FP8_E5M2 | FP8_E8M0 (uint8) |

### 规则与约束

- 仅支持标准 OCP MX 量化算法，不支持其他 scale 计算方法
- `blocksize` 必须为 32 的倍数，最大 1024，不能为 0
- axis 维不是 blocksize 整数倍时，块尾不足部分按 0 补齐（axis 维不足 blocksize 同样按 0 补齐）
- FP8 目标类型仅支持 `round_mode="rint"`
- FP4 输出时输入最后一维必须为偶数
- 确定性计算：aclnnDynamicMxQuantV2 默认确定性实现

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `rank(x)`（输入维度数） | 1 ~ 7 | NPU API 硬性要求；cases.yaml 实测 2 ~ 5 维 |
| 各维度大小 `dim_i` | 1 ~ 65536 | cases.yaml 实测最小 2、最大 16384 |
| 张量总元素数 | 1 ~ 2^30 | cases.yaml 实测最大约 16M（16×64×16384） |
| 量化轴维度大小 | 1 ~ 65536 | 不足一块按 0 补齐；cases.yaml 实测 64 ~ 16384 |
| blocksize | 32, 64, 128, 256, 512, 1024 | 32 的倍数，不能为 0 |
| axis | [-rank, rank-1] | 负值自动规范化；cases.yaml 实测 -1, 0, 1, 2, 3 |
| dst_type | 35, 36, 40, 41 | 35=FP8_E5M2, 36=FP8_E4M3FN, 40=FP4_E2M1, 41=FP4_E1M2 |
| 输入 dtype | float16 / bfloat16 | NPU API 不支持 float32 |
| FP4 最后一维 | 偶数 | dst_type=40/41 时最后一维必须为偶数 |

## 4. 精度要求

本算子输出为低精度量化格式的 uint8 字节表示，采用**字节级精确匹配**进行验证。

**验证方式**：
- 将 golden 和 NPU 的输出均转为 uint8 字节表示
- y 输出：FP8 直接 view 为 uint8；FP4 两两打包为 uint8（低 nibble + 高 nibble << 4）
- mxscale 输出：FP8_E8M0 直接 view 为 uint8
- 逐字节精确比较，mismatch_count = 0 判定为通过

**通过标准**：

| 指标 | 通过条件 |
|------|---------|
| mismatch_count (y) | = 0 |
| mismatch_count (mxscale) | = 0 |
| max_abs_diff | = 0 |

## 5. 标准 Golden 代码（精简摘录，完整实现见 `golden.py`）

```python
import torch

# dst_type (int64 枚举) -> (torch dtype 名, emax, exp_bits, mantissa_bits)
# emax: 目标类型最大正则数的指数；exp_bits/mantissa_bits 用于尾数舍入仿真
# FP8 (35/36) 有 torch 原生 dtype；FP4 (40/41) 无原生 dtype，由 _FP4_POS_TABLE 查表编码
_DST_TYPE_INFO = {
    35: ("float8_e5m2", 15, 5, 2),
    36: ("float8_e4m3fn", 8, 4, 3),
    40: ("float4_e2m1", 2, 2, 1),
    41: ("float4_e1m2", 0, 1, 2),
}

# FP4 正值查找表: 表下标即无符号 code；负值 code = 正值 code + 8；+0.0 -> 0、-0.0 -> 8（signbit 区分正负零）
_FP4_POS_TABLE = {
    "float4_e2m1": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
    "float4_e1m2": (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
}


def _shared_exp_alg0(amax: torch.Tensor, ele_emax: int) -> torch.Tensor:
    """标准 OCP 算法: shared_exp = floor(log2(amax)) - emax；全零块为 -inf。

    floor(log2(amax)) 直接提取 amax 的 fp32 指数域，正规数下与数学 floor(log2)
    严格一致（float32 log2 在二次幂下边界会舍入到整数 k，使 mxscale 偏大 2 倍）。
    """
    abs_bits = amax.contiguous().view(torch.int32) & 0x7FFFFFFF
    exp = ((abs_bits >> 23) - 127).to(torch.float32) - ele_emax
    exp = torch.where(torch.isnan(amax), torch.full_like(exp, float("nan")), exp)
    exp = torch.where(torch.isinf(amax), torch.full_like(exp, float("inf")), exp)
    # 全零块为 -inf（mxscale 编码 0x00）
    return torch.where(amax == 0, torch.full_like(exp, -float("inf")), exp)


def _encode_fp4(y: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """float32 量化值 -> FP4 packed uint8。

    量化构造保证 y 的每个元素可被目标 FP4 格式精确表示，直接查表编码：
    无符号 code 为 |y| 在正值表中的下标，负值（含 -0.0，signbit 区分）code+8；
    末维两两打包为 byte = code[2i] | (code[2i+1] << 4)，
    输出 shape = y.shape[:-1] + (y.shape[-1] // 2,)。
    """
    pos_vals = torch.tensor(_FP4_POS_TABLE[dtype_name], dtype=torch.float32)
    match = y.abs().unsqueeze(-1) == pos_vals
    codes = torch.argmax(match.to(torch.uint8), dim=-1).to(torch.uint8)
    codes = codes + torch.where(torch.signbit(y), 8, 0).to(torch.uint8)  # 负值 code+8
    codes = codes.reshape(y.shape[:-1] + (y.shape[-1] // 2, 2))
    return codes[..., 0] | (codes[..., 1] << 4)


def dynamic_mx_quant(
    x: torch.Tensor,
    axis: int = -1,
    round_mode: str = "rint",
    dst_type: int = 40,
    blocksize: int = 32,
):
    """MX 动态量化 (FP8/FP4 目标类型)。

    参数校验: axis 范围 [-D, D-1]；blocksize 为 32 的倍数且 <= 1024（float32
    输入仅支持 32）；round_mode in {rint, floor, round}；FP4 目标类型要求 x
    最后一维为偶数。

    核心流程:
    1. 统一到 float32，把量化轴换到末尾，块尾不足按 0 补齐 -> [.., n_blocks, blocksize]
    2. 每块 amax -> _shared_exp_alg0 -> E8M0 [-127, 127] 钳制 -> mxscale_val = 2^share_exp
    3. 元素级量化：去尺度 -> 私有指数截断到最小正则指数（FP8: -(2^(exp_bits-1))+2；
       FP4: -(2^(exp_bits-1))+exp_bits）-> 尾数定点化后按 round_mode 舍入 -> 还原
    4. 目标格式编码：FP8 用 _to_fp8 位级编码为 fp8 dtype；FP4 用 _encode_fp4
       查表编码并两两打包为 uint8
    5. mxscale 布局：scale_val = mxscale_val.squeeze(-1)，非尾轴时 movedim(-1, axis)，
       再 _to_fp8(scale_val, torch.float8_e8m0fnu)；逐块尺度张量，量化轴维 = n_blocks，
       其余维度与 x 一致（无补偶/配对/交织）
    """
    ...  # 参数校验、分块、量化与编码细节见 golden.py

    # mxscale 布局：逐块尺度张量，量化轴维 = n_blocks，其余维度与 x 一致（无补偶/配对/交织）
    scale_val = mxscale_val.squeeze(-1)
    if not tail_axis:
        scale_val = scale_val.movedim(-1, axis)
    mxscale = _to_fp8(scale_val, torch.float8_e8m0fnu)

    return y, mxscale
```

完整实现（含 `_pow2` / `_to_fp8` / `_round_mantissa` 等辅助函数与参数校验）见 `golden.py`。

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# 尾轴 FP4_E2M1 量化（y 为两两打包的 uint8）
x = torch.randn(128, 1024, dtype=torch.float16, device="npu")
y, mxscale = cann_bench.dynamic_mx_quant(x, axis=-1, round_mode="rint", dst_type=40, blocksize=32)
# y.shape = (128, 512), y.dtype = torch.uint8（两元素打包为一个 uint8）
# mxscale.shape = (128, 32), mxscale.dtype = torch.float8_e8m0fnu

# 尾轴 FP8_E4M3FN 量化
x = torch.randn(32, 4096, dtype=torch.float16, device="npu")
y, mxscale = cann_bench.dynamic_mx_quant(x, axis=-1, round_mode="rint", dst_type=36, blocksize=32)
# y.shape = (32, 4096), y.dtype = torch.float8_e4m3fn
# mxscale.shape = (32, 128), mxscale.dtype = torch.float8_e8m0fnu

# 非尾轴量化 (axis=0)
x = torch.randn(64, 512, dtype=torch.float16, device="npu")
y, mxscale = cann_bench.dynamic_mx_quant(x, axis=0, round_mode="rint", dst_type=41, blocksize=32)
# y.shape = (64, 256), mxscale.shape = (2, 512)
```
