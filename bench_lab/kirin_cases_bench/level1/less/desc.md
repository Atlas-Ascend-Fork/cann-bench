# Less 算子 API 描述

## 1. 算子简介

Less：逐元素“小于”比较，输出 bool。

**开发目标**：Kirin 端侧 NPU（AscendC 直调 kernel）。

**算子特征**：
- 难度等级：L1（Elementwise）
- 类别：L1-Elementwise
- 双输入单输出，`x1`/`x2` 同 shape 同 dtype，输出与输入同 shape
- Kernel 将输入展平为 1D 处理，tiling 仅依赖总元素数 `totalLen`

## 2. 算子定义

逐元素比较：

```
y[i] = (x1[i] < x2[i])
```

**规则与约束**：
- 输出 shape 与输入 shape 完全一致，无 broadcasting
- 无额外属性（`attrs = {}`）
- bool 输出在 kernel 内用 `uint8_t` 的 0x00/0x01 表示，算子原型和接口签名中仍体现逻辑类型bool
- IEEE754 语义：NaN 参与比较结果 False；`+0 == -0`；int/uint 不经 float 中转
- `x1`/`x2` 必须同 shape、同 dtype；kernel 应将输入展平为 1D 处理（`totalLen = N*M`）

## 3. 接口规范

### 算子原型

```
REG_OP(Less)
.INPUT(x1, TensorType({ DT_FLOAT 等 }))
.INPUT(x2, TensorType({ DT_FLOAT 等 }))
.OUTPUT(y, TensorType({ DT_BOOL }))
.OP_END()
```

### 接口签名

```
signature: less(x1: Tensor[..., float32/等], x2: Tensor[..., float32/等]) -> Tensor[..., bool]
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x1 | Tensor | 必选 | 输入张量，支持 ND 格式 |
| x2 | Tensor | 必选 | 输入张量，支持 ND 格式 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x1 相同 | bool | 输出张量，bool 类型（kernel 内 uint8 0x00/0x01） |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | bool |
| float16 | bool |
| int32 | bool |
| int16 | bool |
| int8 | bool |
| uint32 | bool |
| uint16 | bool |
| uint8 | bool |

### 支持范围（shape_range）

`shape_range` 是范围，kernel 必须支持范围内所有合法 shape，不是只支持某一个典型 shape。
开发完成后会使用范围内的多个 shape + 多个 dtype 测试。

```yaml
inputs:
  x1:
    ndim: 2
    dims:
      N:
      - 1
      - 256
      M:
      - 1
      - 1024
    total_elements:
    - 1
    - 262144
  x2:
    ndim: 2
    dims:
      N:
      - 1
      - 256
      M:
      - 1
      - 1024
    total_elements:
    - 1
    - 262144
```

对应支持范围（cases.csv 实测）：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x1`/`x2` 维度数 (ndim) | 2D [N, M] | - |
| `x1` 维度大小 | N∈[1,256]，M∈[1,1024] | - |
| `x1` 总元素数 | [1, 262144] | - |

约束：输出 shape 与输入 shape 完全一致，无 broadcasting；无额外属性。

### TilingData 契约

```
size_bytes: 4

struct LessTilingData {
    uint32_t totalLen;   // 展平后总元素数
};
```

直调通道：host 每个用例按输入 shape 序列化 4 字节（小端），经最后一个 `GM_ADDR tiling`
形参传给 kernel；`totalLen = N*M`（展平后总元素数），kernel 的 TilingData 必须与上述结构一致。

## 4. 精度要求

| dtype | atol | rtol |
|-------|------|------|
| float32 | 0.0 | 0.0 |
| float16 | 0.0 | 0.0 |
| int32 | 0.0 | 0.0 |
| int16 | 0.0 | 0.0 |
| int8 | 0.0 | 0.0 |
| uint32 | 0.0 | 0.0 |
| uint16 | 0.0 | 0.0 |
| uint8 | 0.0 | 0.0 |

全部 dtype 均为逐字节精确比较语义（bool 输出 0x00/0x01）。

## 5. 标准 Golden 参考

- 参考实现：`kirin_cases_bench/level1/less/golden.py`（Python numpy 实现）
- 契约：`golden(x1, x2)`；顶层别名 `less(x1, x2, **kw)`；输出 uint8（0/1）
- kernel 输出必须与 golden 在 atol/rtol 范围内一致

## 6. 额外信息

### 数据构造提示

`x2` 应基于 `x1` 构造，以覆盖 小于/大于/相等/IEEE754 边界（±0/±Inf/NaN）等场景。
