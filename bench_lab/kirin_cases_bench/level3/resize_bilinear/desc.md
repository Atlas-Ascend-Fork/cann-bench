# ResizeBilinear 算子 API 描述

## 1. 算子简介

ResizeBilinear：单通道 2D uint8 双线性缩放，标杆为 OpenCV
`cv2.resize(x, (out_w, out_h), interpolation=INTER_LINEAR)` 的 8U 实现
（`modules/imgproc/src/resize.cpp` 的 CV_8U 路径，4.x/5.x 同码），**逐字节一致验收（atol=0）**。

**开发目标**：Kirin 端侧 NPU（AscendC 直调 kernel）。

**算子特征**：
- 难度等级：L3（ImageResize）
- 三条互斥分派路径，舍入规则各不相同，必须完全复刻：
  1. 恒等（out == in）→ 拷贝（本题目约束已排除）
  2. 精确 2x 双维下采样（H==2*out_h 且 W==2*out_w）→ 2x2 盒式平均 `y=(a+b+c+d+2)>>2`
  3. 通用定点双线性：11bit 系数 half-even 舍入、水平轴左/右缘权重强制 (2048,0)、
     垂直轴只裁剪行号不清零权重（不对称！）、融合舍入 `>>4` 截断不可省略
- W 可为质数/奇数，尾块搬运须 DataCopyPad/按字节处理

## 2. 算子定义

**验收基准**：目标不是 `INTER_LINEAR_EXACT`（float 半像素 + 逐点舍入），也不是“任意双线性
近似”——验收是逐字节精确比对（atol=0），必须复刻下述定点管线，包括 `>>4` 截断和 half-even
权重舍入。golden.py 已与 cv2 5.0.0 在 2.5 万组尺寸（含全组合小尺寸、精确 2x、质数边、
1 像素、1080p/4K）下逐字节交叉验证一致。

### 分派逻辑（三条互斥路径，必须完全复刻）

1. **恒等**（`out_h == H` 且 `out_w == W`）→ 逐字节拷贝。
   （本 spec 的约束已排除恒等，列出仅为完整性。）
2. **精确 2x 双维下采样**（`H == 2*out_h` 且 `W == 2*out_w`）→ 2x2 盒式平均：

   ```
   y = (a + b + c + d + 2) >> 2
   ```

   a,b,c,d 是对应 2x2 输入块四像素（无饱和风险，max `(255*4+2)>>2 = 255`）。
   这是 OpenCV 在 `is_area_fast` 且 iscale==2x2 时把 INTER_LINEAR 提升为
   `resizeAreaFast` 8U 的特殊路径——舍入与通用路径**不同**，漏掉会整图偏 1。
3. **其他** → 通用定点双线性（下述算法）。

### 通用定点双线性

符号：`in/inH/inW` 输入，`out/outH/outW` 输出。

**坐标**（水平，`d = 0..outW-1`；垂直同，`d = 0..outH-1`）：

```
scale = in / out                     （double 除法）
fx = float32((d + 0.5) * scale - 0.5) （double 算完再截断到 f32）
sx = floor(fx)；frac = fx - sx        （f32 运算，frac ∈ [0, 1)）
```

设备端等价（已验证与 double→f32 管线逐位一致，推荐整数形式）：

```
N = (2d + 1) * in - out；D = 2 * out（均 int32，N 可为负）
fx = float32(N / D)                  （单次 f32 除法）
```

**权重**（11 位定点，half-even 舍入）：

```
w0 = rint((1.0f - frac) * 2048)；w1 = rint(frac * 2048)
```

（f32 乘法后取最近偶数；`w0 + w1 == 2048`）

**水平轴边缘**（与垂直轴不对称，是逐字节一致的魔鬼细节）：

```
sx < 0        → sx = 0，权重强制 (2048, 0)（左缘 replicate）
sx >= inW - 1 → sx = inW - 1，权重强制 (2048, 0)（右缘 replicate）
```

**垂直轴【不做】上述权重清零**：`sy` 保留原值（可为 -1 或 outH-1），权重按原始 `frac`
计算，行号裁剪：

```
r0 = clip(sy, 0, inH - 1)；r1 = clip(sy + 1, 0, inH - 1)
```

（两行可相同 = replicate。垂直权重拆分影响下方 `>>4` 截断，清零会得错误结果——务必区别于水平轴。）

**两段式计算**（全部 int32，无溢出：中间量 < 2^27）：

水平 pass（对 r0/r1 两行，精确整数，无舍入）：

```
Hr[dx] = S_r[sx[dx]] * w0[dx] + S_r[sx[dx]+1] * w1[dx]
```

垂直 pass（融合舍入；`>>4` 先截断低 4 位是 INTER_LINEAR 与 INTER_LINEAR_EXACT 差异的根源，
不可省略/不可改舍入）：

```
y = ((b0 * (H0 >> 4)) >> 16 + (b1 * (H1 >> 4)) >> 16 + 2) >> 2
```

（b0/b1 是垂直权重，H0/H1 是 r0/r1 行的水平结果；最后饱和到 uint8，数学上不会超 255。）

## 3. 接口规范

### 算子原型

```
REG_OP(ResizeBilinear)
.INPUT(x, TensorType({ DT_UINT8 }))
.REQUIRED_ATTR(out_h, AttrValue::INT)
.REQUIRED_ATTR(out_w, AttrValue::INT)
.OUTPUT(y, TensorType({ DT_UINT8 }))
.OP_END()
```

### 接口签名

```
signature: resize_bilinear(x: Tensor[H, W], uint8, out_h: int, out_w: int) -> Tensor[out_h, out_w], uint8
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入图像，2D `[H, W]` uint8 |
| out_h | int | 必选 | 输出高 |
| out_w | int | 必选 | 输出宽 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[out_h, out_w]` | uint8 | 缩放结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| uint8 | uint8 |

### 属性

| 属性 | 类型 | 范围 | 描述 |
|------|------|------|------|
| out_h | int | `[4, 2160]`（用例实际取值见 cases.yaml/cases.csv） | 输出高 |
| out_w | int | 同上 | 输出宽 |

### 支持范围（shape_range）

`shape_range` 是范围，kernel 必须支持范围内所有合法 shape，不是只支持某一个典型 shape。
开发完成后会使用范围内的多个 shape + 多个 dtype 测试。

```yaml
inputs:
  x:
    ndim: 2
    dims:
      H:
      - 8
      - 2160
      W:
      - 8
      - 3840
    total_elements:
    - 64
    - 8294400
    constraints:
    - (H != out_h) or (W != out_w)
    - H <= 32 * out_h and W <= 32 * out_w
    - out_h <= 8 * H and out_w <= 8 * W
    - out_h * out_w <= 4194304
    - (out_h != 8 or H == 16) and (out_w != 8 or W == 16)
    - (out_h != 32 or H == 64) and (out_w != 32 or W == 64)
    - (out_h != 512 or H == 1024) and (out_w != 512 or W == 1024)
    - (out_h != 1024 or H == 2048) and (out_w != 1024 or W == 2048)
```

对应支持范围：
- H∈[8,2160]，W∈[8,3840]，输入总元素∈[64, 8294400]
- out_h/out_w∈[4,2160]，缩小≤32x、放大≤8x，out_h*out_w≤4194304
- 盒式锚点：out∈{8,32,512,1024} 时输入强制 2 倍（spec constraints）
- W 可为质数/奇数，尾块搬运须 DataCopyPad/按字节处理

### TilingData 契约

```
size_bytes: 20

struct ResizeBilinearTilingData {
    uint32_t inHeight;   // 输入高 H
    uint32_t inWidth;    // 输入宽 W
    uint32_t outHeight;  // 输出高 out_h
    uint32_t outWidth;   // 输出宽 out_w
    uint32_t dtype;      // 0 = uint8
};
```

直调通道：host 每个用例按输入 shape 与 attrs 序列化 20 字节（5×uint32，小端），经最后一个
`GM_ADDR tiling` 形参传给 kernel；kernel 的 TilingData 必须与上述结构一致。

## 4. 精度要求

| dtype | atol | rtol |
|-------|------|------|
| uint8 | 0.0 | 0.0 |

逐字节精确比对：kernel 输出必须与 golden 完全一致。

## 5. 标准 Golden 参考

- 参考实现：`kirin_cases_bench/level3/resize_bilinear/golden.py`（Python numpy 实现）
- 契约：`golden(x, out_h=, out_w=)`；辅助钩子 `output_shape(shape, attrs)` /
  `gen_tiling(shape, attrs, dtype_name)`
- golden.py 已与 cv2 5.0.0 在 2.5 万组尺寸下逐字节交叉验证一致
- kernel 输出必须与 golden 在 atol/rtol 范围内一致

## 6. 额外信息
