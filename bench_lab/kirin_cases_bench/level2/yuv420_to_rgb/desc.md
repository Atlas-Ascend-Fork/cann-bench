# Yuv420ToRgb 算子 API 描述

## 1. 算子简介

Yuv420ToRgb：YUV420 平面（I420/YU12）→ RGB 交错图像颜色变换，标杆为 OpenCV
`cvtColor(src, COLOR_YUV2RGB_I420)` 标量算法（BT.601 limited range，常数 ITUR_BT_601_*），
逐字节复刻 `cvtYuv42xxp2RGB8`。

**开发目标**：Kirin 端侧 NPU（AscendC 直调 kernel）。

**算子特征**：
- 难度等级：L2（ImageColorTransform）
- 输入为 [3H/2, W] uint8 紧凑缓冲：Y 平面 [H,W] + U 平面 [H/2,W/2] + V 平面 [H/2,W/2]
- 输出为 [H, W, 3] uint8 RGB 交错
- 纯逐字节精确（atol=0），必须使用整数定点算法，不能用浮点近似

## 2. 算子定义

**输入平面布局**：输入是单个 `[3H/2, W]` uint8 紧凑缓冲（与 OpenCV 的 I420 单 Mat 表示一致）：

```
Y 平面: 字节偏移 [0, H*W)
U 平面: 字节偏移 [H*W, H*W + H*W/4)
V 平面: 字节偏移 [H*W + H*W/4, 3*H*W/2)
```

注意平面边界按【字节平坦偏移】计算，不是行号×W——当 `H % 4 != 0` 时 U/V 平面起始偏移
不落在整行边界上。

**色度上采样**：2x2 最近邻（每个 U/V 样本覆盖对应的 2x2 luma 块，非双线性）。

**换算公式**（BT.601 limited range，整数定点；中间量 int32 即可，峰值约 5.1e8 < 2^31）：

```
y00 = max(0, Y - 16) * 1220542
R = clip((y00 + 2^19 + 1673527 * (V - 128)) >> 20, 0, 255)
G = clip((y00 + 2^19 - 852492 * (V - 128) - 409993 * (U - 128)) >> 20, 0, 255)
B = clip((y00 + 2^19 + 2116026 * (U - 128)) >> 20, 0, 255)
```

其中 `>>` 为算术右移（负数向负无穷取整后再 clip 到 0..255，与 C++ 一致）。

> ⚠️ G 的 U 系数是 **-409993**（-0.391*2^20 精确取整；极旧版 OpenCV 曾误用 -408994，
> 差 999，会让 ~4% 像素的 G 偏 1——勿抄旧帖子）。

**输出**：[H, W, 3] uint8，RGB 三通道交错（R 在前）。

**验收要求**：逐字节精确比对（atol=0），必须用上述整数定点算法，不要用
1.164/1.596/0.813/0.391/2.018 浮点近似（舍入后会有 ±1 偏差）。

## 3. 接口规范

### 算子原型

```
REG_OP(Yuv420ToRgb)
.INPUT(x, TensorType({ DT_UINT8 }))
.OUTPUT(y, TensorType({ DT_UINT8 }))
.OP_END()
```

### 接口签名

```
signature: yuv420_to_rgb(x: Tensor[3H/2, W], uint8) -> Tensor[H, W, 3], uint8 (RGB 交错)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，2D `[3H/2, W]` uint8 紧凑 I420 缓冲 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[H, W, 3]` | uint8 | RGB 交错输出（R 在前） |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| uint8 | uint8 |

### 属性

无（`attrs = {}`）。输出尺寸由输入 shape 推导：`H = x.shape[0] * 2 // 3`，`W = x.shape[1]`。

### 支持范围（shape_range）

`shape_range` 是范围，kernel 必须支持范围内所有合法 shape，不是只支持某一个典型 shape。
开发完成后会使用范围内的多个 shape + 多个 dtype 测试。

```yaml
inputs:
  x:
    ndim: 2
    dims:
      H:
      - 16
      - 1080
      - step 2
      W:
      - 16
      - 1920
      - step 2
    total_elements:
    - 384
    - 3110400
    constraints:
    - H % 2 == 0
    - W % 2 == 0
```

对应支持范围：`H∈[16,1080]`（偶数），`W∈[16,1920]`（偶数），输入总元素 ∈ [384, 3110400]。

> 注：shape_range 中的 `H`/`W` 为**逻辑图像尺寸**（即输出 `[H, W, 3]` 的高宽）；
> cases.yaml/cases.csv 的 `input_shape` 为**物理输入** `[3H/2, W]`（Y/U/V 平面拼接的平坦缓冲），
> 如 H=1080 对应输入首维 `1080*3//2 = 1620`。两者勿混用。

### 规则与约束

- 输入必须是单个 `[3H/2, W]` uint8 缓冲（Y/U/V 平面按字节平坦偏移拼接）
- H、W 必须为偶数（保证 U/V 平面为整数尺寸）
- U/V 上采样为 2x2 最近邻，不使用插值
- 输出 [H, W, 3] uint8，RGB 交错（R 在前）
- 逐字节精确（atol=0），禁止浮点近似

### TilingData 契约

```
size_bytes: 20

struct Yuv420ToRgbTilingData {
    uint32_t height;   // 输出高 H（输入行数 = 3H/2）
    uint32_t width;    // 输出宽 W（= 输入列数）
    uint32_t ySize;    // Y 平面字节数 = H*W（= U 平面起始平坦偏移）
    uint32_t uSize;    // U 平面字节数 = H*W/4（V 平面起始平坦偏移 = ySize + uSize）
    uint32_t dtype;    // 0 = uint8
};
```

直调通道：host 每个用例按输入 shape 序列化 20 字节（5×uint32，小端），经最后一个
`GM_ADDR tiling` 形参传给 kernel；kernel 的 TilingData 必须与上述结构一致。

## 4. 精度要求

| dtype | atol | rtol |
|-------|------|------|
| uint8 | 0.0 | 0.0 |

逐字节精确比对：kernel 输出必须与 golden 完全一致。

## 5. 标准 Golden 参考

- 参考实现：`kirin_cases_bench/level2/yuv420_to_rgb/golden.py`（Python numpy 实现）
- 契约：`golden(x, **attrs)`（内部按 `H = x.shape[0] // 3 * 2` 推导）；辅助钩子
  `physical_shapes(dims, attrs)` / `output_shape(shape, attrs)` / `gen_tiling(...)`
- golden.py 已与 cv2 4.10/4.12/5.0 的 `cvtColor(COLOR_YUV2RGB_I420)` 在多尺寸
  （含 1080p、H%4=2）下逐字节交叉验证一致
- kernel 输出必须与 golden 在 atol/rtol 范围内一致
