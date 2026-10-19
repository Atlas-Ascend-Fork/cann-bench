# NMS 算子 API 描述

## 1. 算子简介

对候选框执行非极大值抑制 (Non-Maximum Suppression)，根据置信度分数和 IoU 阈值过滤重叠的候选框。

**主要应用场景**：
- 目标检测模型（如 YOLO、Faster R-CNN）的后处理去重
- 实例分割中候选区域的筛选
- 人脸检测等需要消除重叠检测框的场景

**算子特征**：
- 难度等级：L3（SortSelect）
- 双输入单输出，根据置信度排序并基于 IoU 阈值迭代过滤重叠框，输出保留框的索引

## 2. 算子定义

### 数学公式

$$
\text{keep\_indices} = \text{NMS}(\text{boxes}, \text{scores}, \text{iou\_threshold})
$$

### 处理流程

1. 按 scores 从高到低对候选框排序
2. 选取得分最高的框加入保留列表
3. 计算该框与剩余所有框的 IoU（交并比）
4. 移除 IoU 大于 `iou_threshold` 的重叠框
5. 重复步骤 2-4，直至所有框被处理

其中 IoU 定义为：

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

## 3. 接口规范

### 算子原型

```python
cann_bench.nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor keep_indices
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| boxes | Tensor | 必选 | 输入候选框，格式为 [x1, y1, x2, y2]，shape 为 [N, 4] |
| scores | Tensor | 必选 | 每个候选框的置信度分数，shape 为 [N] |
| iou_threshold | float | 必选 | IoU 阈值，用于过滤重叠框 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| keep_indices | [M] | int64 | NMS 后保留的框索引，M <= N |

### 数据类型

| 输入 (boxes) dtype | 输入 (scores) dtype | 输出 dtype |
|-------------------|-------------------|-----------|
| float32 | float32 | int64 |

### 规则与约束

- `boxes` 的形状必须为 [N, 4]，每行为 [x1, y1, x2, y2] 格式
- `boxes` 坐标必须为有限 float32 值；输入前处理统一保证 x1 < x2、y1 < y2
- `scores` 的形状必须为 [N]，且 N 与 boxes 的第一维一致；所有分数必须为有限值且位于 [0, 1]
- `iou_threshold` 取值范围为 (0, 1)，值越小过滤越严格
- 输出 `keep_indices` 为 1D int64 张量，长度 M 取决于过滤后保留的框数
- 输出索引按置信度从高到低排序
- 等分（tie）行为：多个候选框 `scores` 相等时的排序与保留次序**未定义**（排序非稳定，golden 与任意合法实现可能不同）；计分 case 保证 `scores` 两两互不相等

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `boxes` shape | `[N, 4]`，N ∈ [1, 8192] | cases.csv 实测 N ∈ [511, 4096]，包含对齐 (1024/4096) 与质数非对齐 (1009/3001/4001) |
| `scores` shape | `[N]`，与 boxes 第一维一致 | cases.csv 实测 N ∈ [511, 4096] |
| `boxes` dtype | float32 | cases.csv 实测仅 float32 |
| `scores` dtype | float32 | cases.csv 实测仅 float32 |
| `boxes` value | float32 有限值范围 | cases.csv 实测 [-1000000, 1000000]，覆盖大有限值域与接近零的小值域，不包含 NaN / Inf |
| `scores` value | [0, 1] | cases.csv 实测均为 [0, 1]（置信度语义） |
| `iou_threshold` | (0, 1) | cases.csv 实测 0.05 ~ 0.9，覆盖严格 / 宽松阈值 |
| 输出 `keep_indices` | `[M]`，M ≤ N，int64 | 长度由过滤结果决定，按 scores 从高到低排序 |

约束：`scores.shape[0]` 必须等于 `boxes.shape[0]`；`iou_threshold` 须严格落在开区间 (0, 1) 内。

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。


## 5. 标准 Golden 代码

```python
#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch

"""
NMS 算子 Torch Golden 参考实现

对候选框执行非极大值抑制 (Non-Maximum Suppression)
公式：keep_indices = nms(boxes, scores, iou_threshold)
"""
def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float
) -> torch.Tensor:
    """
    对候选框执行非极大值抑制

    公式：keep_indices = nms(boxes, scores, iou_threshold)

    Args:
        boxes: 输入候选框，格式为 [x1, y1, x2, y2]，shape 为 [N, 4]
        scores: 每个候选框的置信度分数，shape 为 [N]
        iou_threshold: IoU 阈值，用于过滤重叠框

    Returns:
        keep_indices: NMS 后保留的框索引，shape 为 [M]
    """

    # 确保输入格式正确
    assert boxes.dim() == 2 and boxes.shape[1] == 4, "boxes shape must be [N, 4]"
    assert scores.dim() == 1 and scores.shape[0] == boxes.shape[0], "scores shape must be [N]"

    # Pure PyTorch NMS (no torchvision, to dodge ABI issues).
    #
    # Greedy NMS is sequential, but the original element-by-element loop did a
    # `.item()` + `nonzero` every step — each forces a device->host sync, so on NPU
    # it serialises ~2N stream syncs and the op runs for tens of minutes (golden-
    # candidate ST timed out at 1801s). Split the work instead: compute the full IoU
    # matrix in ONE vectorised pass (stays on the input device — NPU when this runs
    # as a candidate), then do the inherently-sequential greedy pick on CPU after a
    # SINGLE device->host copy. Keep set is bit-identical to the per-element loop.
    n = scores.shape[0]
    dev = boxes.device
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Pairwise IoU[i, j] — the heavy arithmetic, vectorised on `dev`.
    x1 = torch.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = torch.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = torch.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = torch.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    iou = inter / (areas[:, None] + areas[None, :] - inter + 1e-6)

    _, order = scores.sort(descending=True)

    # Greedy selection on CPU: one sync (the .cpu() copies), then none.
    iou_cpu = iou.cpu()
    order_cpu = order.cpu()
    suppressed = torch.zeros(n, dtype=torch.bool)
    keep = []
    for idx_t in order_cpu:
        idx = int(idx_t)
        if suppressed[idx]:
            continue
        keep.append(idx)
        # Original loop keeps survivors with iou <= thr, so suppress the rest.
        # Use ~(iou <= thr) (not iou > thr) so nan/inf IoU suppress, matching it.
        # No explicit self-mark needed: each idx is visited once (order is a
        # permutation), so suppressed[idx] is never read after this point.
        suppressed |= ~(iou_cpu[idx] <= iou_threshold)

    return torch.tensor(keep, dtype=torch.long, device=dev)


def get_input(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    **kwargs,
):
    """规整测试输入,使 NMS 良定义且可被真实单算子复现(同时替换 golden 与候选输入,公平)。

    两处规整:
    1. **合法框**:cases.csv 对 boxes 的 [x1,y1,x2,y2] 用同一 value_range **独立**取值,
       故约半数框 x1>x2 / y1>y2 —— 退化(负面积)框。golden 的 IoU 在负面积上是实现特定
       行为,**任何真实 NMS 单算子(npu_nms_v4 等)都无法复现** → 基线只能用 torch 重拼
       golden。这里把每个框归一到 x1<x2, y1<y2,使 NMS 良定义、真实单算子能匹配 golden。
    2. **去并列分数**:keep 顺序在分数并列时不确定(tie-break 在 golden 与候选间不稳定)。
       常量分数或有限精度随机值都可能产生并列，仅在存在并列时把分数替换为确定性互异序列。

    kernel_eval 用输入名+attrs 作为关键字调用本函数，并用返回值（按 golden 签名的
    Tensor 顺序）同时替换 golden 与候选的输入，故比较公平。

    Returns:
        [boxes, scores]，顺序与 nms 签名的 (boxes, scores) 张量一致。
    """
    # 1) 合法框:逐框排序 x 对、y 对，并以 dtype epsilon 保证严格正宽高。
    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    eps = torch.finfo(boxes.dtype).eps
    x2 = torch.maximum(x2, x1 + eps * torch.maximum(x1.abs(), torch.ones_like(x1)))
    y2 = torch.maximum(y2, y1 + eps * torch.maximum(y1.abs(), torch.ones_like(y1)))
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # 2) 去并列分数(仅在存在并列时)。
    n = int(scores.shape[0])
    if n > 1 and int(scores.reshape(-1).unique().numel()) < n:
        g = torch.Generator().manual_seed(0)  # 确定性，跨 eval 复现
        ranks = (torch.randperm(n, generator=g) + 1).to(torch.float32) / float(n)
        scores = ranks.reshape(scores.shape).to(dtype=scores.dtype, device=scores.device)
    return [boxes, scores]
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

coords = torch.rand(1000, 4, dtype=torch.float32, device="npu") * 100
top_left = torch.minimum(coords[:, :2], coords[:, 2:])
bottom_right = torch.maximum(coords[:, :2], coords[:, 2:]) + 1e-3
boxes = torch.cat([top_left, bottom_right], dim=1)
scores = torch.rand(1000, dtype=torch.float32, device="npu")
keep = cann_bench.nms(boxes, scores, iou_threshold=0.5)

# 低 IoU 阈值（更严格的过滤）
keep = cann_bench.nms(boxes, scores, iou_threshold=0.3)
```
