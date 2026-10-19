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
- `scores` 的形状必须为 [N]，且 N 与 boxes 的第一维一致
- `iou_threshold` 取值范围为 (0, 1)，值越小过滤越严格
- 输出 `keep_indices` 为 1D int64 张量，长度 M 取决于过滤后保留的框数
- 输出索引按置信度从高到低排序

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `boxes` shape | `[N, 4]`，N ∈ [1, 8192] | cases.csv 实测 N ∈ [511, 4096]，包含对齐 (1024/4096) 与质数非对齐 (1009/3001/4001) |
| `scores` shape | `[N]`，与 boxes 第一维一致 | cases.csv 实测 N ∈ [511, 4096] |
| `boxes` dtype | float32 | cases.csv 实测仅 float32 |
| `scores` dtype | float32 | cases.csv 实测仅 float32 |
| `boxes` value | float32 有限值范围 | cases.csv 实测 [-65504, 65504]，含 float16 边界 / ±inf / nan / 全零等特殊值 |
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

    # 纯 PyTorch 实现 NMS，避免 torchvision ABI 兼容问题
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 0:
            break
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            order = order.new_empty(0)
            break

        xx1 = boxes[order[1:], 0].clamp(min=boxes[i, 0])
        yy1 = boxes[order[1:], 1].clamp(min=boxes[i, 1])
        xx2 = boxes[order[1:], 2].clamp(max=boxes[i, 2])
        yy2 = boxes[order[1:], 3].clamp(max=boxes[i, 3])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze(1)

        order = order[inds + 1]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

boxes = torch.rand(1000, 4, dtype=torch.float32, device="npu") * 100
scores = torch.rand(1000, dtype=torch.float32, device="npu")
keep = cann_bench.nms(boxes, scores, iou_threshold=0.5)

# 低 IoU 阈值（更严格的过滤）
keep = cann_bench.nms(boxes, scores, iou_threshold=0.3)
```
