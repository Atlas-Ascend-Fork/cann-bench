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
       golden。这里把每个框归一到 x1≤x2, y1≤y2,使 NMS 良定义、真实单算子能匹配 golden。
    2. **去并列分数**:keep 顺序在分数并列时不确定(tie-break 在 golden 与候选间不稳定)——
       case 15(value_range [0,0] 全并列)、case 9/13([0,1] 内 bit-identical)。仅在存在并列
       时把分数替换为确定性互异序列。

    kernel_eval 用输入名+attrs 作为关键字调用本函数，并用返回值（按 golden 签名的
    Tensor 顺序）同时替换 golden 与候选的输入，故比较公平。

    Returns:
        [boxes, scores]，顺序与 nms 签名的 (boxes, scores) 张量一致。
    """
    # 1) 合法框:逐框排序 x 对、y 对 → x1≤x2, y1≤y2(对已合法的框为幂等,零扰动)。
    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # 2) 去并列分数(仅在存在并列时)。
    n = int(scores.shape[0])
    if n > 1 and int(scores.reshape(-1).unique().numel()) < n:
        g = torch.Generator().manual_seed(0)  # 确定性，跨 eval 复现
        ranks = (torch.randperm(n, generator=g) + 1).to(torch.float32) / float(n)
        scores = ranks.reshape(scores.shape).to(dtype=scores.dtype, device=scores.device)
    return [boxes, scores]
