#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Eq.4 权重档（weight profiles）——单一事实来源。

`benches/cann_scoring.py` 与 `report/scoring.py` 的 ``aggregate_eq4`` 共用
本模块。提供两套并存的权重档：

- ``season2``（**默认**，season-2 赛季现行口径）::

      w_c=0.05（编译/运行）  w_f=0.20（精度）  w_b=0.05（全用例精度通过 bonus）
      w_p=0.70（性能）

  即"5 编译 + 20 精度 + 5 全通过 bonus + 70 性能"。bonus 项为
  ``w_b · δ_all · 100``，``δ_all = ∏_i δ_acc,i``（全部 N 个用例精度通过才
  为 1，漏跑视为未通过）；反作弊置零路径同样清零 bonus。

- ``season1``（历史档，season-1 口径，可切换用于旧数据复算）::

      w_c=0.20（编译/运行）  w_f=0.30（精度）  w_p=0.50（性能）  w_b=0（无 bonus）

权重档只作用于 Eq.4 聚合，和单用例公式（``eq3`` / ``eq3_log``，见
``perf_metric.py``）正交——任一权重档下两条公式均可用，互不影响。

切换方式（按优先级）：

1. 代码侧显式传参：``aggregate_eq4(..., weights="season1")``；
2. 环境变量：``CANN_BENCH_SCORE_WEIGHTS=season1``；
3. 均未指定时取 ``season2``（season-2 为现行赛季）。
"""

import os
from dataclasses import dataclass
from typing import Optional

SCORE_WEIGHTS_SEASON1 = "season1"
SCORE_WEIGHTS_SEASON2 = "season2"
VALID_SCORE_WEIGHT_PROFILES = (SCORE_WEIGHTS_SEASON1, SCORE_WEIGHTS_SEASON2)

#: 环境变量名：未显式传 weights 时从这里取默认权重档。
SCORE_WEIGHTS_ENV = "CANN_BENCH_SCORE_WEIGHTS"


@dataclass(frozen=True)
class ScoreWeights:
    """一套 Eq.4 权重。``wc + wf + wp + wb = 1``，单算子名义满分 100。"""

    wc: float  # 编译/运行正确性
    wf: float  # 精度正确性
    wp: float  # 性能优化性
    wb: float  # 全用例精度通过 bonus（w_b · δ_all · 100）


WEIGHT_PROFILES = {
    SCORE_WEIGHTS_SEASON1: ScoreWeights(wc=0.2, wf=0.3, wp=0.5, wb=0.0),
    SCORE_WEIGHTS_SEASON2: ScoreWeights(wc=0.05, wf=0.20, wp=0.70, wb=0.05),
}


def resolve_score_weights(profile: Optional[str] = None) -> ScoreWeights:
    """解析生效的权重档。

    ``profile`` 为 None 时读环境变量 ``CANN_BENCH_SCORE_WEIGHTS``（每次调用
    实时读取，便于测试与运行期切换），仍未设置则回落 ``season2``——
    season-2 为现行赛季口径；复算 season-1 历史数据时显式切 ``season1``。
    非法取值直接抛 ``ValueError``——静默回落会让打分口径悄悄漂移。
    """
    if profile is None:
        profile = os.environ.get(SCORE_WEIGHTS_ENV, SCORE_WEIGHTS_SEASON2)
    profile = str(profile).strip().lower()
    if profile not in WEIGHT_PROFILES:
        raise ValueError(
            f"unknown score weight profile {profile!r}; "
            f"expected one of {VALID_SCORE_WEIGHT_PROFILES}"
        )
    return WEIGHT_PROFILES[profile]
