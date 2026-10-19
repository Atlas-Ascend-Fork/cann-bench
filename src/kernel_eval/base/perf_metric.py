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

"""单用例性能分公式（Eq.3 家族）——单一事实来源。

`benches/cann_scoring.py` 与 `report/scoring.py` 的 ``per_case_sol_score``
共用本模块：两处调用方保留各自的锚点守卫（NaN / 非正锚点 / 缺基线
fallback / 日志去重），本模块只负责"锚点均有效之后"的公式本体。

提供两个同构版本：

- ``eq3``（默认，与历史行为逐位一致）::

      score = (B - H) / ((C - H) + (B - H))

- ``eq3_log``（对数同构版，候选评估中，默认不启用）::

      score = lg(B / H) / (lg(C / H) + lg(B / H))

其中 B = T_baseline、C = T_cand、H = T_HW。两式在 C = B 处同为 0.5、
C = H 处同为 1，对 C 单调性一致；差别在远离下界的区间——log 版把
"每快 10x"映射为等距分差，宽时距算子（实测距下界多个数量级）的
分辨率显著更好。

切换方式（按优先级）：

1. 代码侧显式传参：``compute_perf_score(..., metric="eq3_log")``；
2. 环境变量：``CANN_BENCH_PERF_METRIC=eq3_log``；
3. 均未指定时取 ``eq3``——默认行为与引入本模块之前完全一致。
"""

import math
import os
from typing import Optional

PERF_METRIC_EQ3 = "eq3"
PERF_METRIC_EQ3_LOG = "eq3_log"
VALID_PERF_METRICS = (PERF_METRIC_EQ3, PERF_METRIC_EQ3_LOG)

#: 环境变量名：未显式传 metric 时从这里取默认公式。
PERF_METRIC_ENV = "CANN_BENCH_PERF_METRIC"

#: 单用例 HAP 奖励封顶（见 ``hap_reward``）。HAP = T_HW / T_cand 达到一个
#: 量级（10x）以上，几乎必然意味着该用例的 t_hw_us 标定有误，而不是真实
#: 创新；封顶防止坏锚点通过聚合支配算子总分。数值为维护者可调项。
PER_CASE_SCORE_CAP = 10.0

#: 单算子综合分封顶（Eq.4 聚合后生效）。名义满分 100，HAP > 1 的奖励
#: 允许总分突破 100 以激励"快过硬件下界"的合法创新，但以本常数兜底，
#: 保证单算子分对 Level 分 / 总分的贡献有界。数值为维护者可调项。
OPERATOR_SCORE_CAP = 120.0


def resolve_perf_metric(metric: Optional[str] = None) -> str:
    """解析生效的公式名。

    ``metric`` 为 None 时读环境变量 ``CANN_BENCH_PERF_METRIC``（每次调用
    实时读取，便于测试与运行期切换），仍未设置则回落 ``eq3``。
    非法取值直接抛 ``ValueError``——静默回落会让打分口径悄悄漂移。
    """
    if metric is None:
        metric = os.environ.get(PERF_METRIC_ENV, PERF_METRIC_EQ3)
    metric = str(metric).strip().lower()
    if metric not in VALID_PERF_METRICS:
        raise ValueError(
            f"unknown perf metric {metric!r}; expected one of {VALID_PERF_METRICS}"
        )
    return metric


def eq3_score(t_baseline: float, t_cand: float, t_hw: float) -> Optional[float]:
    """Eq.3 减法版（历史公式）。分母 ≤ 0 时返回 None，由调用方告警。"""
    denom = (t_cand - t_hw) + (t_baseline - t_hw)
    if denom <= 0:
        return None
    return (t_baseline - t_hw) / denom


def eq3_log_score(t_baseline: float, t_cand: float, t_hw: float) -> Optional[float]:
    """Eq.3 对数同构版。

    需要 B > H（lg(B/H) > 0）；B ≤ H 属于锚点标定问题，与减法版的
    "denom ≤ 0" 同类，返回 None 交由调用方按既有 None 路径处理。
    """
    if t_baseline <= t_hw:
        return None
    num = math.log10(t_baseline / t_hw)
    denom = math.log10(t_cand / t_hw) + num
    if denom <= 0:
        return None
    return num / denom


def hap_reward(t_cand: float, t_hw: float) -> float:
    """T_cand < T_HW（突破硬件下界，HAP > 1）区间的奖励分。

    只以 T_HW 为锚：``score = T_HW / T_cand``，封顶 ``PER_CASE_SCORE_CAP``。

    动机（issue #288）：两个公式在 T_cand < T_HW 区间都有极点，且极点位置
    取决于 baseline——eq3 在 ``2·T_HW − T_baseline``，eq3_log 在
    ``T_HW² / T_baseline``。基线贴近下界时，小幅击穿即爆分、更大击穿反而
    落入 None→0，既不单调也不可比。两个极点都位于 T_HW 之下，因此统一改为：
    一旦 T_cand < T_HW，脱离基线锚，直接按硬件加速比 HAP 给奖励分——

    - 在 T_cand = T_HW 处与两公式的取值 1 连续衔接；
    - 对 T_cand 严格单调（更快一定更高分）；
    - 与 baseline 取值无关，不再有可被基线位置操纵的区间。
    """
    return min(t_hw / t_cand, PER_CASE_SCORE_CAP)


def compute_perf_score(
    t_baseline: float,
    t_cand: float,
    t_hw: float,
    metric: Optional[str] = None,
) -> Optional[float]:
    """按生效公式计算单用例性能分。

    前置条件（由调用方守卫保证）：三个锚点均为正、非 NaN。
    返回 None 表示该公式在此锚点组合下无定义（调用方按既有
    None 语义告警并计 0 性能分）。

    T_cand < T_HW（HAP > 1）时两个公式共用 ``hap_reward`` 分支（见其
    docstring）；该分支使 eq3 / eq3_log 各自的极点区间（均位于 T_HW 之下）
    不再可达。
    """
    metric = resolve_perf_metric(metric)
    if t_cand < t_hw:
        return hap_reward(t_cand, t_hw)
    if metric == PERF_METRIC_EQ3_LOG:
        return eq3_log_score(t_baseline, t_cand, t_hw)
    return eq3_score(t_baseline, t_cand, t_hw)
