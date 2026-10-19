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

"""双公式（eq3 / eq3_log）并存的行为验证。

核心目标：
1. 默认行为与历史逐位一致（eq3 减法版）；
2. eq3_log 在锚点处与 eq3 同构（C=B → 0.5，C=H → 1.0），单调性一致；
3. 切换通道（显式参数 / 环境变量）工作正常，非法取值报错；
4. 两个调用方模块（benches / report）共享同一公式实现。
"""

import math

import pytest

from kernel_eval.base.perf_metric import (
    PERF_METRIC_ENV,
    compute_perf_score,
    eq3_log_score,
    eq3_score,
    resolve_perf_metric,
)


def _legacy_eq3(t_baseline, t_cand, t_hw):
    """历史公式的字面实现，用来钉死默认行为。"""
    denom = (t_cand - t_hw) + (t_baseline - t_hw)
    if denom <= 0:
        return None
    return (t_baseline - t_hw) / denom


class TestDefaultBehaviorUnchanged:
    """默认（不传 metric、不设环境变量）必须与历史逐位一致。"""

    @pytest.mark.parametrize("b,c,h", [
        (100.0, 100.0, 50.0),
        (100.0, 50.0, 50.0),
        (100.0, 1000.0, 50.0),
        (15.0, 5.2, 10.0),          # 极点邻域样本（基线贴近下界）
        (3.0, 7.0, 1.0),
    ])
    def test_matches_legacy_literal(self, b, c, h, monkeypatch):
        monkeypatch.delenv(PERF_METRIC_ENV, raising=False)
        expected = _legacy_eq3(b, c, h)
        got = compute_perf_score(b, c, h)
        if expected is None:
            assert got is None
        else:
            assert got == pytest.approx(expected, rel=0, abs=0)

    def test_per_case_sol_score_default_unchanged(self, monkeypatch):
        monkeypatch.delenv(PERF_METRIC_ENV, raising=False)
        from kernel_eval.benches.cann_scoring import per_case_sol_score as bench_score
        from kernel_eval.report.scoring import per_case_sol_score as report_score

        b, c, h = 300.0, 120.0, 60.0
        expected = _legacy_eq3(b, c, h)
        assert bench_score(b, c, h) == pytest.approx(expected)
        assert report_score(b, c, h) == pytest.approx(expected)


class TestLogMetricIsomorphism:
    """eq3_log 与 eq3 在锚点处同构。"""

    @pytest.mark.parametrize("b,h", [(100.0, 50.0), (300.0, 10.0), (176.0, 140.0)])
    def test_anchor_at_baseline_is_half(self, b, h):
        assert eq3_score(b, b, h) == pytest.approx(0.5)
        assert eq3_log_score(b, b, h) == pytest.approx(0.5)

    @pytest.mark.parametrize("b,h", [(100.0, 50.0), (300.0, 10.0), (176.0, 140.0)])
    def test_anchor_at_hw_is_one(self, b, h):
        assert eq3_score(b, h, h) == pytest.approx(1.0)
        assert eq3_log_score(b, h, h) == pytest.approx(1.0)

    def test_monotone_decreasing_in_cand(self):
        b, h = 1000.0, 10.0
        cands = [10.0, 20.0, 100.0, 1000.0, 10000.0, 100000.0]
        log_scores = [eq3_log_score(b, c, h) for c in cands]
        eq3_scores = [eq3_score(b, c, h) for c in cands]
        assert log_scores == sorted(log_scores, reverse=True)
        assert eq3_scores == sorted(eq3_scores, reverse=True)

    def test_log_keeps_resolution_far_from_floor(self):
        """宽基线下 10x 真实进步：log 版分差显著，减法版塌缩。"""
        b, h = 1000.0, 1.0   # 基线距下界 1000x
        s_slow_log = eq3_log_score(b, 100000.0, h)
        s_fast_log = eq3_log_score(b, 10000.0, h)
        s_slow_eq3 = eq3_score(b, 100000.0, h)
        s_fast_eq3 = eq3_score(b, 10000.0, h)
        assert (s_fast_log - s_slow_log) > 0.05      # log 版可读的分差
        assert (s_fast_eq3 - s_slow_eq3) < 0.09      # 减法版几乎不动

    def test_log_baseline_not_above_hw_returns_none(self):
        assert eq3_log_score(50.0, 60.0, 50.0) is None
        assert eq3_log_score(40.0, 60.0, 50.0) is None


class TestMetricSelection:
    def test_explicit_arg_overrides(self, monkeypatch):
        monkeypatch.delenv(PERF_METRIC_ENV, raising=False)
        b, c, h = 1000.0, 100.0, 10.0
        assert compute_perf_score(b, c, h, metric="eq3_log") == pytest.approx(
            eq3_log_score(b, c, h))
        assert compute_perf_score(b, c, h, metric="eq3") == pytest.approx(
            eq3_score(b, c, h))

    def test_env_switch(self, monkeypatch):
        b, c, h = 1000.0, 100.0, 10.0
        monkeypatch.setenv(PERF_METRIC_ENV, "eq3_log")
        assert compute_perf_score(b, c, h) == pytest.approx(eq3_log_score(b, c, h))
        monkeypatch.setenv(PERF_METRIC_ENV, "eq3")
        assert compute_perf_score(b, c, h) == pytest.approx(eq3_score(b, c, h))

    def test_env_switch_reaches_both_callers(self, monkeypatch):
        from kernel_eval.benches.cann_scoring import per_case_sol_score as bench_score
        from kernel_eval.report.scoring import per_case_sol_score as report_score

        b, c, h = 1000.0, 100.0, 10.0
        monkeypatch.setenv(PERF_METRIC_ENV, "eq3_log")
        expected = eq3_log_score(b, c, h)
        assert bench_score(b, c, h) == pytest.approx(expected)
        assert report_score(b, c, h) == pytest.approx(expected)

    def test_explicit_metric_arg_on_callers(self, monkeypatch):
        monkeypatch.delenv(PERF_METRIC_ENV, raising=False)
        from kernel_eval.benches.cann_scoring import per_case_sol_score as bench_score
        from kernel_eval.report.scoring import per_case_sol_score as report_score

        b, c, h = 1000.0, 100.0, 10.0
        expected = eq3_log_score(b, c, h)
        assert bench_score(b, c, h, metric="eq3_log") == pytest.approx(expected)
        assert report_score(b, c, h, metric="eq3_log") == pytest.approx(expected)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            resolve_perf_metric("eq4")

    def test_invalid_env_raises(self, monkeypatch):
        monkeypatch.setenv(PERF_METRIC_ENV, "banana")
        with pytest.raises(ValueError):
            compute_perf_score(100.0, 50.0, 10.0)


class TestNonePathPreserved:
    """公式无定义 → None → 聚合按 0 计入性能项（既有语义不变）。"""

    def test_eq3_denom_nonpositive_none(self, monkeypatch):
        monkeypatch.delenv(PERF_METRIC_ENV, raising=False)
        from kernel_eval.benches.cann_scoring import per_case_sol_score
        # T_baseline ≈ T_HW 且 T_cand 远小于 T_HW → denom ≤ 0
        assert per_case_sol_score(10.0, 1.0, 10.0) is None

    def test_log_b_below_h_none(self, monkeypatch):
        monkeypatch.delenv(PERF_METRIC_ENV, raising=False)
        from kernel_eval.benches.cann_scoring import per_case_sol_score
        assert per_case_sol_score(40.0, 60.0, 50.0, metric="eq3_log") is None

    def test_nan_rejected_before_formula(self, monkeypatch):
        monkeypatch.setenv(PERF_METRIC_ENV, "eq3_log")
        from kernel_eval.report.scoring import per_case_sol_score
        assert per_case_sol_score(float("nan"), 50.0, 10.0) is None
        assert per_case_sol_score(100.0, float("nan"), 10.0) is None
        assert per_case_sol_score(100.0, 50.0, float("nan")) is None
