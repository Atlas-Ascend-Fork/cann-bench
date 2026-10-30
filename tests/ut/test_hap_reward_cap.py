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

"""HAP > 1 奖励分支与封顶（单用例 / 单算子）的行为验证。

核心目标（issue #288）：
1. T_cand < T_HW 时两个公式共用 HAP 奖励：score = T_HW / T_cand，只锚 T_HW；
2. 在 T_cand = T_HW 处与两公式连续衔接（=1），全程对 T_cand 单调；
3. 旧极点区间（eq3: 2H−B；eq3_log: H²/B）不再可达——不再爆分、不再
   "更快反而 None→0"；
4. 单用例封顶 PER_CASE_SCORE_CAP、单算子总分封顶 OPERATOR_SCORE_CAP。
"""

import pytest

from kernel_eval.base.perf_metric import (


    OPERATOR_SCORE_CAP,
    PER_CASE_SCORE_CAP,
    PERF_METRIC_ENV,
    compute_perf_score,
    hap_reward,
)

@pytest.fixture(autouse=True)
def _pin_1_weights(monkeypatch):
    """本文件的数值断言按 tasks 1.x 权重档（0.2/0.3/0.5，无 bonus）书写。

    该档作为历史口径永久保留、逐位不变（经 CANN_BENCH_SCORE_WEIGHTS=1
    或 weights="1" 切换）；tasks 2.x 档的行为由
    test_score_weights.py 钉死，内容解析见 test_score_weights_version.py。
    """
    monkeypatch.setenv("CANN_BENCH_SCORE_WEIGHTS", "1")



class TestHapRewardBranch:
    @pytest.mark.parametrize("metric", ["eq3", "eq3_log"])
    def test_below_hw_uses_hap_both_metrics(self, metric):
        b, c, h = 100.0, 25.0, 50.0
        assert compute_perf_score(b, c, h, metric=metric) == pytest.approx(50.0 / 25.0)

    def test_issue_288_example(self):
        """issue #288 的示例组合：t_hw=10、baseline=15，极点在 2H−B=5。

        旧 eq3 在极点邻域爆分（C=5.2 → 25.0）；新分支给出有物理含义的
        HAP 奖励 10/5.2 ≈ 1.923。
        """
        b, c, h = 15.0, 5.2, 10.0
        score = compute_perf_score(b, c, h)
        assert score == pytest.approx(h / c)
        assert 1.0 < score < PER_CASE_SCORE_CAP

    def test_monotone_across_old_pole(self):
        """旧极点（2H−B=5.0）两侧必须单调：更快一定分更高（旧实现是爆分→None→0）。"""
        b, h = 15.0, 10.0
        cands = [9.9, 8.0, 5.2, 5.0, 4.9, 2.0, 1.1]
        scores = [compute_perf_score(b, c, h) for c in cands]
        assert all(s is not None for s in scores)
        assert scores == sorted(scores)

    @pytest.mark.parametrize("metric", ["eq3", "eq3_log"])
    def test_continuous_at_hw(self, metric):
        b, h = 100.0, 50.0
        eps = 1e-6
        at = compute_perf_score(b, h, h, metric=metric)
        below = compute_perf_score(b, h - eps, h, metric=metric)
        above = compute_perf_score(b, h + eps, h, metric=metric)
        assert at == pytest.approx(1.0)
        assert below == pytest.approx(1.0, abs=1e-4)
        assert above == pytest.approx(1.0, abs=1e-4)

    def test_baseline_independent(self):
        c, h = 25.0, 50.0
        assert compute_perf_score(100.0, c, h) == compute_perf_score(10000.0, c, h)

    def test_per_case_cap(self):
        assert hap_reward(1.0, 1e6) == pytest.approx(PER_CASE_SCORE_CAP)
        assert compute_perf_score(100.0, 1.0, 50.0) == pytest.approx(
            min(50.0, PER_CASE_SCORE_CAP))

    def test_per_case_cap_value_is_three(self):
        """封顶值是评分口径，改动必须显式过这条断言。

        常规用例（T_cand ≥ T_HW）单例分上界为 1，聚合按用例等权求和，
        故该常数 = 一条用例最多顶几条常规用例。取 3。
        """
        assert PER_CASE_SCORE_CAP == 3.0
        # 2x 击穿仍在封顶内，线性给奖励；10x 被收到 3。
        assert hap_reward(5.0, 10.0) == pytest.approx(2.0)
        assert hap_reward(1.0, 10.0) == pytest.approx(3.0)

    def test_callers_get_hap(self, monkeypatch):
        monkeypatch.delenv(PERF_METRIC_ENV, raising=False)
        from kernel_eval.benches.cann_scoring import per_case_sol_score as bench_score
        from kernel_eval.report.scoring import per_case_sol_score as report_score
        b, c, h = 15.0, 5.2, 10.0
        assert bench_score(b, c, h) == pytest.approx(h / c)
        assert report_score(b, c, h) == pytest.approx(h / c)


class TestOperatorScoreCap:
    @pytest.mark.parametrize("module_path", [
        "kernel_eval.benches.cann_scoring",
        "kernel_eval.report.scoring",
    ])
    def test_total_capped_and_uncapped_reported(self, module_path):
        import importlib
        aggregate_eq4 = importlib.import_module(module_path).aggregate_eq4
        # 20 个 case 全部拿到单用例封顶分：未截断总分 = 20 + 30 + 50*3 = 200
        case_scores = [(True, PER_CASE_SCORE_CAP)] * 20
        agg = aggregate_eq4(
            compile_passed=True, total_cases=20, case_scores=case_scores,
            rel_path="level2/synthetic_op",
        )
        assert agg["total_score"] == pytest.approx(OPERATOR_SCORE_CAP)
        assert agg["total_score_uncapped"] == pytest.approx(
            20.0 + 30.0 + 50.0 * PER_CASE_SCORE_CAP)

    @pytest.mark.parametrize("module_path", [
        "kernel_eval.benches.cann_scoring",
        "kernel_eval.report.scoring",
    ])
    def test_normal_total_unchanged(self, module_path):
        import importlib
        aggregate_eq4 = importlib.import_module(module_path).aggregate_eq4
        case_scores = [(True, 0.8)] * 20
        agg = aggregate_eq4(
            compile_passed=True, total_cases=20, case_scores=case_scores,
            rel_path="level2/synthetic_op",
        )
        expected = 20.0 + 30.0 + 0.8 * 50.0
        assert agg["total_score"] == pytest.approx(expected)
        assert agg["total_score_uncapped"] == pytest.approx(expected)

    @pytest.mark.parametrize("module_path", [
        "kernel_eval.benches.cann_scoring",
        "kernel_eval.report.scoring",
    ])
    def test_moderate_hap_bonus_passes_through(self, module_path):
        """单个 HAP≈1.32 的 case 把总分带到 ~100.8，不应被封顶截掉。"""
        import importlib
        aggregate_eq4 = importlib.import_module(module_path).aggregate_eq4
        case_scores = [(True, 1.0)] * 19 + [(True, 1.3213)]
        agg = aggregate_eq4(
            compile_passed=True, total_cases=20, case_scores=case_scores,
            rel_path="level2/synthetic_op",
        )
        expected = 20.0 + 30.0 + (19 * 1.0 + 1.3213) / 20.0 * 100.0 * 0.5
        assert agg["total_score"] == pytest.approx(expected)
        assert agg["total_score"] > 100.0
        assert agg["total_score"] < OPERATOR_SCORE_CAP
