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

"""tasks 大版本双权重档（2 / 1）并存的行为验证。

核心目标：
1. tasks 2.x 档为 2（0.05/0.20/0.70 + 0.05 全通过 bonus，tasks 2.x 现行
   口径），数值正确，bonus 仅在全部用例精度通过时生效；
2. 1 历史档（0.2/0.3/0.5，无 bonus 项）经切换后与历史行为逐位一致
   ——tasks 1.x 口径永久保留不变；
3. 切换通道（显式参数 / 环境变量）工作正常，非法取值报错，显式单权重
   覆盖优先于权重档；
4. 权重档与单用例公式（eq3 / eq3_log）正交；
5. 两个调用方模块（benches / report）行为一致；反作弊置零与
   OPERATOR_SCORE_CAP 封顶在两档下语义一致。
"""

import pytest

from kernel_eval.base.perf_metric import (
    OPERATOR_SCORE_CAP,
    PER_CASE_SCORE_CAP,
    PERF_METRIC_ENV,
)
from kernel_eval.base.score_weights import (
    SCORE_WEIGHTS_ENV,
    WEIGHT_PROFILES,
    resolve_score_weights,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(SCORE_WEIGHTS_ENV, raising=False)
    monkeypatch.delenv(PERF_METRIC_ENV, raising=False)


def _agg_fns():
    from kernel_eval.benches.cann_scoring import aggregate_eq4 as bench_agg
    from kernel_eval.report.scoring import aggregate_eq4 as report_agg
    return bench_agg, report_agg


class TestResolveProfile:
    def test_missing_source_raises(self):
        with pytest.raises(ValueError, match="profile is required"):
            resolve_score_weights()

    def test_env_switch(self, monkeypatch):
        monkeypatch.setenv(SCORE_WEIGHTS_ENV, "1")
        assert resolve_score_weights() is WEIGHT_PROFILES["1"]

    def test_explicit_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv(SCORE_WEIGHTS_ENV, "1")
        assert resolve_score_weights("2") is WEIGHT_PROFILES["2"]

    def test_profiles_sum_to_one(self):
        for profile in WEIGHT_PROFILES.values():
            assert profile.wc + profile.wf + profile.wp + profile.wb == pytest.approx(1.0)

    def test_2_values(self):
        p = WEIGHT_PROFILES["2"]
        assert (p.wc, p.wf, p.wb, p.wp) == (0.05, 0.20, 0.05, 0.70)

    def test_invalid_profile_raises(self):
        with pytest.raises(ValueError):
            resolve_score_weights("season3")

    def test_invalid_env_raises(self, monkeypatch):
        monkeypatch.setenv(SCORE_WEIGHTS_ENV, "banana")
        with pytest.raises(ValueError):
            resolve_score_weights()


class TestMajor1Unchanged:
    """1 历史档经切换后必须与历史行为逐位一致（口径永久保留）。"""

    @pytest.mark.parametrize("how", ["arg", "env"])
    def test_matches_legacy_literal(self, how, monkeypatch):
        # N=4：3 个精度通过（其中 1 个缺锚点 None），1 个失败。
        case_scores = [(True, 0.5), (True, 0.8), (True, None), (False, None)]
        kwargs = {}
        if how == "arg":
            kwargs["weights"] = "1"
        else:
            monkeypatch.setenv(SCORE_WEIGHTS_ENV, "1")
        for agg_fn in _agg_fns():
            agg = agg_fn(compile_passed=True, total_cases=4,
                         case_scores=case_scores, **kwargs)
            # 历史公式字面展开：wc=0.2, wf=0.3, wp=0.5
            assert agg["compilation_score"] == pytest.approx(0.2 * 100.0)
            assert agg["function_score"] == pytest.approx(3 * 0.3 / 4 * 100.0)
            assert agg["performance_score"] == pytest.approx(1.3 * 0.5 / 4 * 100.0)
            assert agg["bonus_score"] == 0.0
            assert agg["total_score"] == pytest.approx(
                20.0 + 22.5 + 16.25)

    def test_1_all_pass_has_no_bonus(self):
        """1 档 w_b=0：即使全用例通过，bonus 恒为 0，总分口径不变。"""
        case_scores = [(True, 1.0), (True, 1.0)]
        for agg_fn in _agg_fns():
            agg = agg_fn(compile_passed=True, total_cases=2,
                         case_scores=case_scores, weights="1")
            assert agg["bonus_score"] == 0.0
            assert agg["total_score"] == pytest.approx(20.0 + 30.0 + 50.0)


class TestMajor2ContentDefault:
    """入口解析 tasks 2.x 后，聚合通过环境变量获得档 2。"""

    def test_default_all_pass_full_score_is_100(self, monkeypatch):
        monkeypatch.setenv(SCORE_WEIGHTS_ENV, "2")
        case_scores = [(True, 1.0), (True, 1.0)]
        for agg_fn in _agg_fns():
            agg = agg_fn(compile_passed=True, total_cases=2,
                         case_scores=case_scores)
            assert agg["compilation_score"] == pytest.approx(5.0)
            assert agg["function_score"] == pytest.approx(20.0)
            assert agg["bonus_score"] == pytest.approx(5.0)
            assert agg["performance_score"] == pytest.approx(70.0)
            assert agg["total_score"] == pytest.approx(100.0)


class TestMajor2Profile:
    def test_all_pass_full_score_is_100(self):
        """5 编译 + 20 精度 + 5 全通过 bonus + 70 性能 = 100。"""
        case_scores = [(True, 1.0), (True, 1.0)]
        for agg_fn in _agg_fns():
            agg = agg_fn(compile_passed=True, total_cases=2,
                         case_scores=case_scores, weights="2")
            assert agg["compilation_score"] == pytest.approx(5.0)
            assert agg["function_score"] == pytest.approx(20.0)
            assert agg["bonus_score"] == pytest.approx(5.0)
            assert agg["performance_score"] == pytest.approx(70.0)
            assert agg["total_score"] == pytest.approx(100.0)

    def test_one_accuracy_fail_drops_bonus(self):
        case_scores = [(True, 1.0), (False, None)]
        for agg_fn in _agg_fns():
            agg = agg_fn(compile_passed=True, total_cases=2,
                         case_scores=case_scores, weights="2")
            assert agg["bonus_score"] == 0.0
            assert agg["function_score"] == pytest.approx(10.0)
            assert agg["performance_score"] == pytest.approx(35.0)

    def test_missing_case_drops_bonus(self):
        """漏跑（case_scores 少于 total_cases）视为未全通过。"""
        case_scores = [(True, 1.0), (True, 1.0)]
        for agg_fn in _agg_fns():
            agg = agg_fn(compile_passed=True, total_cases=3,
                         case_scores=case_scores, weights="2")
            assert agg["bonus_score"] == 0.0

    def test_env_switch_reaches_both_modules(self, monkeypatch):
        monkeypatch.setenv(SCORE_WEIGHTS_ENV, "2")
        case_scores = [(True, 1.0), (True, 1.0)]
        for agg_fn in _agg_fns():
            agg = agg_fn(compile_passed=True, total_cases=2,
                         case_scores=case_scores)
            assert agg["total_score"] == pytest.approx(100.0)

    def test_explicit_single_weight_overrides_profile(self):
        """显式传入的单个权重优先于权重档的对应项。"""
        case_scores = [(True, 1.0)]
        for agg_fn in _agg_fns():
            agg = agg_fn(compile_passed=True, total_cases=1,
                         case_scores=case_scores, weights="2", wc=0.2)
            assert agg["compilation_score"] == pytest.approx(20.0)
            assert agg["performance_score"] == pytest.approx(70.0)

    def test_invalid_env_raises_in_aggregate(self, monkeypatch):
        monkeypatch.setenv(SCORE_WEIGHTS_ENV, "banana")
        for agg_fn in _agg_fns():
            with pytest.raises(ValueError):
                agg_fn(compile_passed=True, total_cases=1,
                       case_scores=[(True, 1.0)])

    def test_operator_cap_still_applies(self):
        """2 + HAP 大额奖励分 → 仍被 OPERATOR_SCORE_CAP 封顶。"""
        case_scores = [(True, PER_CASE_SCORE_CAP)]  # 单用例封顶的 HAP 奖励
        for agg_fn in _agg_fns():
            agg = agg_fn(compile_passed=True, total_cases=1,
                         case_scores=case_scores, weights="2")
            uncapped = 5.0 + 20.0 + 5.0 + PER_CASE_SCORE_CAP * 0.70 * 100.0
            assert agg["total_score_uncapped"] == pytest.approx(uncapped)
            assert agg["total_score"] == pytest.approx(OPERATOR_SCORE_CAP)

    def test_anti_cheat_zero_path_zeroes_bonus(self):
        """反作弊置零路径在 2 档下同样清零（含 bonus）。"""
        case_scores = [(True, None), (True, None)]
        for agg_fn in _agg_fns():
            agg = agg_fn(compile_passed=True, total_cases=2,
                         case_scores=case_scores, weights="2",
                         n_no_perf_pass=2)
            assert agg["total_score"] == 0.0
            assert agg["bonus_score"] == 0.0
            assert agg["zeroed_by_no_npu_perf"] is True

    def test_compile_fail_is_zero(self):
        for agg_fn in _agg_fns():
            agg = agg_fn(compile_passed=False, total_cases=2,
                         case_scores=[(False, None), (False, None)],
                         weights="2")
            assert agg["total_score"] == 0.0
            assert agg["bonus_score"] == 0.0


class TestOrthogonalToPerfMetric:
    """权重档与单用例公式（eq3 / eq3_log）正交——任意组合均可用。"""

    @pytest.mark.parametrize("metric", ["eq3", "eq3_log"])
    @pytest.mark.parametrize("weights", ["1", "2"])
    def test_all_four_combinations(self, metric, weights, monkeypatch):
        from kernel_eval.report.scoring import aggregate_eq4, per_case_sol_score

        monkeypatch.setenv(PERF_METRIC_ENV, metric)
        monkeypatch.setenv(SCORE_WEIGHTS_ENV, weights)
        b, c, h = 300.0, 120.0, 60.0
        score_i = per_case_sol_score(b, c, h)
        agg = aggregate_eq4(compile_passed=True, total_cases=1,
                            case_scores=[(True, score_i)])
        p = WEIGHT_PROFILES[weights]
        expected = (p.wc + p.wf + p.wb + p.wp * score_i) * 100.0
        assert agg["total_score"] == pytest.approx(expected)

    def test_metric_choice_does_not_change_weights(self, monkeypatch):
        """同一 case_scores 下，总分只由权重档决定，与 perf metric 环境无关。"""
        bench_agg, _ = _agg_fns()
        case_scores = [(True, 0.5), (True, 0.5)]
        monkeypatch.setenv(SCORE_WEIGHTS_ENV, "2")
        monkeypatch.setenv(PERF_METRIC_ENV, "eq3")
        total_eq3 = bench_agg(compile_passed=True, total_cases=2,
                              case_scores=case_scores)["total_score"]
        monkeypatch.setenv(PERF_METRIC_ENV, "eq3_log")
        total_log = bench_agg(compile_passed=True, total_cases=2,
                              case_scores=case_scores)["total_score"]
        assert total_eq3 == total_log == pytest.approx(
            5.0 + 20.0 + 5.0 + 0.5 * 0.70 * 100.0)


class TestCalculatorPassthrough:
    """ScoringCalculator 默认构造必须跟随环境变量权重档。"""

    def test_default_calculator_resolves_at_call_time(self, monkeypatch):
        from kernel_eval.benches.cann_scoring import ScoringCalculator as BenchCalc
        from kernel_eval.report.scoring import ScoringCalculator as ReportCalc

        for calc_cls in (BenchCalc, ReportCalc):
            calc = calc_cls()
            monkeypatch.setenv(SCORE_WEIGHTS_ENV, "2")
            assert calc._effective_weights() == (0.05, 0.20, 0.70, 0.05)
            monkeypatch.setenv(SCORE_WEIGHTS_ENV, "1")
            assert calc._effective_weights() == (0.2, 0.3, 0.5, 0.0)

    def test_explicit_calculator_weights_win(self, monkeypatch):
        from kernel_eval.report.scoring import ScoringCalculator

        monkeypatch.setenv(SCORE_WEIGHTS_ENV, "2")
        calc = ScoringCalculator(wc=0.2, wf=0.3, wp=0.5, wb=0.0)
        assert calc._effective_weights() == (0.2, 0.3, 0.5, 0.0)
