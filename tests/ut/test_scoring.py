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

"""
Scoring 模块单元测试

测试对象：kernel_eval.report.scoring
覆盖：Eq.3 (per_case_sol_score)、Eq.4 (aggregate_eq4 / ScoringCalculator)、
Eq.5 (calculate_overall_score / calculate_level_score)。
"""

from dataclasses import asdict

import pytest

from kernel_eval.base.result import (
    AccuracyResult,
    FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
    FAILURE_TYPE_PRECISION_MISMATCH,
)
from kernel_eval.eval.results import EvalCaseResult, EvalOperatorResult
from kernel_eval.eval.perf_eval import PerfResult
from kernel_eval.report.report_generator import OperatorReport
from kernel_eval.report.scoring import (
    NO_NPU_PERF_ERROR,
    NO_NPU_PERF_ERROR_CODE,
    WEIGHT_COMPILATION,
    WEIGHT_FUNCTION,
    WEIGHT_PERFORMANCE,
    OperatorScoreInfo,
    ScoringCalculator,
    aggregate_eq4,
    per_case_sol_score,
)
from kernel_eval.report.summary_generator import calculate_operator_summary, generate_summary

@pytest.fixture(autouse=True)
def _pin_season1_weights(monkeypatch):
    """本文件的数值断言按 season-1 权重档（0.2/0.3/0.5，无 bonus）书写。

    该档作为历史口径永久保留、逐位不变（经 CANN_BENCH_SCORE_WEIGHTS=season1
    或 weights="season1" 切换）；season-2 默认档的行为由
    test_score_weights_season2.py 钉死。
    """
    monkeypatch.setenv("CANN_BENCH_SCORE_WEIGHTS", "season1")



class TestPerCaseSolScore:
    """Eq.3: score_i = (T_base - T_HW) / ((T_cand - T_HW) + (T_base - T_HW))"""

    def test_score_at_baseline_equals_half(self):
        # T_cand == T_baseline ⇒ score == 0.5
        assert per_case_sol_score(100, 100, 50) == pytest.approx(0.5)

    def test_score_at_hw_equals_one(self):
        # T_cand == T_HW ⇒ score == 1.0
        assert per_case_sol_score(100, 50, 50) == pytest.approx(1.0)

    def test_score_can_exceed_one_when_cand_below_hw(self):
        # T_cand < T_HW（突破硬件下界）允许 score > 1.0
        score = per_case_sol_score(t_baseline=100, t_cand=25, t_hw=50)
        assert score > 1.0

    def test_zero_anchor_returns_none(self):
        # baseline=0 现在走 fallback，不再返回 None；只剩 t_cand=0 / t_hw=0 返回 None
        assert per_case_sol_score(100, 0, 50) is None
        assert per_case_sol_score(100, 80, 0) is None

    def test_negative_anchor_returns_none(self):
        # 负 t_cand / 负 t_hw 仍返回 None；负 baseline 等同缺失，走 fallback
        assert per_case_sol_score(100, -1, 50) is None
        assert per_case_sol_score(100, 80, -1) is None

    def test_baseline_missing_uses_fallback(self):
        # baseline ≤ 0 时：fallback = max(t_hw*10, 10)
        # t_hw=50 → fallback=max(500,10)=500；t_cand=100
        # score = (500-50) / ((100-50) + (500-50)) = 450/500 = 0.9
        assert per_case_sol_score(t_baseline=0, t_cand=100, t_hw=50) == pytest.approx(0.9)

    def test_baseline_missing_fallback_uses_10us_floor(self):
        # t_hw 很小时 fallback 退到 10 us 地板
        # t_hw=1 → fallback=max(10,10)=10（10us 地板）；t_cand=5
        # score = (10-1) / ((5-1) + (10-1)) = 9/13
        assert per_case_sol_score(t_baseline=0, t_cand=5, t_hw=1) == pytest.approx(9.0 / 13.0)

    def test_baseline_negative_treated_as_missing(self):
        # 负 baseline 与 0 等效，走 fallback
        score = per_case_sol_score(t_baseline=-1, t_cand=100, t_hw=50)
        assert score is not None

    def test_denom_zero_returns_none(self):
        # (t_cand - t_hw) + (t_base - t_hw) == 0
        assert per_case_sol_score(t_baseline=50, t_cand=50, t_hw=50) is None

    def test_baseline_below_hw_emits_warning(self, caplog):
        # F059: _warn_baseline_below_hw 改用 _logger.warning → stderr
        score = per_case_sol_score(t_baseline=40, t_cand=80, t_hw=50, rel_path="test")
        assert score is not None
        assert "T_baseline" in caplog.text
        assert "< T_HW" in caplog.text


class TestAggregateEq4:
    """Eq.4 聚合（dict-input 与 EvalOperatorResult 共用同一路径）"""

    def test_all_pass_full_score_when_cand_at_hw(self):
        # 所有用例 score_i=1.0 ⇒ total_score=100
        agg = aggregate_eq4(
            compile_passed=True,
            total_cases=3,
            case_scores=[(True, 1.0), (True, 1.0), (True, 1.0)],
        )
        assert agg["compilation_score"] == pytest.approx(20.0)
        assert agg["function_score"] == pytest.approx(30.0)
        assert agg["performance_score"] == pytest.approx(50.0)
        assert agg["total_score"] == pytest.approx(100.0)

    def test_compile_fail_zeros_everything(self):
        # 编译失败：function/performance 都置零
        agg = aggregate_eq4(
            compile_passed=False,
            total_cases=3,
            case_scores=[(True, 1.0), (True, 1.0), (True, 1.0)],
        )
        assert agg["compilation_score"] == 0.0
        assert agg["function_score"] == 0.0
        assert agg["performance_score"] == 0.0
        assert agg["total_score"] == 0.0
        assert agg["per_case_scores"] == [None, None, None]

    def test_none_score_counted_as_zero_for_performance(self):
        # score_i=None（缺锚点）：功能通过但性能项按 0 计入
        agg = aggregate_eq4(
            compile_passed=True,
            total_cases=1,
            case_scores=[(True, None)],
        )
        assert agg["compilation_score"] == pytest.approx(20.0)
        assert agg["function_score"] == pytest.approx(30.0)
        assert agg["performance_score"] == 0.0
        assert agg["total_score"] == pytest.approx(50.0)

    def test_partial_pass_uses_total_cases_as_denominator(self):
        # 5 个用例只过 2 个，分母用 5 而不是 2
        agg = aggregate_eq4(
            compile_passed=True,
            total_cases=5,
            case_scores=[(True, 1.0), (True, 1.0), (False, None),
                         (False, None), (False, None)],
        )
        assert agg["function_score"] == pytest.approx(2 * WEIGHT_FUNCTION / 5 * 100)
        assert agg["performance_score"] == pytest.approx(2 * WEIGHT_PERFORMANCE / 5 * 100)

    def test_compile_runtime_failures_deduct_compile_runtime_per_case(self):
        agg = aggregate_eq4(
            compile_passed=True,
            total_cases=5,
            case_scores=[(True, 1.0), (False, None), (False, None),
                         (False, None), (False, None)],
            n_compile_runtime_fail=2,
        )
        assert agg["compilation_score"] == pytest.approx(3 * WEIGHT_COMPILATION / 5 * 100)
        assert agg["n_compile_runtime_fail"] == 2

    def test_precision_failures_do_not_deduct_compile_runtime_score(self):
        agg = aggregate_eq4(
            compile_passed=True,
            total_cases=5,
            case_scores=[(True, 1.0), (False, None), (False, None),
                         (False, None), (False, None)],
            n_compile_runtime_fail=0,
        )
        assert agg["compilation_score"] == pytest.approx(WEIGHT_COMPILATION * 100)

    def test_cand_below_hw_inflates_score_above_100(self):
        # T_cand < T_HW 时 score_i > 1，总分可超过 100
        agg = aggregate_eq4(
            compile_passed=True,
            total_cases=1,
            case_scores=[(True, 2.0)],
        )
        assert agg["total_score"] > 100.0


class TestScoringCalculator:
    """ScoringCalculator.calculate_operator_score 与 dict 路径一致性"""

    @staticmethod
    def _make_case(success, baseline_us, t_hw, elapsed_us, failure_type=None,
                   perf_collection_failed=None):
        """Build an EvalCaseResult for scoring tests.

        perf_collection_failed:
          - None (default): perf=None when elapsed_us<=0 (profiler 未运行/采集异常)
          - True:  perf with elapsed_us=0, metadata perf_collection_failed=True (采集异常)
          - False: perf with elapsed_us=0, metadata perf_collection_failed=False (CSV 已产出但无 NPU kernel = 作弊)
        """
        if elapsed_us and elapsed_us > 0:
            perf = PerfResult(
                elapsed_us=elapsed_us,
                metadata={'baseline_us': baseline_us, 't_hw_us': t_hw},
            )
        elif perf_collection_failed is not None:
            perf = PerfResult(
                elapsed_us=0.0,
                metadata={
                    'baseline_us': baseline_us,
                    't_hw_us': t_hw,
                    'perf_collection_failed': perf_collection_failed,
                },
            )
        else:
            perf = None
        return EvalCaseResult(
            case_id="c", rel_path="level1/exp", operator="Exp",
            case_num=0, success=success, perf_result=perf,
            baseline_perf_us=baseline_us, t_hw_us=t_hw,
            failure_type=failure_type,
        )

    def test_all_pass(self):
        cases = [
            self._make_case(True, 100, 50, 50),   # score=1.0
            self._make_case(True, 100, 50, 75),   # score=0.667
        ]
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=2, passed_cases=2, failed_cases=0, skipped_cases=0,
            results=cases, pass_rate=1.0, avg_speedup=1.5,
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.compile_passed is True
        assert info.compilation_score == pytest.approx(20.0)
        assert info.function_score == pytest.approx(30.0)
        # perf_score_sum = 1.0 + 2/3 = 5/3; (5/3 * 0.5 / 2) * 100 = 41.67
        assert info.performance_score == pytest.approx(5 / 3 * WEIGHT_PERFORMANCE / 2 * 100)

    def test_total_cases_uses_max_of_declared_and_actual(self):
        # 声明 total_cases=10，但 results 只有 2 条——分母用 10
        cases = [self._make_case(True, 100, 50, 50)] * 2
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=10, passed_cases=2, failed_cases=0, skipped_cases=8,
            results=cases, pass_rate=0.2, avg_speedup=1.0,
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.total_cases == 10
        assert info.function_score == pytest.approx(2 * WEIGHT_FUNCTION / 10 * 100)

    def test_accuracy_pass_csv_present_no_npu_kernel_zeroes_operator(self):
        """CSV 已产出但无 NPU kernel（perf_collection_failed=False）→ 反作弊触发，整算子 0 分。"""
        cases = [
            self._make_case(True, 100, 50, 50),
            self._make_case(True, 100, 50, 0, perf_collection_failed=False),
        ]
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=2, passed_cases=2, failed_cases=0, skipped_cases=0,
            results=cases, pass_rate=1.0, avg_speedup=1.0,
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.compilation_score == 0.0
        assert info.function_score == 0.0
        assert info.performance_score == 0.0
        assert info.total_score == 0.0
        assert info.score_error_code == NO_NPU_PERF_ERROR_CODE
        assert info.zeroed_by_no_npu_perf is True

        report = OperatorReport.from_eval_operator_result(
            op, info.total_score, score_info=info,
        )
        assert report.score_error_code == NO_NPU_PERF_ERROR_CODE
        assert report.score_error == NO_NPU_PERF_ERROR
        assert report.cases[0].performance_error_msg is None
        assert report.cases[1].performance_error_msg == NO_NPU_PERF_ERROR

    def test_accuracy_pass_perf_none_not_cheating(self):
        """perf_result=None（profiler 未运行/采集异常）→ 不触发反作弊，保留编译/精度分。"""
        cases = [
            self._make_case(True, 100, 50, 50),
            self._make_case(True, 100, 50, 0),  # perf=None
        ]
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=2, passed_cases=2, failed_cases=0, skipped_cases=0,
            results=cases, pass_rate=1.0, avg_speedup=1.0,
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.compilation_score == pytest.approx(WEIGHT_COMPILATION * 100)
        assert info.function_score == pytest.approx(WEIGHT_FUNCTION * 100)
        assert info.performance_score == pytest.approx(
            WEIGHT_PERFORMANCE * 1.0 / 2 * 100
        )
        assert info.score_error_code is None
        assert info.zeroed_by_no_npu_perf is False

    def test_accuracy_pass_collection_failed_not_cheating(self):
        """perf_collection_failed=True（采集异常）→ 不触发反作弊，保留编译/精度分。"""
        cases = [
            self._make_case(True, 100, 50, 50),
            self._make_case(True, 100, 50, 0, perf_collection_failed=True),
        ]
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=2, passed_cases=2, failed_cases=0, skipped_cases=0,
            results=cases, pass_rate=1.0, avg_speedup=1.0,
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.compilation_score == pytest.approx(WEIGHT_COMPILATION * 100)
        assert info.function_score == pytest.approx(WEIGHT_FUNCTION * 100)
        assert info.performance_score == pytest.approx(
            WEIGHT_PERFORMANCE * 1.0 / 2 * 100
        )
        assert info.score_error_code is None
        assert info.zeroed_by_no_npu_perf is False

    def test_collection_failure_survives_process_result_roundtrip(self):
        """子进程 JSON 往返不能丢失采集异常标记并误触发反作弊。"""
        case = self._make_case(
            True, 100, 50, 0, perf_collection_failed=True,
        )
        case.perf_result.error_msg = "kernel_details.csv not found"
        roundtrip = EvalCaseResult.from_dict(case.to_dict())
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=1, passed_cases=1, failed_cases=0, skipped_cases=0,
            results=[roundtrip], pass_rate=1.0, avg_speedup=0.0,
        )

        assert roundtrip.perf_result is not None
        assert roundtrip.perf_result.error_msg == "kernel_details.csv not found"
        assert roundtrip.perf_result.metadata["perf_collection_failed"] is True
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.score_error_code is None
        assert info.zeroed_by_no_npu_perf is False
        assert info.total_score == pytest.approx(
            (WEIGHT_COMPILATION + WEIGHT_FUNCTION) * 100
        )

    def test_empty_operator_returns_zero(self):
        # F062: 空壳算子（0 声明 + 0 实测）直接 0 分，不适用 max(..., 1) floor
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=0, passed_cases=0, failed_cases=0, skipped_cases=0,
            results=[], pass_rate=0.0, avg_speedup=0.0,
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.total_cases == 0
        assert info.total_score == 0.0

    def test_compile_failed_zeroes_function_and_perf(self):
        cases = [self._make_case(True, 100, 50, 50)]
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=1, passed_cases=1, failed_cases=0, skipped_cases=0,
            results=cases, pass_rate=1.0, avg_speedup=1.0,
            compilation_error="compile failed",
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.compile_passed is False
        assert info.compilation_score == 0.0
        assert info.function_score == 0.0
        assert info.performance_score == 0.0

    def test_runtime_case_failure_deducts_compile_runtime_component(self):
        cases = [
            self._make_case(True, 100, 50, 50),
            self._make_case(False, 100, 50, 0, FAILURE_TYPE_COMPILE_RUNTIME_ERROR),
        ]
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=2, passed_cases=1, failed_cases=1, skipped_cases=0,
            results=cases, pass_rate=0.5, avg_speedup=1.0,
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.compile_runtime_fail_cases == 1
        assert info.compilation_score == pytest.approx(WEIGHT_COMPILATION / 2 * 100)

    def test_precision_case_failure_keeps_compile_runtime_component(self):
        cases = [
            self._make_case(True, 100, 50, 50),
            self._make_case(False, 100, 50, 0, FAILURE_TYPE_PRECISION_MISMATCH),
        ]
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=2, passed_cases=1, failed_cases=1, skipped_cases=0,
            results=cases, pass_rate=0.5, avg_speedup=1.0,
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.compile_runtime_fail_cases == 0
        assert info.compilation_score == pytest.approx(WEIGHT_COMPILATION * 100)

    def test_subprocess_failed_zeroes_compilation_score(self):
        """子进程失败的算子不应获得编译分"""
        cases = [self._make_case(True, 100, 50, 50)]
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=1, passed_cases=0, failed_cases=1, skipped_cases=0,
            results=cases, pass_rate=0.0, avg_speedup=0.0,
            subprocess_failure_reason="subprocess timeout",
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.compile_passed is False
        assert info.compilation_score == 0.0
        assert info.function_score == 0.0
        assert info.performance_score == 0.0
        assert info.total_score == 0.0

    def test_runtime_failure_in_case_error_msg_deducts_compile_runtime(self):
        """Runtime execution failure recorded on the case-level ``error_msg``
        (aclnn call failed / AclOpKernelInit failed) — with ``failure_type``
        unset and a bare accuracy stub — must deduct the compile/runtime
        component, not default to a precision mismatch that keeps the
        ``wc*100`` compile credit.

        Regression: previously the case-level ``error_msg`` was only consulted
        in the dict-input branch of ``is_compile_runtime_case_failure``; live
        ``EvalCaseResult`` objects fell through to ``get_accuracy_failure_type``
        which only inspects ``accuracy_result.error_msg``, so a runtime-failed
        operator with a bare accuracy stub kept the 20-point compile credit and
        inflated the total score.
        """
        runtime_case = EvalCaseResult(
            case_id="c", rel_path="level2/softmax", operator="Softmax",
            case_num=0, success=False, perf_result=None,
            baseline_perf_us=100, t_hw_us=50,
            accuracy_result=AccuracyResult(passed=False, metadata={"dtype": "06"}),
            error_msg=(
                "AI算子执行失败: call aclnnSoftmax failed, detail:EZ9999: "
                "Inner Error! AclOpKernelInit failed opType. SoftmaxAiCore failed."
            ),
            failure_type=None,
        )
        op = EvalOperatorResult(
            rel_path="level2/softmax", operator="Softmax",
            total_cases=1, passed_cases=0, failed_cases=1, skipped_cases=0,
            results=[runtime_case], pass_rate=0.0, avg_speedup=0.0,
        )
        info = ScoringCalculator().calculate_operator_score(op)
        assert info.compile_runtime_fail_cases == 1
        assert info.compilation_score == 0.0
        assert info.function_score == 0.0
        assert info.performance_score == 0.0
        assert info.total_score == 0.0


class TestSummaryDictScoring:
    def test_nested_summary_preserves_profiler_collection_failure(self):
        op = EvalOperatorResult(
            rel_path="level1/exp",
            operator="Exp",
            total_cases=2,
            passed_cases=2,
            failed_cases=0,
            skipped_cases=0,
            results=[
                EvalCaseResult(
                    case_id="case_1",
                    rel_path="level1/exp",
                    operator="Exp",
                    case_num=1,
                    success=True,
                    perf_result=PerfResult(elapsed_us=50),
                    baseline_perf_us=100,
                    t_hw_us=50,
                ),
                EvalCaseResult(
                    case_id="case_2",
                    rel_path="level1/exp",
                    operator="Exp",
                    case_num=2,
                    success=True,
                    perf_result=PerfResult(
                        elapsed_us=0,
                        metadata={"perf_collection_failed": True},
                    ),
                    baseline_perf_us=100,
                    t_hw_us=50,
                ),
            ],
            pass_rate=1.0,
            avg_speedup=1.0,
        )
        summary = generate_summary({"operators": [op.to_dict()]})

        assert summary.operators[0].composite_score == pytest.approx(75.0)

        no_npu_perf = op.to_dict()
        no_npu_perf["results"][1]["perf"]["metadata"]["perf_collection_failed"] = False
        summary = generate_summary({"operators": [no_npu_perf]})
        assert summary.operators[0].composite_score == pytest.approx(0.0)

    def test_flat_report_summary_preserves_profiler_collection_failure(self):
        op = EvalOperatorResult(
            rel_path="level1/exp",
            operator="Exp",
            total_cases=2,
            passed_cases=2,
            failed_cases=0,
            skipped_cases=0,
            results=[
                EvalCaseResult(
                    case_id="case_1",
                    rel_path="level1/exp",
                    operator="Exp",
                    case_num=1,
                    success=True,
                    perf_result=PerfResult(elapsed_us=50),
                    baseline_perf_us=100,
                    t_hw_us=50,
                ),
                EvalCaseResult(
                    case_id="case_2",
                    rel_path="level1/exp",
                    operator="Exp",
                    case_num=2,
                    success=True,
                    perf_result=PerfResult(
                        elapsed_us=0,
                        metadata={"perf_collection_failed": True},
                    ),
                    baseline_perf_us=100,
                    t_hw_us=50,
                ),
            ],
            pass_rate=1.0,
            avg_speedup=1.0,
        )
        score_info = ScoringCalculator().calculate_operator_score(op)
        report = OperatorReport.from_eval_operator_result(
            op,
            score_info.total_score,
            score_info=score_info,
        )
        summary = generate_summary({"operators": [asdict(report)]})

        assert score_info.total_score == pytest.approx(75.0)
        assert summary.operators[0].composite_score == pytest.approx(score_info.total_score)

    def test_json_summary_counts_compile_runtime_failures(self):
        summary = calculate_operator_summary({
            "operator": "Exp",
            "rel_path": "level1/exp",
            "total_cases": 2,
            "passed_cases": 1,
            "cases": [
                {
                    "status": "success",
                    "baseline_perf_us": 100,
                    "t_hw_us": 50,
                    "elapsed_us": 50,
                },
                {
                    "status": "failed",
                    "failure_type": FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
                    "accuracy": None,
                },
            ],
        })
        assert summary.compile_runtime_fail_cases == 1
        assert summary.compilation_score == pytest.approx(WEIGHT_COMPILATION / 2 * 100)

    def test_json_summary_classifies_runtime_error_without_explicit_failure_type(self):
        summary = calculate_operator_summary({
            "operator": "Exp",
            "rel_path": "level1/exp",
            "total_cases": 2,
            "passed_cases": 1,
            "cases": [
                {
                    "status": "success",
                    "baseline_perf_us": 100,
                    "t_hw_us": 50,
                    "elapsed_us": 50,
                },
                {
                    "status": "failed",
                    "accuracy": {
                        "passed": False,
                        "error_msg": "RuntimeError: dtype unsupported",
                    },
                },
            ],
        })
        assert summary.compile_runtime_fail_cases == 1
        assert summary.compilation_score == pytest.approx(WEIGHT_COMPILATION / 2 * 100)

    def test_json_summary_classifies_ai_execution_failure_without_explicit_failure_type(self):
        summary = calculate_operator_summary({
            "operator": "Exp",
            "rel_path": "level1/exp",
            "total_cases": 2,
            "passed_cases": 1,
            "cases": [
                {
                    "status": "success",
                    "baseline_perf_us": 100,
                    "t_hw_us": 50,
                    "elapsed_us": 50,
                },
                {
                    "status": "failed",
                    "error_msg": "AI算子执行失败: npuSynchronizeDevice failed",
                    "accuracy": {"passed": False},
                },
            ],
        })
        assert summary.compile_runtime_fail_cases == 1
        assert summary.compilation_score == pytest.approx(WEIGHT_COMPILATION / 2 * 100)

    def test_json_summary_classifies_shape_mismatch_without_explicit_failure_type(self):
        summary = calculate_operator_summary({
            "operator": "DynamicQuant",
            "rel_path": "level2/dynamic_quant",
            "total_cases": 2,
            "passed_cases": 1,
            "cases": [
                {
                    "status": "success",
                    "baseline_perf_us": 100,
                    "t_hw_us": 50,
                    "elapsed_us": 50,
                },
                {
                    "status": "failed",
                    "accuracy": {
                        "passed": False,
                        "output_results": [
                            {"passed": False, "error_msg": "形状不匹配: output=torch.Size([1022]), golden=torch.Size([2, 511])"}
                        ],
                    },
                },
            ],
        })
        assert summary.compile_runtime_fail_cases == 1
        assert summary.compilation_score == pytest.approx(WEIGHT_COMPILATION / 2 * 100)

    def test_json_summary_keeps_compile_runtime_for_precision_failures(self):
        summary = calculate_operator_summary({
            "operator": "Exp",
            "rel_path": "level1/exp",
            "total_cases": 2,
            "passed_cases": 1,
            "cases": [
                {
                    "status": "success",
                    "baseline_perf_us": 100,
                    "t_hw_us": 50,
                    "elapsed_us": 50,
                },
                {
                    "status": "failed",
                    "failure_type": FAILURE_TYPE_PRECISION_MISMATCH,
                    "accuracy": {
                        "passed": False,
                        "metadata": {"failure_type": FAILURE_TYPE_PRECISION_MISMATCH},
                    },
                },
            ],
        })
        assert summary.compile_runtime_fail_cases == 0
        assert summary.compilation_score == pytest.approx(WEIGHT_COMPILATION * 100)


class TestLevelAndOverallScores:
    """Eq.5: overall = Σ EachOperatorScore; level_score(level) 过滤 rel_path."""

    @staticmethod
    def _make_info(rel_path, total_score):
        return OperatorScoreInfo(operator="op", rel_path=rel_path, total_score=total_score)

    def test_overall_score_sums_all(self):
        infos = [
            self._make_info("level1/a", 10),
            self._make_info("level2/b", 20),
            self._make_info("level3/c", 30),
        ]
        assert ScoringCalculator().calculate_overall_score(infos) == 60

    def test_level_score_filters_by_rel_path(self):
        infos = [
            self._make_info("level1/a", 10),
            self._make_info("level1/b", 15),
            self._make_info("level2/c", 20),
        ]
        sc = ScoringCalculator()
        assert sc.calculate_level_score(infos, "level1") == 25
        assert sc.calculate_level_score(infos, "level2") == 20
        assert sc.calculate_level_score(infos, "level3") == 0  # 无匹配

    def test_overall_equals_sum_of_levels(self):
        infos = [
            self._make_info("level1/a", 10),
            self._make_info("level1/b", 15),
            self._make_info("level2/c", 20),
            self._make_info("level3/d", 5),
        ]
        sc = ScoringCalculator()
        total = sum(sc.calculate_level_score(infos, lv) for lv in sc.list_levels(infos))
        assert total == sc.calculate_overall_score(infos)

    def test_list_levels_preserves_first_occurrence_order(self):
        infos = [
            self._make_info("level3/a", 1),
            self._make_info("level1/b", 1),
            self._make_info("level3/c", 1),
            self._make_info("level2/d", 1),
        ]
        assert ScoringCalculator().list_levels(infos) == ["level3", "level1", "level2"]
