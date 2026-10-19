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
验证 ScoreInfo → OperatorScoreInfo / CaseScoreInfo 重命名后的导入兼容性

P0-1: 两处 ScoreInfo 重命名，消除同名混淆。
"""

import pytest


class TestOperatorScoreInfoImport:
    """scoring.py 的 ScoreInfo 应重命名为 OperatorScoreInfo"""

    def test_can_import_operator_score_info_from_scoring(self):
        """scoring.py 应导出 OperatorScoreInfo"""
        from kernel_eval.report.scoring import OperatorScoreInfo
        assert OperatorScoreInfo is not None

    def test_operator_score_info_has_operator_level_fields(self):
        """OperatorScoreInfo 应有算子级别的专用字段"""
        from kernel_eval.report.scoring import OperatorScoreInfo
        info = OperatorScoreInfo(operator="Exp", rel_path="level1/exp", total_score=85.0)
        assert info.operator == "Exp"
        assert info.rel_path == "level1/exp"
        assert info.total_score == 85.0
        assert hasattr(info, 'compilation_score')
        assert hasattr(info, 'function_score')
        assert hasattr(info, 'performance_score')
        assert hasattr(info, 'per_case_scores')

    def test_operator_score_info_to_dict_includes_all_fields(self):
        """to_dict 应包含三轴得分"""
        from kernel_eval.report.scoring import OperatorScoreInfo
        info = OperatorScoreInfo(
            operator="Exp", rel_path="level1/exp",
            compilation_score=20.0, function_score=30.0, performance_score=35.0,
            total_score=85.0, pass_rate=1.0, avg_speedup=2.0,
            compile_passed=True, passed_cases=3, total_cases=3,
        )
        d = info.to_dict()
        assert d['compilation_score'] == 20.0
        assert d['function_score'] == 30.0
        assert d['performance_score'] == 35.0
        assert d['total_score'] == 85.0

    def test_scoring_calculator_uses_operator_score_info(self):
        """ScoringCalculator.calculate_operator_score 应返回 OperatorScoreInfo"""
        from kernel_eval.report.scoring import ScoringCalculator, OperatorScoreInfo
        from kernel_eval.eval.results import EvalOperatorResult
        op = EvalOperatorResult(
            rel_path="level1/exp", operator="Exp",
            total_cases=1, passed_cases=1, failed_cases=0, skipped_cases=0,
            results=[], pass_rate=1.0, avg_speedup=2.0,
        )
        info = ScoringCalculator(weights="2").calculate_operator_score(op)
        assert isinstance(info, OperatorScoreInfo)


class TestCaseScoreInfoImport:
    """scoring_scheme.py 的 ScoreInfo 应重命名为 CaseScoreInfo"""

    def test_can_import_case_score_info_from_scheme(self):
        """scoring_scheme.py 应导出 CaseScoreInfo"""
        from kernel_eval.base.scoring import CaseScoreInfo
        assert CaseScoreInfo is not None

    def test_case_score_info_has_case_level_fields(self):
        """CaseScoreInfo 应有用例级别的专用字段"""
        from kernel_eval.base.scoring import CaseScoreInfo
        info = CaseScoreInfo(
            operator="Exp", rel_path="level1/exp",
            passed=True, elapsed_us=50.0, baseline_us=100.0, score=0.5,
        )
        assert info.operator == "Exp"
        assert info.passed is True
        assert info.elapsed_us == 50.0
        assert info.baseline_us == 100.0
        assert info.score == 0.5
        assert info.metadata == {}

    def test_case_score_info_used_by_cann_scoring(self):
        """cann_scoring.py 的 aggregate_operator_scores 应接受 CaseScoreInfo 列表"""
        from kernel_eval.base.scoring import CaseScoreInfo
        from kernel_eval.benches.cann_scoring import CannScoringScheme
        scheme = CannScoringScheme()
        scores = [
            CaseScoreInfo(passed=True, score=0.8),
            CaseScoreInfo(passed=True, score=0.6),
            CaseScoreInfo(passed=False),
        ]
        result = scheme.aggregate_operator_scores(scores, compile_passed=True)
        assert result >= 0

    def test_report_init_exports_renamed_classes(self):
        """OperatorScoreInfo 应从 benches.cann 导入，CaseScoreInfo 应从 base 导入"""
        from kernel_eval.benches import OperatorScoreInfo
        from kernel_eval.base import CaseScoreInfo
        assert OperatorScoreInfo is not None
        assert CaseScoreInfo is not None
