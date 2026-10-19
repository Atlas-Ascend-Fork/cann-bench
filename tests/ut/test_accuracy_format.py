#!/usr/bin/python3
# coding=utf-8

"""
测试 summary_generator 和 report_generator 对 accuracy 字段的读取路径。

核心问题：
- AccuracyResult.to_dict() 将 mare/mere/max_diff/total_count 放在 metadata 子字典中
- 但 summary_generator.calculate_operator_summary() 和 report_generator 从 accuracy 顶层读取
- 导致报告中的 mare/mere/max_diff 始终为 0

本测试验证：
1. 新格式（metadata 嵌套）能被正确读取
2. 旧格式（顶层）保持兼容
3. Markdown 报告中的 max_diff 正确显示
"""

import pytest
import torch

from kernel_eval.eval.results import EvalCaseResult, EvalOperatorResult, summarize_case_results
from kernel_eval.base.result import AccuracyResult
from kernel_eval.checkers.relative_error_checker import RelativeErrorChecker
from kernel_eval.report.summary_generator import calculate_operator_summary
from kernel_eval.report.report_generator import (
    EvalReport,
    EvalResult,
    OperatorReport,
    ReportGenerator,
)


class TestSummaryGeneratorAccuracyFormat:
    """测试 summary_generator 对 accuracy 字段的读取"""

    @pytest.fixture(autouse=True)
    def _score_profile(self, monkeypatch):
        monkeypatch.setenv("CANN_BENCH_SCORE_WEIGHTS", "2")

    def test_mare_mere_from_metadata_nested_format(self):
        """新格式：mare/mere 在 metadata 中，应能正确读取"""
        # 构造一个精度失败的 case
        torch.manual_seed(42)
        x = torch.randn(100, dtype=torch.float64) * 2
        golden = torch.exp(x + 2.0)
        output = torch.exp(x)

        checker = RelativeErrorChecker()
        accuracy = checker.check(
            ai_outputs=output.float().half(),
            golden_outputs=golden.float().half(),
            dtype='float16',
            threshold=0.009765625,
        )

        case_result = EvalCaseResult(
            case_id='exp_10',
            rel_path='level1/exp',
            operator='Exp',
            case_num=10,
            success=accuracy.is_passed(),
            accuracy_result=accuracy,
            baseline_perf_us=100.0,
            t_hw_us=10.0,
        )

        # 构造 op_dict
        summary = summarize_case_results([case_result])
        op_result = EvalOperatorResult(
            rel_path='level1/exp',
            operator='Exp',
            total_cases=1,
            passed_cases=summary.passed,
            failed_cases=summary.failed,
            skipped_cases=summary.skipped,
            results=[case_result],
            pass_rate=summary.pass_rate,
            avg_speedup=summary.avg_speedup,
        )
        op_dict = op_result.to_dict()

        # 验证 accuracy 结构
        acc_section = op_dict['results'][0]['accuracy']
        assert 'metadata' in acc_section, "accuracy 应包含 metadata 子字典"
        assert 'mare' in acc_section['metadata'], "metadata 应包含 mare"
        assert 'mere' in acc_section['metadata'], "metadata 应包含 mere"
        assert acc_section['metadata']['mare'] > 0, "mare 应为非零值"
        assert acc_section['metadata']['mere'] > 0, "mere 应为非零值"

        # 验证 calculate_operator_summary 能正确读取
        op_summary = calculate_operator_summary(op_dict)
        assert op_summary.mare_avg > 0, f"mare_avg 应为非零，实际为 {op_summary.mare_avg}"
        assert op_summary.mere_avg > 0, f"mere_avg 应为非零，实际为 {op_summary.mere_avg}"
        assert abs(op_summary.mare_avg - acc_section['metadata']['mare']) < 1e-6, \
            f"mare_avg 应与 metadata.mare 一致"

    def test_mare_mere_from_top_level_legacy_format(self):
        """旧格式：mare/mere 在顶层，应保持兼容"""
        # 构造旧格式的 accuracy dict
        old_format_accuracy = {
            'passed': False,
            'threshold': 0.001,
            'mare': 0.5,
            'mere': 0.3,
            'max_diff': 100.0,
            'output_results': [],
        }

        case_dict = {
            'case_id': 'test_1',
            'rel_path': 'level1/test',
            'operator': 'Test',
            'case_num': 1,
            'success': False,
            'accuracy': old_format_accuracy,
            'perf': {'elapsed_us': 100, 'speedup': 2.0},
        }

        op_dict = {
            'rel_path': 'level1/test',
            'operator': 'Test',
            'total_cases': 1,
            'passed_cases': 0,
            'failed_cases': 1,
            'results': [case_dict],
        }

        op_summary = calculate_operator_summary(op_dict)
        assert op_summary.mare_avg == 0.5, f"旧格式 mare_avg 应为 0.5，实际为 {op_summary.mare_avg}"
        assert op_summary.mere_avg == 0.3, f"旧格式 mere_avg 应为 0.3，实际为 {op_summary.mere_avg}"

    def test_allclose_checker_no_mare_mere(self):
        """AllCloseChecker 不产生 mare/mere，列表应为空"""
        accuracy = AccuracyResult(
            passed=False,
            threshold=0.01,
            error_msg="allclose failed",
            metadata={'checker_name': 'allclose', 'atol': 0.01, 'rtol': 0.01},
        )

        case_result = EvalCaseResult(
            case_id='test_1',
            rel_path='level1/test',
            operator='Test',
            case_num=1,
            success=False,
            accuracy_result=accuracy,
        )

        summary = summarize_case_results([case_result])
        op_result = EvalOperatorResult(
            rel_path='level1/test',
            operator='Test',
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
            skipped_cases=0,
            results=[case_result],
            pass_rate=0.0,
            avg_speedup=0.0,
        )
        op_dict = op_result.to_dict()

        op_summary = calculate_operator_summary(op_dict)
        # AllCloseChecker 不产生 mare/mere，所以 avg 应为 0
        assert op_summary.mare_avg == 0.0, "AllCloseChecker 无 mare，avg 应为 0"
        assert op_summary.mere_avg == 0.0, "AllCloseChecker 无 mere，avg 应为 0"


class TestReportGeneratorAccuracyFormat:
    """测试 report_generator 对 accuracy 字段的读取"""

    @staticmethod
    def _render_markdown(eval_result, tmp_path):
        """通过真实 Markdown 入口渲染一个最小报告"""
        operator_report = OperatorReport(
            rel_path=eval_result.rel_path,
            operator=eval_result.operator,
            total_cases=1,
            failed_cases=1,
            cases=[eval_result],
        )
        report = EvalReport(
            framework_version="test",
            tasks_version="test",
            eval_code="accuracy_format",
            timestamp="test",
            device="cpu",
            total_operators=1,
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
            overall_score=0.0,
            operators=[operator_report],
            summary={"pass_rate": 0.0},
        )
        generator = ReportGenerator(output_dir=tmp_path, eval_code="accuracy_format")
        return generator._generate_markdown_content(report)

    def test_max_diff_from_metadata_nested_format(self, tmp_path):
        """新格式：max_diff 在 metadata 中，Markdown 报告应正确显示"""
        torch.manual_seed(42)
        x = torch.randn(100, dtype=torch.float64) * 2
        golden = torch.exp(x + 2.0)
        output = torch.exp(x)

        checker = RelativeErrorChecker()
        accuracy = checker.check(
            ai_outputs=output.float().half(),
            golden_outputs=golden.float().half(),
            dtype='float16',
            threshold=0.009765625,
        )

        case_result = EvalCaseResult(
            case_id='exp_10',
            rel_path='level1/exp',
            operator='Exp',
            case_num=10,
            success=accuracy.is_passed(),
            accuracy_result=accuracy,
            baseline_perf_us=100.0,
            t_hw_us=10.0,
        )

        # 转换为 EvalResult
        eval_result = EvalResult.from_eval_case_result(case_result)

        # 验证 accuracy 结构
        assert eval_result.accuracy is not None
        assert 'metadata' in eval_result.accuracy
        assert 'max_diff' in eval_result.accuracy['metadata']
        assert eval_result.accuracy['metadata']['max_diff'] > 0

        # 通过真实 Markdown 入口验证用户可见的精度误差列
        content = self._render_markdown(eval_result, tmp_path)
        expected = f"{eval_result.accuracy['metadata']['max_diff']:.6f}"
        assert f"| exp_10 |" in content
        assert f"| {expected} |" in content

    def test_max_diff_from_top_level_legacy_format(self, tmp_path):
        """旧格式：max_diff 在顶层，应保持兼容"""
        old_format_accuracy = {
            'passed': False,
            'max_diff': 50.0,
            'output_results': [],
        }

        eval_result = EvalResult(
            rel_path='level1/test',
            operator='Test',
            case_id='test_1',
            status='failed',
            accuracy=old_format_accuracy,
        )

        content = self._render_markdown(eval_result, tmp_path)
        assert '| 50.000000 |' in content

    def test_accuracy_none_case(self, tmp_path):
        """accuracy 为 None 时不应崩溃"""
        case_result = EvalCaseResult(
            case_id='test_1',
            rel_path='level1/test',
            operator='Test',
            case_num=1,
            success=False,
            accuracy_result=None,
            error_msg="AI算子执行失败",
        )

        eval_result = EvalResult.from_eval_case_result(case_result)
        assert eval_result.accuracy is None
        content = self._render_markdown(eval_result, tmp_path)
        assert 'AI算子执行失败' in content


class TestAccuracyResultSerialization:
    """测试 AccuracyResult 序列化后的结构"""

    def test_accuracy_result_to_dict_structure(self):
        """验证 AccuracyResult.to_dict() 的结构"""
        torch.manual_seed(42)
        x = torch.randn(100, dtype=torch.float64)
        golden = torch.exp(x)
        output = torch.exp(x + 0.1)

        checker = RelativeErrorChecker()
        accuracy = checker.check(
            ai_outputs=output.float(),
            golden_outputs=golden.float(),
            dtype='float32',
            threshold=0.0001,
        )

        acc_dict = accuracy.to_dict()

        # 验证必需字段
        assert 'passed' in acc_dict
        assert 'output_results' in acc_dict
        assert 'metadata' in acc_dict

        # 验证 metadata 包含聚合指标
        assert 'mare' in acc_dict['metadata']
        assert 'mere' in acc_dict['metadata']
        assert 'max_diff' in acc_dict['metadata']
        assert 'total_count' in acc_dict['metadata']

        # 验证 output_results 非空（正常场景）
        assert len(acc_dict['output_results']) > 0

        # 验证 output_results[0] 包含必要字段
        first_output = acc_dict['output_results'][0]
        assert 'passed' in first_output
        assert 'dtype' in first_output
        assert 'total_count' in first_output

    def test_accuracy_result_json_roundtrip(self):
        """验证 AccuracyResult 经过 JSON 序列化/反序列化后结构完整"""
        import json

        torch.manual_seed(42)
        x = torch.randn(100, dtype=torch.float64)
        golden = torch.exp(x)
        output = torch.exp(x + 0.1)

        checker = RelativeErrorChecker()
        accuracy = checker.check(
            ai_outputs=output.float(),
            golden_outputs=golden.float(),
            dtype='float32',
            threshold=0.0001,
        )

        # 序列化
        acc_dict = accuracy.to_dict()
        json_str = json.dumps(acc_dict, ensure_ascii=False)

        # 反序列化
        acc_from_json = json.loads(json_str)

        # 验证结构完整
        assert 'metadata' in acc_from_json
        assert 'mare' in acc_from_json['metadata']
        assert 'mere' in acc_from_json['metadata']
        assert acc_from_json['metadata']['mare'] > 0
        assert len(acc_from_json['output_results']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
