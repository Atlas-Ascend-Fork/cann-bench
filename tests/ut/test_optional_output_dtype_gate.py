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

"""proto ``optional: true`` 输出的 dtype 闸行为（issue #310）。

Unique 在 ``return_inverse=false`` 时按契约只返回 y（golden/desc 同此约定，
proto 将 inverse 声明为 optional）。修复前 ``_get_output_dtype_error`` 对
缺位输出一律按 "实际 missing" 判负——所有正确实现的 return_inverse=false
用例被系统性误判（公开 20 例中 11 例、隐藏 80 例中 56 例）。

本文件钉死：
1. optional 输出缺位 → 不触发 dtype 闸（端到端精度通过）；
2. 必选输出缺位 → 照旧判负（missing）；
3. optional 输出存在但 dtype 错 → 照旧判负；
4. golden 产出而候选缺位（return_inverse=true 少返 inverse）→ 照旧判负
   （由输出数量比对裁决，不因 optional 而放行）。
"""

import pytest
import torch

from kernel_eval.eval.accuracy_eval import AccuracyEvaluator

EXPECTED = [["float16"], ["int64"]]
NAMES = ["y", "inverse"]
OPTIONAL = [False, True]


@pytest.fixture(autouse=True)
def _ensure_default_checker():
    """幂等补注册默认精度判断器。

    个别测试文件会清空全局 checker 注册表且不恢复（注册表隔离是另一个
    已知的测试债），端到端用例依赖 ``relative_error``——与
    ``benches/cann.py`` 同款的幂等注册兜底，保证本文件不受运行顺序影响。
    """
    from kernel_eval.registry import CheckerRegistry
    from kernel_eval.checkers.relative_error_checker import RelativeErrorChecker

    if "relative_error" not in CheckerRegistry.get_all():
        CheckerRegistry.register("relative_error", RelativeErrorChecker())


@pytest.fixture()
def ev():
    return AccuracyEvaluator()


class TestDtypeGateUnit:
    def test_optional_missing_is_allowed(self, ev):
        y = torch.arange(4, dtype=torch.float16)
        err = ev._get_output_dtype_error([y], EXPECTED, NAMES, OPTIONAL)
        assert err is None

    def test_required_missing_still_fails(self, ev):
        err = ev._get_output_dtype_error([], EXPECTED, NAMES, OPTIONAL)
        assert err is not None and "y" in err and "missing" in err

    def test_optional_present_wrong_dtype_still_fails(self, ev):
        y = torch.arange(4, dtype=torch.float16)
        inverse = torch.zeros(4, dtype=torch.int32)  # 契约要求 int64
        err = ev._get_output_dtype_error([y, inverse], EXPECTED, NAMES, OPTIONAL)
        assert err is not None and "inverse" in err and "int32" in err

    def test_optional_present_correct_dtype_passes(self, ev):
        y = torch.arange(4, dtype=torch.float16)
        inverse = torch.zeros(4, dtype=torch.int64)
        assert ev._get_output_dtype_error([y, inverse], EXPECTED, NAMES, OPTIONAL) is None

    def test_no_flags_keeps_legacy_strictness(self, ev):
        """未传 optional 标志（旧调用方）时行为与修复前一致：缺位即判负。"""
        y = torch.arange(4, dtype=torch.float16)
        err = ev._get_output_dtype_error([y], EXPECTED, NAMES)
        assert err is not None and "inverse" in err and "missing" in err


class TestEndToEnd:
    def test_return_inverse_false_contract_passes(self, ev):
        """unique 场景：golden 与候选都只返回 y → 应通过。"""
        y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        result = ev.evaluate(
            ai_output=[y.clone()],
            golden_output=[y.to(torch.float64)],
            dtype="float16",
            expected_output_dtypes=EXPECTED,
            output_names=NAMES,
            output_optional_flags=OPTIONAL,
        )
        assert result.passed, result.error_msg

    def test_golden_has_inverse_but_candidate_missing_fails(self, ev):
        """return_inverse=true 时候选少返 inverse → 仍须判负（数量比对裁决）。"""
        y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        inverse = torch.tensor([0, 1, 2], dtype=torch.int64)
        result = ev.evaluate(
            ai_output=[y.clone()],
            golden_output=[y.to(torch.float64), inverse],
            dtype="float16",
            expected_output_dtypes=EXPECTED,
            output_names=NAMES,
            output_optional_flags=OPTIONAL,
        )
        assert not result.passed


class TestOptionalFlagPlumbing:
    """optional 标志的全链路（检视意见：此前仅闸本体识别标志，上游从未填充——
    OutputSpec 无字段、loader 不读 proto、序列化不携带，实际运行恒 [False, ...]）。"""

    def test_loader_reads_proto_optional(self):
        """真实加载 tasks/level3/unique：inverse 的 optional 必须为 True。"""
        from pathlib import Path
        from kernel_eval.benches.cann_loader import CannTaskLoader

        tasks = Path(__file__).resolve().parents[2] / "tasks"
        op = CannTaskLoader(str(tasks)).get_operator("level3/unique")
        flags = [bool(getattr(o, "optional", False)) for o in op.outputs]
        names = [o.name for o in op.outputs]
        assert names == ["y", "inverse"]
        assert flags == [False, True], flags

    def test_output_spec_serialization_roundtrip(self):
        from kernel_eval.base.models import OutputSpec

        spec = OutputSpec(name="inverse", dtype="int64", optional=True)
        data = spec.to_dict()
        assert data["optional"] is True
        back = OutputSpec.from_dict(data)
        assert back.optional is True
        # 默认值与旧数据兼容：无 optional 键的旧字典反序列化为 False
        legacy = {k: v for k, v in data.items() if k != "optional"}
        assert OutputSpec.from_dict(legacy).optional is False

    def test_end_to_end_unique_contract_via_loader(self, ev):
        """用真实 proto 构造标志走完 dtype 闸：return_inverse=false 只返 y 应通过。"""
        from pathlib import Path
        from kernel_eval.benches.cann_loader import CannTaskLoader

        tasks = Path(__file__).resolve().parents[2] / "tasks"
        op = CannTaskLoader(str(tasks)).get_operator("level3/unique")
        flags = [bool(getattr(o, "optional", False)) for o in op.outputs]
        dtypes = [getattr(o, "dtypes", None) or [o.dtype] for o in op.outputs]
        names = [o.name for o in op.outputs]
        y = torch.arange(4, dtype=torch.float16)
        assert ev._get_output_dtype_error([y], dtypes, names, flags) is None
        err = ev._get_output_dtype_error([], dtypes, names, flags)
        assert err is not None and "y" in err  # 必选输出缺位仍判负
