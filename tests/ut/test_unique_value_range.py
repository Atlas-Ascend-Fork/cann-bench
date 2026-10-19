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

"""唯一整数值域 ["unique", min, max] 的生成行为（D9.2）。

Season2 公开集 level2/scatter 的 reduce=null 用例以该值域声明"索引无重复"
（重复索引下哪次写入生效取决于执行顺序，精度比对不可判定）。本文件钉死：

1. 生成结果无重复、落在声明区间内、同种子可复现；
2. 多输入 case 路径（_generate_tensors 的 vr_item 过滤）不会把三元值域
   静默替换成默认区间；
3. 容量不足 / 非整数 dtype 大声失败；
4. 公开 scatter 用例的全部 unique 声明均满足容量约束。
"""

from pathlib import Path

import pytest
import torch
import yaml

from kernel_eval.data.data_generator import DataGenerator

TASKS = Path(__file__).resolve().parents[2] / "tasks"


def test_unique_range_basic_properties():
    gen = DataGenerator(seed=0)
    t = gen.generate_input_tensor([4096], "int64", ["unique", 0, 1048575])
    assert t.dtype == torch.int64
    assert t.numel() == 4096
    assert t.unique().numel() == t.numel()
    assert int(t.min()) >= 0 and int(t.max()) <= 1048575


def test_unique_range_deterministic_under_case_seed():
    a = DataGenerator().generate_input_tensors_from_case(
        [[64]], ["int64"], [["unique", 0, 4095]], seed=7)[0]
    b = DataGenerator().generate_input_tensors_from_case(
        [[64]], ["int64"], [["unique", 0, 4095]], seed=7)[0]
    assert torch.equal(a, b)


def test_unique_range_survives_multi_input_case_path():
    """三元值域不得被 _generate_tensors 的 vr_item 过滤静默置 None。

    置 None 会落到 _parse_range 的默认区间 (0, 100)——对 scatter 的百万级
    索引输入意味着海量重复 + 工作集坍缩，正是本测试要钉死的回归。
    """
    gen = DataGenerator(seed=0)
    tensors = gen.generate_input_tensors_from_case(
        [[128], [1024], [1024]],
        ["float16", "int64", "float16"],
        [[-1, 1], ["unique", 0, 1048575], [-1, 1]],
        seed=123,
    )
    idx = tensors[1]
    assert idx.unique().numel() == idx.numel()
    # 默认区间 (0, 100) 下 1024 个值必然重复且 max <= 100；全域生成则以
    # 压倒性概率越过 100。两条断言合起来排除静默回退。
    assert int(idx.max()) > 100


def test_unique_range_capacity_shortfall_fails_loudly():
    gen = DataGenerator(seed=0)
    with pytest.raises(ValueError):
        gen.generate_input_tensor([16], "int64", ["unique", 0, 7])


def test_unique_range_rejects_non_int_dtype():
    gen = DataGenerator(seed=0)
    with pytest.raises(ValueError):
        gen.generate_input_tensor([8], "float32", ["unique", 0, 100])


def test_public_scatter_unique_cases_generate_duplicate_free():
    """公开 scatter 用例：unique 声明容量全部足额，最小用例实际生成验证。"""
    doc = yaml.safe_load(
        (TASKS / "level2" / "scatter" / "cases.yaml").read_text(encoding="utf-8"))
    unique_cases = []
    for case in doc["cases"]:
        vrs = case.get("value_range") or []
        for i, vr in enumerate(vrs):
            if isinstance(vr, list) and len(vr) == 3 and vr[0] == "unique":
                numel = 1
                for s in case["input_shape"][i]:
                    numel *= int(s)
                span = int(vr[2]) - int(vr[1]) + 1
                assert span >= numel, (
                    f"case {case['case_id']} input{i}: unique 区间容量 {span} "
                    f"小于元素数 {numel}")
                unique_cases.append((case, i, numel))
    assert unique_cases, "scatter 公开用例应包含 unique 值域声明"

    case, idx_pos, _ = min(unique_cases, key=lambda t: t[2])
    tensors = DataGenerator().generate_input_tensors_from_case(
        case["input_shape"], case["dtype"], case["value_range"], seed=1)
    idx = tensors[idx_pos]
    vr = case["value_range"][idx_pos]
    assert idx.unique().numel() == idx.numel()
    assert int(idx.min()) >= int(vr[1]) and int(idx.max()) <= int(vr[2])


class TestNumericStringRanges:
    """YAML 1.1 无符号指数陷阱：``-1.0e38`` 被解析为字符串。

    归一前这类端点落入特殊值路径，实际按 [-1, 1] 生成——声明的近溢出
    角点（conv/matmul 家族 8 个公开用例）被静默替换。归一后走数值路径：
    全跨度均匀采样 + torch.finfo 夹取。
    """

    def test_numeric_string_range_generates_full_span(self):
        gen = DataGenerator(seed=0)
        t = gen.generate_input_tensor([4096], "float32", ["-1.0e38", "1.0e38"])
        assert float(t.abs().max()) > 1.0e37      # 归一前上界是 1.0
        assert float(t.min()) < 0 < float(t.max())

    def test_numeric_string_range_respects_dtype_finfo(self):
        gen = DataGenerator(seed=0)
        t = gen.generate_input_tensor([4096], "float16", ["-1.0e34", "1.0e34"])
        assert torch.isfinite(t).all()            # 夹到 fp16 finfo，而非溢出成 inf
        assert float(t.abs().max()) > 6.0e4       # 贴近 fp16 上界 65504

    def test_special_markers_keep_special_path(self):
        gen = DataGenerator(seed=0)
        t = gen.generate_input_tensor([64], "float32", ["nan", "nan"])
        assert torch.isnan(t).any() and (~torch.isnan(t)).any()
        t = gen.generate_input_tensor([64], "float32", [0.1, "inf"])
        assert torch.isinf(t).any()
        assert float(t[torch.isfinite(t)].min()) >= 0.1

    def test_public_near_overflow_cases_load_and_generate_full_span(self):
        """公开集字符串科学计数法值域的用例，经加载器端到端生成后达到声明量级。"""
        from kernel_eval.benches.cann_loader import CannCaseLoader

        loader = CannCaseLoader(str(TASKS))
        checked = 0
        for rel_path, case_num, scale in [
            ("level3/unique", 14, 1.0e38),
            ("level3/top_k", 14, 1.0e38),
            ("level3/grouped_matmul", 18, 1.5e18),
        ]:
            specs = [s for s in loader.scan_by_rel_path(rel_path)
                     if getattr(s, "case_num", 0) == case_num]
            if not specs:
                continue
            s = specs[0]
            tensors = DataGenerator().generate_input_tensors_from_case(
                s.input_shapes, s.dtypes, s.value_ranges, seed=11)
            first = tensors[0] if not isinstance(tensors[0], list) else tensors[0][0]
            finfo_max = torch.finfo(first.dtype).max
            expect = min(scale, float(finfo_max))
            assert float(first.abs().max()) > expect / 10, (
                f"{rel_path} c{case_num}: max abs {float(first.abs().max()):.3e} "
                f"远低于声明量级 {expect:.3e}")
            checked += 1
        assert checked >= 2
