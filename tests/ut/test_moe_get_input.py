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

"""MoE 类算子结构化输入契约的 get_input 测试

测试对象：
1. archives/season1/tasks/level3/moe_finalize_routing/golden.py::get_input
   —— expanded_src_to_dst_row 是"源行 -> 展开缓冲区行"的单射映射（drop_less 下是
      完整置换），单区间 value_range 表达不了"互异"，通用生成器有放回采样出大量重复。
2. tasks/level3/moe_re_routing/golden.py::get_input
   —— proto 要求 expert_token_num_per_rank 元素必须大于 0，A < N*E 时原实现静默产出 0。
"""

import importlib.util

import pytest
import torch

from kernel_eval.config import get_project_root


def _load(op_rel, *, archived=False):
    suite = "archives/season1/tasks" if archived else "tasks"
    path = get_project_root() / suite / op_rel / "golden.py"
    spec = importlib.util.spec_from_file_location(f"_golden_{op_rel.replace('/', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def finalize_routing():
    # Keep historical Golden regression coverage without reintroducing the task.
    return _load("level3/moe_finalize_routing", archived=True)


@pytest.fixture(scope="module")
def re_routing():
    return _load("level3/moe_re_routing")


# ---------------------------------------------------------------------------
# MoeFinalizeRouting：expanded_src_to_dst_row 单射契约
# ---------------------------------------------------------------------------


def _drop_less_case(num_rows=16, k=4, hidden=8, seed=0):
    """drop_less（mode 0/2）：num_dst == NK，映射应为 [0, NK) 的完整置换"""
    g = torch.Generator().manual_seed(seed)
    nk = num_rows * k
    epr = torch.rand(nk, hidden, generator=g)
    esdr = torch.randint(0, nk, (nk,), generator=g, dtype=torch.int32)
    scales = torch.rand(num_rows, k, generator=g)
    skip1 = torch.rand(num_rows, hidden, generator=g)
    return epr, esdr, scales, skip1


def _drop_pad_case(num_rows=32, k=1, experts=8, capacity=10, hidden=8, seed=0):
    """drop_pad（mode 1/3）：epr 为 (E, C, H)，num_dst = E*C > NK，-1 表示丢弃"""
    g = torch.Generator().manual_seed(seed)
    nk = num_rows * k
    epr = torch.rand(experts, capacity, hidden, generator=g)
    esdr = torch.randint(-1, experts * capacity, (nk,), generator=g, dtype=torch.int32)
    scales = torch.rand(num_rows, k, generator=g)
    skip1 = torch.rand(num_rows, hidden, generator=g)
    return epr, esdr, scales, skip1


class TestFinalizeRoutingInjectivity:
    def test_drop_less_becomes_full_permutation(self, finalize_routing):
        epr, esdr, scales, skip1 = _drop_less_case()
        nk = esdr.numel()
        assert esdr.unique().numel() < nk, "构造失败：原始映射本应有重复"
        out = finalize_routing.get_input(epr, esdr, skip1=skip1, scales=scales, drop_pad_mode=0)
        new = out[1]
        assert sorted(new.tolist()) == list(range(nk))

    def test_drop_pad_non_sentinel_entries_distinct(self, finalize_routing):
        epr, esdr, scales, skip1 = _drop_pad_case()
        out = finalize_routing.get_input(epr, esdr, skip1=skip1, scales=scales, drop_pad_mode=1)
        new = out[1]
        keep = new != -1
        assert int(keep.sum()) > 0
        assert new[keep].unique().numel() == int(keep.sum())

    def test_drop_pad_sentinel_positions_preserved(self, finalize_routing):
        """-1 的位置原样保留，维持既有的丢弃覆盖"""
        epr, esdr, scales, skip1 = _drop_pad_case()
        out = finalize_routing.get_input(epr, esdr, skip1=skip1, scales=scales, drop_pad_mode=1)
        assert torch.equal(esdr == -1, out[1] == -1)

    def test_indices_within_destination_space(self, finalize_routing):
        epr, esdr, scales, skip1 = _drop_pad_case()
        out = finalize_routing.get_input(epr, esdr, skip1=skip1, scales=scales, drop_pad_mode=1)
        new = out[1]
        keep = new != -1
        num_dst = epr.numel() // epr.shape[-1]
        assert int(new[keep].min()) >= 0
        assert int(new[keep].max()) < num_dst

    def test_shape_and_dtype_unchanged(self, finalize_routing):
        epr, esdr, scales, skip1 = _drop_less_case()
        out = finalize_routing.get_input(epr, esdr, skip1=skip1, scales=scales, drop_pad_mode=0)
        assert out[1].shape == esdr.shape
        assert out[1].dtype == esdr.dtype

    def test_reproducible_across_calls(self, finalize_routing):
        epr, esdr, scales, skip1 = _drop_less_case()
        a = finalize_routing.get_input(epr, esdr, skip1=skip1, scales=scales, drop_pad_mode=0)[1]
        b = finalize_routing.get_input(epr, esdr, skip1=skip1, scales=scales, drop_pad_mode=0)[1]
        assert torch.equal(a, b)

    def test_other_inputs_passed_through(self, finalize_routing):
        epr, esdr, scales, skip1 = _drop_less_case()
        bias = torch.rand(4, epr.shape[-1])
        efsr = torch.randint(0, 4, (16, 4), dtype=torch.int32)
        out = finalize_routing.get_input(epr, esdr, skip1=skip1, skip2=None, bias=bias,
                                         scales=scales, expert_for_source_row=efsr,
                                         drop_pad_mode=0)
        assert len(out) == 7
        assert out[0] is epr and out[2] is skip1 and out[3] is None
        assert out[4] is bias and out[5] is scales and out[6] is efsr

    def test_extra_kwargs_tolerated(self, finalize_routing):
        epr, esdr, scales, skip1 = _drop_less_case()
        finalize_routing.get_input(epr, esdr, skip1=skip1, scales=scales,
                                   drop_pad_mode=0, skip2_exist=True)

    def test_impossible_injection_passthrough(self, finalize_routing):
        """抽屉原理：源行多于目的行时保持原样而非报错"""
        g = torch.Generator().manual_seed(0)
        epr = torch.rand(4, 8, generator=g)          # num_dst = 4
        esdr = torch.randint(0, 4, (16,), generator=g, dtype=torch.int32)
        out = finalize_routing.get_input(epr, esdr, drop_pad_mode=0)
        assert out[1] is esdr

    def test_gather_semantics_unchanged(self, finalize_routing):
        """重建后 golden 仍可正常执行，输出规格不变"""
        epr, esdr, scales, skip1 = _drop_less_case()
        out = finalize_routing.get_input(epr, esdr, skip1=skip1, scales=scales, drop_pad_mode=0)
        y = finalize_routing.moe_finalize_routing(out[0], out[1], skip1=out[2], scales=out[5],
                                                  drop_pad_mode=0)
        assert y.shape == skip1.shape


# ---------------------------------------------------------------------------
# MoeReRouting：expert_token_num_per_rank 元素必须 > 0
# ---------------------------------------------------------------------------


class TestReRoutingGuard:
    def test_feasible_case_sums_to_token_count(self, re_routing):
        tokens = torch.rand(1024, 16)
        etn = torch.zeros(8, 8, dtype=torch.int32)
        out = re_routing.get_input(tokens, etn)
        assert int(out[1].sum()) == tokens.shape[0]
        assert int(out[1].min()) > 0

    def test_remainder_still_absorbed(self, re_routing):
        """A 不整除 N*E 时总和仍须等于 A（沿用原有的"余数并入最后一格"）"""
        tokens = torch.rand(1009, 16)
        etn = torch.zeros(8, 8, dtype=torch.int32)
        out = re_routing.get_input(tokens, etn)
        assert int(out[1].sum()) == 1009
        assert int(out[1].min()) > 0

    def test_infeasible_case_raises_instead_of_emitting_zeros(self, re_routing):
        """A < N*E：无法让每格都 >0，应显式报错而非静默产出违反契约的 0"""
        tokens = torch.rand(32, 16)          # A=32 < N*E=64
        etn = torch.zeros(8, 8, dtype=torch.int32)
        with pytest.raises(ValueError, match="N\\*E"):
            re_routing.get_input(tokens, etn)

    def test_boundary_a_equals_cells(self, re_routing):
        """A == N*E 恰好每格 1 个，属可行边界"""
        tokens = torch.rand(64, 16)
        etn = torch.zeros(8, 8, dtype=torch.int32)
        out = re_routing.get_input(tokens, etn)
        assert int(out[1].min()) == 1 and int(out[1].sum()) == 64

    def test_per_token_scales_passed_through(self, re_routing):
        tokens = torch.rand(1024, 16)
        etn = torch.zeros(8, 8, dtype=torch.int32)
        scales = torch.rand(1024)
        out = re_routing.get_input(tokens, etn, per_token_scales=scales)
        assert out[0] is tokens and out[2] is scales
