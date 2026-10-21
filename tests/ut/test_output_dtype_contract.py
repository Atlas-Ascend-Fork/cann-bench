#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import csv
import json
import runpy
from pathlib import Path

import pytest
import torch

from kernel_eval.benches.cann_loader import CannTaskLoader
from kernel_eval.base.result import FAILURE_TYPE_COMPILE_RUNTIME_ERROR, get_accuracy_failure_type
from kernel_eval.eval.accuracy_eval import AccuracyEvaluator
from kernel_eval.eval.evaluator import Evaluator


def test_candidate_output_must_match_proto_dtype_contract():
    tasks_root = Path(__file__).resolve().parents[2] / "tasks"
    nms = CannTaskLoader(tasks_root=str(tasks_root)).get_task("level3/nms")
    expected_dtypes = Evaluator._get_output_dtype_contracts(nms)
    output_names = [output.name for output in nms.outputs]

    golden = torch.tensor([0, 3], dtype=torch.int64)
    wrong_dtype = torch.tensor([0, 3], dtype=torch.int32)
    evaluator = AccuracyEvaluator()

    # 数值完全相同的跨整数 dtype 输出在未应用 proto 契约的旧路径上会通过。
    legacy_result = evaluator.evaluate(wrong_dtype, golden, dtype="int32")
    assert legacy_result.passed

    mismatch = evaluator.evaluate(
        wrong_dtype,
        golden,
        dtype="int32",
        expected_output_dtypes=expected_dtypes,
        output_names=output_names,
    )
    assert not mismatch.passed
    assert "keep_indices" in mismatch.error_msg
    assert "int64" in mismatch.error_msg
    assert "int32" in mismatch.error_msg
    assert get_accuracy_failure_type(mismatch) == FAILURE_TYPE_COMPILE_RUNTIME_ERROR

    transformed_output_mismatch = evaluator.evaluate(
        golden.clone(),
        golden,
        dtype="int64",
        expected_output_dtypes=expected_dtypes,
        output_names=output_names,
        output_dtype_source=wrong_dtype,
    )
    assert not transformed_output_mismatch.passed

    matching = evaluator.evaluate(
        golden.clone(),
        golden,
        dtype="int64",
        expected_output_dtypes=expected_dtypes,
        output_names=output_names,
    )
    assert matching.passed

    multi_dtype = evaluator.evaluate(
        wrong_dtype,
        golden,
        dtype="int32",
        expected_output_dtypes=[["int32", "int64"]],
        output_names=output_names,
    )
    assert multi_dtype.passed


def test_multi_output_contracts_check_each_declared_dtype():
    evaluator = AccuracyEvaluator()
    wrong_dtype_outputs = (
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([0, 3], dtype=torch.int32),
    )
    golden_outputs = (
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([0, 3], dtype=torch.int64),
    )

    mismatch = evaluator.evaluate(
        wrong_dtype_outputs,
        golden_outputs,
        dtype="int32",
        expected_output_dtypes=[[], ["int64"]],
        output_names=["score", "keep_indices"],
    )
    assert not mismatch.passed
    assert "keep_indices" in mismatch.error_msg

    unconstrained = evaluator.evaluate(
        wrong_dtype_outputs[1],
        golden_outputs[1],
        dtype="int32",
        expected_output_dtypes=[[]],
    )
    assert unconstrained.passed


def _load_esa_select_topk_spec():
    bench_lab_root = Path(__file__).resolve().parents[2] / "bench_lab"
    op_dir = bench_lab_root / "omni-ops_bench" / "esa_select_topk"
    spec = CannTaskLoader(bench_root=str(bench_lab_root)).get_task(
        "omni-ops_bench/esa_select_topk"
    )
    assert spec is not None
    return op_dir, spec


def test_esa_select_topk_layout_support_matches_cases_and_golden():
    op_dir, spec = _load_esa_select_topk_spec()
    with (op_dir / "cases.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    layouts = {json.loads(row["attrs"])["input_layout"] for row in rows}
    assert layouts == {"TND"}
    assert "TND" in spec.shape_support
    assert "BSND" not in spec.shape_support
    assert "PA_BSND" not in spec.shape_support

    golden_module = runpy.run_path(str(op_dir / "golden.py"))
    query = torch.ones((1, 1, 1), dtype=torch.float16)
    key = torch.ones((1, 4, 1, 1), dtype=torch.float16)
    for layout in ("BSND", "PA_BSND"):
        with pytest.raises(
            ValueError, match="CPU golden currently supports TND layout"
        ):
            golden_module["esa_select_topk"](
                query,
                key,
                blk_size=2,
                init_blk_num=0,
                local_blk_num=0,
                topk=0,
                input_layout=layout,
                actual_seq_q_len_optional=[1],
                actual_seq_k_len_optional=[4],
                actual_cmp_seq_k_len_optional=[4],
                compress_blk_size=1,
            )


def test_esa_select_topk_proto_contract_rejects_float_index_output():
    _, spec = _load_esa_select_topk_spec()
    expected_dtypes = Evaluator._get_output_dtype_contracts(spec)
    output_names = [output.name for output in spec.outputs]
    assert expected_dtypes == [["int32"]]

    evaluator = AccuracyEvaluator()
    golden = torch.tensor([[[1]]], dtype=torch.int32)
    for dtype_name in ("float16", "bfloat16", "float32"):
        candidate = golden.to(getattr(torch, dtype_name))
        mismatch = evaluator.evaluate(
            candidate,
            golden,
            dtype=dtype_name,
            expected_output_dtypes=expected_dtypes,
            output_names=output_names,
        )
        assert not mismatch.passed
        assert "int32" in mismatch.error_msg
        assert dtype_name in mismatch.error_msg

    matching = evaluator.evaluate(
        golden.clone(),
        golden,
        dtype="int32",
        expected_output_dtypes=expected_dtypes,
        output_names=output_names,
    )
    assert matching.passed
