#!/usr/bin/python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See the LICENSE file in the root of the software repository for the full text.

import ast
from pathlib import Path

import pytest
import yaml


REPOSITORY = Path(__file__).resolve().parents[2]
OPERATORS = REPOSITORY / "bench_lab" / "cv_agent_fa_bench"
OPTIONAL_POSITION_BIAS_CASE_IDS = {8, 13, 15, 17, 19, 20}


def _function(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def _assert_parameter_defaults_to_none(function, name):
    parameters = function.args.posonlyargs + function.args.args
    parameter_names = [parameter.arg for parameter in parameters]
    parameter_index = parameter_names.index(name)
    first_default_index = len(parameters) - len(function.args.defaults)
    assert parameter_index >= first_default_index
    default = function.args.defaults[parameter_index - first_default_index]
    assert isinstance(default, ast.Constant) and default.value is None


@pytest.mark.parametrize(
    ("operator", "function_name"),
    (
        ("crossformer_attention", "crossformer_attention"),
        ("window_attention", "window_attention"),
    ),
)
def test_position_bias_optional_contract(operator, function_name):
    folder = OPERATORS / operator
    proto = yaml.safe_load((folder / "proto.yaml").read_text(encoding="utf-8"))["operator"]
    field = next(item for item in proto["inputs"] if item["name"] == "position_bias")
    schema = " ".join(proto["schema"].split())
    assert field["optional"] is True
    assert field["shape"] == "[H, N, N]"
    assert f"Tensor? position_bias=None" in schema

    cases = yaml.safe_load((folder / "cases.yaml").read_text(encoding="utf-8"))["cases"]
    no_bias_cases = [case for case in cases if case["attrs"].get("has_position_bias") is False]
    assert {case["case_id"] for case in no_bias_cases} == OPTIONAL_POSITION_BIAS_CASE_IDS
    for case in no_bias_cases:
        assert case["input_shape"][3] is None
        assert case["dtype"][3] is None
        assert case["value_range"][3] is None

    golden_source = (folder / "golden.py").read_text(encoding="utf-8")
    golden_tree = ast.parse(golden_source)
    _assert_parameter_defaults_to_none(_function(golden_tree, "forward"), "position_bias")
    _assert_parameter_defaults_to_none(_function(golden_tree, function_name), "position_bias")

    description = (folder / "desc.md").read_text(encoding="utf-8")
    assert f"{function_name}(Tensor q, Tensor k, Tensor v, Tensor? position_bias=None) -> Tensor output" in description
    assert "| `position_bias` | Tensor? | `float16` | `[H, N, N]` | 可选输入；默认 `None`" in description
    assert (
        f"def {function_name}(\n"
        "    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, position_bias: torch.Tensor | None = None\n"
        ") -> torch.Tensor:"
    ) in description
