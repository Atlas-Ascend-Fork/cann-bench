#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See the repository LICENSE for the full text.

"""Historical proposal contracts and their explicit Season2 release transition."""

import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "docs/design/next_suite_20260924.json").read_text(encoding="utf-8"))
OPERATORS = MANIFEST["operators"]
RELEASE = json.loads((ROOT / "tasks/suite_manifest.json").read_text(encoding="utf-8"))


def test_selection_counts_and_unique_identifiers():
    assert MANIFEST["rename_scope"] == "selected_bench_lab_only"
    assert len(OPERATORS) == 60
    assert len({r["id"] for r in OPERATORS}) == 60
    assert len({r["path"] for r in OPERATORS}) == 60
    assert Counter(r["selection"] for r in OPERATORS) == {"retained": 40, "added": 20}
    assert Counter(r["planned_level"] for r in OPERATORS) == {"L1": 2, "L2": 14, "L3": 27, "L4": 12, "L5": 5}
    micro = [r for r in OPERATORS if r["micro"]]
    assert len(micro) == 16
    assert Counter(r["planned_level"] for r in micro) == {"L1": 1, "L2": 6, "L3": 5, "L4": 3, "L5": 1}
    assert {r["id"] for r in micro if r["selection"] == "added"} == {
        "image_resize_normalize", "dynamic_mx_quant", "linear_cross_entropy", "kimi_delta_attention"
    }


@pytest.mark.parametrize("record", OPERATORS, ids=lambda r: r["id"])
def test_delivery_names_and_golden_entry(record):
    name = record["id"]
    retained = record["selection"] == "retained"
    operator_name = record["previous_name"] if retained else name
    assert record["path"].startswith("tasks/" if retained else "bench_lab/")
    if retained:
        assert record["path"] == record["previous_path"]
    # The dated proposal is historical, not the live Season2 task manifest.
    # Its retained Season1 deliveries remain testable in the unchanged archive.
    folder = ROOT / ("archives/season1" if retained else "") / record["path"]
    assert folder.resolve().is_relative_to(ROOT.resolve())
    assert re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name)
    assert folder.name == name
    for filename in ("proto.yaml", "cases.yaml", "cases.csv", "golden.py", "desc.md"):
        assert (folder / filename).is_file()
    proto = yaml.safe_load((folder / "proto.yaml").read_text(encoding="utf-8"))["operator"]
    assert proto["name"] == operator_name
    assert proto["difficulty"] == record["repository_level"]
    entry = re.match(r"(\w+)\s*\(", proto["schema"]).group(1)
    tree = ast.parse((folder / "golden.py").read_text(encoding="utf-8-sig"))
    functions = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert entry in functions
    cases = yaml.safe_load((folder / "cases.yaml").read_text(encoding="utf-8"))["cases"]
    with (folder / "cases.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_cases = list(csv.DictReader(stream))
    assert cases and len(cases) == len(csv_cases)
    assert {c["operator"] for c in cases} == {operator_name}
    assert {c["operator"] for c in csv_cases} == {operator_name}


def test_historical_proposal_transitions_to_final_season2_selection():
    proposed = {r["id"] for r in OPERATORS}
    released = {r["id"] for r in RELEASE["operators"]}
    assert proposed - released == {"resize_bilinear"}
    assert released - proposed == {"attention_residual"}
    assert len(released) == 60
    assert Counter(r["level"] for r in RELEASE["operators"]) == {
        "L1": 2, "L2": 13, "L3": 27, "L4": 13, "L5": 5,
    }
    retained = {p.parent.name for p in (ROOT / "archives/season1/tasks").glob("level*/*/proto.yaml")}
    assert len(released & retained) == 39
    assert len(released - retained) == 21


def test_duplicate_kept_in_lab_but_not_selected():
    excluded = MANIFEST["excluded_overlap"][0]
    assert excluded["path"] not in {r["path"] for r in OPERATORS}
    assert excluded["represented_by"] in {r["id"] for r in OPERATORS}
    assert (ROOT / excluded["path"] / "golden.py").is_file()


def test_kda_optional_hook_names_follow_schema():
    source = (ROOT / "bench_lab/model_attention_bench/kimi_delta_attention/golden.py").read_text(encoding="utf-8")
    names = {n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    assert {"kimi_delta_attention", "kimi_delta_attention_oracle", "kimi_delta_attention_bench"} <= names


@pytest.mark.parametrize("record", [
    r for r in OPERATORS if r["id"] in {
        "aggregate_hidden_grad", "scatter_block_update",
        "fused_causal_conv1d", "q4_0_dequant_matmul",
    }
], ids=lambda r: r["id"])
def test_renamed_operator_descriptions_use_current_name(record):
    for filename in ("desc.md", "proto.yaml", "golden.py"):
        source = (ROOT / record["path"] / filename).read_text(encoding="utf-8-sig")
        assert record["previous_name"] not in source
