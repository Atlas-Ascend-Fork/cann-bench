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

"""级别选择的一致性: 空选择的产物, 以及 ST 侧的级别清单

两组契约:

1. `eval` 在 "筛不到东西" 时, CPU 与 NPU 两条路径的观感必须一致 -- 都是退出码 0
   且照常产出一份报告. 否则下游 (run_evaluation.sh 结束时 `ls -t reports/*.md`)
   在两条路径上看到的东西不一样.
2. ST harness 的级别清单只能有一份. 历史上有三份 (eval_run.LEVELS /
   test_golden_npu_mock._LEVELS / select_from_changes._TASK_RE), 且已经漂开.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from kernel_eval.config import get_project_root


_ST_DIR = get_project_root() / "tests" / "st"


def _load_st_collector():
    """按文件路径加载 ST 收集器, 用独立模块名避免与 pytest 自己的收集冲突."""
    if str(_ST_DIR) not in sys.path:
        sys.path.insert(0, str(_ST_DIR))
    spec = importlib.util.spec_from_file_location(
        "_st_golden_npu_mock_probe", _ST_DIR / "test_golden_npu_mock.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_harness(name):
    if str(_ST_DIR) not in sys.path:
        sys.path.insert(0, str(_ST_DIR))
    return importlib.import_module(f"harness.{name}")


def _write_op(op_dir: Path) -> None:
    op_dir.mkdir(parents=True)
    (op_dir / "proto.yaml").write_text(
        f"operator:\n  name: DummyOp\n  difficulty: L1\n"
        f"  schema: {op_dir.name}(Tensor x) -> Tensor y\n",
        encoding="utf-8",
    )
    (op_dir / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
    (op_dir / "golden.py").write_text(
        f"def {op_dir.name}(x):\n    return x\n", encoding="utf-8"
    )


class TestEmptySelectionStillReports:
    """筛不到算子时 CPU 路径的产物, 要与 NPU 路径的 "无匹配用例" 对齐."""

    @staticmethod
    def _run_cpu_eval_with_empty_level(monkeypatch, tmp_path):
        import kernel_eval.cli as cli

        (tmp_path / "metadata").mkdir()
        (tmp_path / "metadata" / "VERSION").write_text("2.0.0\n")

        args = cli.create_parser().parse_args([
            "eval", "--device", "cpu", "--task-dir", str(tmp_path), "--level", "5",
        ])
        monkeypatch.setattr(cli, "get_project_root", lambda: tmp_path)
        monkeypatch.setattr(cli, "resolve_task_dir", lambda task_dir, project_root: (str(tmp_path), ""))
        monkeypatch.setattr(cli, "_operator_names_for_level", lambda *unused: [])

        class DummyConfig:
            reports_dir = tmp_path

        monkeypatch.setattr(cli, "_create_config_from_args", lambda *unused: DummyConfig())

        trace = []

        class DummyReport:
            operators = []
            failed_cases = 0

        class DummyReportGenerator:
            def __init__(self, **unused):
                pass

            def generate(self):
                trace.append("generate")
                return DummyReport()

            def save_all(self, report):
                trace.append("save_all")

            def print_summary(self, report):
                trace.append("print_summary")

        monkeypatch.setattr(cli, "ReportGenerator", DummyReportGenerator)
        monkeypatch.setattr(
            "kernel_eval.simulation.simulate",
            lambda *unused, **unused_kwargs: trace.append("simulate"),
        )
        return cli.cmd_eval(args), trace

    def test_no_evaluation_runs(self, monkeypatch, tmp_path):
        pytest.importorskip("torch")
        rc, trace = self._run_cpu_eval_with_empty_level(monkeypatch, tmp_path)
        assert rc == 0
        assert "simulate" not in trace

    def test_a_report_is_still_produced(self, monkeypatch, tmp_path):
        pytest.importorskip("torch")
        _, trace = self._run_cpu_eval_with_empty_level(monkeypatch, tmp_path)
        assert trace == ["generate", "save_all", "print_summary"]


@pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning")
class TestStLevelListIsSingleSourced:
    """加载 ST 收集器时会带出 level* 标记, 这些标记注册在 tests/st/conftest.py,
    UT 会话里看不到 -- 与本文件要验的东西无关, 故就地忽略."""

    def test_collector_and_harness_agree(self):
        harness_levels = tuple(_import_harness("eval_run").LEVELS)
        collector_levels = tuple(_load_st_collector()._LEVELS)
        assert collector_levels == harness_levels

    def test_selector_regex_covers_every_known_level(self):
        select = _import_harness("select_from_changes")
        levels = _import_harness("eval_run").LEVELS
        for level in levels:
            assert select._TASK_RE.search(f"tasks/{level}/some_op/"), \
                f"选择器认不出 {level}, PR 只改了该级别算子时会退回默认冒烟组"

    def test_selector_regex_rejects_an_unknown_level(self):
        select = _import_harness("select_from_changes")
        assert select._TASK_RE.search("tasks/level_future/some_op/") is None

    def test_levels_track_the_tasks_tree(self):
        levels = _import_harness("eval_run").LEVELS
        on_disk = {p.name for p in (get_project_root() / "tasks").glob("level*") if p.is_dir()}
        assert set(levels) == on_disk


class TestGoldenCandidateToleratesATrimmedTree:
    """`build_golden_candidate` 的 tasks_dir 可以与 LEVELS 的来源不是同一棵树."""

    def test_level_missing_from_the_given_tree_is_skipped(self, tmp_path):
        tasks = tmp_path / "tasks"
        _write_op(tasks / "level1" / "dummy_op")

        build = _import_harness("golden_mock").build_golden_candidate
        dest = build(tmp_path / "candidate", tasks_dir=tasks)

        init_py = (dest / "cann_bench" / "__init__.py").read_text(encoding="utf-8")
        assert "dummy_op" in init_py
