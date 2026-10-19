#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details in the root directory of this software repository.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# ----------------------------------------------------------------------------------------------------------

from types import SimpleNamespace

import pytest

import kernel_eval.cli as cli


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), (0, {"case_id": 0}), (1, {"case_id": 1})],
)
def test_cpu_case_id_filter_preserves_zero(monkeypatch, tmp_path, value, expected):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "VERSION").write_text("2.0.0\n")
    command = ["eval", "--device", "cpu"]
    if value is not None:
        command += ["--case-id", str(value)]
    args = cli.create_parser().parse_args(command)
    captured = {}

    class DummyReport:
        operators = []
        failed_cases = 0

    class DummyReportGenerator:
        def __init__(self, **unused):
            pass

        def generate(self):
            return DummyReport()

        def save_all(self, report):
            pass

        def print_summary(self, report):
            pass

    monkeypatch.setattr(cli, "get_project_root", lambda: "/tmp/project")
    monkeypatch.setattr(cli, "resolve_task_dir", lambda task_dir, project_root: (str(tmp_path), None))
    monkeypatch.setattr(
        cli,
        "_create_config_from_args",
        lambda args, bench_root: SimpleNamespace(reports_dir="/tmp/reports"),
    )
    monkeypatch.setattr(cli, "ReportGenerator", DummyReportGenerator)

    def fake_simulate(config, bench_name, operator_filter, case_filter, report_generator):
        captured["case_filter"] = case_filter

    monkeypatch.setattr("kernel_eval.simulation.simulate", fake_simulate)

    assert cli.cmd_eval(args) == 0
    assert captured["case_filter"] == expected


@pytest.mark.parametrize(
    ("value", "expected_case_nums"),
    [(None, [1, 2]), (0, []), (1, [1])],
)
def test_npu_case_id_filter_preserves_zero(monkeypatch, value, expected_case_nums):
    cases = [
        SimpleNamespace(case_num=1, rel_path="level1/dummy", operator="Dummy"),
        SimpleNamespace(case_num=2, rel_path="level1/dummy", operator="Dummy"),
    ]
    captured = {}

    class DummyLoader:
        def scan_all(self):
            return list(cases)

    class DummyCoordinator:
        card_count = 1
        total_processes = 1

        def __init__(self, **unused):
            pass

        def evaluate_task_units(self, task_units):
            captured["case_nums"] = [case.case_num for unit in task_units for case in unit.cases]
            return []

        def shutdown(self):
            pass

    args = SimpleNamespace(
        bench_name="cann",
        device_id=0,
        processes_per_card=1,
        max_cases_per_task_unit=64,
        timeout_per_operator=300,
        source_dir=None,
        skip_install=False,
        level=None,
        operator=None,
        case_id=value,
        no_perf=True,
        reports_dir=None,
        warmup=1,
        repeat=1,
    )

    monkeypatch.setattr("kernel_eval.registry.get_case_loader", lambda bench_name, tasks_root: DummyLoader())
    monkeypatch.setattr("kernel_eval.eval.process_pool.ProcessPoolCoordinator", DummyCoordinator)
    monkeypatch.setattr("kernel_eval.eval.subprocess_utils.detect_pypto_pro_submission", lambda: False)

    config = SimpleNamespace()
    report_generator = SimpleNamespace(add_operator_result=lambda result: None)

    assert cli._cmd_eval_npu(args, "/tmp/tasks", None, config, report_generator) == 0
    assert captured.get("case_nums", []) == expected_case_nums
