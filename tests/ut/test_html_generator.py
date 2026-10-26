from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import tempfile

from kernel_eval.report.html_generator import (
    _render_bars,
    _render_level_table,
    _render_operator_tables,
    _render_top_tables,
)
from kernel_eval.report.report_generator import EvalReport, EvalResult, OperatorReport


class _TableShapeParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.headers = []
        self.rows = []
        self._section = None
        self._current_row = None

    def handle_starttag(self, tag, attrs):
        if tag in {"thead", "tbody"}:
            self._section = tag
        elif tag == "tr":
            self._current_row = []
        elif tag in {"th", "td"} and self._current_row is not None:
            self._current_row.append(tag)

    def handle_endtag(self, tag):
        if tag == "tr" and self._current_row is not None:
            if self._section == "thead":
                self.headers.append(self._current_row)
            elif self._section == "tbody":
                self.rows.append(self._current_row)
            self._current_row = None
        elif tag in {"thead", "tbody"}:
            self._section = None


def _report():
    op = OperatorReport(
        rel_path="level1/foo.py",
        operator="foo",
        total_cases=10,
        passed_cases=5,
        failed_cases=5,
        pass_rate=0.5,
        avg_speedup=1.25,
        score=60.0,
    )
    return EvalReport(
        framework_version="1",
        tasks_version="1",
        eval_code="test",
        timestamp="now",
        device="cpu",
        total_operators=1,
        total_cases=10,
        passed_cases=5,
        failed_cases=5,
        overall_score=60.0,
        operators=[op],
        summary={"pass_rate": 0.5},
    )


def test_level_table_does_not_advertise_unavailable_precision_metric():
    html = _render_level_table(_report())
    parser = _TableShapeParser()
    parser.feed(html)

    assert "Avg Precision" not in html
    assert len(parser.headers) == 1
    assert len(parser.rows) == 1
    assert len(parser.headers[0]) == len(parser.rows[0]) == 7


def test_operator_table_does_not_duplicate_pass_rate_as_precision():
    html = _render_operator_tables(_report().operators)
    parser = _TableShapeParser()
    parser.feed(html)

    assert "Avg Precision" not in html
    assert len(parser.headers) == 1
    assert len(parser.rows) == 1
    assert len(parser.headers[0]) == len(parser.rows[0]) == 9


def test_level_bars_render_l4_and_l5():
    report = _report()

    for level in (4, 5):
        report.operators[0].rel_path = f"level{level}/foo.py"
        html = _render_bars(report)

        assert html.count(f"Level {level}") == 3


def _speedup_report(groups):
    operators = []
    total_cases = 0
    passed_cases = 0
    for index, case_specs in enumerate(groups):
        rel_path = f"level1/op{index}"
        operator = f"op{index}"
        cases = [
            EvalResult(
                case_id=f"{operator}_{case_index}",
                rel_path=rel_path,
                operator=operator,
                status=status,
                speedup=speedup,
            )
            for case_index, (speedup, status) in enumerate(case_specs)
        ]
        passed = sum(case.status == "success" for case in cases)
        valid_speedups = [
            case.speedup for case in cases if case.status == "success" and case.speedup > 0
        ]
        operators.append(
            OperatorReport(
                rel_path=rel_path,
                operator=operator,
                total_cases=len(cases),
                passed_cases=passed,
                failed_cases=len(cases) - passed,
                pass_rate=passed / len(cases) if cases else 0.0,
                avg_speedup=sum(valid_speedups) / len(valid_speedups) if valid_speedups else 0.0,
                score=80.0,
                cases=cases,
            )
        )
        total_cases += len(cases)
        passed_cases += passed
    return EvalReport(
        framework_version="1",
        tasks_version="1",
        eval_code="geomean",
        timestamp="now",
        device="cpu",
        total_operators=len(operators),
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=total_cases - passed_cases,
        overall_score=80.0,
        operators=operators,
        summary={"pass_rate": passed_cases / total_cases if total_cases else 0.0},
    )


def _speedup_chart(html):
    section = html.split("Figure 1b.", 1)[1].split("Figure 1c.", 1)[0]
    value = section.split('<span class="bar-val">')[-1].split("</span>", 1)[0]
    width = section.split('style="width:', 1)[-1].split(";", 1)[0]
    return value, width


def test_level_speedup_uses_geomean_of_successful_positive_case_speeds():
    report = _speedup_report(
        [[(1.0, "success"), (9.0, "success"), (0.0, "success"), (99.0, "failed")]]
    )

    value, _ = _speedup_chart(_render_bars(report))

    assert value == "3.00x"


def test_level_speedup_weights_case_speeds_across_operators():
    report = _speedup_report(
        [
            [(1.0, "success"), (9.0, "success")],
            [(9.0, "success"), (9.0, "success"), (9.0, "success")],
        ]
    )

    value, _ = _speedup_chart(_render_bars(report))

    assert value == "5.80x"


def test_level_speedup_geomean_control_and_bar_width():
    equal_report = _speedup_report([[(4.0, "success"), (4.0, "success")]])
    low_report = _speedup_report([[(0.01, "success"), (0.09, "success")]])

    equal_value, _ = _speedup_chart(_render_bars(equal_report))
    low_value, low_width = _speedup_chart(_render_bars(low_report))

    assert equal_value == "4.00x"
    assert low_value == "0.03x"
    assert abs(float(low_width[:-1]) - 3.0) < 1e-9


def _speedup_chart_rows(html):
    section = html.split("Figure 1b.", 1)[1].split("Figure 1c.", 1)[0]
    rows = []
    for row in section.split('<div class="bar-row">')[1:]:
        label = row.split('<span class="bar-label">', 1)[1].split("</span>", 1)[0]
        value = row.split('<span class="bar-val">', 1)[1].split("</span>", 1)[0]
        width = row.split('style="width:', 1)[1].split(";", 1)[0]
        rows.append((label, value, width))
    return rows


def test_gen_report_preserves_cases_for_level_speedup_geomean():
    report = _speedup_report(
        [
            [(1.0, "success"), (9.0, "success")],
            [(4.0, "success"), (4.0, "success")],
        ]
    )
    report.operators[1].rel_path = "level2/op1"
    report_data = report.to_dict()
    repo_root = Path(__file__).resolve().parents[2]
    temp_root = repo_root / ".tmp"
    temp_root.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(dir=temp_root) as temp_dir:
        temp_path = Path(temp_dir)
        json_path = temp_path / "eval.json"
        template_path = temp_path / "template.html"
        json_path.write_text(json.dumps(report_data), encoding="utf-8")
        template_path.write_text("", encoding="utf-8")
        subprocess.run(
            [
                "bash",
                str(repo_root / "scripts/utils/gen_report.sh"),
                "--json",
                str(json_path),
                "--template",
                str(template_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        html = json_path.with_suffix(".html").read_text(encoding="utf-8")

    rows = _speedup_chart_rows(html)

    assert [(label, value) for label, value, _ in rows] == [
        ("Level 1", "3.00x"),
        ("Level 2", "4.00x"),
    ]

def test_html_report_escapes_operator_and_category_text(monkeypatch):
    from html import escape

    injected = '<img src=x onerror="document.body.dataset.reportInjected=1">'
    escaped = escape(injected)
    operators = [
        OperatorReport(
            rel_path="level1/add",
            operator=injected,
            total_cases=1,
            passed_cases=1,
            pass_rate=1.0,
            avg_speedup=1.0,
            score=100.0,
        ),
        OperatorReport(
            rel_path="level1/add",
            operator="Add",
            total_cases=1,
            passed_cases=1,
            pass_rate=1.0,
            avg_speedup=1.0,
            score=100.0,
        ),
    ]
    monkeypatch.setattr(
        "kernel_eval.report.html_generator._get_category", lambda unused_path: injected
    )

    top_tables = _render_top_tables(operators)
    details = _render_operator_tables(operators)

    assert injected not in top_tables + details
    assert top_tables.count(escaped) == 2
    assert details.count(escaped) == 3
    assert "Add" in top_tables + details
