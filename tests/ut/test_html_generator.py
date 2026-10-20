from html.parser import HTMLParser

from kernel_eval.report.html_generator import (
    _render_bars,
    _render_level_table,
    _render_operator_tables,
    _render_top_tables,
)
from kernel_eval.report.report_generator import EvalReport, OperatorReport


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
