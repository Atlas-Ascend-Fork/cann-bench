from types import SimpleNamespace

from kernel_eval import cli


def test_list_operator_filter_is_applied(monkeypatch, capsys, tmp_path):
    operators = [
        SimpleNamespace(name="Exp", difficulty=None, category="Elementwise", description="exp"),
        SimpleNamespace(name="Softmax", difficulty=None, category="Normalization", description="softmax"),
    ]

    class _Loader:
        def list_tasks(self):
            return operators

    monkeypatch.setattr(cli, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "resolve_task_dir", lambda task_dir, root: (str(tmp_path), None))
    monkeypatch.setattr("kernel_eval.registry.get_task_loader", lambda *a, **k: _Loader())

    args = cli.create_parser().parse_args(["list", "--operator", "Exp"])

    assert cli.cmd_list(args) == 0

    output = capsys.readouterr().out
    assert "共 1 个算子" in output
    assert "Exp" in output
    assert "Softmax" not in output
