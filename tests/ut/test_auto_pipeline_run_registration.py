import pytest

import auto_pipeline.core as core
import auto_pipeline.state as state


def _bad_config(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "agent:\n  type: akg-agent\nbenchmark:\n  name: cann\nbroken_key: 1\n",
        encoding="utf-8",
    )
    return path


def test_pre_registration_failure_marks_running_run_as_error(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DEFAULT_CANN_BENCH_ROOT", tmp_path)
    state.upsert_run("bad-run", {"status": "running", "pid": 123, "pid_start_time": 1})

    with pytest.raises(ValueError, match="unsupported config keys"):
        core.run_from_config(
            _bad_config(tmp_path),
            runtime={"run_id": "bad-run", "workspace": str(tmp_path / "ws")},
        )

    run = state.read_json(state.run_file("bad-run"))
    assert run["status"] == "error"
    assert run["error_type"] == "ValueError"
    assert "unsupported config keys" in run["error"]
    assert run["completed_at"]


def test_unregistered_foreground_failure_does_not_create_state(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DEFAULT_CANN_BENCH_ROOT", tmp_path)

    with pytest.raises(ValueError, match="unsupported config keys"):
        core.run_from_config(
            _bad_config(tmp_path),
            runtime={"run_id": "fg-run", "workspace": str(tmp_path / "ws")},
        )

    assert not state.run_file("fg-run").exists()


def test_terminal_run_is_not_overwritten_by_a_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DEFAULT_CANN_BENCH_ROOT", tmp_path)
    state.upsert_run("killed-run", {"status": "killed", "kill_signal": 15})

    with pytest.raises(ValueError, match="unsupported config keys"):
        core.run_from_config(
            _bad_config(tmp_path),
            runtime={"run_id": "killed-run", "workspace": str(tmp_path / "ws")},
        )

    assert state.read_json(state.run_file("killed-run"))["status"] == "killed"
