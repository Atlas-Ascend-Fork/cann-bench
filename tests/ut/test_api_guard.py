#!/usr/bin/python3
# coding=utf-8

"""APIGuard snapshot-state regression tests."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


API_GUARD_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "kernel_eval"
    / "security"
    / "api_guard.py"
)


@pytest.fixture
def api_guard_module(monkeypatch):
    spec = spec_from_file_location("api_guard_under_test", API_GUARD_PATH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    def original(*_args, **_kwargs):
        return 0.0

    parent = SimpleNamespace(elapsed_time=original)
    monkeypatch.delenv("ALLOW_TIMING_TAMPERING", raising=False)
    monkeypatch.setattr(
        module,
        "_CRITICAL_API_ENTRIES",
        [("torch.npu.Event.elapsed_time", parent, "elapsed_time")],
    )
    monkeypatch.setattr(module, "_INIT_DONE", True)
    monkeypatch.setattr(module, "_INIT_TORCH_NPU_AVAILABLE", True)
    monkeypatch.setattr(module, "_ATEXIT_REGISTERED", True)
    return module, parent, original


def test_verify_without_snapshot_fails_closed(api_guard_module):
    module, parent, _ = api_guard_module

    def tampered(*_args, **_kwargs):
        return 999.0

    parent.elapsed_time = tampered

    with pytest.raises(RuntimeError, match="快照未建立"):
        module.APIGuard().verify()

    assert parent.elapsed_time is tampered


def test_snapshot_allows_unchanged_api_and_detects_tampering(api_guard_module):
    module, parent, original = api_guard_module
    guard = module.APIGuard()

    guard.snapshot()

    assert module._SNAPSHOT_TAKEN is True
    assert guard.verify() is True

    def tampered(*_args, **_kwargs):
        return 999.0

    parent.elapsed_time = tampered
    with pytest.raises(RuntimeError, match="torch.npu.Event.elapsed_time"):
        guard.verify()

    assert parent.elapsed_time is original
