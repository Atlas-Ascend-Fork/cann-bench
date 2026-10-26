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

"""Host checks for dynamically loaded reference implementations."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "scripts/baseline/ref_registry.py"


@pytest.fixture
def registry(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "bench_ref_registry_host", REGISTRY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "ref_registry.py"))
    tmp_path.joinpath("refs").mkdir()
    monkeypatch.setitem(
        sys.modules, "_3way_refs_custom", types.ModuleType("previous_reference")
    )
    monkeypatch.delitem(sys.modules, "_3way_refs_custom")
    return module, tmp_path / "refs"


def test_registry_loads_postponed_dataclass_reference(registry):
    module, refs = registry
    (refs / "custom.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\nclass RefConfig:\n    factor: int = 3\n"
        "def reference(inputs, attrs):\n    return inputs[0] * RefConfig().factor\n"
        "REGISTRY = {'level1/custom': reference}\n",
        encoding="utf-8",
    )

    result = module._load()

    assert result["level1/custom"]([7], {}) == 21


def test_registry_loads_plain_reference(registry):
    module, refs = registry
    (refs / "custom.py").write_text(
        "REGISTRY = {'level1/plain': lambda inputs, attrs: inputs[0] + 1}\n",
        encoding="utf-8",
    )

    result = module._load()

    assert result["level1/plain"]([7], {}) == 8


def test_registry_restores_previous_module_after_failed_reference(
    registry, capsys, monkeypatch
):
    module, refs = registry
    previous = types.ModuleType("previous_reference")
    monkeypatch.setitem(sys.modules, "_3way_refs_custom", previous)
    (refs / "custom.py").write_text(
        "raise RuntimeError('incomplete reference')\n", encoding="utf-8"
    )

    result = module._load()

    assert result == {}
    assert sys.modules["_3way_refs_custom"] is previous
    assert "incomplete reference" in capsys.readouterr().out
