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

"""Reference implementation registry — auto-loads refs/level{1,2,3,4}.py.

Each `refs/levelN.py` (or any other module dropped into refs/) must export a
top-level dict named `REGISTRY` mapping operator paths ("levelN/op_dir") to
callables of the form `ref(inputs, attrs) -> outputs`. inputs already lives
on the target NPU device.

When no NPU equivalent exists for an op, simply omit the entry — the harness
will skip method 3 ("npu_ref") for that op.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict, Optional


_REG: Optional[Dict[str, Callable]] = None


def _load() -> Dict[str, Callable]:
    here = Path(__file__).resolve().parent / "refs"
    out: Dict[str, Callable] = {}
    if not here.is_dir():
        return out
    for p in sorted(here.glob("*.py")):
        if p.name == "__init__.py":
            continue
        spec = importlib.util.spec_from_file_location(f"_3way_refs_{p.stem}", p)
        previous = sys.modules.get(spec.name)
        try:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
        except Exception as e:
            if previous is None:
                sys.modules.pop(spec.name, None)
            else:
                sys.modules[spec.name] = previous
            # subagents may write partial files; print but don't abort
            print(f"[ref_registry] WARN: failed to load {p.name}: {e}")
            continue
        reg = getattr(mod, "REGISTRY", None)
        if isinstance(reg, dict):
            out.update(reg)
    return out


def _ensure() -> Dict[str, Callable]:
    global _REG
    if _REG is None:
        _REG = _load()
    return _REG


def get_ref(op_path: str) -> Optional[Callable]:
    return _ensure().get(op_path)


def all_keys():
    return list(_ensure().keys())
