#!/usr/bin/python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Vendor library search entries must be compared as complete paths."""

import os
from pathlib import Path

import pytest

from src.kernel_eval.data.package_manager import _add_vendor_lib_dirs, _get_arch_str


@pytest.mark.parametrize("existing_kind", ["suffix", "child", "exact", "unset"])
def test_vendor_library_paths_are_complete_entries(
    monkeypatch, tmp_path, existing_kind
):
    monkeypatch.chdir(tmp_path)
    vendor = Path("vendor")
    api = vendor / "op_api" / "lib"
    tiling = vendor / "op_impl" / "ai_core" / "tbe" / "op_tiling"
    arch = tiling / "lib" / "linux" / _get_arch_str()
    for directory in (api, arch):
        directory.mkdir(parents=True)

    existing = {
        "suffix": str(api) + "64",
        "child": str(arch),
        "exact": str(api),
        "unset": None,
    }[existing_kind]
    if existing is None:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH", existing + ":unrelated")

    _add_vendor_lib_dirs(vendor)
    entries = os.environ["LD_LIBRARY_PATH"].split(":")
    for directory in (api, tiling, arch):
        assert entries.count(str(directory)) == 1
    if existing is not None:
        assert entries[-2:] == [existing, "unrelated"]

    # Calling the installer helper again must not duplicate exact entries.
    first = os.environ["LD_LIBRARY_PATH"]
    _add_vendor_lib_dirs(vendor)
    assert os.environ["LD_LIBRARY_PATH"] == first
