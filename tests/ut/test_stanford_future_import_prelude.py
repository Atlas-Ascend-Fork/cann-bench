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

"""Public Stanford preparation must not invalidate legal future imports."""

import ast
import sys
from types import SimpleNamespace

import pytest

from auto_pipeline.converter.submission import prepare_submission


MODEL_BODY = """import torch
from torch import nn
from b7_prelude_fixture import WIDTH

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)
    def forward(self, inputs):
        return self.linear(inputs)
"""
TASK_BODY = """from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
    def forward(self, inputs):
        return self.linear(inputs)
def get_init_inputs():
    return []
"""


@pytest.mark.parametrize(
    "prefix",
    [
        "from __future__ import annotations\n",
        '"""Submission module documentation."""\nfrom __future__ import annotations\n',
        "from __future__ import (\n    annotations,\n)\n",
        "",
    ],
)
def test_prepare_keeps_future_imports_and_local_modules(tmp_path, monkeypatch, prefix):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = prefix + MODEL_BODY
    compile(source, "original-ai-op.py", "exec")
    ai_op = submission / "ai_op.py"
    ai_op.write_text(source)
    (submission / "b7_prelude_fixture.py").write_text("WIDTH = 2\n")
    task = tmp_path / "task.py"
    task.write_text(TASK_BODY)
    monkeypatch.delitem(sys.modules, "b7_prelude_fixture", raising=False)
    case = SimpleNamespace(files={"task": task})

    prepare_submission("stanford", case, submission)

    prepared = ai_op.read_text()
    compile(prepared, str(ai_op), "exec")
    if prefix.startswith('"""'):
        tree = ast.parse(prepared)
        assert ast.get_docstring(tree) == "Submission module documentation."
    before_second_run = ai_op.read_bytes()
    prepare_submission("stanford", case, submission)
    assert ai_op.read_bytes() == before_second_run
    # Construction and state_dict validation above ran real PyTorch Linear.
    # The local dependency is a real file, made importable by the real prelude.
    sys.modules.pop("b7_prelude_fixture", None)
