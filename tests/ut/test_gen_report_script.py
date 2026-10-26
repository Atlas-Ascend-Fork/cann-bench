# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("json_name", "template_name"),
    [
        ("report with spaces.json", "template with spaces.html"),
        ("reviewer's report.json", "template.html"),
        ("report.json", "reviewer's template.html"),
        ("reviewer's report.json", "reviewer's template.html"),
    ],
)
def test_gen_report_preserves_path_arguments(tmp_path, json_name, template_name):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required to run the report script")
    root = Path(__file__).resolve().parents[2]
    report = tmp_path / json_name
    report.write_text(
        json.dumps({"device": "cpu", "operators": [], "summary": {"pass_rate": 0.0}}),
        encoding="utf-8",
    )
    template = tmp_path / template_name
    template.write_text(
        "<html><body>path-template-marker</body></html>", encoding="utf-8"
    )
    # Use the same Python environment as pytest, including on systems without python3.
    command_dir = tmp_path / "commands"
    command_dir.mkdir()
    python_command = command_dir / "python3"
    python_path = str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    python_command.write_text(
        f"#!/usr/bin/env bash\nexport PYTHONPATH={shlex.quote(python_path)}\n"
        f'exec {shlex.quote(Path(sys.executable).as_posix())} "$@"\n',
        encoding="utf-8",
    )
    python_command.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(command_dir) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [
            bash,
            (root / "scripts/utils/gen_report.sh").as_posix(),
            "--json",
            report.as_posix(),
            "--template",
            template.as_posix(),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = report.with_suffix(".html")
    assert output.is_file()
    html = output.read_text(encoding="utf-8")
    assert "path-template-marker" in html
    assert "Section" in html or "<html" in html
