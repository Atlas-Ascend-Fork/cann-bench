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

"""
单点版本真相源

从项目根 VERSION 文件和 tasks/metadata/VERSION 文件动态读取版本号，
供 __init__.py 和其他模块引用。

使用方法:
    from kernel_eval._version import FRAMEWORK_VERSION, TASKS_VERSION
    或
    import kernel_eval
    kernel_eval.__version__       # 框架版本
    kernel_eval.TASKS_VERSION     # 评测集版本

修改版本时只需编辑根 VERSION 和 tasks/metadata/VERSION 文件，
所有引用自动跟随变化。
"""

import os
import re
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _read_version(rel_path: str) -> str:
    """从相对路径读取版本号，取第一行非注释内容。

    Args:
        rel_path: 相对项目根目录的路径，如 "VERSION" 或 "tasks/metadata/VERSION"

    Returns:
        版本字符串（如 "0.3.0"）；文件不存在时返回 "0.0.0-dev"
    """
    version_file = os.path.join(_ROOT, rel_path)
    try:
        with open(version_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
        # 文件存在但无有效行
        return "0.0.0-dev"
    except FileNotFoundError:
        return "0.0.0-dev"


def read_tasks_version(bench_root) -> str:
    """严格读取被评测集 VERSION 首行；后续兼容性注释不属于版本号。"""
    version_file = Path(bench_root) / "metadata" / "VERSION"
    try:
        with version_file.open(encoding="utf-8") as stream:
            version = stream.readline().strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read tasks VERSION: {version_file}") from exc
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version):
        raise ValueError(f"invalid tasks VERSION {version!r} in {version_file}; expected X.Y.Z")
    return version


FRAMEWORK_VERSION = _read_version("VERSION")
TASKS_VERSION = _read_version("tasks/metadata/VERSION")
