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

"""Submission format checks owned by converters."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from auto_pipeline.core import CannBenchCase


SUBMISSION_RULES_DOC = "docs/spec/submission_spec.md"
SUBMISSION_RULE_IDS = (
    "AUTO-SUB-001",
    "AUTO-CANN-001",
    "AUTO-CANN-002",
    "AUTO-STANFORD-001",
    "AUTO-STANFORD-002",
    "AUTO-STANFORD-003",
    "AUTO-STANFORD-004",
    "AUTO-STANFORD-005",
    "AUTO-STANFORD-006",
    "AUTO-STANFORD-007",
    "AUTO-STANFORD-008",
    "AUTO-STANFORD-009",
    "AUTO-STANFORD-010",
    "AUTO-STANFORD-011",
    "AUTO-STANFORD-012",
)


@dataclass(frozen=True)
class SubmissionIssue:
    """One actionable violation of the documented submission contract."""

    rule_id: str
    summary: str
    path: Path
    expected: str
    actual: str
    remediation: str

    def render(self) -> str:
        return (
            f"[{self.rule_id}] {self.summary}\n"
            f"  path: {self.path}\n"
            f"  expected: {self.expected}\n"
            f"  actual: {self.actual}\n"
            f"  fix: {self.remediation}"
        )


def prepare_submission(
    target_benchmark: str, case: CannBenchCase, source_dir: Path
) -> None:
    if _normalize_name(target_benchmark) == "stanford":
        issues = _prepare_stanford_submission(case, Path(source_dir))
        if issues:
            raise ValueError(
                _format_submission_issues(
                    "Stanford", target_benchmark, source_dir, issues
                )
            )


def is_submission_dir(target_benchmark: str, source_dir: Path) -> bool:
    return not collect_submission_issues(target_benchmark, source_dir)


def validate_submission(
    target_benchmark: str, case: CannBenchCase, source_dir: Path, *, label: str
) -> None:
    issues = collect_submission_issues(target_benchmark, source_dir)
    if issues:
        raise ValueError(
            _format_submission_issues(label, target_benchmark, source_dir, issues)
        )
    prepare_submission(target_benchmark, case, source_dir)


def collect_submission_issues(
    target_benchmark: str, source_dir: Path
) -> list[SubmissionIssue]:
    """Return all format issues that can be determined without executing a submission."""

    path = Path(source_dir)
    if not path.is_dir():
        return [
            SubmissionIssue(
                "AUTO-SUB-001",
                "submission source_dir is not a directory",
                path,
                "an existing directory containing the benchmark submission",
                "path is missing or is not a directory",
                "pass the extracted submission root directory",
            )
        ]
    if _normalize_name(target_benchmark) == "stanford":
        return _stanford_source_issues(path)
    return _cannbench_source_issues(path)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def _cannbench_source_issues(path: Path) -> list[SubmissionIssue]:
    issues = []
    build_script = path / "build.sh"
    if not build_script.is_file():
        issues.append(
            SubmissionIssue(
                "AUTO-CANN-001",
                "CANN submission is missing build.sh",
                build_script,
                "a regular file executed as `bash build.sh`",
                "missing or not a regular file",
                "add build.sh at the submission root",
            )
        )

    package_dir = path / "cann_bench"
    dist_dir = path / "dist"
    wheels = sorted(dist_dir.glob("cann_bench*.whl")) if dist_dir.is_dir() else []
    if not package_dir.is_dir() and not wheels:
        issues.append(
            SubmissionIssue(
                "AUTO-CANN-002",
                "CANN submission has neither package sources nor a prebuilt wheel",
                path,
                "cann_bench/ directory or dist/cann_bench*.whl",
                f"root entries: {_submission_root_entries(path)}",
                "add the cann_bench package directory or a matching wheel under dist/",
            )
        )
    return issues


def _stanford_source_issues(path: Path) -> list[SubmissionIssue]:
    ai_op_path = path / "ai_op.py"
    if ai_op_path.is_file():
        return []
    return [
        SubmissionIssue(
            "AUTO-STANFORD-001",
            "Stanford submission is missing ai_op.py",
            ai_op_path,
            "a regular Python file defining ModelNew",
            "missing or not a regular file",
            "convert the raw agent artifact into a standard ai_op.py submission",
        )
    ]


def _prepare_stanford_submission(
    case: CannBenchCase, source_dir: Path
) -> list[SubmissionIssue]:
    ai_op_path = source_dir / "ai_op.py"
    if not ai_op_path.is_file():
        return _stanford_source_issues(source_dir)
    source_issues = _stanford_python_source_issues(ai_op_path)
    if source_issues:
        return source_issues
    _ensure_stanford_ai_op_prelude(ai_op_path)
    return _stanford_model_contract_issues(case, source_dir)


def _stanford_python_source_issues(path: Path) -> list[SubmissionIssue]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return [
            SubmissionIssue(
                "AUTO-STANFORD-002",
                "Stanford ai_op.py is not valid UTF-8 Python source",
                path,
                "UTF-8 encoded, syntactically valid Python",
                f"{type(exc).__name__}: {exc}",
                "fix the file encoding or Python syntax before conversion",
            )
        ]

    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            module = "." * node.level + (node.module or "")
            issues.append(
                SubmissionIssue(
                    "AUTO-STANFORD-003",
                    "Stanford ai_op.py uses a relative import",
                    path,
                    "flat/local absolute imports resolvable from source_dir",
                    f"relative import {module!r} at line {node.lineno}",
                    "replace the relative import with a local absolute import",
                )
            )
    return issues


def _stanford_model_contract_issues(
    case: CannBenchCase, source_dir: Path
) -> list[SubmissionIssue]:
    ai_op_path = source_dir / "ai_op.py"
    task_path = case.files.get("task")
    if task_path is None or not Path(task_path).is_file():
        return []

    try:
        task_module = _load_module_from_path("stanford_task_contract", Path(task_path))
    except Exception as exc:
        return [
            SubmissionIssue(
                "AUTO-STANFORD-004",
                "Stanford task contract cannot be imported",
                Path(task_path),
                "an importable task module defining Model",
                f"{type(exc).__name__}: {exc}",
                "fix the task module or its declared dependencies",
            )
        ]
    try:
        ai_module = _load_module_from_path(
            "stanford_ai_contract", ai_op_path, prepend_paths=[source_dir]
        )
    except Exception as exc:
        return [
            SubmissionIssue(
                "AUTO-STANFORD-005",
                "Stanford ai_op.py cannot be imported",
                ai_op_path,
                "an importable module using files contained in source_dir",
                f"{type(exc).__name__}: {exc}",
                "include required local modules and remove import-time failures",
            )
        ]

    return _loaded_stanford_contract_issues(task_module, ai_module, ai_op_path)


def _loaded_stanford_contract_issues(task_module, ai_module, ai_op_path):
    task_model = getattr(task_module, "Model", None)
    ai_model = getattr(ai_module, "ModelNew", None)
    if task_model is None or ai_model is None:
        return [
            SubmissionIssue(
                "AUTO-STANFORD-006",
                "Stanford model classes are incomplete",
                ai_op_path,
                "task.Model and ai_op.ModelNew",
                f"task.Model={task_model is not None}, ai_op.ModelNew={ai_model is not None}",
                "define ModelNew in ai_op.py with the same public contract as task.Model",
            )
        ]

    issues = []
    issues.extend(
        _method_signature_issues(task_model, ai_model, "__init__", ai_op_path)
    )
    issues.extend(_method_signature_issues(task_model, ai_model, "forward", ai_op_path))
    issues.extend(
        _state_dict_contract_issues(task_module, task_model, ai_model, ai_op_path)
    )
    return issues


def _load_module_from_path(
    name: str, path: Path, *, prepend_paths: Iterable[Path] = ()
):
    old_path = list(sys.path)
    for entry in reversed([str(Path(item)) for item in prepend_paths]):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    try:
        module_name = f"{name}_{abs(hash(str(path.resolve())))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


def _method_signature_issues(
    task_model: object, ai_model: object, method_name: str, path: Path
) -> list[SubmissionIssue]:
    rule_id = "AUTO-STANFORD-007" if method_name == "__init__" else "AUTO-STANFORD-008"
    try:
        task_signature = _normalized_signature(getattr(task_model, method_name))
        ai_signature = _normalized_signature(getattr(ai_model, method_name))
    except (AttributeError, TypeError, ValueError) as exc:
        return [
            SubmissionIssue(
                rule_id,
                f"Stanford ModelNew.{method_name} signature cannot be inspected",
                path,
                f"a callable {method_name} matching task.Model.{method_name}",
                f"{type(exc).__name__}: {exc}",
                f"define ModelNew.{method_name} with an inspectable Python signature",
            )
        ]
    if task_signature != ai_signature:
        return [
            SubmissionIssue(
                rule_id,
                f"Stanford ModelNew.{method_name} signature does not match task.Model.{method_name}",
                path,
                repr(task_signature),
                repr(ai_signature),
                f"copy the parameter order, kinds and defaults from task.Model.{method_name}",
            )
        ]
    return []


def _normalized_signature(method: object) -> list[tuple[str, str, bool, str]]:
    signature = inspect.signature(method)
    normalized = []
    for param in signature.parameters.values():
        has_default = param.default is not inspect._empty
        default = repr(param.default) if has_default else ""
        normalized.append((param.name, str(param.kind), has_default, default))
    return normalized


def _state_dict_contract_issues(
    task_module: object, task_model_cls: object, ai_model_cls: object, path: Path
) -> list[SubmissionIssue]:
    try:
        get_init_inputs = getattr(task_module, "get_init_inputs", None)
        init_inputs = list(get_init_inputs() if callable(get_init_inputs) else [])
        task_model = task_model_cls(*init_inputs)
        ai_model = ai_model_cls(*init_inputs)
        task_state = task_model.state_dict()
        ai_state = ai_model.state_dict()
    except Exception as exc:
        return [
            SubmissionIssue(
                "AUTO-STANFORD-009",
                "Stanford models cannot be constructed for state_dict validation",
                path,
                "Model and ModelNew constructible from get_init_inputs() with state_dict() support",
                f"{type(exc).__name__}: {exc}",
                "align constructors and ensure both model classes expose state_dict()",
            )
        ]

    issues = []
    task_keys = list(task_state.keys())
    ai_keys = list(ai_state.keys())
    if task_keys != ai_keys:
        issues.append(
            SubmissionIssue(
                "AUTO-STANFORD-010",
                "Stanford ModelNew state_dict keys do not match task.Model",
                path,
                repr(task_keys),
                repr(ai_keys),
                "register the same parameters and buffers in the same order",
            )
        )

    for key in [item for item in task_keys if item in ai_state]:
        task_tensor = task_state[key]
        ai_tensor = ai_state[key]
        if tuple(task_tensor.shape) != tuple(ai_tensor.shape):
            issues.append(
                SubmissionIssue(
                    "AUTO-STANFORD-011",
                    f"Stanford state_dict shape mismatch for {key}",
                    path,
                    repr(tuple(task_tensor.shape)),
                    repr(tuple(ai_tensor.shape)),
                    "create the parameter or buffer with the task model shape",
                )
            )
        if task_tensor.dtype != ai_tensor.dtype:
            issues.append(
                SubmissionIssue(
                    "AUTO-STANFORD-012",
                    f"Stanford state_dict dtype mismatch for {key}",
                    path,
                    str(task_tensor.dtype),
                    str(ai_tensor.dtype),
                    "create the parameter or buffer with the task model dtype",
                )
            )
    return issues


def _render_stanford_ai_op(source: str) -> str:
    prelude = (
        "from pathlib import Path as _Path\n"
        "import sys as _sys\n"
        "_op_dir = str(_Path(__file__).resolve().parent)\n"
        "if _op_dir not in _sys.path:\n"
        "    _sys.path.insert(0, _op_dir)\n"
        "del _Path, _sys, _op_dir\n"
        "\n"
    )
    tree = ast.parse(source)
    insertion_line = 0
    for index, node in enumerate(tree.body):
        is_docstring = (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if not is_docstring and not is_future:
            break
        insertion_line = node.end_lineno
    lines = source.splitlines(keepends=True)
    prefix = "".join(lines[:insertion_line])
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    return prefix + prelude + "".join(lines[insertion_line:])


def _ensure_stanford_ai_op_prelude(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if "_sys.path.insert(0, _op_dir)" in source:
        return
    path.write_text(_render_stanford_ai_op(source), encoding="utf-8")


def _submission_root_entries(path: Path) -> str:
    if not path.is_dir():
        return "<not a directory>"
    entries = [child.name + ("/" if child.is_dir() else "") for child in path.iterdir()]
    return ", ".join(sorted(entries)) or "<empty>"


def _format_submission_issues(
    label: str,
    target_benchmark: str,
    source_dir: Path,
    issues: Iterable[SubmissionIssue],
) -> str:
    rendered = "\n".join(f"- {issue.render()}" for issue in issues)
    return (
        f"{label} submission is not valid for benchmark {target_benchmark}.\n"
        f"Submission root: {Path(source_dir)}\n"
        f"Root entries: {_submission_root_entries(Path(source_dir))}\n"
        f"Violations:\n{rendered}\n"
        f"Rules: {SUBMISSION_RULES_DOC}"
    )
