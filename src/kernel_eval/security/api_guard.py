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
Timing API防护模块

职责：
1. 在submission代码运行前快照关键Timing API的身份
2. 安装wheel后验证API未被篡改
3. 程序退出前恢复原始API

防护原理：
- 攻击者可能通过monkey-patch修改torch.npu.Event.elapsed_time等API
- 在运行submission代码前快照API的原始callable，安装后验证是否一致
- 如果发现篡改，先恢复原始API再报错，避免atexit崩溃

参考evaluation/evaluate.py中的防篡改机制
"""

import atexit
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple


# 关键Timing API列表
_CRITICAL_API_ENTRIES: List[Tuple[str, Any, str]] = []

# 标记 _init_critical_apis 是否检测到 torch_npu 不可用——
# 用于在 snapshot 时拒绝继续而不是静默通过。
_INIT_TORCH_NPU_AVAILABLE: bool = False
_INIT_DONE: bool = False

# 快照存储
_API_SNAPSHOT: Dict[str, Tuple[Any, str, Any]] = {}
_SNAPSHOT_TAKEN: bool = False

# 模块级状态读写互斥锁——snapshot / verify / restore 都会访问
# 快照状态，多线程评测路径下需要互斥防止读到半截的状态。
_API_SNAPSHOT_LOCK = threading.Lock()

# snapshot 时固定下来的 ALLOW_TIMING_TAMPERING 状态，避免 TOCTOU——
# 攻击者在 snapshot 之后、verify 之前设置环境变量并重新实例化 APIGuard 来绕过校验。
_SNAPSHOT_ALLOW_TAMPERING: Optional[bool] = None

# atexit 钩子是否已注册——避免重复注册
_ATEXIT_REGISTERED: bool = False


def _init_critical_apis():
    """初始化关键API列表（延迟加载，避免import时torch_npu不可用）"""
    global _CRITICAL_API_ENTRIES, _INIT_TORCH_NPU_AVAILABLE, _INIT_DONE

    if _INIT_DONE:
        return
    _INIT_DONE = True

    try:
        import torch
        import torch_npu

        _CRITICAL_API_ENTRIES = [
            # Legacy event-based timing
            ("torch.npu.Event.elapsed_time", torch.npu.Event, "elapsed_time"),
            ("torch.npu.Event.record", torch.npu.Event, "record"),
            ("torch.npu.synchronize", torch.npu, "synchronize"),
            # Current profiler-based timing
            ("torch_npu.profiler.profile", torch_npu.profiler, "profile"),
            ("torch_npu.profiler.schedule", torch_npu.profiler, "schedule"),
            ("torch_npu.profiler.tensorboard_trace_handler", torch_npu.profiler, "tensorboard_trace_handler"),
            ("torch_npu.profiler._ExperimentalConfig", torch_npu.profiler, "_ExperimentalConfig"),
        ]
        _INIT_TORCH_NPU_AVAILABLE = True
    except ImportError as e:
        # torch_npu 不可用——记录状态。snapshot 时若要求安全保护会拒绝继续。
        print(
            f"[ERROR] APIGuard: torch_npu import failed ({e}); "
            "Timing API security checks DISABLED. NPU evaluation results cannot be trusted.",
            file=sys.stderr,
            flush=True,
        )
        _INIT_TORCH_NPU_AVAILABLE = False


def snapshot_timing_apis() -> None:
    """
    快照Timing API身份

    在submission代码运行前调用，保存原始callable。同时：
    - 固化 ALLOW_TIMING_TAMPERING 环境变量值（防 TOCTOU）
    - 注册 atexit 钩子以确保异常退出时 API 也会被恢复（防止 torch_npu atexit 用到被篡改的 API）
    """
    global _API_SNAPSHOT, _SNAPSHOT_TAKEN, _SNAPSHOT_ALLOW_TAMPERING, _ATEXIT_REGISTERED

    _init_critical_apis()

    with _API_SNAPSHOT_LOCK:
        # 固化 ALLOW_TIMING_TAMPERING——之后 verify 读这个固化值
        _SNAPSHOT_ALLOW_TAMPERING = (os.environ.get("ALLOW_TIMING_TAMPERING") == "1")

        _API_SNAPSHOT = {}
        _SNAPSHOT_TAKEN = False
        for name, parent, attr in _CRITICAL_API_ENTRIES:
            try:
                original = getattr(parent, attr)
                _API_SNAPSHOT[name] = (parent, attr, original)
            except AttributeError:
                # API不存在时跳过
                pass

        _SNAPSHOT_TAKEN = True

        # 注册一次 atexit，确保任何退出路径（包括 SystemExit / 未捕获异常）
        # 都会还原 timing API；torch_npu 自己的 atexit 钩子才能用回真身。
        if not _ATEXIT_REGISTERED and _API_SNAPSHOT:
            atexit.register(restore_timing_apis)
            _ATEXIT_REGISTERED = True


def is_torch_npu_available_for_guard() -> bool:
    """供调用方检查：APIGuard 是否能真正提供安全保护。"""
    _init_critical_apis()
    return _INIT_TORCH_NPU_AVAILABLE


def verify_timing_apis() -> List[str]:
    """
    验证Timing API完整性

    Returns:
        被篡改的API名称列表（空列表表示通过）。
        若 torch_npu 在初始化时不可用 → snapshot 是空 dict，无任何可校验对象，
        直接判定为不可信，返回 ["__torch_npu_unavailable__"] sentinel；若尚未调用
        snapshot，则返回 ["__snapshot_not_taken__"] sentinel。
    """
    # F091: torch_npu 不可用时不能 silent-pass —— 没有 snapshot 对象就什么都校验不了，
    # 评测结果不可信。强制返回非空列表让调用方触发现有的篡改处理路径。
    _init_critical_apis()
    if not _INIT_TORCH_NPU_AVAILABLE:
        return ["__torch_npu_unavailable__"]

    changed = []

    with _API_SNAPSHOT_LOCK:
        if not _SNAPSHOT_TAKEN:
            return ["__snapshot_not_taken__"]
        snapshot_items = list(_API_SNAPSHOT.items())

    for name, (parent, attr, original) in snapshot_items:
        try:
            current = getattr(parent, attr)
            if current is not original:
                changed.append(name)
        except AttributeError:
            # API被删除也算篡改
            changed.append(name)

    if changed:
        # 先恢复原始API，避免atexit崩溃
        restore_timing_apis()

    return changed


def restore_timing_apis() -> None:
    """
    恢复原始Timing API

    在程序退出前调用，确保torch_npu的atexit钩子使用原始API
    """
    with _API_SNAPSHOT_LOCK:
        snapshot_items = list(_API_SNAPSHOT.items())

    for name, (parent, attr, original) in snapshot_items:
        try:
            setattr(parent, attr, original)
        except Exception as e:
            # F008: 旧版 silent pass 吞 setattr 失败 → restore 失败完全不可见。
            # 攻击者可让目标属性变只读描述符强制 restore 失败；运维也无法察觉。
            # 至少记录一条 stderr 日志保留诊断线索。仍然继续循环（best-effort
            # restore — 一个 API 失败不影响其他 API 的恢复）。
            print(
                f"[WARN] api_guard.restore_timing_apis: setattr {name} failed: {e}",
                file=sys.stderr, flush=True,
            )


class APIGuard:
    """
    Timing API防护器

    使用方法：
        guard = APIGuard()
        guard.snapshot()           # 安装wheel前
        install_wheel(path)        # 安装submission
        guard.verify()             # 检查完整性
        # ... 执行评测 ...
        guard.restore()            # 程序退出前
    """

    def __init__(self):
        # 不再在 __init__ 读环境变量——TOCTOU 安全漏洞：
        # 攻击者可在 snapshot() 之后、verify() 之前设置 ALLOW_TIMING_TAMPERING=1
        # 然后重新实例化一个 APIGuard 来让 verify() 直接 return True。
        # ALLOW_TIMING_TAMPERING 现在在 snapshot_timing_apis() 中固化为模块级值。
        pass

    def snapshot(self) -> None:
        """快照API身份"""
        snapshot_timing_apis()

    def verify(self) -> bool:
        """
        验证API完整性

        Returns:
            True表示通过，False表示被篡改

        Raises:
            RuntimeError: 如果快照未建立或检测到篡改且未设置ALLOW_TIMING_TAMPERING
        """
        # 读 snapshot 时固化的 allow_tampering 值——非每次 verify 重新读 env
        if _SNAPSHOT_ALLOW_TAMPERING is True:
            return True

        changed = verify_timing_apis()
        if changed:
            if changed == ["__snapshot_not_taken__"]:
                raise RuntimeError(
                    "[SECURITY] Timing API快照未建立，无法验证完整性。\n"
                    "评测结果不可信，已终止执行。\n"
                    "请在校验前调用 snapshot() 建立基线。"
                )
            raise RuntimeError(
                f"[SECURITY] Timing API被篡改: {changed}\n"
                "评测结果不可信，已终止执行。\n"
                "如需调试，可在 snapshot() 之前设置环境变量 ALLOW_TIMING_TAMPERING=1"
            )
        return True

    def restore(self) -> None:
        """恢复原始API"""
        restore_timing_apis()

    def __enter__(self):
        """上下文管理器入口"""
        self.snapshot()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.restore()
        return False
