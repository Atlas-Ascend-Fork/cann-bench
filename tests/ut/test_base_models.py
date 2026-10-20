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

from kernel_eval.base.models import TaskSpec


def test_task_spec_to_dict_preserves_path_and_category():
    task = TaskSpec(
        task_id="t1",
        name="Demo",
        rel_path="tasks/level2/demo",
        category="activation",
    )

    data = task.to_dict()

    assert data["rel_path"] == "tasks/level2/demo"
    assert data["category"] == "activation"


def test_task_spec_to_dict_includes_default_path_and_category():
    data = TaskSpec(task_id="t1", name="Demo").to_dict()

    assert data["rel_path"] == ""
    assert data["category"] == ""
