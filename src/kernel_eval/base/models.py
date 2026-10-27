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
数据模型基类（统一）

包含：
- AttrSpec: 算子属性规格
- InputSpec: 输入规格
- OutputSpec: 输出规格
- TaskSpec: 任务规格基类
- CaseSpec: 用例规格基类
- SolutionSpec: 解决方案规格基类

Why: 为所有评测体系提供统一的数据模型定义接口
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import DifficultyLevel, BackendType, SourceType


# === 属性规格 ===

@dataclass
class AttrSpec:
    """算子属性规格"""
    name: str
    type: str
    default: Optional[Any] = None
    description: str = ""
    required: bool = False
    choices: Optional[list] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'type': self.type,
            'default': self.default,
            'description': self.description,
            'required': self.required,
            'choices': self.choices,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttrSpec':
        """从字典创建"""
        return cls(
            name=data.get('name', ''),
            type=data.get('type', ''),
            default=data.get('default'),
            description=data.get('description', ''),
            required=data.get('required', False),
            choices=data.get('choices'),
            metadata=data.get('metadata', {}),
        )


# === 输入/输出规格 ===

@dataclass
class InputSpec:
    """输入规格"""
    name: str
    dtype: str = ""
    shape: Optional[List[int]] = None
    description: str = ""
    optional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'dtype': self.dtype,
            'shape': self.shape,
            'description': self.description,
            'optional': self.optional,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InputSpec':
        return cls(
            name=data.get('name', ''),
            dtype=data.get('dtype', ''),
            shape=data.get('shape'),
            description=data.get('description', ''),
            optional=data.get('optional', False),
            metadata=data.get('metadata', {}),
        )


@dataclass
class OutputSpec:
    """输出规格"""
    name: str
    dtype: str = ""
    shape: Optional[List[int]] = None
    description: str = ""
    compare: bool = True
    # proto 声明 optional: true 的输出允许整体缺位（如 Unique 在
    # return_inverse=false 时只返回 y）；缺位合法性由与 golden 的输出
    # 数量/逐值比对裁决，dtype 契约闸对缺位的 optional 输出跳过。
    optional: bool = False
    # 索引类输出的"指向值"校验声明（如 TopK/ArgSort 的索引输出）。
    # 为 None 时不做索引校验。结构见 eval/index_check.py，例如：
    #   {'input': 'x', 'dim_attr': 'dim', 'value_output': 'y'}
    index_gather: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'dtype': self.dtype,
            'shape': self.shape,
            'description': self.description,
            'compare': self.compare,
            'optional': self.optional,
            'index_gather': self.index_gather,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutputSpec':
        return cls(
            name=data.get('name', ''),
            dtype=data.get('dtype', ''),
            shape=data.get('shape'),
            description=data.get('description', ''),
            compare=data.get('compare', True),
            optional=data.get('optional', False),
            index_gather=data.get('index_gather'),
            metadata=data.get('metadata', {}),
        )


# === 任务规格基类 ===

@dataclass
class TaskSpec:
    """任务规格基类"""
    task_id: str
    name: str
    rel_path: str = ""                      # 相对路径（用于定位算子目录）
    difficulty: DifficultyLevel = DifficultyLevel.L1
    description: str = ""
    category: str = ""                      # 算子类别（如 activation, normalization 等）
    inputs: List[InputSpec] = field(default_factory=list)
    outputs: List[OutputSpec] = field(default_factory=list)
    attrs: List[AttrSpec] = field(default_factory=list)
    reference: Optional[str] = None
    precision_thresholds: Dict[str, float] = field(default_factory=dict)  # 自定义精度阈值
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_level_id(self) -> int:
        # Why 不再用字面量表: 那张表与 DifficultyLevel 平行, 且 .get 默认值 1 会把
        # 新增级别静默算成 L1. 序号直接取自枚举成员名, 加级别时无需第二处改动.
        return int(self.difficulty.value[1:])

    def get_input_names(self) -> List[str]:
        return [inp.name for inp in self.inputs]

    def get_output_names(self) -> List[str]:
        return [out.name for out in self.outputs]

    def get_attr_names(self) -> List[str]:
        return [attr.name for attr in self.attrs]

    def get_compare_output_indices(self) -> List[int]:
        return [i for i, out in enumerate(self.outputs) if out.compare]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'name': self.name,
            'rel_path': self.rel_path,
            'difficulty': self.difficulty.value,
            'description': self.description,
            'category': self.category,
            'inputs': [inp.to_dict() for inp in self.inputs],
            'outputs': [out.to_dict() for out in self.outputs],
            'attrs': [attr.to_dict() for attr in self.attrs],
            'reference': self.reference,
            'precision_thresholds': self.precision_thresholds,
            'metadata': self.metadata,
        }


# === 用例规格基类 ===

@dataclass
class CaseSpec:
    """用例规格基类"""
    case_id: str
    operator: str = ""                     # 算子名称
    rel_path: str = ""                     # 相对路径
    case_num: int = 0                      # 用例编号
    baseline_perf_us: float = 0.0          # 基线性能（微秒）
    t_hw_us: float = 0.0                   # 理论硬件下界（微秒）
    input_shapes: List[List[int]] = field(default_factory=list)
    dtypes: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    value_ranges: List[Dict[str, float]] = field(default_factory=list)
    tolerance: Dict[str, float] = field(default_factory=lambda: {"rtol": 1e-4, "atol": 1e-4})
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_case_id_str(self) -> str:
        """获取用例标识字符串（基类默认实现）"""
        return self.case_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            'case_id': self.case_id,
            'operator': self.operator,
            'rel_path': self.rel_path,
            'case_num': self.case_num,
            'baseline_perf_us': self.baseline_perf_us,
            't_hw_us': self.t_hw_us,
            'input_shapes': self.input_shapes,
            'dtypes': self.dtypes,
            'attrs': self.attrs,
            'value_ranges': self.value_ranges,
            'tolerance': self.tolerance,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CaseSpec':
        """从字典重建"""
        return cls(
            case_id=data.get('case_id', ''),
            operator=data.get('operator', ''),
            rel_path=data.get('rel_path', ''),
            case_num=int(data.get('case_num', 0)),
            baseline_perf_us=float(data.get('baseline_perf_us', 0.0)),
            t_hw_us=float(data.get('t_hw_us', 0.0)),
            input_shapes=data.get('input_shapes', []),
            dtypes=data.get('dtypes', []),
            attrs=data.get('attrs', {}),
            value_ranges=data.get('value_ranges', []),
            tolerance=data.get('tolerance', {"rtol": 1e-4, "atol": 1e-4}),
            metadata=data.get('metadata', {}),
        )


# === 解决方案规格基类 ===

@dataclass
class SolutionSpec:
    """解决方案规格（通用基类）

    定义待评测的实现方案。

    支持场景:
    - AI 生成的算子实现
    - 手动编写的 kernel
    - 已有的库实现
    - Golden 实现自验证

    通用字段:
    - solution_id: 解决方案标识
    - task_id: 关联任务
    - name: 解决方案名称
    - backend: Backend 类型
    - source_type: 源码类型
    - source: 源码路径或内容
    - description: 描述
    - metadata: 扩展字段
    """
    solution_id: str                    # 解决方案标识
    task_id: str                        # 关联任务
    name: str = ""                      # 解决方案名称
    backend: BackendType = BackendType.TORCH_NPU  # Backend 类型
    source_type: SourceType = SourceType.FILE  # 源码类型
    source: str = ""                    # 源码路径/内容/模块名
    description: str = ""               # 描述信息
    metadata: Dict[str, Any] = field(default_factory=dict)  # 扩展字段

    def is_file_source(self) -> bool:
        """是否为文件来源"""
        return self.source_type == SourceType.FILE

    def is_code_source(self) -> bool:
        """是否为代码内容"""
        return self.source_type == SourceType.CODE or self.source_type == SourceType.GENERATED

    def is_module_source(self) -> bool:
        """是否为模块来源"""
        return self.source_type == SourceType.MODULE

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'solution_id': self.solution_id,
            'task_id': self.task_id,
            'name': self.name,
            'backend': self.backend.value,
            'source_type': self.source_type.value,
            'source': self.source,
            'description': self.description,
            'metadata': self.metadata,
        }
