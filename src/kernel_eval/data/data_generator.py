#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
测试数据生成器

职责：
1. 根据shape和dtype生成输入张量
2. 根据value_range填充数据
3. 支持确定性种子（通过 torch.Generator）确保评测可复现
"""

import math
from typing import Any, List, Optional

import torch

from ..utils.dtype_mapper import str_to_torch_dtype, is_float_dtype, is_int_dtype, is_bool_dtype


#: 支持的输入数据分布。键为规范名，值为可接受的 CLI 别名。
INPUT_DISTRIBUTIONS = {
    "uniform": ("uniform", "u"),
    "normal": ("normal", "norm", "gaussian", "n"),
}

#: 别名 -> 规范名
_DIST_ALIASES = {a: canon for canon, aliases in INPUT_DISTRIBUTIONS.items() for a in aliases}

#: argparse choices 的单一来源——由 INPUT_DISTRIBUTIONS 派生，新增分布时无需再改 CLI
INPUT_DIST_CHOICES = sorted(_DIST_ALIASES)


def normalize_input_dist(dist: Any) -> str:
    """把 CLI/配置传入的分布名规整为规范名，非法值直接报错

    >>> normalize_input_dist("norm")
    'normal'
    """
    if dist is None:
        return "uniform"
    key = str(dist).strip().lower()
    if key not in _DIST_ALIASES:
        raise ValueError(
            f"不支持的输入分布: {dist!r}。可选: "
            + "、".join(f"{c}（别名 {', '.join(a[1:])}）" if len(a) > 1 else c
                        for c, a in INPUT_DISTRIBUTIONS.items())
        )
    return _DIST_ALIASES[key]


class DataGenerator:
    """测试数据生成器

    Args:
        seed: 全局随机种子（可选）
        input_dist: 浮点输入的采样分布，"uniform"（默认）或 "normal"（别名 norm /
            gaussian）。默认值走的是与历史完全相同的代码路径与随机数流，生成结果
            逐位一致。整数 / bool 输入不受本项影响，始终均匀采样。
    """

    def __init__(self, seed: int = None, input_dist: str = "uniform"):
        if seed is not None:
            torch.manual_seed(seed)
        self.input_dist = normalize_input_dist(input_dist)

    def generate_input_tensor(self, shape: List[int], dtype: str, value_range: Any = None,
                              generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """生成输入张量

        Args:
            shape: 张量形状
            dtype: 数据类型字符串
            value_range: 值范围 [min, max] 或 [[min1, max1], ...]
            generator: torch.Generator 用于确定性随机数生成（可选）
        """
        # 支持 shape 为 None 或包含 None 的情况，表示不使用该参数
        if shape is None:
            return None
        if isinstance(shape, list) and None in shape:
            return None

        torch_dtype = str_to_torch_dtype(dtype)

        # 唯一整数索引（D9.2，scatter reduce=null 等）：value_range 形如
        # ["unique", min, max]，生成 [min, max] 内无重复的随机整数。
        if self._is_unique_range(value_range):
            return self._gen_unique_int(shape, torch_dtype, dtype,
                                        value_range[1], value_range[2],
                                        generator=generator)

        min_val, max_val = self._parse_range(value_range, dtype)

        if is_float_dtype(dtype):
            # 特殊值（NaN/inf）必须优先处理，避免 'nan' == 'nan' 走全量填充路径
            if isinstance(min_val, str) or isinstance(max_val, str):
                return self._gen_float(shape, torch_dtype, min_val, max_val, generator=generator)
            if self._is_special_float(min_val) or self._is_special_float(max_val):
                return self._gen_float(shape, torch_dtype, min_val, max_val, generator=generator)
            if min_val == max_val:
                return torch.full(shape, float(min_val), dtype=torch_dtype)
            return self._gen_float(shape, torch_dtype, float(min_val), float(max_val), generator=generator)
        elif is_int_dtype(dtype):
            if min_val == max_val:
                return torch.full(shape, int(min_val), dtype=torch_dtype)
            min_int, max_int = int(min_val), int(max_val)
            int64_min = -(1 << 63)
            int64_max = (1 << 63) - 1
            if (
                dtype.lower() == "int64"
                and max_int == int64_max
                and int64_min <= min_int < max_int
            ):
                return self._gen_int64_inclusive_max(
                    shape, min_int, generator=generator
                )
            # torch.randint 支持 generator 参数（generator=None 时等同默认 RNG），
            # 直接透传即可保证确定性，无需 rand 缩放
            return torch.randint(
                min_int,
                max_int + 1,
                shape,
                generator=generator,
                dtype=torch_dtype,
            )
        elif is_bool_dtype(dtype):
            return self._gen_bool(shape, min_val, max_val, generator=generator)
        else:
            return torch.zeros(shape, dtype=torch_dtype)

    def generate_input_tensors_from_case(self, input_shapes: List, dtypes: List, value_ranges: List,
                                         seed: Optional[int] = None) -> List:
        """根据用例信息生成输入数据

        假设输入已由 CaseLoader 规范化：
        - input_shapes: 嵌套列表 [[shape1], [shape2], ...]
        - dtypes: 列表 [dtype1, dtype2, ...]
        - value_ranges: 列表 [[min1, max1], [min2, max2], ...]

        Args:
            input_shapes: 输入形状列表
            dtypes: 数据类型列表
            value_ranges: 值范围列表
            seed: 确定性种子（可选）。设置后使用 torch.Generator 确保可复现。
                  None 表示使用全局随机状态（不保证可复现）。
        """
        input_tensors = []
        num_inputs = self._count_inputs(input_shapes)

        # 创建 local Generator 用于确定性种子
        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)

        # 扩展dtypes
        if not isinstance(dtypes, list) or (dtypes and isinstance(dtypes[0], str) and len(dtypes) < num_inputs):
            if isinstance(dtypes, str):
                dtypes = [dtypes]
            elif not isinstance(dtypes, list):
                dtypes = ['float32']
        if isinstance(dtypes, list) and len(dtypes) < num_inputs:
            dtypes = dtypes + [dtypes[-1]] * (num_inputs - len(dtypes)) if dtypes else ['float32'] * num_inputs

        # 规范化 value_ranges: 区分单输入 [min, max] 和多输入 [[min1, max1], [min2, max2], ...]
        value_ranges = self._normalize_value_ranges(value_ranges, num_inputs)

        self._generate_tensors(input_shapes, dtypes, value_ranges, input_tensors, generator=gen)

        return input_tensors

    def _normalize_value_ranges(self, value_ranges: List, num_inputs: int) -> List:
        """规范化 value_ranges 为每个输入一个范围"""
        if not value_ranges:
            return [None] * num_inputs

        # 判断是标量区间 [min, max] 还是逐输入 [[min1, max1], ...]
        first_item = value_ranges[0]
        is_single_range = not isinstance(first_item, list)

        if is_single_range:
            # 标量区间 [min, max]：广播到**全部**输入。
            #
            # 此前写作 `[value_ranges] + [None] * (num_inputs - 1)`，注释标的是
            # "单输入算子"——该分支只考虑了单输入场景。而 cases.yaml 里
            # `value_range: [-1, 1]` 是"所有输入同一区间"的简写，7 个多输入算子
            # （RmsNorm / GroupNorm / MHA / GQA / MLA / MlaProlog /
            # SparseFlashAttention，共 140 个用例）都用了这种写法，结果除第 1 个
            # 输入外全部拿不到声明区间，静默落到 _parse_range 的默认值
            # （float [0,1] / int [0,100]）——注意默认值是**非负**的：
            #   MHA/GQA  的 key / value 落在 [0,1]，注意力少了抵消项覆盖；
            #   MLA      5 个输入里 4 个非负；
            #   MlaProlog 9 个输入里 8 个非负（含 rope_sin/rope_cos 只剩第一象限）；
            #   RmsNorm/GroupNorm 的 gamma / beta 非负，而用例 note 写的是
            #   "symmetric-small"（对称小值域）——声明与实际直接矛盾。
            # 逐输入区间不同的场景仍走下面的 list-of-list 形式，不受影响。
            return [value_ranges] * num_inputs
        else:
            # 多输入算子: value_ranges 是 [[min1, max1], [min2, max2], ...]
            if len(value_ranges) < num_inputs:
                value_ranges = value_ranges + [value_ranges[-1]] * (num_inputs - len(value_ranges))
            return value_ranges

    @staticmethod
    def _gen_int64_inclusive_max(
        shape: List[int],
        min_val: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """在包含 INT64_MAX 的闭区间内均匀生成 int64。"""
        int64_min = -(1 << 63)
        int64_max = (1 << 63) - 1
        half_range = 1 << 63
        interval_size = int64_max - min_val + 1

        # torch.randint 的上界是排他的且必须能用有符号 long 表示。
        # 对不超过 2**63 个值的区间，直接在偏移量域采样。
        if interval_size < half_range:
            offsets = torch.randint(
                0, interval_size, shape, generator=generator, dtype=torch.int64
            )
            return offsets + min_val
        if interval_size == half_range:
            offsets = torch.randint(
                int64_min, 0, shape, generator=generator, dtype=torch.int64
            )
            return offsets.bitwise_xor(int64_min)

        # 更宽的区间由两个等大的 2**63 域组成。对大于请求区间的偏移拒绝重采，
        # 使剩余偏移在整个闭区间内保持均匀；分块限制额外工作内存。
        remainder = interval_size - half_range
        upper_base = min_val - int64_min
        result = torch.empty(shape, dtype=torch.int64)
        flat_result = result.reshape(-1)
        chunk_size = 1 << 20

        for start in range(0, flat_result.numel(), chunk_size):
            end = min(start + chunk_size, flat_result.numel())
            chunk_shape = (end - start,)
            offsets = torch.randint(
                int64_min, 0, chunk_shape, generator=generator, dtype=torch.int64
            ).bitwise_xor(int64_min)
            upper_half = torch.randint(
                0, 2, chunk_shape, generator=generator, dtype=torch.int8
            )

            if remainder < half_range:
                invalid = upper_half.bool() & (offsets >= remainder)
                while bool(invalid.any().item()):
                    replacement_offsets = torch.randint(
                        int64_min,
                        0,
                        chunk_shape,
                        generator=generator,
                        dtype=torch.int64,
                    ).bitwise_xor(int64_min)
                    replacement_half = torch.randint(
                        0, 2, chunk_shape, generator=generator, dtype=torch.int8
                    )
                    offsets = torch.where(invalid, replacement_offsets, offsets)
                    upper_half = torch.where(invalid, replacement_half, upper_half)
                    invalid = upper_half.bool() & (offsets >= remainder)

                # Mask rejected offsets before adding, so even unused intermediate values
                # remain within the signed int64 range.
                upper_offsets = torch.where(
                    offsets < remainder, offsets, torch.zeros_like(offsets)
                )
            else:
                # The full int64 domain: both halves are valid.
                upper_offsets = offsets

            lower_values = offsets + min_val
            upper_values = upper_offsets + upper_base
            flat_result[start:end] = torch.where(
                upper_half.bool(), upper_values, lower_values
            )

        return result

    def _parse_range(self, value_range: Any, dtype: str) -> tuple:
        """解析值范围"""
        if value_range is None:
            return (0.0, 1.0) if is_float_dtype(dtype) else (0, 100)

        if isinstance(value_range, list):
            if len(value_range) >= 2:
                return (self._coerce_endpoint(value_range[0]),
                        self._coerce_endpoint(value_range[1]))
            elif len(value_range) == 1:
                v = self._coerce_endpoint(value_range[0])
                return (v, v)

        v = self._coerce_endpoint(value_range)
        return (v, v)

    @staticmethod
    def _coerce_endpoint(val: Any) -> Any:
        """把数值型字符串端点归一为 float。

        YAML 1.1 把无符号指数的科学计数法（如 ``-1.0e38``，缺 ``e+38`` 的
        正号）解析成**字符串**而非 float。若不归一，这类端点会被误路由到
        特殊值路径 ``_gen_special``——它只按 [-1, 1] 生成普通样本再 clamp，
        声明的近溢出区间被静默替换掉。``nan`` / ``inf`` 特殊标记保持字符串
        原样，仍由既有特殊值路径处理（随机数消费顺序不变）。
        """
        if not isinstance(val, str):
            return val
        if val.strip().lower() in ("nan", "inf", "-inf", "+inf"):
            return val
        try:
            return float(val)
        except ValueError:
            return val

    @staticmethod
    def _is_unique_range(value_range: Any) -> bool:
        """检测唯一整数值域声明 ["unique", min, max]"""
        return (isinstance(value_range, list) and len(value_range) == 3
                and value_range[0] == "unique")

    def _gen_unique_int(self, shape: List[int], torch_dtype: torch.dtype, dtype: str,
                        min_val: Any, max_val: Any,
                        generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """生成 [min, max] 闭区间内无重复的随机整数张量（D9.2）。

        要求 dtype 为整数且区间容量 >= 元素数；不满足时大声失败而非静默回退，
        保证"索引唯一"这一 case 声明不会被悄悄破坏。
        """
        if not is_int_dtype(dtype):
            raise ValueError(f"unique 值域仅支持整数 dtype，收到 {dtype}")
        lo, hi = int(min_val), int(max_val)
        numel = 1
        for s in shape:
            numel *= int(s)
        span = hi - lo + 1
        if span < numel:
            raise ValueError(
                f"unique 值域容量不足: 区间 [{lo}, {hi}] 共 {span} 个值，"
                f"张量需要 {numel} 个唯一值")
        perm = torch.randperm(span, generator=generator)[:numel]
        return (perm + lo).reshape(shape).to(torch_dtype)

    @staticmethod
    def _is_special_float(val: Any) -> bool:
        """检测 float NaN/inf 特殊值"""
        import math
        return isinstance(val, float) and (math.isnan(val) or math.isinf(val))

    def _is_tensor_list(self, shape_item: Any) -> bool:
        """判断是否为张量列表"""
        return (isinstance(shape_item, list) and shape_item and
                isinstance(shape_item[0], list) and shape_item[0] and
                isinstance(shape_item[0][0], int))

    def _gen_float(self, shape: List[int], dtype: torch.dtype, min_val: float, max_val: float,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """生成浮点张量

        Args:
            shape: 张量形状
            dtype: torch 数据类型
            min_val: 最小值
            max_val: 最大值
            generator: torch.Generator 用于确定性随机数生成（可选）
        """
        # 处理特殊值（字符串形式或 float NaN/inf）
        if isinstance(min_val, str) or isinstance(max_val, str):
            return self._gen_special(shape, dtype, min_val, max_val, generator=generator)
        if self._is_special_float(min_val) or self._is_special_float(max_val):
            return self._gen_special(shape, dtype, min_val, max_val, generator=generator)

        # 使用 torch.finfo 获取 dtype 的精确范围
        finfo = torch.finfo(dtype)
        dmin = finfo.min
        dmax = finfo.max

        # 裁剪到 dtype 可表示范围
        min_val = max(min_val, dmin)
        max_val = min(max_val, dmax)

        # 正态分布：仅在显式指定时走这条路。input_dist == "uniform"（默认）时下面的
        # 均匀路径与历史完全一致——包括随机数消费顺序，故既有评测输入逐位不变。
        if self.input_dist == "normal" and max_val > min_val:
            return self._gen_float_normal(shape, dtype, min_val, max_val, generator=generator)

        # 统一使用 torch.rand + 缩放 路径，支持 generator 参数
        # 使用 float64 中间值；端点跨度溢出时改用凸组合计算。
        range_val = max_val - min_val
        rand_f64 = torch.rand(shape, dtype=torch.float64, generator=generator)
        if math.isfinite(range_val):
            tensor_f64 = rand_f64 * range_val + min_val
        else:
            # 两个有限的 float64 端点跨过零时，端点差可能溢出；凸组合不需要该差值。
            tensor_f64 = (1.0 - rand_f64) * min_val + rand_f64 * max_val
        # clamp 确保值严格在 dtype 范围内，避免转换溢出
        tensor_f64 = torch.clamp(tensor_f64, dmin, dmax)
        return tensor_f64.to(dtype)

    def _gen_float_normal(self, shape: List[int], dtype: torch.dtype,
                          min_val: float, max_val: float,
                          generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """在声明区间内按正态分布生成浮点张量

        真实业务里的激活、权重多接近正态分布，而通用生成器只产出均匀分布——两者的
        尾部行为、动态范围、异常值密度都不同，kernel 在其上的精度表现也不同。

        分布参数由 cases.yaml 的 value_range 推出，不额外引入配置项：
            mu    = (min_val + max_val) / 2
            sigma = (max_val - min_val) / 6      # ±3σ 覆盖声明区间
        采样后 clamp 回 [min_val, max_val]：value_range 常带语义（索引边界、dtype
        上界等），不能因为换了分布就突破。代价是约 0.27% 的样本会被截断堆在边界上，
        这是保边界的必要取舍。

        与均匀路径同样通过 float64 中间量计算，避免 range 超出 dtype 时溢出。

        Args:
            shape: 张量形状
            dtype: torch 数据类型
            min_val: 区间下界（已裁剪到 dtype 可表示范围）
            max_val: 区间上界（已裁剪到 dtype 可表示范围）
            generator: torch.Generator 用于确定性随机数生成（可选）
        """
        finfo = torch.finfo(dtype)
        range_val = max_val - min_val
        if math.isfinite(range_val):
            mu = min_val + range_val / 2.0
            sigma = range_val / 6.0
        else:
            # 端点差溢出时，用半量计算相同的中点和标准差，避免中间值溢出。
            mu = min_val / 2.0 + max_val / 2.0
            sigma = max_val / 6.0 - min_val / 6.0

        z = torch.randn(shape, dtype=torch.float64, generator=generator)
        tensor_f64 = z * sigma + mu
        # 先夹回声明区间，再夹到 dtype 可表示范围，避免转换溢出
        tensor_f64 = torch.clamp(tensor_f64, min_val, max_val)
        tensor_f64 = torch.clamp(tensor_f64, finfo.min, finfo.max)
        return tensor_f64.to(dtype)

    def _gen_special(self, shape: List[int], dtype: torch.dtype, min_val: Any, max_val: Any,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """生成包含特殊值的张量

        Args:
            shape: 张量形状
            dtype: torch 数据类型
            min_val: 最小值（可能是特殊值字符串或 float NaN/inf）
            max_val: 最大值（可能是特殊值字符串或 float NaN/inf）
            generator: torch.Generator 用于确定性随机数生成（可选）
        """
        import math

        # 转换特殊值字符串
        def to_float(v):
            if v == 'inf': return float('inf')
            if v == '-inf': return float('-inf')
            if v == 'nan': return float('nan')
            return float(v)

        min_f = to_float(min_val) if isinstance(min_val, str) else min_val
        max_f = to_float(max_val) if isinstance(max_val, str) else max_val

        # 处理 NaN 范围：混合 NaN 与正常随机值，避免全 NaN 触发 kernel 边界 bug
        if math.isnan(min_f) and math.isnan(max_f):
            if generator is not None:
                # 确定性路径：使用 torch.rand + generator 生成基础值
                rand_base = torch.rand(shape, dtype=torch.float32, generator=generator)
                tensor = (rand_base * 2 - 1).to(dtype)  # 简化的正态近似
            else:
                tensor = torch.randn(shape, dtype=torch.float32).to(dtype)
            # 使用确定性 seed 生成 NaN mask
            if generator is not None:
                nan_rand = torch.rand(shape, dtype=torch.float32, generator=generator)
                nan_mask = nan_rand < 0.5
            else:
                nan_mask = torch.rand(shape, dtype=torch.float32) < 0.5
            tensor[nan_mask] = float('nan')
            return tensor

        # 闭区间退化为单个无穷值时，所有样本都必须保持该值。
        if min_f == max_f and (math.isinf(min_f) or math.isinf(max_f)):
            return torch.full(shape, min_f, dtype=dtype)

        if generator is not None:
            rand_base = torch.rand(shape, dtype=torch.float32, generator=generator)
            tensor = (rand_base * 2 - 1).to(dtype)
        else:
            tensor = torch.randn(shape, dtype=torch.float32).to(dtype)

        # 有限端点仍约束普通样本；无穷只注入它实际所在的那一侧。
        # 例如 [0.1, +inf] 不能混入负值或 -inf，否则会违背用例声明的输入范围。
        finite_min = min_f if math.isfinite(min_f) else None
        finite_max = max_f if math.isfinite(max_f) else None
        if finite_min is not None or finite_max is not None:
            tensor = torch.clamp(tensor, min=finite_min, max=finite_max)

        # 在对应边界填充特殊值。
        flat = tensor.flatten()
        n = max(1, len(flat) // 20)
        if min_f == float('-inf'):
            flat[:n] = float('-inf')
        if max_f == float('inf'):
            flat[-n:] = float('inf')

        return tensor

    def _gen_bool(self, shape: List[int], min_val: Any, max_val: Any,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """生成布尔张量

        bool 既不属于 is_float_dtype 也不属于 is_int_dtype，此前会落到 zeros 分支，
        导致所有 bool 输入被静默生成为全 False、value_range 完全失效（例如
        MoeGatingTopKSoftmax 的 finished 恒为 False，"已完成 token" 分支从未被覆盖）。

        torch.randint 不接受 dtype=torch.bool，故先按 int64 采样再转 bool。
        value_range 被夹到 {0, 1}：[0, 1] → 随机；[0, 0] → 全 False；[1, 1] → 全 True。
        未指定 value_range 时 _parse_range 返回 (0, 100)，夹取后等价于 [0, 1] 随机。

        Args:
            shape: 张量形状
            min_val: 最小值（夹到 0/1）
            max_val: 最大值（夹到 0/1）
            generator: torch.Generator 用于确定性随机数生成（可选）
        """
        lo = min(1, max(0, int(min_val)))
        hi = min(1, max(0, int(max_val)))
        if lo > hi:
            lo, hi = hi, lo

        if lo == hi:
            return torch.full(shape, bool(lo), dtype=torch.bool)
        return torch.randint(
            lo, hi + 1, shape, generator=generator, dtype=torch.int64,
        ).to(torch.bool)

    # ---- 嵌套结构支持（GroupedMatmul 等算子） ----

    def _count_inputs(self, shapes: Any) -> int:
        """统计输入张量总数（顶层列表长度）"""
        if isinstance(sizes := shapes, list):
            return len(sizes)
        return 1

    def _generate_tensors(self, shapes, dtypes, value_ranges, output: list,
                          generator: Optional[torch.Generator] = None):
        """递归生成张量，支持嵌套结构

        处理三种情况：
        1. 单个形状 [N, C, H, W] -> 单个 tensor
        2. TensorList [[shape1], [shape2], ...] 其中每个 shape_i 是 [dims...] -> 保持为 list
        3. 扁平形状列表 [[N,C,H,W], [N,C,H,W], ...] -> 每个元素一个 tensor

        Args:
            shapes: 形状列表
            dtypes: 数据类型列表
            value_ranges: 值范围列表
            output: 输出张量列表（追加）
            generator: torch.Generator 用于确定性随机数生成（可选）
        """
        if not isinstance(shapes, list):
            return

        # 单个形状 [N, C, H, W] —— 直接生成单个张量
        if shapes and isinstance(shapes[0], int):
            dtype_item = dtypes if isinstance(dtypes, str) else (
                dtypes[0] if isinstance(dtypes, list) and dtypes else 'float32'
            )
            vr_item = value_ranges if (
                isinstance(value_ranges, list) and len(value_ranges) == 2
                and not isinstance(value_ranges[0], list)
            ) or self._is_unique_range(value_ranges) else None
            output.append(self.generate_input_tensor(shapes, dtype_item, vr_item, generator=generator))
            return

        # 判断每个元素是否为 TensorList（嵌套）还是单个 shape
        for i, s in enumerate(shapes):
            d = dtypes[i] if isinstance(dtypes, list) and i < len(dtypes) else dtypes
            vr = value_ranges[i] if isinstance(value_ranges, list) and i < len(value_ranges) else value_ranges

            if s is None:
                # None 形状占位符
                output.append(None)
            elif isinstance(s, list) and (not s or isinstance(s[0], int)):
                # s 是单个 shape [N, C, H, W] 或 rank-0 标量 shape [] -> 单个 tensor
                dtype_item = d if isinstance(d, str) else (d[0] if isinstance(d, list) and d else 'float32')
                vr_item = vr if (isinstance(vr, list) and len(vr) == 2 and not isinstance(vr[0], list)) \
                    or self._is_unique_range(vr) else None
                output.append(self.generate_input_tensor(s, dtype_item, vr_item, generator=generator))
            elif isinstance(s, list) and s and isinstance(s[0], list) and s[0] and isinstance(s[0][0], int):
                # s 是 TensorList [[shape1], [shape2], ...] 其中 shape_i 是 int list
                # 递归生成每个元素，保持为 list 结构
                sub_list = []
                sub_dtypes = d if isinstance(d, list) else d
                if isinstance(vr, list):
                    # case schema declares one range per logical input. For a
                    # TensorList input, broadcast a scalar [min, max] range to
                    # every member instead of dropping it and using defaults.
                    is_scalar_range = (
                        len(vr) == 2
                        and not isinstance(vr[0], (list, tuple))
                        and not isinstance(vr[1], (list, tuple))
                    )
                    sub_vr = [vr] * len(s) if is_scalar_range else vr
                else:
                    sub_vr = None
                for j, sub_shape in enumerate(s):
                    sub_d = sub_dtypes[j] if isinstance(sub_dtypes, list) and j < len(sub_dtypes) else sub_dtypes
                    if sub_vr and isinstance(sub_vr, list) and j < len(sub_vr):
                        item_vr = sub_vr[j]
                    else:
                        item_vr = sub_vr
                    sub_list.append(self.generate_input_tensor(sub_shape, sub_d, item_vr, generator=generator))
                output.append(sub_list)  # 保持为 TensorList
            elif isinstance(s, list) and s and isinstance(s[0], list) and s[0] and isinstance(s[0][0], list):
                # 三层嵌套：每个元素本身是 TensorList -> 递归处理
                sub_list = []
                for j, sub_item in enumerate(s):
                    sub_d = d[j] if isinstance(d, list) and j < len(d) else d
                    sub_vr_item = vr[j] if isinstance(vr, list) and j < len(vr) else vr
                    self._generate_tensors(sub_item, sub_d, sub_vr_item, sub_list, generator=generator)
                output.append(sub_list)
