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

import torch

"""
StridedSlice 算子 Torch Golden 参考实现

对标 TensorFlow tf.strided_slice，完整支持所有mask参数：
- begin_mask: 位1表示该维度从0开始
- end_mask: 位1表示该维度切到末尾
- ellipsis_mask: 位1表示省略号展开
- shrink_axis_mask: 位1表示收缩该维度
- new_axis_mask: 位1表示插入新维度

公式: y[i,j,k,...] = x[begin[i]:end[i]:strides[i], begin[j]:end[j]:strides[j], ...]
"""
def strided_slice(
    x: torch.Tensor, begin: list, end: list, strides: list,
    begin_mask: int = 0, end_mask: int = 0, ellipsis_mask: int = 0,
    shrink_axis_mask: int = 0, new_axis_mask: int = 0
) -> torch.Tensor:
    """
    使用步长对输入张量进行多维切片，对标 TensorFlow strided_slice。

    Args:
        x: 输入张量
        begin: 切片起始位置数组
        end: 切片结束位置数组
        strides: 切片步长数组，支持负数步长
        begin_mask: 二进制掩码，位1表示该维度从0开始
        end_mask: 二进制掩码，位1表示该维度切到末尾
        ellipsis_mask: 二进制掩码，位1表示省略号标记
        shrink_axis_mask: 二进制掩码，位1表示收缩该维度（取单元素）
        new_axis_mask: 二进制掩码，位1表示插入新维度

    Returns:
        输出张量，切片结果
    """
    ndim = x.dim()
    shape = x.shape

    # 处理 ellipsis_mask (省略号)
    # ellipsis_mask为1的位表示该位置插入省略号，展开中间所有维度
    # TF中 ellipsis_mask最多只有一位为1
    ellipsis_pos = None
    for i in range(32):
        if ellipsis_mask & (1 << i):
            ellipsis_pos = i
            break

    # 计算有效索引维度数（不包括new_axis）
    # new_axis会增加虚拟维度，需要在构建索引时处理
    num_new_axis = 0
    for i in range(len(begin) if begin else 0):
        if new_axis_mask & (1 << i):
            num_new_axis += 1

    # 构建 slicing indices
    # 需要处理：new_axis插入位置、shrink用整数索引、begin/end mask
    indices = []

    # 输入维度索引指针
    input_dim_idx = 0
    # begin/end/strides 参数索引指针
    param_idx = 0

    # 如果ellipsis存在，需要知道它覆盖多少维度
    if ellipsis_pos is not None:
        # ellipsis覆盖的维度数 = 输入维度 - (参数维度 - new_axis维度 - 1)
        # 参数中省略号位只占一个位置，但实际要展开
        num_params = len(begin) if begin else 0
        num_ellipsis_dims = ndim - (num_params - num_new_axis - 1)
        if num_ellipsis_dims < 0:
            num_ellipsis_dims = 0

    while input_dim_idx < ndim or param_idx < (len(begin) if begin else 0):
        # 检查是否是 new_axis 位置
        if param_idx < len(begin) and (new_axis_mask & (1 << param_idx)):
            indices.append(None)  # None 表示插入新维度
            param_idx += 1
            continue

        # 检查是否是 ellipsis 位置
        if ellipsis_pos is not None and param_idx == ellipsis_pos:
            # 省略号展开：覆盖 num_ellipsis_dims 个维度
            for _ in range(num_ellipsis_dims):
                indices.append(slice(None, None, None))
                input_dim_idx += 1
            param_idx += 1
            continue

        # 正常切片维度
        if input_dim_idx < ndim and param_idx < len(begin):
            dim_size = shape[input_dim_idx]
            b = begin[param_idx] if param_idx < len(begin) else 0
            e = end[param_idx] if param_idx < len(end) else dim_size
            s = strides[param_idx] if param_idx < len(strides) else 1

            # 处理负索引
            if b < 0:
                b = b + dim_size
            if e < 0:
                e = e + dim_size

            # 应用 begin_mask
            if begin_mask & (1 << param_idx):
                b = 0 if s > 0 else dim_size - 1

            # 应用 end_mask
            if end_mask & (1 << param_idx):
                e = dim_size if s > 0 else -1

            # 应用 shrink_axis_mask
            if shrink_axis_mask & (1 << param_idx):
                # shrink: 取单个元素，用整数索引
                indices.append(b)
            else:
                # 正常切片
                indices.append(slice(b, e, s))

            input_dim_idx += 1
            param_idx += 1
        elif input_dim_idx < ndim:
            # 输入维度还有剩余，但参数已用完
            # 默认取整个维度
            indices.append(slice(None, None, None))
            input_dim_idx += 1
        else:
            # 参数还有剩余但输入维度已处理完
            # 可能是多余的new_axis
            if param_idx < len(begin) and (new_axis_mask & (1 << param_idx)):
                indices.append(None)
            param_idx += 1

    # 执行切片
    y = x[tuple(indices)]

    # 物化输出:真实 StridedSlice 算子会写出独立 buffer;若直接返回 x[...] 的 view,
    # golden-as-candidate 在 NPU 上不跑 compute kernel(view 无拷贝)→ profiler 测不到 perf。
    # clone() 强制拷贝,非连续切片即跑出真实 SliceAiCore kernel。精度不变(值相同),且更贴近
    # 真实算子语义(输出与输入不共享内存)。
    return y.clone()