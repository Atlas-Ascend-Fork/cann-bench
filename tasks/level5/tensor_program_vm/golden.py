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
TensorProgramVm 算子 Torch Golden 参考实现

张量程序虚拟机：kernel 即解释器，程序即数据。输入 program [P, 4] int32 是一段
待执行的指令序列，init_regs [R, W] 是寄存器堆（R 个行向量寄存器，每个宽 W）。
对 t = 0..P-1 顺序执行，译码规则（mod 保证任意非负 int32 都是合法指令）:
    op  = program[t, 0] mod 8
    dst = program[t, 1] mod R
    a   = program[t, 2] mod R
    b   = program[t, 3] mod R
ISA（所有操作数为寄存器堆的行向量 [W]，sat(x) = clamp(x, -10000, 10000)）:
    0 ADD:    regs[dst] = sat(regs[a] + regs[b])
    1 MUL:    regs[dst] = sat(regs[a] * regs[b])
    2 RELU:   regs[dst] = max(regs[a], 0)                （b 忽略）
    3 MEANB:  regs[dst] = broadcast(mean(regs[a]))       （沿 W 求均值后广播；b 忽略）
    4 MAX:    regs[dst] = elementwise_max(regs[a], regs[b])
    5 ROLL:   regs[dst][j] = regs[a][(j-1+W) mod W]      （向右循环移 1 位；b 忽略）
    6 MULADD: regs[dst] = sat(regs[dst] + regs[a] * regs[b])   （读 dst，写 dst）
    7 COPY:   regs[dst] = regs[a]                        （b 忽略）
输出 final_regs = 全部指令执行完后的寄存器堆 [R, W]。
饱和仅作用于 ADD/MUL/MULADD，保证任意随机程序数值有界（|值| ≤ max(10000, |init|)）、
结果良定义。指令间存在真依赖（后一条可能读前一条的结果），golden 逐指令顺序执行；
每条指令是 [W] 向量操作，寄存器堆全程张量化。
plain golden 内部升 fp32 计算（bench 语义：寄存器堆全程 fp32 驻留，仅输出转回输入
dtype）；tensor_program_vm_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为
fp64 真值），不硬编码 .float()。
"""

_SAT_BOUND = 10000.0


def _tensor_program_vm_core(program, init_regs, compute_dtype):
    """核心计算：以 compute_dtype 精度逐指令顺序执行，返回 final_regs [R, W]。"""
    P = program.shape[0]
    R, W = init_regs.shape

    prog = program.to(torch.int64)
    op_v = (prog[:, 0] % 8).tolist()
    dst_v = (prog[:, 1] % R).tolist()
    a_v = (prog[:, 2] % R).tolist()
    b_v = (prog[:, 3] % R).tolist()

    hi = torch.tensor(_SAT_BOUND, dtype=compute_dtype, device=init_regs.device)
    lo = torch.tensor(-_SAT_BOUND, dtype=compute_dtype, device=init_regs.device)
    zero = torch.zeros((), dtype=compute_dtype, device=init_regs.device)

    regs = init_regs.to(compute_dtype).clone()
    for t in range(P):
        op, dst, a, b = op_v[t], dst_v[t], a_v[t], b_v[t]
        if op == 0:                                          # ADD
            regs[dst] = torch.clamp(regs[a] + regs[b], lo, hi)
        elif op == 1:                                        # MUL
            regs[dst] = torch.clamp(regs[a] * regs[b], lo, hi)
        elif op == 2:                                        # RELU
            regs[dst] = torch.maximum(regs[a], zero)
        elif op == 3:                                        # MEANB（沿 W 求均值后广播）
            regs[dst] = regs[a].mean().expand(W).clone()
        elif op == 4:                                        # MAX
            regs[dst] = torch.maximum(regs[a], regs[b])
        elif op == 5:                                        # ROLL（向右循环移 1 位）
            regs[dst] = torch.roll(regs[a], shifts=1, dims=0)
        elif op == 6:                                        # MULADD（读 dst，写 dst）
            regs[dst] = torch.clamp(regs[dst] + regs[a] * regs[b], lo, hi)
        else:                                                # 7 COPY
            regs[dst] = regs[a].clone()
    return regs


def tensor_program_vm(program: torch.Tensor, init_regs: torch.Tensor) -> torch.Tensor:
    """
    张量程序虚拟机 golden reference（逐指令顺序执行；plain golden = bench：fp32 寄存器堆）

    Args:
        program: [P, 4] int32 指令序列，任意非负整数合法（评测取值范围 [0, 1073741823]），
            每行经 mod 译码为 (op, dst, a, b)
        init_regs: [R, W] float32/float16/bfloat16 寄存器堆初值（评测取值范围 [-1, 1]）

    Returns:
        final_regs: [R, W] 全部指令执行完后的寄存器堆，dtype 与 init_regs 一致
    """
    out = _tensor_program_vm_core(program, init_regs, torch.float32)
    return out.to(init_regs.dtype)


def tensor_program_vm_oracle(program: torch.Tensor, init_regs: torch.Tensor) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _tensor_program_vm_core(program, init_regs, init_regs.dtype)
