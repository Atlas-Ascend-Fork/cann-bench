# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Elementwise add: y = x1 + x2.

dtype float16 / bfloat16 / float32 / int32, any rank (including 0-dim
scalars), same shape and dtype for both inputs.

Flatten to 1D, split into tiles across AI cores, loop tiles per core and
process each tile in VL-lane vector segments with tail masking.

Dynamic shape: the element count n is a runtime kernel argument bound to a
``Dim("n")`` leaf of the host tensor spec, so one compiled binary serves
every shape (no per-shape recompilation). The core count is queried from
the CANN runtime at trace time (``get_platform_info().core_num``, stream
core-quota aware) instead of being hardcoded. The compiled binary goes
through the DSL content-addressed compile cache: the memory tier is always
on and the disk tier is enabled here via ``CANNBOTDSL_CACHE_DIR``, so a
second process reuses the cached .so instead of recompiling.
"""

import os

import torch

# Opt in to the DSL disk compile cache (L2) before the first compilation.
# The memory tier (L1) is on by default; the disk tier only activates when
# CANNBOTDSL_CACHE_DIR is set. Pre-existing user configuration wins.
os.environ.setdefault("CANNBOTDSL_CACHE_DIR", os.path.expanduser("~/.cache/cannbotdsl"))

import cannbotdsl
from cannbotdsl import Dim, TensorSpec, dtypes, get_mem_size, get_platform_info
from cannbotdsl.aot.cache import compile_cache
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    RoundingMode,
    full_mask,
    mask_counter,
    update_mask,
    vadd,
    vcast,
    vload,
    vload_unpack,
    vstore,
    vstore_pack,
)
from cannbotdsl.tensor import MemLoc, Tensor, tile_slice

VL = 64
UB_RETAIN_BYTES = 1024

_TORCH_TO_DSL = {
    torch.float16: dtypes.float16,
    torch.bfloat16: dtypes.bfloat16,
    torch.float32: dtypes.float32,
    torch.int32: dtypes.int32,
}

_IS_16BIT = (dtypes.float16, dtypes.bfloat16)


def _choose_tile(dtype):
    elem_bytes = 2 if dtype in _IS_16BIT else 4
    budget = (get_mem_size("ub") - UB_RETAIN_BYTES) // (3 * elem_bytes)
    return max(VL, (budget // VL) * VL)


def _core_count():
    """Actual AI core count from the CANN runtime.

    Clamped by the current stream's core quota (rtsGetStreamResLimit), so
    tiling works against what the launch may actually use. Only sensible
    with a CANN runtime attached; kernels cannot run without one anyway.
    """
    core_num = get_platform_info().core_num
    if core_num <= 0:
        raise RuntimeError(
            "cannot determine the AI core count; is the CANN runtime set up?"
        )
    return core_num


@kernel
class VecAddKernel:
    def __init__(self, dtype=dtypes.float32):
        self._dtype = dtype
        self._is_16bit = dtype in _IS_16BIT
        self._tile = _choose_tile(dtype)
        self._seg = self._tile // VL

        self._x = Buffer(MemLoc.UB, (self._tile,), dtype)
        self._y = Buffer(MemLoc.UB, (self._tile,), dtype)
        self._o = Buffer(MemLoc.UB, (self._tile,), dtype)

    def __call__(self, gm_x: Tensor, gm_y: Tensor, gm_out: Tensor, n):
        tile = self._tile
        bi = get_block_idx()
        bn = get_block_num()
        n_tiles = (n + tile - 1) // tile

        for t in cannbotdsl.range(bi, n_tiles, bn):
            remaining = n - t * tile

            mem_copy(self._x, tile_slice(gm_x, (tile,), (t,)))
            mem_copy(self._y, tile_slice(gm_y, (tile,), (t,)))

            self._add(self._x, self._y, self._o, remaining)

            mem_copy(tile_slice(gm_out, (tile,), (t,)), self._o)

    @jit
    def _add(self, x, y, o, remaining):
        with vf(mode="simd"):
            full = full_mask()
            cnt = mask_counter(remaining)
            for s in range(self._seg):
                m, cnt = update_mask(cnt, elem_bits=32)
                off = s * VL
                if const_expr(self._is_16bit):
                    xu = vload_unpack(x, off, unpack_mode=UnpackMode.B16_TO_B32)
                    yu = vload_unpack(y, off, unpack_mode=UnpackMode.B16_TO_B32)
                    xv = vcast(xu, dtypes.float32, mask=full)
                    yv = vcast(yu, dtypes.float32, mask=full)
                    acc = vadd(xv, yv, mask=m)
                    packed = vcast(
                        acc, self._dtype, mask=full, rounding=RoundingMode.RN
                    )
                    vstore_pack(o, off, packed, m, pack_mode=PackMode.B32_TO_B16)
                else:
                    xv = vload(x, off)
                    yv = vload(y, off)
                    acc = vadd(xv, yv, mask=m)
                    vstore(o, off, acc, m)


class VecAdd:
    def __init__(self, dtype):
        self.dtype = dtype

    @host
    def run(self, gm_x, gm_y, gm_out):
        n = gm_x.shape[0]
        block_dim = _core_count()
        VecAddKernel(self.dtype)[block_dim](gm_x, gm_y, gm_out, n)


@compile_cache
def _add_provider(dtype):
    """Compile the add program once per dtype for a dynamic element count.

    The host tensors are declared with a ``Dim("n")`` leaf, so ``n`` reaches
    the kernel as a runtime argument and one binary serves every shape. The
    factory result is reused across calls in this process; the underlying
    .so is shared through the DSL compile cache (memory tier always on,
    disk tier via CANNBOTDSL_CACHE_DIR) across processes and runs.
    """
    spec = TensorSpec((Dim("n", min=1),), dtype)
    return cannbotdsl.compile(VecAdd(dtype).run, spec, spec, spec)


def add(x1, x2):
    """Elementwise addition: y = x1 + x2.

    Supports float16 / bfloat16 / float32 / int32 tensors on npu, any rank
    (including 0-dim scalars), same shape and dtype for both inputs.
    """
    if x1.shape != x2.shape:
        raise ValueError(f"shape mismatch: {tuple(x1.shape)} vs {tuple(x2.shape)}")
    if x1.dtype != x2.dtype:
        raise TypeError(f"dtype mismatch: {x1.dtype} vs {x2.dtype}")
    if x1.dtype not in _TORCH_TO_DSL:
        raise TypeError(
            "only float16 / bfloat16 / float32 / int32 are supported"
        )
    if not x1.is_contiguous() or not x2.is_contiguous():
        raise ValueError("inputs must be contiguous")

    out = torch.empty_like(x1)
    n = x1.numel()
    if n == 0:
        return out

    dsl_dtype = _TORCH_TO_DSL[x1.dtype]

    # No host-side padding: tile_slice clips the final partial tile to the
    # remaining extent, so the kernel handles the tail without any extra
    # fill/copy/pad ops.
    _add_provider(dsl_dtype)(x1.reshape(n), x2.reshape(n), out.reshape(n))
    return out
