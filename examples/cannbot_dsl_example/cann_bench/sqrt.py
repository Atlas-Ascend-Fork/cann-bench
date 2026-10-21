# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Elementwise sqrt: y = sqrt(x), x >= 0, dtype float16 / float32.

Standard form: dynamic core count, dynamic shape, one shared compilation,
compile cache enabled.

- Dynamic core count: the AI core count is queried at runtime from the CANN
  runtime via ``get_platform_info().core_num`` (hardware ai_core_cnt clamped
  by the current stream's core quota), not hardcoded. The launch uses exactly
  that many blocks and the kernel walks tiles with a grid-stride loop.
- Dynamic shape: the element count n never enters compilation. The ``@host``
  entry is compiled once per dtype through ``cannbotdsl.compile()`` against
  ``TensorSpec((Dim("n"),), ...)`` descriptors, and every call reuses that
  single ProviderCallable for any shape. n is read from ``gm_x.shape[0]``
  (a runtime i64) and passed to the kernel as a launch argument; the
  in-kernel tile-loop bound is ``ceil_div(n, tile)`` and the tail tile is
  clipped by ``tile_slice`` + masked by ``mask_counter``.
- Compile cache: the in-memory tier is always on; setting
  ``CANNBOTDSL_CACHE_DIR`` (below, when unset) enables the persistent disk
  tier under ``$CANNBOTDSL_CACHE_DIR/v2/``, so later processes and runs
  reuse compiled artifacts instead of recompiling. Clear with
  ``cannbotdsl.clear_compile_cache()``.
"""

import os

# Enable the persistent on-disk compile cache tier before importing
# cannbotdsl. The in-memory tier is on by default; CANNBOTDSL_CACHE_DIR
# additionally publishes every compilation to $CANNBOTDSL_CACHE_DIR/v2/
# so subsequent processes and evaluation runs hit the cache instead of
# recompiling (see cannbotdsl.core.compiler.cache).
os.environ.setdefault(
    "CANNBOTDSL_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "cannbotdsl"),
)

import torch

import cannbotdsl
from cannbotdsl import Dim, TensorSpec, dtypes, get_mem_size, get_platform_info
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
    mask_counter,
    update_mask,
    vcast,
    vload,
    vload_unpack,
    vsqrt,
    vstore,
    vstore_pack,
)
from cannbotdsl.ops.reg.cast import RoundingMode
from cannbotdsl.tensor import MemLoc, Tensor, ceil_div, tile_slice

VL = 64
UB_MARGIN = 1024

_TORCH_TO_DSL = {
    torch.float16: dtypes.float16,
    torch.float32: dtypes.float32,
}


def _elem_bytes(dtype):
    return 2 if dtype == dtypes.float16 else 4


def _choose_tile(dtype, ub_size):
    # x / y both in UB with double space headroom, keep 1KB margin
    budget = (ub_size - UB_MARGIN) // (2 * _elem_bytes(dtype))
    return max(VL, (budget // VL) * VL)


@kernel
class SqrtKernel:
    def __init__(self, tile, dtype=dtypes.float32):
        self._tile = tile
        self._dtype = dtype
        self._is_16bit = dtype == dtypes.float16
        self._seg = tile // VL
        self._x = Buffer(MemLoc.UB, (tile,), dtype)
        self._y = Buffer(MemLoc.UB, (tile,), dtype)

    def __call__(self, gm_x: Tensor, gm_y: Tensor, n):
        tile = self._tile
        n_tiles = ceil_div(n, tile)
        bi = get_block_idx()
        bn = get_block_num()

        for t in cannbotdsl.range(bi, n_tiles, bn):
            # n - t * tile >= tile on non-last tiles, so every mask of this
            # segment loop stays full; the last tile gets the tail mask.
            actual = n - t * tile
            mem_copy(self._x, tile_slice(gm_x, (tile,), (t,)))
            self._compute(self._x, self._y, actual)
            mem_copy(tile_slice(gm_y, (tile,), (t,)), self._y)

    @jit
    def _compute(self, x, y, actual):
        with vf(mode="simd"):
            remaining = mask_counter(actual)
            for s in range(self._seg):
                m, remaining = update_mask(remaining, elem_bits=32)
                off = s * VL
                if const_expr(self._is_16bit):
                    xu = vload_unpack(x, off, unpack_mode=UnpackMode.B16_TO_B32)
                    xv = vcast(xu, dtypes.float32, mask=m)
                    r = vsqrt(xv, mask=m)
                    h = vcast(r, self._dtype, mask=m, rounding=RoundingMode.RN)
                    vstore_pack(y, off, h, m, pack_mode=PackMode.B32_TO_B16)
                else:
                    xv = vload(x, off)
                    r = vsqrt(xv, mask=m)
                    vstore(y, off, r, m)


class Sqrt:
    def __init__(self, dtype=dtypes.float32):
        self.dtype = dtype
        self._tile = _choose_tile(dtype, get_mem_size("ub"))
        # Real AI core count from the CANN runtime, clamped by the current
        # stream's core quota; the launch never exceeds it.
        self._core_num = max(1, get_platform_info().core_num)

    @host
    def run(self, gm_x, gm_y):
        n = gm_x.shape[0]
        op = SqrtKernel(self._tile, self.dtype)
        op[self._core_num](gm_x, gm_y, n)

    def program(self):
        """Compile this dtype's entry once for every shape.

        The x/y extents share one dynamic Dim("n"), so the resulting
        ProviderCallable serves any n >= 1 without recompiling.
        """

        spec = TensorSpec((Dim("n", min=1),), self.dtype)
        return cannbotdsl.compile(self.run, spec, spec)


_PROGRAMS = {}


def _get_program(dtype):
    """Return the (cached, one-per-dtype) dynamic-shape provider."""

    program = _PROGRAMS.get(dtype)
    if program is None:
        program = Sqrt(dtype).program()
        _PROGRAMS[dtype] = program
    return program


def sqrt(x):
    """Elementwise sqrt on an NPU tensor. Returns a tensor of the same
    shape and dtype."""
    if x.dtype not in _TORCH_TO_DSL:
        raise TypeError(f"x dtype must be float16 or float32, got {x.dtype}")
    n = x.numel()
    if n == 0:
        return torch.empty_like(x)
    xc = x if x.is_contiguous() else x.contiguous()
    dtype = _TORCH_TO_DSL[xc.dtype]

    x1d = xc.reshape(n)
    out = torch.empty_like(xc)
    # tile_slice clips partial tail tiles to the remaining length and the
    # kernel's mask_counter tail mask covers the last VL segment.
    _get_program(dtype)(x1d, out.reshape(n))
    return out
