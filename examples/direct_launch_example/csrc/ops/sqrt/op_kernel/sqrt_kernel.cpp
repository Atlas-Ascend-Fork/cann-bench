/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sqrt_host.cpp
 * \brief Sqrt host code - kernel launch and tiling (compiled with bisheng + -xasc)
 *
 * AscendC::Sqrt has no native fp16/bf16 overload, so for those dtypes we
 * cast the input to fp32, run Sqrt, and cast the result back.
 */

#include <tuple>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t PIPELINE_DEPTH = 2;

template <typename T>
class KernelSqrt {
public:
    __aicore__ inline KernelSqrt() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
    {
        xGm_.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
        zGm_.SetGlobalBuffer((__gm__ T *)z + blockLength * AscendC::GetBlockIdx());
        // tileSize is element count (not bytes)
        pipe_.InitBuffer(inQueueX_,  PIPELINE_DEPTH, tileSize * sizeof(T));
        pipe_.InitBuffer(outQueueZ_, PIPELINE_DEPTH, tileSize * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(xFloatBuf_, tileSize * sizeof(float));
            pipe_.InitBuffer(zFloatBuf_, tileSize * sizeof(float));
        }
        int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
        if (currentBlockLength > blockLength) currentBlockLength = blockLength;
        if (currentBlockLength < 0) currentBlockLength = 0;
        elementNumPerTile_ = tileSize;
        tileNum_ = currentBlockLength / elementNumPerTile_;
        tailTileElementNum_ = currentBlockLength - tileNum_ * elementNumPerTile_;
    }

    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < tileNum_; ++i) {
            CopyIn(i * elementNumPerTile_, elementNumPerTile_);
            Compute(elementNumPerTile_);
            CopyOut(i * elementNumPerTile_, elementNumPerTile_);
        }
        if (tailTileElementNum_ > 0) {
            CopyIn(tileNum_ * elementNumPerTile_, tailTileElementNum_);
            Compute(tailTileElementNum_);
            CopyOut(tileNum_ * elementNumPerTile_, tailTileElementNum_);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        auto xLocal = inQueueX_.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = inQueueX_.DeQue<T>();
        auto zLocal = outQueueZ_.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Sqrt(zLocal, xLocal, count);
        } else {
            auto xF = xFloatBuf_.Get<float>();
            auto zF = zFloatBuf_.Get<float>();
            AscendC::Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Sqrt(zF, xF, count);
            constexpr auto roundMode = std::is_same<T, bfloat16_t>::value
                ? AscendC::RoundMode::CAST_RINT : AscendC::RoundMode::CAST_NONE;
            AscendC::Cast(zLocal, zF, roundMode, count);
        }
        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        auto zLocal = outQueueZ_.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(zGm_[offset], zLocal, copyParams);
        outQueueZ_.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> xGm_, zGm_;
    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> outQueueZ_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFloatBuf_, zFloatBuf_;
    int64_t elementNumPerTile_ = 0, tileNum_ = 0, tailTileElementNum_ = 0;
};

template <typename T>
__global__ __aicore__ __vector__ void sqrt_kernel(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    KernelSqrt<T> op;
    op.Init(x, z, totalLength, blockLength, tileSize);
    op.Process();
}

// Returns (numBlocks, blockLength, tileSize) where tileSize is element count
// per UB tile, derived at runtime from the UB capacity.
//
// Unlike add, the per-element UB cost here depends on T (queue depth
// included): fp32 needs only the in/out queues, while fp16/bf16 also need the
// two fp32 cast buffers. The tiling function cannot see T, so budget for the
// worst case -- fp16/bf16 -- which is also safe (just a smaller tile) for fp32.
std::tuple<int64_t, int64_t, int64_t> calc_sqrt_tiling_params(int64_t totalLength)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
    // Full per-element UB cost, queue depth already folded in (see Init()):
    //   fp32      -- in/out queues: 2 * PIPELINE_DEPTH * 4 B            = 16 B
    //   fp16/bf16 -- in/out queues: 2 * PIPELINE_DEPTH * 2 B    =  8 B,
    //                plus the two fp32 cast buffers 2 * 4 B           =  8 B
    // Both paths land on 16 B per element, so a single depth-inclusive tile
    // budget covers every dtype. Do NOT divide by PIPELINE_DEPTH again below;
    // it is already part of these terms.
    constexpr static int64_t BYTES_PER_ELEM_FP32 = 2 * PIPELINE_DEPTH * sizeof(float);
    constexpr static int64_t BYTES_PER_ELEM_FP16 =
        2 * PIPELINE_DEPTH * sizeof(uint16_t) + 2 * sizeof(float);
    constexpr static int64_t BYTES_PER_ELEM_MAX =
        std::max(BYTES_PER_ELEM_FP32, BYTES_PER_ELEM_FP16);

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    numBlocks = std::max(numBlocks, static_cast<int64_t>(1));
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    int64_t tileSize = static_cast<int64_t>(ubSize) / BYTES_PER_ELEM_MAX;
    // Enforce strict 32 B alignment on every InitBuffer size (the GetUbBlockSize()
    // granularity ops-nn aligns to): the narrowest supported T is fp16 at 2 B, so
    // flooring the element count to 32 B / 2 B = 16 makes tileSize * sizeof(T) an
    // exact multiple of 32 B for every dtype (fp16 buffers land exactly on 32 B,
    // the 4 B fp32/cast buffers on 64 B) and TPipe applies no padding.
    // A no-op on 910B3: 196352 / 16 = 12272 = 16 * 767, so the aligned tile
    // fills the UB exactly (196352 B allocated, ~100% utilization).
    constexpr static int64_t TILE_ALIGN_ELEMS = 32 / sizeof(uint16_t);
    tileSize -= tileSize % TILE_ALIGN_ELEMS;
    // A zero tileSize would divide by zero in Init().
    if (tileSize < 1) tileSize = 1;
    return std::make_tuple(numBlocks, blockLength, tileSize);
}

extern "C" {

void launch_sqrt_kernel_float(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    sqrt_kernel<float><<<numBlocks, nullptr, stream>>>(x, z, totalLength, blockLength, tileSize);
}

void launch_sqrt_kernel_half(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    sqrt_kernel<half><<<numBlocks, nullptr, stream>>>(x, z, totalLength, blockLength, tileSize);
}

void launch_sqrt_kernel_bfloat16(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    sqrt_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, z, totalLength, blockLength, tileSize);
}

}
