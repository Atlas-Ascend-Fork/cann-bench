# Direct Launch Example

基于 `<<<>>>` 语法的 AscendC 自定义算子示例，支持算子自注册机制。

## Supported SOC Versions

| SOC | `NPU_ARCH` Value | Bisheng `--npu-arch` | AICore 微架构 | Ascend Chip |
|------|---------------|----------------------|---------------|-------------|
| Ascend 910B | `ascend910b` | `dav-2201` | c220 (cube+vec) | 910B1/B2/B3/B4 |
| Ascend 910_93 | `ascend910_93` | `dav-2201` | c220 (共享 910B 微架构) | 910_93 series |
| Ascend 950 | `ascend950` | `dav-3510` | c310 (cube+vec) | 950 series |

## 目录结构

```
direct_launch_example/
├── cann_bench/         # Python包
├── cmake/              # 公共CMake配置(不感知算子)
│   ├── func.cmake      # 注册宏定义
│   └── ...
├── csrc/
│   ├── extension.cpp   # Python扩展入口
│   └── ops/            # 算子目录
│       ├── CMakeLists.txt  # 自动发现算子
│       ├── add/            # Add算子
│       │   ├── CMakeLists.txt  # 算子自注册
│       │   ├── op_kernel/
│       │   │   ├── add_kernel.cpp  # Kernel+Tiling+Launch(bisheng)
│       │   │   └ add_launch.h      # Launch声明(g++可见)
│       │   └── op_plugin/
│       │       └ add_plugin.cpp    # Python bindings(g++)
│       └── sqrt/           # Sqrt算子(结构相同)
├── dist/               # 输出目录
│   └── cann_bench-1.0.0-cp38-abi3-linux_aarch64.whl
├── scripts/
│   └── build_wheel.sh
├── tests/
├── build.sh            # 统一构建入口
└── setup.py
```

## 构建方法

```bash
bash build.sh                              # Auto-detect SOC, build wheel
bash build.sh --soc=ascend910b             # Build for 910B explicitly
bash build.sh --soc=ascend910_93           # Build for 910_93
bash build.sh --soc=ascend950              # Build for 950
bash build.sh --soc=ascend910b --install   # Build + install for 910B
```

If SOC cannot be auto-detected, the script will exit with an error. Use `--soc=` to specify manually.

> **CMake 版本兼容性说明**：要求 CMake >= 3.16。CMake 3.31 起默认向链接器传递
> `--dependency-file` 参数（`CMAKE_LINK_DEPENDS_USE_LINKER=ON`），而毕昇链接器
> （cce-ld）不支持该参数，链接 `_C.so` 会报
> `ignoring unknown argument: --dependency-file=...` 导致构建失败。
> 本工程 `CMakeLists.txt` 已在所有 `add_library()` 之前设置
> `CMAKE_LINK_DEPENDS_USE_LINKER=FALSE` 规避（对 CMake < 3.31 无副作用），
> 无需额外操作；若基于本工程新建 CMake 工程，请保留该设置。

---

## 新增算子详细步骤

### 第一步：创建算子目录结构

```bash
# 以新增 Mul 算子为例
cd csrc/ops
mkdir -p mul/op_kernel
mkdir -p mul/op_plugin
```

**目录说明：**
| 目录 | 用途 | 编译器 |
|------|------|--------|
| `op_kernel/` | Kernel实现 + Tiling计算 + Launch函数 | bisheng (-xasc) |
| `op_plugin/` | Python bindings (torch.library) | g++ |

### 第二步：编写op_kernel文件

#### 2.1 mul_kernel.cpp (bisheng编译)

此文件包含三部分：Kernel实现、Tiling计算、Launch函数声明。

```cpp
/**
 * Mul算子 - Kernel + Tiling + Launch
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t PIPELINE_DEPTH = 2;

// ========== Kernel实现 (AscendC) ==========
template <typename T>
__global__ __aicore__ void mul_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, 
    int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm, yGm, zGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outQueueZ;
    
    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(inQueueY, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQueueZ, PIPELINE_DEPTH, tileSize);
    
    xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
    yGm.SetGlobalBuffer((__gm__ T *)y + blockLength * AscendC::GetBlockIdx());
    zGm.SetGlobalBuffer((__gm__ T *)z + blockLength * AscendC::GetBlockIdx());
    
    // ... Kernel实现逻辑
}

// ========== Tiling计算函数 ==========
std::tuple<int64_t, int64_t, int64_t> calc_mul_tiling_params(int64_t totalLength)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
    constexpr static int64_t BUFFER_NUM = 3;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    int64_t tileSize = ubSize / PIPELINE_DEPTH / BUFFER_NUM;
    return std::make_tuple(numBlocks, blockLength, tileSize);
}

// ========== Launch函数 (extern "C" 供g++调用) ==========
extern "C" {

void launch_mul_kernel_float(GM_ADDR x, GM_ADDR y, GM_ADDR z, 
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    mul_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, z, totalLength, blockLength, tileSize);
}

void launch_mul_kernel_half(GM_ADDR x, GM_ADDR y, GM_ADDR z,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    mul_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, z, totalLength, blockLength, tileSize);
}

}
```

#### 2.2 mul_launch.h (头文件，g++可见)

```cpp
/**
 * Launch function declarations for g++
 */

#ifndef MUL_LAUNCH_H
#define MUL_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Tiling function declaration
std::tuple<int64_t, int64_t, int64_t> calc_mul_tiling_params(int64_t totalLength);

// Launch function declarations
extern "C" {
void launch_mul_kernel_float(GM_ADDR x, GM_ADDR y, GM_ADDR z, 
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
void launch_mul_kernel_half(GM_ADDR x, GM_ADDR y, GM_ADDR z,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
}

// Convenience macros
#define MUL_KERNEL_LAUNCH_FLOAT(x, y, z, len, blocks, blkLen, tileSz, stream) \
    launch_mul_kernel_float(x, y, z, len, blocks, blkLen, tileSz, stream)

#define MUL_KERNEL_LAUNCH_HALF(x, y, z, len, blocks, blkLen, tileSz, stream) \
    launch_mul_kernel_half(x, y, z, len, blocks, blkLen, tileSz, stream)

#endif
```

### 第三步：编写op_plugin文件

#### 3.1 mul_plugin.cpp (g++编译)

```cpp
/**
 * Mul Python bindings
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/mul_launch.h"

namespace cann_bench {

// Schema注册
TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("mul(Tensor x, Tensor y) -> Tensor");
}

// Meta函数
torch::Tensor mul_meta(const torch::Tensor &x, const torch::Tensor &y)
{
    TORCH_CHECK(x.sizes() == y.sizes(), "Shapes must match.");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "Dtypes must match.");
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("mul", mul_meta);
}

// NPU实现
torch::Tensor mul_npu(const torch::Tensor &x, const torch::Tensor &y)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto z = mul_meta(x, y);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    
    int64_t totalLength = x.numel();
    int64_t numBlocks, blockLength, tileSize;
    std::tie(numBlocks, blockLength, tileSize) = calc_mul_tiling_params(totalLength);
    
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();
    auto z_ptr = (GM_ADDR)z.data_ptr();

    auto acl_call = [=]() -> int {
        if (x.scalar_type() == torch::kFloat32) {
            MUL_KERNEL_LAUNCH_FLOAT(x_ptr, y_ptr, z_ptr, totalLength, numBlocks, blockLength, tileSize, stream);
        } else if (x.scalar_type() == torch::kFloat16) {
            MUL_KERNEL_LAUNCH_HALF(x_ptr, y_ptr, z_ptr, totalLength, numBlocks, blockLength, tileSize, stream);
        }
        return 0;
    };
    
    at_npu::native::OpCommand::RunOpApi("Mul", acl_call);
    return z;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("mul", mul_npu);
}

}
```

### 第四步：编写算子CMakeLists.txt

**csrc/ops/mul/CMakeLists.txt**：
```cmake
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Mul算子自注册

# Kernel源文件(bisheng编译: --npu-arch and -xasc are set centrally from NPU_ARCH)
set(MUL_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mul_kernel.cpp
)

# Plugin源文件(g++编译)
set(MUL_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/mul_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${MUL_KERNEL_SRCS}"    # Kernel源文件列表
    op_kernel               # Kernel include目录(相对路径)
    "${MUL_PLUGIN_SRCS}"    # Plugin源文件列表
    op_kernel               # Plugin include目录(相对路径)
)
```

### 第五步：重新构建

```bash
bash build.sh --install
```

**无需修改任何公共CMakeLists.txt文件！**

---

## 算子自注册机制原理

### 注册宏定义 (cmake/func.cmake)

```cmake
macro(register_direct_launch_op KERNEL_SRCS KERNEL_INCLUDE_DIR PLUGIN_SRCS PLUGIN_INCLUDE_DIR)
    # 将kernel源文件添加到 ALL_KERNEL_SRCS
    # 将plugin源文件添加到 ALL_PLUGIN_SRCS
    # 将include目录添加到全局列表
endmacro()
```

### 自动发现算子 (csrc/ops/CMakeLists.txt)

```cmake
file(GLOB SUB_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${SUB_DIRS})
    if(IS_DIRECTORY ${SUB_DIR})
        add_subdirectory(${SUB_DIR})  # 调用算子的CMakeLists.txt
    endif()
endforeach()
```

### 公共CMakeLists.txt不感知算子

顶层 `CMakeLists.txt` 仅使用全局变量：
```cmake
# Kernel编译(bisheng)
set_source_files_properties(${ALL_KERNEL_SRCS} PROPERTIES COMPILE_FLAGS "--npu-arch=${BISHENG_NPU_ARCH} -xasc")
add_library(all_kernels_obj OBJECT ${ALL_KERNEL_SRCS})

# Plugin编译(g++)
add_library(all_plugins_obj OBJECT ${ALL_PLUGIN_SRCS})

# 合并为共享库
add_library(_C SHARED $<TARGET_OBJECTS:all_kernels_obj> $<TARGET_OBJECTS:all_plugins_obj>)
```

---

## Python API

```python
import cann_bench
z = cann_bench.add(x, y)  # Add算子
r = cann_bench.sqrt(x)    # Sqrt算子
m = cann_bench.mul(x, y)  # 新增Mul算子

# 或通过torch.ops调用
z = torch.ops.cann_bench.add(x, y)
m = torch.ops.cann_bench.mul(x, y)
```

---

## 文件职责总结

| 文件 | 编译器 | 职责 |
|------|--------|------|
| `op_kernel/*.cpp` | bisheng | Kernel实现 + Tiling + Launch extern "C" |
| `op_kernel/*.h` | bisheng/g++ | Launch函数声明(g++可见) |
| `op_plugin/*.cpp` | g++ | torch.library注册 + Meta + NPU实现 |
| `CMakeLists.txt` | cmake | 调用register_direct_launch_op() |