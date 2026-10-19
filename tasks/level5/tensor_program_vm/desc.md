# TensorProgramVm 算子 API 描述

## 1. 算子简介

张量程序虚拟机：**kernel 即解释器，程序即数据**。输入不是固定的算子参数，而是一段待执行的指令序列 `program`——kernel 必须在片上译码这段程序，并按指令逐条驱动寄存器堆（R 个宽 W 的行向量寄存器）上的向量运算。指令间存在真依赖（后一条指令可以读前一条刚写入的寄存器），执行顺序是规格的一部分。

这一形态是新一代 megakernel 推理基础设施的核心机制：Hazy Research 的低延迟 megakernel 在每个计算单元上运行一个持久化解释器，把整网前向编译成指令流按序消费；Mirage MPK 把 LLM 推理编译成单个持久 kernel，由片上调度器逐任务译码分发（见 §6 参考文献）。这类系统消灭了逐算子 kernel launch 的开销与中间结果的 GM 往返，本算子把其中"片上解释器"这一关键机制抽象为可评测的算子。

**主要应用场景**：
- Megakernel / 持久化 kernel 推理系统的片上指令流解释与任务调度
- 以数据驱动控制流的算子融合运行时（指令序列由上层编译器生成，kernel 侧只负责高效执行）
- 张量程序的硬件行为建模与调度器验证

**算子特征**：
- 难度等级：L5（FusedComposite）
- 双输入（program int32、init_regs 浮点）单输出（final_regs）
- 控制流由数据决定：8 条指令的 ISA，经 mod 译码后任意非负 int32 程序都合法
- 饱和算术（sat = clamp ±10000）保证任意随机程序数值有界、结果良定义

**为何是 L5**：
- **规格自足，不考知识**：全部 ISA 语义、译码规则与常数在本文档给全（§2），实现不需要查任何资料；难点不在数学而在"数据依赖的控制流 + 真依赖指令流"的工程映射
- **正确 ≠ 快（性能墙）**：按 §2 逐指令直译、每条指令下发一次独立 kernel 是完全正确的实现，能通过全部精度用例，但 P 条指令意味着 P 次 kernel launch 与 P 次寄存器堆的 GM 往返，性能只能锚定朴素 baseline（预期得分 ~0.5 档）。要逼近硬件下界，必须把整个执行做成**持久化解释器**：程序只从 GM 读一次，寄存器堆全程驻留片上，标量单元译码指令流驱动向量单元执行——这正是 megakernel 解释器的核心机制。指令间真依赖（含 MULADD 的读改写）决定了不能简单按指令并行，可并行维度必须从数据布局中挖掘
- **实现完整规格**：实现必须遵循完整 ISA、译码 mod 语义、MULADD 读写顺序、ROLL 方向、MEANB 广播和饱和边界，不应依赖公开样本的特定配置。
- **精度契约明确**：内部计算精度为 fp32（§2），阈值按实测误差下限收紧（§4），数值行为无自由发挥空间

## 2. 算子定义

### 执行模型

寄存器堆 $\mathrm{regs} \in \mathbb{R}^{R \times W}$，初值为 `init_regs`。对 $t = 0, 1, \dots, P-1$ **严格按序**执行 `program` 的第 $t$ 行；第 $t+1$ 条指令读取的寄存器值是第 $t$ 条执行完后的状态（真依赖）。全部 P 条指令执行完后的寄存器堆即输出 `final_regs`。

### 译码规则

`program` 的每行 4 个 int32（任意非负整数都是合法指令）：

$$
\mathrm{op} = \mathrm{program}[t,0] \bmod 8, \quad
\mathrm{dst} = \mathrm{program}[t,1] \bmod R, \quad
a = \mathrm{program}[t,2] \bmod R, \quad
b = \mathrm{program}[t,3] \bmod R
$$

program 取值非负（评测取值范围 [0, 1073741823]），mod 无符号歧义。

### ISA（8 条指令）

所有操作数为寄存器堆的行向量 $[W]$。饱和函数 $\mathrm{sat}(x) = \mathrm{clamp}(x, -10000.0, 10000.0)$ 逐元素作用。

| op | 名称 | 语义 | 说明 |
|----|------|------|------|
| 0 | ADD | regs[dst] = sat(regs[a] + regs[b]) | 逐元素加，饱和 |
| 1 | MUL | regs[dst] = sat(regs[a] * regs[b]) | 逐元素乘，饱和 |
| 2 | RELU | regs[dst] = max(regs[a], 0) | b 忽略 |
| 3 | MEANB | regs[dst] = broadcast(mean(regs[a])) | 沿 W 求均值后广播为常数行；b 忽略 |
| 4 | MAX | regs[dst] = elementwise_max(regs[a], regs[b]) | 逐元素取大 |
| 5 | ROLL | regs[dst][j] = regs[a][(j−1+W) mod W] | 向右循环移 1 位；b 忽略 |
| 6 | MULADD | regs[dst] = sat(regs[dst] + regs[a] * regs[b]) | **读 dst，写 dst**（读改写） |
| 7 | COPY | regs[dst] = regs[a] | b 忽略 |

**本算子的精确约定**：
- **饱和语义**：sat 仅作用于 ADD / MUL / MULADD 三条会扩大数值范围的指令。这保证任意随机程序下所有中间值有界（$|x| \le \max(10000, \max|{\rm init}|)$）、结果良定义；RELU / MEANB / MAX / ROLL / COPY 不会扩大值域，不做饱和
- **MULADD 读改写**：先读 dst 的当前值，加上 regs[a] ⊙ regs[b]，饱和后写回 dst——它是唯一读三个寄存器的指令
- **ROLL 方向**：向右循环移 1 位，即输出下标 j 取输入下标 (j−1+W) mod W；连续 W 次 ROLL 恒等
- **MEANB**：先沿 W 求算术均值（一个标量），再广播成整行；对常数行幂等（数学恒等，浮点下可差 1 ulp）
- **dst = a 或 dst = b 合法**：语义仍是"先读源、后写 dst"（每条指令的读全部发生在写之前）
- **计算精度**：全部指令运算以 fp32 执行；bfloat16 / float16 输入先升 fp32，寄存器堆以 fp32 保存，仅最终输出转回输入 dtype（这是规格的一部分，golden 与评测均按此定义）

## 3. 接口规范

### 算子原型

```python
tensor_program_vm(Tensor program, Tensor init_regs) -> Tensor final_regs
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| program | Tensor | 是 | int32 | [P, 4] | 指令序列，每行经 mod 译码为 (op, dst, a, b)，任意非负整数合法（评测取值范围 [0, 1073741823]） |
| init_regs | Tensor | 是 | float32/float16/bfloat16 | [R, W] | 寄存器堆初值（公开样本取值范围见 cases.csv） |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| final_regs | 与 init_regs 一致 | [R, W] | 全部 P 条指令执行完后的寄存器堆 |

### 数据类型

| init_regs dtype | final_regs dtype | 内部计算 |
|-----------------|------------------|----------|
| bfloat16 | bfloat16 | fp32（寄存器堆全程 fp32 驻留，仅输出转回） |
| float16 | float16 | fp32 |
| float32 | float32 | fp32 |

### 规则与约束

- program 取值非负（由 value_range 保证）；经 mod 译码后**任意程序都合法**，算子不做合法性检查
- 指令必须严格按 t = 0..P-1 的顺序生效：第 t+1 条指令读到的寄存器状态 = 第 t 条执行完后的状态
- 每条指令内部"先读后写"：dst 与 a / b 重合时，源操作数取写入前的值
- 寄存器堆与全部指令运算以 fp32 保存/执行（§2 计算精度约定）
- MEANB 沿 W 的求和顺序不限（不同归约树均可），误差由 §4 阈值覆盖
- 未被任何指令写入的寄存器在输出中保持初值
- 输出须为 contiguous 张量

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `P`（程序长度） | 1 ~ 4096 | cases.csv 实测 1 ~ 4096 |
| `R`（寄存器数） | 1 ~ 64 | cases.csv 实测 1 ~ 64 |
| `W`（寄存器宽度） | 1 ~ 8192 | cases.csv 实测 1 ~ 8192 |
| dtype | bfloat16 / float16 / float32 | 三种均覆盖 |
| `program` 取值 | [0, 1073741823] | 公开样本见 cases.csv；实现应支持规格域内的任意合法程序 |
| `init_regs` 取值 | [-1, 1] | 公开样本见 cases.csv；实现应遵循完整输入规格 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

阈值经评测框架 checker 实测确定（plain golden 的 fp32 数据通路 vs fp64 oracle，测点覆盖最大规模随机程序、4096 条同寄存器 MULADD 最长依赖链、窄值域非饱和指令混合链、fp16 边界初值与非对齐宽度）：bf16/fp16 全部测点 MERE=MARE=0（fp32 与 fp64 的差异不跨低精度量化格），fp32 最坏 MERE≈3.7e-7 / MARE≈3.4e-5。

| 数据类型 | FLOAT32 | FLOAT16 | BFLOAT16 |
|----------|---------|---------|----------|
| **通过阈值(Threshold)** | 0.0001 | 0.001 | 0.008 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过（MARE 判据对实测最坏值留 ≥30 倍裕量，覆盖独立 fp32 实现在归约顺序 / FMA 上的合理差异）。RELU / MUL 衰减链产生的小值域场景由评测框架的兜底标准处理。**达标前提**：按 §2 约定以 fp32 执行全部指令运算（寄存器堆 fp32 驻留）。

## 5. 标准 Golden 代码

```python
import torch

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
```

## 6. 额外信息

### 算子调用示例

```python
import torch

P, R, W = 1024, 32, 4096

program = torch.randint(0, 1073741824, (P, 4), dtype=torch.int32, device="npu")
init_regs = (torch.rand(R, W, dtype=torch.bfloat16, device="npu") * 2 - 1)

final_regs = tensor_program_vm(program, init_regs)
# final_regs.shape: [R, W]，dtype 与 init_regs 一致
```

### 可用于自检的性质（均已数值验证）

- 程序 `[[7, 1, 0, 0]]`（COPY r1←r0）执行一次与两次结果相同，且 final_regs[1] == init_regs[0]
- 对同一寄存器连续 W 次 ROLL 后恒等；单次 ROLL 满足 out[j] = in[(j−1+W) mod W]
- MEANB 幂等（对已广播的常数行再做 MEANB 结果不变，浮点下至多差 1 ulp）
- init_regs 全 1 时，程序 `[[0,0,0,0]] * 20`（r0 = r0 + r0 连加）使 r0 严格按 2 的幂增长并在 +10000 饱和；全 −1 时对称地在 −10000 饱和
- op=8 与 op=0 等价、dst=R+1 与 dst=1 等价（mod 译码）
- 未被写入的寄存器保持初值

### 关于长随机程序的数值形态

全域随机程序（op 均匀分布）中 ADD 的饱和增长与 MAX 的传播使寄存器值随 P 增大向吸收态（±10000、0）收敛——这是 ISA 的固有性质，不是实现缺陷；实现必须遵循 ISA 对任意合法程序的定义。

### 参考文献

- Hazy Research (Stanford), "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B" (2025)：单 kernel 内的持久化指令解释器，每个计算单元按序消费指令流（本算子"kernel 即解释器"形态的直接原型）
- Mirage Persistent Kernel (MPK), "Compiling LLMs into a MegaKernel: A Path to Low-Latency Inference" (2025, CMU/Mirage 项目)：把 LLM 推理编译为单个持久 kernel，片上调度器逐任务译码分发（SM 级任务调度的原型）
- Gupta, K. et al. (2012). "A Study of Persistent Threads Style GPU Programming for GPGPU Workloads". InPar 2012（持久化线程编程模型的早期系统化研究）
