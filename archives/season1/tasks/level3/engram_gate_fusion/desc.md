# EngramGateFusion 算子 API 描述

## 1. 算子简介

EngramGateFusion 来自 DeepSeek Engram 模块（arXiv:2601.07372），该模块通过 N-gram 哈希查表为 Transformer 提供**条件记忆**。Engram forward 关键路径包含双路 RMSNorm、缩放点积门控、Sigmoid 非线性、门控广播乘法、因果扩张深度可分离卷积（ShortConv）、SiLU 与残差加法共七个子步骤。

**融合目标**：将七个子步骤合并为一个 Ascend 自定义算子，消除中间 tensor 的 HBM 反复读写，降低 kernel launch 开销。

**主要应用场景**：
- DeepSeek Engram 模块的条件记忆计算
- LLM 推理中的 N-gram 条件门控
- Prefill 和 Decode 两个阶段均需支持（Decode 通过 `conv_state` 缓存历史卷积上下文）

**算子特征**：
- 难度等级：L3（VVFusion）
- 7 必选张量输入 + 1 可选状态输入 + 5 个超参数
- 双输出：融合后的特征 + 下一步 Decode 所需的 ShortConv 状态

## 2. 算子定义

### 数学公式

设超参数：$HC$（hyper-connection 通道数）、$D$（hidden size）、$K$（卷积核大小）、$\text{dil}$（扩张率）、$\varepsilon$（RMSNorm epsilon）。

**Step 1 & 2：双路 RMSNorm**

$$
\hat{k}_{b,l,hc,d} = \text{RMSNorm}(\mathbf{keys}_{b,l,hc,:},\; \mathbf{w}^{(1)}_{hc,:},\; \varepsilon)_d
\qquad
\hat{q}_{b,l,hc,d} = \text{RMSNorm}(\mathbf{hidden}_{b,l,hc,:},\; \mathbf{w}^{(2)}_{hc,:},\; \varepsilon)_d
$$

$$
\text{RMSNorm}(\mathbf{x}, \mathbf{w}, \varepsilon) = \frac{\mathbf{x}}{\sqrt{\frac{1}{D}\sum_{d} x_d^2 + \varepsilon}} \odot \mathbf{w}
$$

**Step 3：缩放点积门控**

$$
r_{b,l,hc} = \frac{\sum_{d=0}^{D-1}\hat{k}_{b,l,hc,d} \cdot \hat{q}_{b,l,hc,d}}{\sqrt{D}}
$$

**Step 4：非线性变换 + Sigmoid**

$$
g_{b,l,hc} = \sigma\!\left(\sqrt{\max(|r_{b,l,hc}|,\;10^{-6})} \cdot \text{sign}(r_{b,l,hc})\right)
$$

**Step 5：门控广播乘法**

$$
V^{g}_{b,l,hc,d} = g_{b,l,hc} \cdot \mathbf{value}_{b,l,d}
$$

**Step 6：ShortConv（因果扩张深度可分离卷积 + SiLU）**

$$
\mathbf{y}_{b,c,l} = \text{SiLU}\!\left(\sum_{k=0}^{K-1} W^{(\text{conv})}_{c,k} \cdot \tilde{V}^g_{b,c,\;l-k\cdot\text{dil}}\right), \quad \tilde{V}^g = \text{RMSNorm}(V^g, \mathbf{w}^{(c)}, \varepsilon)
$$

**Step 7：残差相加**

$$
\mathbf{output}_{b,l,hc,d} = V^{g}_{b,l,hc,d} + \text{conv\_out}_{b,l,hc,d}
$$

### 关键性质

- **因果性**：ShortConv 仅做左侧 padding，位置 $l$ 的输出仅依赖 $l$ 及之前的位置
- **Decode 状态缓存**：单步 Decode 需使用 `conv_state`（长度为 $(K-1)\cdot\text{dil}$ 的历史缓存）以保留历史上下文
- **无 bias**：所有 RMSNorm 和 Conv1d 均无 bias 参数
- **门控保护**：`clamp_min(1e-6)` 用于避免极小值附近不稳定并与参考实现保持一致

## 3. 接口规范

### 算子原型

```python
cann_bench.engram_gate_fusion(
    Tensor keys,
    Tensor hidden_states,
    Tensor value,
    Tensor norm1_weight,
    Tensor norm2_weight,
    Tensor conv_norm_weight,
    Tensor conv_weight,
    Tensor? conv_state=None,
    int hc_mult=4,
    int hidden_size=1024,
    int kernel_size=4,
    int dilation=3,
    float norm_eps=1e-5
) -> (Tensor output, Tensor conv_state_out)
```

### 输入参数说明

| 参数 | 类型 | 必需 | Shape | dtype | 描述 |
|------|------|------|-------|-------|------|
| keys | Tensor | 是 | [B, L, HC, D] | bfloat16 | key_proj 输出 |
| hidden_states | Tensor | 是 | [B, L, HC, D] | bfloat16 | Transformer 隐层（query） |
| value | Tensor | 是 | [B, L, D] | bfloat16 | value_proj 输出 |
| norm1_weight | Tensor | 是 | [HC, D] | float32 | key-RMSNorm 的 γ 权重 |
| norm2_weight | Tensor | 是 | [HC, D] | float32 | query-RMSNorm 的 γ 权重 |
| conv_norm_weight | Tensor | 是 | [HC, D] | float32 | ShortConv 内 RMSNorm 的 γ 权重 |
| conv_weight | Tensor | 是 | [HC·D, 1, K] | float32 | 深度可分离 Conv1d 权重（无 bias） |
| conv_state | Tensor | 否 | [B, HC·D, (K-1)·dil] | bfloat16 | Decode 阶段的 ShortConv 历史缓存；Prefill 可为 None |

### 超参数说明

| 参数 | 类型 | 默认值 | 约束 |
|------|------|--------|------|
| hc_mult | int | 4 | hyper-connection 通道数 HC，须等于 `keys.shape[2]` |
| hidden_size | int | 1024 | 每通道隐层维度 D，须等于 `keys.shape[3]` |
| kernel_size | int | 4 | ShortConv 卷积核大小 K，须等于 `conv_weight.shape[2]` |
| dilation | int | 3 | ShortConv 扩张率 = max_ngram_size |
| norm_eps | float | 1e-5 | 所有 RMSNorm 的 ε |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| output | [B, L, HC, D] | bfloat16 | 融合后的 Engram 输出 |
| conv_state_out | [B, HC·D, (K-1)·dil] | bfloat16 | 更新后的 ShortConv 缓存，供下一步 Decode 使用 |

### 数据类型

| 输入 | dtype |
|------|-------|
| 激活（keys / hidden_states / value / conv_state） | bfloat16 |
| 权重（norm1_weight / norm2_weight / conv_norm_weight / conv_weight） | float32（计算时可 cast 到激活类型） |
| 输出（output / conv_state_out） | bfloat16（与 value 一致） |

### 规则与约束

- `keys`, `hidden_states` 必须为 4D 张量，shape 为 [B, L, HC, D]
- `value` 必须为 3D 张量，shape 为 [B, L, D]
- 维度联动：`HC == hc_mult`，`D == hidden_size`
- `keys` / `hidden_states` / `value` 的 B、L、D 必须一致
- `conv_weight` shape 须为 [HC·D, 1, K]，且 K == kernel_size
- L 为动态维度，算子须支持可变序列长度
- 若为 Decode 单步模式（L=1），必须提供 `conv_state`；Prefill 模式 `conv_state` 可为 None（内部左 padding 处理）
- `raw_gate` 中 `clamp_min(1e-6)` 不可省略
- 所有 RMSNorm 与 Conv1d 均无 bias 参数

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch，`keys` / `hidden_states` / `value` / `conv_state` 共享） | 1 ~ 256 | cases.csv 实测 1 ~ 8 |
| `L`（序列长度，`keys` / `hidden_states` / `value` 共享） | 1 ~ 4096 | cases.csv 实测 1 ~ 2048；Decode 单步 `L=1` 必须提供 `conv_state` |
| `HC`（hyper-connection，`keys` / `hidden_states` / `norm1_weight` / `norm2_weight` / `conv_norm_weight` 共享） | 1 ~ 16 | cases.csv 实测 2 / 4 / 8；须等于 `hc_mult` |
| `D`（hidden size，`keys` / `hidden_states` / `value` / `norm*_weight` 共享） | 64 ~ 2048 | cases.csv 实测 256 / 512 / 769 / 1024；须等于 `hidden_size`；不要求 2 的幂（实测含质数 769） |
| `conv_weight.shape[0]`（深度可分离 Conv1d 通道） | = `HC * D` | cases.csv 实测 1024 / 2048 / 3076 / 4096 / 8192 |
| `conv_weight.shape[1]` | 固定 1 | depthwise，`groups = HC*D` |
| `K`（`conv_weight.shape[2]`，卷积核） | 1 ~ 16 | cases.csv 实测 4 / 8；须等于 `kernel_size` |
| `conv_state.shape[2]` | = `(K-1) * dilation` | cases.csv 实测 9（K=4, dil=3） |
| `hc_mult` | 1 ~ 16 | cases.csv 实测 2 / 4 / 8 |
| `hidden_size` | 64 ~ 2048 | cases.csv 实测 256 / 512 / 769 / 1024 |
| `kernel_size` | 1 ~ 16 | cases.csv 实测 4 / 8 |
| `dilation` | 1 ~ 16 | cases.csv 实测 1 / 3 |
| `norm_eps` | (0, 1) | cases.csv 实测 1e-5 |
| 激活值域（`keys` / `hidden_states` / `value` / `conv_state`） | bfloat16 可表示范围 | cases.csv 实测 [-0.1, 0.1] / [-0.01, 0.01] |
| 权重值域（`norm*_weight` / `conv_weight`） | float32 可表示范围 | cases.csv 实测 [-0.1, 0.1] / [-0.01, 0.01] |

约束：`HC == hc_mult`、`D == hidden_size`、`conv_weight.shape == [HC*D, 1, kernel_size]`；Decode 模式（`L=1`）必须传入 shape 为 `[B, HC*D, (K-1)*dilation]` 的 `conv_state`，Prefill 模式 `conv_state` 可为 None（内部左 padding）。

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

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

### 关键精度策略

- 点积门控（`raw_gate`）在 FP32 中进行乘法与累加
- `sqrt`、`sigmoid` 在 FP32 中计算后再转换到激活 dtype
- 卷积输入与 `conv_weight` 需同 dtype（推荐将 `conv_weight` cast 到激活 dtype）

## 5. 标准 Golden 代码

```python
import math
import torch
import torch.nn.functional as F


def engram_gate_fusion(
    keys: torch.Tensor,
    hidden_states: torch.Tensor,
    value: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    conv_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor = None,
    hc_mult: int = 4,
    hidden_size: int = 1024,
    kernel_size: int = 4,
    dilation: int = 3,
    norm_eps: float = 1e-5,
):
    """EngramGateFusion: dual RMSNorm gate + broadcast multiply + ShortConv + residual.

    Returns (output, conv_state_out)."""
    B, L, HC, D = keys.shape
    state_len = (kernel_size - 1) * dilation

    assert HC == hc_mult and D == hidden_size
    assert hidden_states.shape == (B, L, HC, D)
    assert value.shape == (B, L, D)
    assert norm1_weight.shape == (HC, D)
    assert norm2_weight.shape == (HC, D)
    assert conv_norm_weight.shape == (HC, D)
    assert conv_weight.shape == (HC * D, 1, kernel_size)
    if conv_state is not None:
        assert conv_state.shape == (B, HC * D, state_len)

    input_dtype = keys.dtype
    output_dtype = input_dtype
    compute_dtype = torch.float32

    def rms_norm(x, w, eps):
        x_hp = x.to(compute_dtype)
        w_view = w.view(1, 1, HC, D).to(compute_dtype)
        rms = x_hp.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        return x_hp / rms * w_view

    # Step 1 & 2: dual RMSNorm
    normed_keys = rms_norm(keys, norm1_weight, norm_eps)
    normed_qs = rms_norm(hidden_states, norm2_weight, norm_eps)

    # Step 3: scaled dot-product gate
    raw_gate = (normed_keys * normed_qs).sum(dim=-1) / math.sqrt(D)

    # Step 4: nonlinear + sigmoid gate
    safe_abs = raw_gate.abs().clamp_min(1e-6)
    gate = torch.sigmoid(safe_abs.sqrt() * raw_gate.sign()).unsqueeze(-1)

    # Step 5: broadcast gating
    value_hp = value.to(compute_dtype)
    value_gated = gate * value_hp.unsqueeze(2)

    # Step 6: ShortConv
    normed_vg = rms_norm(value_gated, conv_norm_weight, norm_eps)
    x = normed_vg.permute(0, 2, 3, 1).reshape(B, HC * D, L)

    if conv_state is None:
        x_cat = F.pad(x, (state_len, 0))
    else:
        x_cat = torch.cat([conv_state.to(compute_dtype), x], dim=-1)

    conv_state_out_hp = x_cat[:, :, -state_len:].contiguous() if state_len > 0 else x_cat[:, :, :0]

    y = F.conv1d(
        x_cat,
        conv_weight.to(compute_dtype),
        dilation=dilation,
        groups=HC * D,
    )
    y = F.silu(y)
    conv_out = y.reshape(B, HC, D, L).permute(0, 3, 1, 2)

    # Step 7: residual
    output_hp = value_gated + conv_out

    output = output_hp.to(output_dtype)
    conv_state_out = conv_state_out_hp.to(output_dtype)
    return output, conv_state_out
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import torch_npu

B, L, HC, D, K, dil = 1, 13, 4, 1024, 4, 3
state_len = (K - 1) * dil  # = 9

keys           = torch.randn(B, L, HC, D, dtype=torch.bfloat16, device="npu")
hidden_states  = torch.randn(B, L, HC, D, dtype=torch.bfloat16, device="npu")
value          = torch.randn(B, L, D,     dtype=torch.bfloat16, device="npu")
norm1_weight   = torch.randn(HC, D,        dtype=torch.float32,  device="npu")
norm2_weight   = torch.randn(HC, D,        dtype=torch.float32,  device="npu")
conv_norm_w    = torch.randn(HC, D,        dtype=torch.float32,  device="npu")
conv_weight    = torch.randn(HC * D, 1, K, dtype=torch.float32,  device="npu")

# Prefill (no state)
output, conv_state = torch_npu.npu_engram_gate_fusion(
    keys, hidden_states, value,
    norm1_weight, norm2_weight, conv_norm_w, conv_weight,
    conv_state=None,
    hc_mult=HC, hidden_size=D, kernel_size=K, dilation=dil, norm_eps=1e-5,
)

# Decode (single token, with state from previous step)
out_dec, conv_state = torch_npu.npu_engram_gate_fusion(
    keys[:, :1], hidden_states[:, :1], value[:, :1],
    norm1_weight, norm2_weight, conv_norm_w, conv_weight,
    conv_state=conv_state,
    hc_mult=HC, hidden_size=D, kernel_size=K, dilation=dil, norm_eps=1e-5,
)
```

## 7. 参考

- DeepSeek-AI (2025). "Engram: A Framework for Conditional Memory in Large Language Models". arXiv:2601.07372.
- 相关算子：`torch_npu.npu_rms_norm`，`torch.nn.functional.conv1d`
