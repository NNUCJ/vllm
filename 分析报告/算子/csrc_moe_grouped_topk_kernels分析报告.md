# csrc/moe/grouped_topk_kernels.cu 详细分析报告

> **版本**: vLLM 0.19.0  
> **硬件**: 8× RTX 4090 (SM89, Compute 8.9)  
> **模型**: DeepSeek-V3 (256 experts, 8 groups, topk=8, topk_group=4, bf16)  
> **分析入口**: `examples/offline_inference/data_parallel.py`

---

## 目录

- [1. 概述](#1-概述)
- [2. 调用链路:从 data_parallel.py 到 CUDA kernel](#2-调用链路从-data_parallelpy-到-cuda-kernel)
- [2.5 输入/输出参数完整说明](#25-输入输出参数完整说明)
  - [2.5.1 Python API 签名](#251-python-api-签名)
  - [2.5.2 输入参数逐项说明](#252-输入参数逐项说明)
  - [2.5.3 输出参数逐项说明](#253-输出参数逐项说明)
  - [2.5.4 C++ Kernel 参数映射](#254-c-kernel-参数映射)
  - [2.5.5 参数校验规则](#255-参数校验规则)
  - [2.5.6 典型模型参数配置表](#256-典型模型参数配置表)
- [3. 算子总体架构](#3-算子总体架构)
  - [3.1 两条执行路径](#31-两条执行路径)
  - [3.2 路径选择逻辑 invokeNoAuxTc](#32-路径选择逻辑-invokenoauxtc)
- [4. 路径一:small expert count kernel(DeepSeek-V3 走此路径）](#4-路径一small-expert-count-kerneldeepseek-v3-走此路径)
  - [4.1 Kernel 启动参数](#41-kernel-启动参数)
  - [4.2 共享内存布局](#42-共享内存布局)
  - [4.3 阶段一:每组 Top-2 分数计算](#43-阶段一每组-top-2-分数计算)
  - [4.4 阶段二:选择 Top-K 组](#44-阶段二选择-top-k-组)
  - [4.5 阶段三:从选中组中选出全局 Top-K 专家](#45-阶段三从选中组中选出全局-top-k-专家)
  - [4.6 阶段四:写出最终结果(renormalize + scaling)](#46-阶段四写出最终结果renormalize--scaling)
  - [4.7 具体数据流实例（DeepSeek-V3）](#47-具体数据流实例deepseek-v3)
- [5. 路径二:grouped_topk_fused_kernel（大专家数通用路径）](#5-路径二grouped_topk_fused_kernel大专家数通用路径)
  - [5.1 Kernel 启动参数](#51-kernel-启动参数)
  - [5.2 Phase 1:每组 Top-2 分数（warp 级 cooperative_groups reduce）](#52-phase-1每组-top-2-分数warp-级-cooperative_groups-reduce)
  - [5.3 Phase 2:WarpSelect 选组 + 选专家](#53-phase-2warpselect-选组--选专家)
- [6. 核心算法:reduce_topk 详解](#6-核心算法reduce_topk-详解)
  - [6.1 打包比较技术 TopKRedType](#61-打包比较技术-topkredtype)
  - [6.2 K 轮 warp reduce 选 Top-K](#62-k-轮-warp-reduce-选-top-k)
- [7. 核心算法:WarpSelect 详解](#7-核心算法warpselect-详解)
  - [7.1 Bitonic Sort 在 warp 内的实现](#71-bitonic-sort-在-warp-内的实现)
  - [7.2 WarpSelect::add() 的缓冲合并机制](#72-warpselectadd-的缓冲合并机制)
- [8. Scoring 函数:sigmoid_accurate](#8-scoring-函数sigmoid_accurate)
- [9. 性能特征与对比](#9-性能特征与对比)
- [10. 架构图](#10-架构图)
- [11. 总结](#11-总结)

---

## 1. 概述

`grouped_topk_kernels.cu` 实现了 **Grouped Top-K 路由算子**——MoE（Mixture of Experts）模型的核心路由逻辑。与 `topk_softmax_kernels.cu` 中的简单 Top-K 不同,此算子支持**分组竞争**机制:

1. 将 N 个专家分成 G 组
2. 先选出 Top-K_group 个组（基于组内最高分之和）
3. 再从选中组内选出全局 Top-K 个专家

这是 **DeepSeek-V2/V3** 等模型采用的路由策略,目的是保证专家多样性（避免所有 Top-K 都来自同一组）。

**核心参数**（以 DeepSeek-V3 为例）:

| 参数 | 含义 | DeepSeek-V3 值 |
|---|---|---|
| `num_experts` | 总专家数 | 256 |
| `n_group` | 分组数 | 8 |
| `num_experts_per_group` | 每组专家数 | 256/8 = 32 |
| `topk_group` | 选取的组数 | 4 |
| `topk` | 最终选取专家数 | 8 |
| `scoring_func` | 激活函数 | SIGMOID (1) |
| `renormalize` | 是否归一化 | true |
| `routed_scaling_factor` | 路由缩放因子 | 1.0 (通常) |

---

## 2. 调用链路:从 data_parallel.py 到 CUDA kernel

```
data_parallel.py
  └→ LLM(model="deepseek-v3", ...)
      └→ DeepSeekV2MoE.forward()                        # models/deepseek_v2.py
          └→ SharedFusedMoE(FusedMoE).forward_cuda()    # layers/fused_moe/layer.py
              └→ DefaultMoERunner.forward()              # default_moe_runner.py
                  └→ router.select_experts()             # base_router.py
                      └→ GroupedTopKRouter._compute_routing()  # grouped_topk_router.py
                          └→ grouped_topk()              # @torch.compile 包装
                              └→ fused_grouped_topk()
                                  └→ ops.grouped_topk()  # _custom_ops.py
                                      └→ torch.ops._moe_C.grouped_topk()
                                          └→ invokeNoAuxTc()        # C++ dispatch
                                              └→ grouped_topk_fused_small_expert_count_kernel<<<>>>
```

**关键决策节点**:
- `GroupedTopKRouter` 在 `router_factory.py` 中根据 `use_grouped_topk=True` 被创建
- DeepSeek-V3 的 `scoring_func=1` (SIGMOID) 和 `e_score_correction_bias` 非空触发融合 kernel
- `invokeNoAuxTc` 根据 `num_experts` 和 `n_group` 选择具体 kernel 实例

---

## 2.5 输入/输出参数完整说明

本节详细解释 `grouped_topk` 算子从 Python API 到 CUDA kernel 每一层的参数含义、形状、数据类型和物理意义,让算子新手能够准确理解每个参数的角色。

### 2.5.1 Python API 签名

**顶层 Python 入口** ([vllm/_custom_ops.py#L2322](../../vllm/_custom_ops.py)):

```python
def grouped_topk(
    scores: torch.Tensor,           # 输入 1: 路由分数 [num_tokens, num_experts]
    num_expert_group: int,          # 输入 2: 分组数 G
    topk_group: int,                # 输入 3: 选取的组数
    topk: int,                      # 输入 4: 每个 token 选取的专家数 K
    renormalize: bool,              # 输入 5: 是否对 top-k 权重归一化
    routed_scaling_factor: float,   # 输入 6: 权重缩放因子
    bias: torch.Tensor,             # 输入 7: 专家路由 bias [num_experts]
    scoring_func: int = 0,          # 输入 8: 激活函数类型 (0=无, 1=sigmoid)
) -> tuple[torch.Tensor, torch.Tensor]:
    # 返回 (topk_values, topk_indices)
    return torch.ops._moe_C.grouped_topk(...)
```

**C++ 绑定签名** ([csrc/moe/torch_bindings.cpp#L122](../../csrc/moe/torch_bindings.cpp)):

```cpp
"grouped_topk(Tensor scores, int n_group, int topk_group, int topk,"
"              bool renormalize, float routed_scaling_factor,"
"              Tensor bias, int scoring_func=0) -> (Tensor, Tensor)"
```

### 2.5.2 输入参数逐项说明

#### 参数 1:`scores` — 路由分数张量

| 属性 | 值 |
|---|---|
| 形状 | `[num_tokens, num_experts]` |
| 数据类型 | `bfloat16` / `float16` / `float32`（kernel 自动 dispatch） |
| 内存布局 | row-major, CUDA device memory |
| 物理意义 | 每个 token 对每个专家的路由**原始分数**(可能是 logits 或已激活的分数) |

**典型值域**:
- 当 `scoring_func=SCORING_NONE` 时:已在上游做过激活(如 softmax),值域通常 `[0, 1]`
- 当 `scoring_func=SCORING_SIGMOID` 时:原始 logits,值域 `(-∞, +∞)`,kernel 内部会做 sigmoid

**示例**（DeepSeek-V3, num_tokens=128）:
```
scores.shape = [128, 256]
scores.dtype = torch.bfloat16
scores.device = cuda:0
内存占用 = 128 × 256 × 2 bytes = 64 KB
```

#### 参数 2:`n_group` (C++) / `num_expert_group` (Python) — 分组数 G

| 属性 | 值 |
|---|---|
| 类型 | `int64_t` |
| 约束 | `n_group > 0` 且 `n_group ≤ 32` 且 `num_experts % n_group == 0` |
| 物理意义 | 将 `num_experts` 个专家**等分**为 G 组,同一组内的专家通常在同一 GPU/node 上 |

**为什么需要分组?**
- **负载均衡**:DeepSeek-V3 将 256 专家分 8 组,每组 32 专家可分配到 32-way EP 的不同 GPU
- **路由多样性**:强制从多个组中选择,避免所有 top-k 全集中在一组(导致某些 GPU 过载)

**典型值**:
| 模型 | n_group |
|---|---|
| DeepSeek-V2-Lite | 1（无分组） |
| DeepSeek-V2 | 8 |
| DeepSeek-V3/R1 | 8 |
| Kimi-K2 | 1 |

#### 参数 3:`topk_group` — 选取的组数

| 属性 | 值 |
|---|---|
| 类型 | `int64_t` |
| 约束 | `topk_group > 0` 且 `topk_group ≤ n_group` 且 `topk_group ≤ 4` (small kernel 路径) |
| 物理意义 | 从 G 组中挑出**分数最高的 `topk_group` 组**作为候选池 |

**组分数定义**: 组内**最高两个专家分数之和**(Top-2 sum)。源码 `topk_with_k2` 函数:
```cpp
// 组内找 max1 和 max2
// group_score = max1 + max2
smemGroupScores[warpIdx] = max1 + max2;
```

为什么用 Top-2 而不是 Top-1?
- 单个高分专家可能是噪声,Top-2 更稳健地反映组的整体竞争力
- 如果 topk_group × num_experts_per_group 足够大,Top-2 也能避免候选池不足

#### 参数 4:`topk` — 每 token 最终选取的专家数 K

| 属性 | 值 |
|---|---|
| 类型 | `int64_t` |
| 约束 | `topk > 0` 且 `topk ≤ 32` 且 `topk ≤ topk_group × (num_experts / n_group)` |
| 物理意义 | 每个 token 被路由到**最相关的 K 个专家**,K 越大推理质量越高,但 MoE 计算量线性增长 |

**约束的物理含义**: 候选池大小 = `topk_group × experts_per_group`,必须 ≥ `topk`,否则候选不够。

**典型值**:
| 模型 | topk |
|---|---|
| DeepSeek-V2 | 6 |
| DeepSeek-V3 | 8 |
| Mixtral-8×7B | 2 |
| Nemotron MoE | 22（特殊单组大 K） |

#### 参数 5:`renormalize` — 权重归一化开关

| 属性 | 值 |
|---|---|
| 类型 | `bool` |
| 物理意义 | 选出 top-k 后,是否将权重归一化为 Σ = 1 |

**开启时的计算**:
```cpp
float redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
float finalScore = scoreNorm * routedScalingFactor / (redNorm + 1e-20);
// 所有 top-k 权重之和 = routed_scaling_factor
```

**关闭时**: `finalScore = scoreNorm * routedScalingFactor`,直接使用原始 sigmoid 分数。

**实践**:
- DeepSeek 系列:**true** — 确保每个 token 的专家权重分布可比较
- 部分模型:false — 信任原始 sigmoid/softmax 分数

#### 参数 6:`routed_scaling_factor` — 路由缩放因子

| 属性 | 值 |
|---|---|
| 类型 | `double` (C++) / `float` (Python) |
| 物理意义 | 对 top-k 权重**整体乘一个缩放因子**,补偿 MoE 层激活值大小 |

**典型值**:
- DeepSeek-V3: `2.5` — 因为 8 个专家平均激活,需要放大以匹配 dense 模型的激活规模
- DeepSeek-V2: `1.0` — 默认无缩放
- 大部分 Mixtral: `1.0`

**数学上**: 最终 MoE 输出 = Σᵢ (topk_weight[i] × expert_output[i])。缩放因子让训练/推理保持一致的输出范围。

#### 参数 7:`bias` — 专家路由 correction bias

| 属性 | 值 |
|---|---|
| 形状 | `[num_experts]` |
| 数据类型 | `bfloat16` / `float16` / `float32`（独立 dispatch） |
| 物理意义 | 每个专家的**路由修正 bias**,**仅影响选择,不影响输出权重** |

**关键设计**（DeepSeek-V3 的 auxiliary-loss-free balancing）:
```cpp
// 选择时使用 biased score
T scoreBias = sigmoid(score) + bias[expert];
// 但最终权重用 unbiased score
float scoreNorm = smemScoreSigmoid[expertIdx];  // 无 bias
```

**作用**: 在训练中动态调整 `bias` 可以**主动把流量推离过载的专家**(给过载专家加负 bias),实现 load balancing 而无需辅助损失函数。

**示例**:
```
expert 0: bias =  0.02 (低负载,鼓励路由)
expert 5: bias = -0.15 (高负载,抑制路由)
expert 10: bias = 0.0 (均衡)
```

#### 参数 8:`scoring_func` — 激活函数类型

| 值 | 枚举名 | 含义 |
|---|---|---|
| `0` | `SCORING_NONE` | 输入已激活,kernel 内不做任何激活 |
| `1` | `SCORING_SIGMOID` | kernel 内对 scores 做 `sigmoid_accurate` |

**为什么用整数而不是字符串?** PyTorch custom op 的 schema 不支持枚举,用 int 最稳健。

**典型选择**:
| 模型 | scoring_func |
|---|---|
| DeepSeek-V3 (noisy gate) | 1 (SIGMOID) |
| DeepSeek-V2 | 0 (NONE,上游已 softmax) |

### 2.5.3 输出参数逐项说明

#### 输出 1:`topk_values` — 路由权重

| 属性 | 值 |
|---|---|
| 形状 | `[num_tokens, topk]` |
| 数据类型 | **固定 `torch.float32`**（无论输入 dtype） |
| 物理意义 | 每个 token 选中的 top-k 专家的**最终路由权重** |

**为什么固定 float32?**
```cpp
// torch_bindings 层强制创建 fp32 输出
torch::Tensor topk_values = torch::empty(
    {num_tokens, topk}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
```
- 消除下游 Python 层的类型转换(原来需要 `.float()` 再传给 fused MoE)
- 精度足够累加权重 × expert_output(权重通常是小数)

**示例**（DeepSeek-V3, 128 tokens, topk=8）:
```
topk_values.shape = [128, 8]
topk_values[0] = [0.142, 0.131, 0.128, 0.125, 0.124, 0.120, 0.118, 0.112]
                 # 和 ≈ 1.0 (若 renormalize=true, scaling=1.0)
```

#### 输出 2:`topk_indices` — 专家索引

| 属性 | 值 |
|---|---|
| 形状 | `[num_tokens, topk]` |
| 数据类型 | **固定 `torch.int32`** |
| 物理意义 | 每个 token 选中的 top-k 专家的**全局 expert_id** |
| 值域 | `[0, num_experts - 1]` |

**示例**:
```
topk_indices.shape = [128, 8]
topk_indices[0] = [72, 205, 142, 18, 85, 198, 130, 7]
                  # 这些是 scores 第二维的索引
```

**对应关系**:
```
对于 token i, 第 j 个选中的专家:
    expert_id = topk_indices[i, j]
    weight    = topk_values[i, j]
    → 在 MoE forward 中:
       contribution[i] += weight × expert[expert_id].forward(x[i])
```

**不排序的说明**: top-k 的 8 个专家**按分数降序排列**(输出顺序 = 分数从高到低),但 expert_id 本身不排序。

### 2.5.4 C++ Kernel 参数映射

从 Python 一路传到 CUDA kernel 的参数映射:

```cpp
// Python: grouped_topk(scores, n_group, topk_group, topk, renormalize,
//                     routed_scaling_factor, bias, scoring_func)
//   ↓
// C++: grouped_topk(scores, n_group, topk_group, topk, renormalize,
//                  routed_scaling_factor, bias, scoring_func)
//   ↓
// invokeNoAuxTc<T, BiasT, IdxT, SF>(
//     T*           scores,                 // 输入分数 (device ptr)
//     float*       topk_values,            // 输出权重 (device ptr)
//     IdxT*        topk_indices,           // 输出索引 (device ptr)
//     BiasT const* bias,                   // bias (device ptr)
//     int64_t      num_tokens,             // 从 scores.shape[0] 推导
//     int64_t      num_experts,            // 从 scores.shape[1] 推导
//     int64_t      n_group,
//     int64_t      topk_group,
//     int64_t      topk,
//     bool         renormalize,
//     double       routed_scaling_factor,
//     bool         enable_pdl = false,
//     cudaStream_t stream = 0)
//   ↓
// Kernel<<<num_tokens, 256>>>(
//     scores, topk_values, topk_indices, bias,
//     num_tokens, n_group, topk_group, topk, num_experts,
//     num_experts / n_group,    // numExpertsPerGroup 在此计算
//     renormalize, routed_scaling_factor)
```

**推导关系**:
| Kernel 参数 | 来源 |
|---|---|
| `num_tokens` | `scores.size(0)` |
| `num_experts` | `scores.size(1)` |
| `numExpertsPerGroup` | `num_experts / n_group`(运行时除法) |

### 2.5.5 参数校验规则

在 `grouped_topk()` C++ 函数入口做的 `TORCH_CHECK` 校验（[grouped_topk_kernels.cu#L1020](../../csrc/moe/grouped_topk_kernels.cu)):

| 校验 | 规则 | 失败原因 |
|---|---|---|
| 1 | `scores` 是 2D 张量 | 必须是 `[num_tokens, num_experts]` |
| 2 | `n_group > 0` | 至少 1 组 |
| 3 | `topk > 0` | 至少选 1 个专家 |
| 4 | `topk_group > 0` | 至少选 1 组 |
| 5 | `topk_group ≤ n_group` | 不能选超过总组数的组 |
| 6 | `num_experts % n_group == 0` | 专家必须能整除分组,保证每组大小相等 |
| 7 | `n_group ≤ 32` | 每组 1 warp 映射,n_group ≤ 32 意味着单 block 最多 1024 线程 |
| 8 | `topk ≤ 32` | warp-level reduce 要求 topk < WARP_SIZE |
| 9 | `topk ≤ topk_group × (num_experts / n_group)` | 候选池要够大 |
| 10 | `scoring_func ∈ {0, 1}` | 目前仅支持 NONE 和 SIGMOID |

**触发示例**:
```python
ops.grouped_topk(scores, n_group=8, topk_group=4, topk=8, ...)
# 若 num_experts = 257 (非 8 的倍数) → TORCH_CHECK fail:
#   "num_experts should be divisible by n_group"
```

### 2.5.6 典型模型参数配置表

| 模型 | num_experts | n_group | experts_per_group | topk_group | topk | scoring | renorm | scaling | 走的 kernel |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V2 | 160 | 8 | 20 | 3 | 6 | NONE (上游已 softmax) | true | 1.0 | small (multi-group) |
| DeepSeek-V3 / R1 | 256 | 8 | 32 | 4 | 8 | SIGMOID | true | 2.5 | **small (multi-group)** |
| Kimi-K2 | 384 | 1 | 384 | 1 | 8 | SIGMOID | true | 1.0 | small (single-group) |
| Nemotron MoE | 512 | 1 | 512 | 1 | 22 | SIGMOID | true | 1.0 | small (single-group, 特殊 K=22) |
| Mixtral-8×7B (无分组) | 8 | 1 | 8 | 1 | 2 | softmax (上游) | true | 1.0 | 走 `topk_softmax_kernels.cu` |

---

## 3. 算子总体架构

### 3.1 两条执行路径

```
invokeNoAuxTc()
  ├─ is_multi_group == true  → grouped_topk_fused_small_expert_count_kernel
  │   条件: n_group > 1, num_experts ≤ 256,
  │          experts_per_group ≤ 32, topk ≤ 8, topk_group ≤ 4
  │   ★ DeepSeek-V3 (256 experts, 8 groups) 走此路径
  │
  ├─ is_single_group == true → grouped_topk_fused_small_expert_count_kernel (UseGroups=false)
  │   条件: n_group == 1, topk_group == 1, num_experts ≤ 512
  │   适用: Nemotron (512 experts, topk=22), Kimi-K2 (384 experts)
  │
  └─ else → grouped_topk_fused_kernel (通用路径)
      条件: 以上都不满足（超大专家数或超大 topk）
```

### 3.2 路径选择逻辑 invokeNoAuxTc

```cpp
// DeepSeek-V3 的判断:
is_multi_group = (n_group > 1)           // 8 > 1 ✓
    && (num_experts <= 256)              // 256 ≤ 256 ✓
    && (experts_per_group <= 32)         // 32 ≤ 32 ✓
    && (experts_per_group * topk_group <= 128)  // 32*4=128 ≤ 128 ✓
    && (topk <= 8)                       // 8 ≤ 8 ✓
    && (topk_group <= 4)                 // 4 ≤ 4 ✓
→ true, 选择 small expert count kernel

// 模板参数:
MaxNumExperts = NumDeepseekExperts = 256
UseGroups = true
num_threads = 256
```

---

## 4. 路径一:small expert count kernel（DeepSeek-V3 走此路径）

### 4.1 Kernel 启动参数

#### 4.1.1 Kernel 函数签名（模板）

```cpp
template <typename T, typename BiasT, typename IdxT, ScoringFunc SF,
          int MaxNumExperts, bool UseGroups,
          int MaxNumTopExperts = DefaultMaxNumTopExperts>
__global__ void grouped_topk_fused_small_expert_count_kernel(
    T*           scores,              // [输入] raw 分数  [num_tokens, num_experts]
    float*       topkValues,          // [输出] 权重      [num_tokens, topk], fp32
    IdxT*        topkIndices,         // [输出] 专家索引  [num_tokens, topk], int32
    BiasT const* routingBias,         // [输入] bias      [num_experts]
    int64_t      numTokens,           // [输入] token 数
    int64_t      numGroup,            // [输入] 组数 G
    int64_t      topkGroup,           // [输入] 选中组数
    int64_t      topk,                // [输入] top-K
    int64_t      numExperts,          // [输入] 专家总数
    int64_t      numExpertsPerGroup,  // [输入] 每组专家数 = numExperts/numGroup
    bool         renormalize,         // [输入] 是否归一化
    double       routedScalingFactor  // [输入] 权重缩放因子
);
```

**模板参数（编译期常量）**:

| 模板参数 | DeepSeek-V3 实例化值 | 含义 |
|---|---|---|
| `T` | `__nv_bfloat16` | scores 的数据类型 |
| `BiasT` | `__nv_bfloat16` | bias 的数据类型（可与 T 不同） |
| `IdxT` | `int32_t` | topk_indices 的索引类型 |
| `SF` | `SCORING_SIGMOID` (=1) | 激活函数 |
| `MaxNumExperts` | `256` (=NumDeepseekExperts) | 专家数上界,决定 blockDim 和 shared memory 大小 |
| `UseGroups` | `true` | 是否使用分组(multi-group path 为 true) |
| `MaxNumTopExperts` | `8` | top-K 上界 |

**运行时参数（kernel 调用时传入）**: 见上表「输入」标注的参数,这些是每次 kernel 启动时可变的。

#### 4.1.2 Grid/Block 维度

```
gridDim  = num_tokens     （每个 block 处理 1 个 token）
blockDim = 256            （= MaxNumExperts = NumDeepseekExperts）
动态 shared memory = 0    （全部用静态 shared memory）
```

**具体场景**: 假设 batch 中有 128 个 token:
```
gridDim  = 128
blockDim = 256
→ 128 个 block × 256 线程 = 32,768 个线程
→ 每个 block 用 8 个 warp (256/32 = 8)
```

#### 4.1.3 线程到专家的映射

在 `UseGroups=true` 路径下:
```
threadIdx.x = [0..255]
warpIdx = threadIdx.x / 32 = [0..7]  → 对应 group [0..7]
laneIdx = threadIdx.x % 32 = [0..31] → 对应组内 expert [0..31]

threadExpert = warpIdx * 32 + laneIdx = threadIdx.x
```

即 **1 个线程 = 1 个专家**,完美的 1:1 映射!

| threadIdx.x | warpIdx (组号) | laneIdx (组内专家号) | 全局 expert_id |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 1 |
| ... | ... | ... | ... |
| 31 | 0 | 31 | 31 |
| 32 | 1 | 0 | 32 |
| 33 | 1 | 1 | 33 |
| ... | ... | ... | ... |
| 255 | 7 | 31 | 255 |

#### 4.1.4 每个线程访问的内存

| 访问目标 | 读写 | 数据量/线程 | 总计/block |
|---|---|---|---|
| `scores[blockIdx.x * 256 + threadIdx.x]` | 读 Global | 2 bytes (bf16) | 512 B |
| `bias[threadIdx.x]` | 读 Global | 2 bytes | 512 B（全 block 广播） |
| `smemScoreSigmoid[threadIdx.x]` | 读/写 Shared | 4 bytes | 1024 B |
| `smemScoreBias[threadIdx.x]` | 读/写 Shared | 4 bytes | 1024 B |
| `smemGroupScores[warpIdx]` | 读/写 Shared | 4 bytes（仅 warp leader 写） | 32 B |
| `topkValues[blockIdx.x * 8 + lane]` | 写 Global | 4 bytes（仅 lane<8） | 32 B |
| `topkIndices[blockIdx.x * 8 + lane]` | 写 Global | 4 bytes（仅 lane<8） | 32 B |

### 4.2 共享内存布局

```cpp
__shared__ float smemScoreSigmoid[256];   // 每个 expert 的 sigmoid(score)
__shared__ float smemScoreBias[256];      // 每个 expert 的 sigmoid(score) + bias
__shared__ float smemGroupScores[8];      // 每组的 Top-2 分数之和
```

总计: 256×4 + 256×4 + 8×4 = **2080 bytes** 静态 shared memory。

### 4.3 阶段一:每组 Top-2 分数计算

**目标**: 每个 warp（=每组 32 个专家）计算组内**最大两个分数之和**,作为组的竞争力评分。

```cpp
// 每个线程读取自己负责的 1 个专家的 score
float score = scores[blockIdx.x * 256 + threadExpert];  // 从 Global Memory 读

// 应用 sigmoid
float scoreSigmoid = sigmoid_accurate(score);
//   = 0.5 * tanh(0.5 * score) + 0.5

// 写入 shared memory 供后续使用
smemScoreSigmoid[threadExpert] = scoreSigmoid;

// 加 bias
float scoreBias = scoreSigmoid + bias[threadExpert];
smemScoreBias[threadExpert] = scoreBias;
```

然后执行**组内 Top-2 归约**:

```cpp
// reduceTopK<2>(warp, topExpGroupScores, topExpGroupIdx, scoreBias, threadExpert, -inf)
// 使用 cg::reduce + TopKRedType 打包比较
// 第一轮: 32 个线程的 scoreBias → 全组最大值 max1
// 第二轮: 屏蔽 max1 → 全组第二大值 max2

// warp 内线程 0 写出:
smemGroupScores[warpIdx] = max1 + max2;
```

**具体数据流示例**（Group 0, 专家 0~31）:

假设组内各专家的 `scoreBias` 值:
```
expert  0: 0.82   expert  1: 0.45   expert  2: 0.91   ...
expert 15: 0.88   expert 16: 0.23   ...
expert 31: 0.56
```

Top-2 归约过程（详见第 6 节）:
```
轮次 1: cg::reduce(32 threads, cg::greater) → max1 = 0.91 (expert 2)
轮次 2: expert 2 的值被屏蔽为 -inf → cg::reduce → max2 = 0.88 (expert 15)
Group 0 score = 0.91 + 0.88 = 1.79
```

**`__syncthreads()`** — 确保所有 8 组的分数都写入 `smemGroupScores`。

### 4.4 阶段二:选择 Top-K 组

**只有 warp 0（32 个线程）参与**,其余 7 个 warp 在此阶段退出。

```cpp
if (warpIdx == 0) {
    // lane 0~7 各读取一个组分数, lane 8~31 读 -inf
    float groupScore = (laneIdx < 8) ? smemGroupScores[laneIdx] : -inf;

    // reduceTopK<4>(warp, topGroups, topGroupIdx, groupScore, laneIdx, -inf)
    // 从 8 个组中选出 top-4 组
}
```

假设 8 组分数:
```
Group 0: 1.79   Group 1: 1.52   Group 2: 2.01   Group 3: 1.33
Group 4: 1.88   Group 5: 1.21   Group 6: 1.95   Group 7: 1.45
```

Top-4 组归约（4 轮 warp reduce）:
```
轮次 1 → topGroups[0] = 2.01, topGroupIdx[0] = 2
轮次 2 → topGroups[1] = 1.95, topGroupIdx[1] = 6
轮次 3 → topGroups[2] = 1.88, topGroupIdx[2] = 4
轮次 4 → topGroups[3] = 1.79, topGroupIdx[3] = 0
```

**选中的 4 组**: Group 2, 6, 4, 0 → 候选专家池 = 4×32 = 128 个专家。

### 4.5 阶段三:从选中组中选出全局 Top-K 专家

仍然**只有 warp 0 工作**。需要从 4 组 × 32 专家 = 128 个候选中选 8 个。

```cpp
// 遍历 4 个选中组
for (int ii = 0; ii < 4; ++ii) {
    int groupIdx = topGroupIdx[ii];
    expertIdxGroup[ii] = groupIdx * 32 + laneIdx;  // 每 lane 负责 1 个专家

    // 从 shared memory 读取 scoreBias
    expertScoreGroup[ii] = smemScoreBias[expertIdxGroup[ii]];
}
```

每个 lane 现在持有 4 个候选（每组 1 个）:

| laneIdx | 候选 0 (Group 2) | 候选 1 (Group 6) | 候选 2 (Group 4) | 候选 3 (Group 0) |
|---|---|---|---|---|
| 0 | expert 64 | expert 192 | expert 128 | expert 0 |
| 1 | expert 65 | expert 193 | expert 129 | expert 1 |
| ... | ... | ... | ... | ... |
| 31 | expert 95 | expert 223 | expert 159 | expert 31 |

然后调用 `reduceTopK<8, float, 4>(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup, -inf, 8)`:

- 32 个线程,每线程 4 个候选值 → 总共 128 个候选
- 通过 8 轮 warp reduce 选出 Top-8

### 4.6 阶段四:写出最终结果（renormalize + scaling）

```cpp
// lane 0~7 各负责 1 个 top expert
int32_t expertIdx = (laneIdx < 8) ? topExperts[laneIdx] : 255;
float scoreNorm = (laneIdx < 8) ? smemScoreSigmoid[expertIdx] : 0.0;

// renormalize: 除以 top-8 的 sigmoid 分数总和
float redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
float finalScore = scoreNorm * routedScalingFactor / redNorm;

// 写出
if (laneIdx < 8) {
    topkValues[laneIdx] = finalScore;
    topkIndices[laneIdx] = expertIdx;
}
```

**关键点**: 最终输出的权重是**无 bias 的 sigmoid 分数**（从 `smemScoreSigmoid` 读取）,不是 `scoreBias`。bias 仅影响选择,不影响权重——与 `topk_softmax_kernels.cu` 的设计一致。

### 4.7 具体数据流实例（DeepSeek-V3）

以 1 个 token 为例，完整的 4 阶段流水线:

```
输入: scores[256] (bf16 raw logits)  +  bias[256] (correction bias)
      ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段一: 8 个 warp 并行, 每 warp 处理 32 个专家               │
│ Warp 0 (experts 0-31):   sigmoid + bias → Top-2 → score=1.79│
│ Warp 1 (experts 32-63):  sigmoid + bias → Top-2 → score=1.52│
│ Warp 2 (experts 64-95):  sigmoid + bias → Top-2 → score=2.01│
│ ...                                                          │
│ Warp 7 (experts 224-255): sigmoid + bias → Top-2 → score=1.45│
│ __syncthreads()                                              │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段二: 仅 Warp 0, 从 8 个组分数中选 Top-4 组               │
│ [1.79, 1.52, 2.01, 1.33, 1.88, 1.21, 1.95, 1.45]          │
│ → 选中: Group 2(2.01), Group 6(1.95),                      │
│         Group 4(1.88), Group 0(1.79)                        │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段三: 仅 Warp 0, 从 4组×32专家=128 候选中选 Top-8         │
│ 每 lane 持有 4 个候选 (scoreBias 值)                        │
│ 8 轮 warp reduce → 全局 Top-8 专家                          │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段四: Lane 0~7 写出                                       │
│ 权重 = sigmoid(score) / Σ(sigmoid) × scaling_factor        │
│ 索引 = 全局 expert_id                                       │
└─────────────────────────────────────────────────────────────┘

输出: topk_values[8] (float32)  +  topk_indices[8] (int32)
```

---

## 5. 路径二:grouped_topk_fused_kernel（大专家数通用路径）

当 `num_experts > 256` 或 `experts_per_group > 32` 时走此路径。

### 5.1 Kernel 启动参数

```cpp
gridDim  = num_tokens
blockDim = n_group × 32    // 每组 1 个 warp
dynamicSmemBytes = WarpSelect内部缓冲 + n_group × sizeof(T)
```

**假设 n_group=8, 专家数 >256 的场景**:
```
blockDim = 8 × 32 = 256
动态 shared memory ≈ 8×32×(2+4) + 16 + 8×2 = 1568 bytes (bf16)
```

### 5.2 Phase 1:每组 Top-2 分数（warp 级 cooperative_groups reduce）

```cpp
// 每个 warp (= 1 组) 的 32 个线程以 stride 循环扫描组内所有专家
// 例如 experts_per_group=64: 每线程处理 2 个专家
for (int i = lane_id; i < 64; i += 32) {
    T value = sigmoid(input[group_offset + i]) + bias[group_offset + i];
    // 维护本地 top-2 (largest, second_largest)
}

// warp reduce 得到组 top-1
T max1 = cg::reduce(tile, largest, cg::greater<T>());
// 屏蔽 max1, 再 reduce 得到 top-2
T max2 = ...;
// 写入 shared memory
s_group_scores[warp_id] = max1 + max2;
```

与 small kernel 的区别: 这里 experts_per_group 可能 > 32,需要 stride 循环。

### 5.3 Phase 2:WarpSelect 选组 + 选专家

只有 warp 0 继续工作,使用 **WarpSelect**（基于 Bitonic Sort 的 warp 级 Top-K）:

```cpp
// Step 1: 选 top-k_group 组
WarpSelect<32, true, T, int32_t, true> group_sel(topk_group, -inf);
group_sel.add(gscore, lane_id);
group_sel.done();

// Step 2: 遍历选中组,将组内所有专家加入 expert_sel
WarpSelect<32, true, T, int32_t, true> expert_sel(topk, -inf);
for (int g = 0; g < topk_group; ++g) {
    int gid = __shfl_sync(FULL_WARP_MASK, sel_gid_lane, g);
    for (int i = lane_id; i < experts_per_group; i += 32) {
        expert_sel.add(cand, offset + i);  // 流式加入
    }
}
expert_sel.done();

// Step 3: 输出 + renormalize（同 small kernel）
```

**WarpSelect 的优势**: 不需要一次性将所有候选放入数组——它是**流式**的,每次 `add()` 一个候选,内部维护一个大小为 k 的排序缓冲,用 Bitonic Merge 合并。内存开销恒定为 O(k),与候选总数无关。

---

## 6. 核心算法:reduce_topk 详解

这是 `moeTopKFuncs.cuh` 中的核心,被 small expert count kernel 调用。

### 6.1 打包比较技术 TopKRedType

**核心创新**: 将 `(value, index)` 打包成一个 64-bit 整数,使得 **标准的 warp max reduce 同时完成 value 比较和 index 传递**。

```cpp
// 对于 float (4 bytes):
TypeCmp = uint64_t  // 高 32 位 = value, 低 32 位 = 65535 - index
IdxT = int32_t

static TypeCmp makeCmpVal(float val, int32_t idx) {
    // 1. TwiddleIn: 将 float 的位模式转为可正确排序的无符号整数
    //    (处理 IEEE 754 的符号位问题)
    auto valueBits = cub::Traits<float>::TwiddleIn(val);

    // 2. 打包: [valueBits(32-bit)] [65535 - idx(32-bit)]
    return (uint64_t(valueBits) << 32) | (65535 - idx);
}
```

**为什么 `65535 - idx`?**
- 当两个 value 相同时,我们希望**更小的 index 胜出**（稳定排序）
- `65535 - idx` 让小 index 对应大值,因此 `max()` 自然选择更小 index
- 例: expert 3 → 65532, expert 5 → 65530, max 选 65532 → expert 3 ✓

### 6.2 K 轮 warp reduce 选 Top-K

```cpp
template <int K, typename Type>
__device__ void reduceTopK(warp, out[K], outIdx[K], value, idx, minValue, actualK) {
    TopKRedType topK{value, idx};  // 当前线程的打包候选
    TypeCmp packedMax{};

    for (int kk = 0; kk < actualK; ++kk) {
        // 屏蔽: 如果本线程的候选等于上一轮的 winner, 替换为 -inf
        topK = (kk > 0 && packedMax == topK.compValIdx) ? {minValue, idx} : topK;

        // warp reduce: 32 线程 → 1 个最大值
        packedMax = topK.reduce(warp);  // = cg::reduce(warp, topK, cg::greater)

        // 解包 winner
        unpack(out[kk], outIdx[kk], packedMax);
    }
}
```

**具体执行过程**（选 Top-4 组,8 个组分数分布在 lane 0~7）:

```
初始状态: lane 0=1.79, lane 1=1.52, lane 2=2.01, lane 3=1.33,
          lane 4=1.88, lane 5=1.21, lane 6=1.95, lane 7=1.45
          lane 8~31 = -inf

轮次 0 (kk=0):
  cg::reduce(max) → packedMax = pack(2.01, 2)
  out[0]=2.01, outIdx[0]=2

轮次 1 (kk=1):
  lane 2 检测: packedMax == topK → 屏蔽为 -inf
  cg::reduce(max) → packedMax = pack(1.95, 6)
  out[1]=1.95, outIdx[1]=6

轮次 2 (kk=2):
  lane 6 被屏蔽
  cg::reduce(max) → packedMax = pack(1.88, 4)
  out[2]=1.88, outIdx[2]=4

轮次 3 (kk=3):
  lane 4 被屏蔽
  cg::reduce(max) → packedMax = pack(1.79, 0)
  out[3]=1.79, outIdx[3]=0
```

**复杂度**: K 轮 × 5 级 warp reduce（log₂32=5 次 `__shfl`）= K×5 条 shuffle 指令。K=8 → 40 条。

### 6.2.1 多候选版本 reduceTopK (N=4)

当每线程持有 N 个候选时（阶段三,每 lane 持有 4 组各 1 个候选）:

```cpp
template <int K, typename Type, int N>
__device__ void reduceTopKFunc(warp, out[K], outIdx[K], value[N], idx[N], minValue, actualK) {
    TopKRedType topK[N];  // 打包 N 个候选

    // 先对本线程的 N 个候选排序(用 Sort 网络)
    Sort<N>::run(topK);  // 4 元素: 5 次比较交换

    for (int kk = 0; kk < actualK; ++kk) {
        // 如果 topK[0] 是上一轮 winner,把队列左移,最后补 -inf
        if (packedMax == topK[0]) {
            topK[0]=topK[1]; topK[1]=topK[2]; topK[2]=topK[3]; topK[3]={-inf,...};
        }
        packedMax = topK[0].reduce(warp);  // 仍然只对 topK[0] 做 warp reduce
        unpack(out[kk], outIdx[kk], packedMax);
    }
}
```

**关键洞察**: 每线程的 N 个候选先本地排序,然后每轮 reduce 只拿 `topK[0]`（本地最大）参与 warp reduce。当本地最大被选走后,整个数组左移,`topK[1]` 上位。这避免了维护全局"已选集合"。

---

## 7. 核心算法:WarpSelect 详解

WarpSelect 用于通用路径（路径二），是一个**流式 warp 级 Top-K 选择器**。

### 7.1 Bitonic Sort 在 warp 内的实现

基本 32 元素 Bitonic Sort 纯用 `__shfl_xor_sync`:

```cpp
// 4 个 stage (2^4 = 16, 处理 32 元素只需 5 个 stage，但这里配合 BitonicMerge)
for (int stage = 0; stage < 4; ++stage) {
    for (int stride = (1 << stage); stride > 0; stride /= 2) {
        bool reverse = (lane >> stage) & 2;
        bool is_second = lane & stride;

        T other = __shfl_xor_sync(FULL_WARP_MASK, *val_arr, stride);
        // 比较交换:根据排序方向决定是否交换
        if (is_better) { *val_arr = other; *idx_arr = other_idx; }
    }
}
```

**全程无 shared memory,纯寄存器 + warp shuffle**。

### 7.2 WarpSelect::add() 的缓冲合并机制

```
候选流: c₀ c₁ c₂ ... cₙ
           ↓
     ┌─── 过滤 ───┐  (只有 > k-th 的值才进入缓冲)
     ↓             ↓
  smem buffer (32 slots)
     ↓
  当 buffer 满:
     BitonicSort(buffer)        → 32 个排序好的候选
     BitonicMerge(buffer, top-k) → 合并到 top-k 数组
     更新 k-th 阈值
```

- **过滤**: `__ballot_sync` 检查哪些 lane 的候选 > 当前 k-th,只有通过的才写入 buffer
- **缓冲**: shared memory 中 32 个槽位,满了就触发合并
- **合并**: Bitonic Merge 将 buffer 和当前 top-k 合并,O(k log k)

这比朴素的 "K 次全扫描"（如 `moeTopK` in `topk_softmax_kernels.cu`）高效得多。

---

## 8. Scoring 函数:sigmoid_accurate

```cpp
__device__ inline float sigmoid_accurate(float x) {
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}
```

**为什么用 `tanh` 而不是 `1/(1+exp(-x))`?**

数学上等价: $\sigma(x) = \frac{1}{1+e^{-x}} = \frac{1}{2}\tanh\frac{x}{2} + \frac{1}{2}$

但 `tanhf` 在 CUDA 中有专门的硬件指令(`MUFU.TANH`),比 `__expf` + 除法更快且精度更好。这来自 TensorRT-LLM 的优化实践。

---

## 9. 性能特征与对比

### Small expert count kernel (DeepSeek-V3)

| 指标 | 值 |
|---|---|
| gridDim | num_tokens |
| blockDim | 256 (= 8 warps) |
| 线程利用率（阶段一） | 256/256 = **100%** |
| 线程利用率（阶段二~四） | 32/256 = **12.5%**（仅 warp 0） |
| Global Memory 读 | 1 次 (scores) + 1 次 (bias) |
| Global Memory 写 | 1 次 (topk_values + topk_indices) |
| Shared Memory | 2080 bytes 静态 |
| __syncthreads() | **1 次**（阶段一→二的交接） |
| Warp shuffle 指令 | ≈ 2×5 + 4×5 + 8×5 = **70** |

### 与 topk_softmax_kernels.cu 的 moeTopK 对比

| 对比项 | grouped_topk (small kernel) | moeTopK (通用路径) |
|---|---|---|
| 分组竞争 | ✓（先选组再选专家） | ✗（直接全局 Top-K） |
| Kernel 数量 | 1 个 | 2 个 (softmax + topk) |
| 归约方式 | Warp shuffle (cg::reduce) | Shared memory (cub::BlockReduce) |
| Top-K 算法 | 打包比较 + K 轮 reduce | K 次全局扫描 |
| 屏蔽已选 | 打包比较自动屏蔽 | 读 Global Memory indices[] |
| Global Memory 访问 | 1 遍读 | softmax 3 遍 + topk K 遍 |
| Shared Memory | 2KB 静态 | BlockReduce tmpStorage |
| 适用模型 | DeepSeek-V2/V3 | 通用 MoE |

---

## 10. 架构图

### 四阶段流水线 + reduceTopK 机制

![grouped-topk-pipeline](grouped-topk-pipeline.png)

> SVG 源文件: [grouped-topk-pipeline.svg](grouped-topk-pipeline.svg)

---

## 11. 总结

`grouped_topk_kernels.cu` 是 vLLM 为 **DeepSeek-V2/V3 等使用分组路由的 MoE 模型**量身优化的 CUDA kernel,核心设计理念:

1. **线程-专家 1:1 映射** — 256 个线程精确对应 256 个专家,无空转（阶段一 100% 利用率）
2. **Warp-组 1:1 映射** — 8 个 warp 分别负责 8 个组,天然并行
3. **打包比较** — value+index 打包成 64-bit,一条 reduce 指令同时完成比较和索引传递
4. **全寄存器+shuffle** — 从阶段二开始完全在寄存器和 warp shuffle 中完成,零 Global Memory 访问
5. **单 kernel 融合** — sigmoid + bias + 选组 + 选专家 + renormalize 全部融合在一个 kernel 中

对比通用路径（`moeSoftmax` + `moeTopK`）,此 kernel 将 Global Memory 访问从 ~10 次降至 2 次,kernel 启动从 2 次降至 1 次,是一个教科书级的 kernel 融合优化案例。
