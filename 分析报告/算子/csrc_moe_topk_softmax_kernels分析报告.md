# vLLM MoE Top-K Softmax CUDA 算子分析报告

> **分析对象**: `csrc/moe/topk_softmax_kernels.cu` (848 行)
> **模块职责**: MoE 模型路由层的核心算子 —— 给每个 token 从 N 个专家里选出 top-k 个专家,并输出对应的权重
> **典型调用方**: `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py::fused_topk`
> **目标读者**: 不熟悉 CUDA 编程的 Python/PyTorch 开发者
> **适用模型示例**: DeepSeek-MoE-16B (64 experts, topk=6), Qwen3-MoE, Mixtral (8 experts, topk=2) 等

---

## 目录

- [0. 背景与问题定义](#0-背景与问题定义)
- [1. CUDA 必备概念速览](#1-cuda-必备概念速览)
- [2. 文件整体结构总览](#2-文件整体结构总览)
- [3. 辅助工具(非算子)](#3-辅助工具非算子)
- [4. 通用路径:3 个独立算子](#4-通用路径3-个独立算子)
  - [4.1 `moeSoftmax` kernel](#41-moesoftmax-kernel)
  - [4.2 `moeSigmoid` kernel](#42-moesigmoid-kernel)
  - [4.3 `moeTopK` kernel](#43-moetopk-kernel)
- [5. 融合快速路径:`topkGating` kernel](#5-融合快速路径topkgating-kernel)
  - [5.4.2a 向量化加载过程详解（以 DeepSeek-MoE-16B bf16 为例）](#542a-向量化加载过程详解以-deepseek-moe-16b-bf16-为例)
  - [5.4.4a Top-K 线程内 argmax 详解（以 DeepSeek-MoE-16B bf16, topk=6 为例）](#544a-top-k-线程内-argmax-详解以-deepseek-moe-16b-bf16-topk6-为例)
- [6. Host 端调度层](#6-host-端调度层)
- [7. Python 端调用链](#7-python-端调用链)
- [8. 完整调用链路图](#8-完整调用链路图)
- [9. DeepSeek-MoE-16B 端到端实战](#9-deepseek-moe-16b-端到端实战)
- [10. 性能设计要点总结](#10-性能设计要点总结)

---

## 0. 背景与问题定义

### 0.1 MoE 路由层做什么

MoE(Mixture of Experts,混合专家)模型在每层有 `N` 个专家(FFN 子模块)。对于每个输入 token,路由器 **只挑选 top-k 个专家** 参与计算,从而用同样的计算量拿到更大的参数规模。

路由层的核心计算流程:

```
输入:  gating_output  形状 [num_tokens, num_experts]    (bf16/fp16/fp32 的路由分数)
步骤 1: 对每行做 softmax(或 sigmoid) → 得到归一化概率
步骤 2: 对每行选出最大的 k 个元素 → 输出专家编号 + 概率值
步骤 3: (可选) 把选中的 k 个概率重新归一化,使其和为 1

输出:
  topk_weights         形状 [num_tokens, topk]           (fp32,选中专家的权重)
  topk_indices         形状 [num_tokens, topk]           (int/int64,选中专家编号)
  token_expert_indices 形状 [num_tokens, topk]           (int,用于后续排序/分组的辅助索引)
```

对应到 DeepSeek-MoE-16B: `num_experts=64`, `topk=6`, 每 step 的 token 数 = batch 内的 prompt/decode token 总数。

### 0.2 为什么不用 PyTorch 原生 `torch.topk(torch.softmax(...))`

两个原因:

1. **算子融合**: PyTorch 原生会生成 2-3 个独立 kernel(softmax、topk、可能还有 renormalize),每个都要读写一次 global memory。本文件用一个 kernel 把全部步骤做完,显著减少显存带宽压力。
2. **针对 MoE 的数据布局优化**: `num_experts` 通常很小(8/16/32/64/128/256),整行数据放得下一个 warp(甚至几个线程)的寄存器,不必走 shared memory。

本文件提供两条路径:
- **融合快速路径** `topkGating`:当 `num_experts` 是 2 的幂(1、2、4、...、512)或 64 的倍数(192、320、384、448、576)时使用,一 kernel 到位。
- **通用慢速路径** `moeSoftmax + moeTopK`(或 `moeSigmoid + moeTopK`):兜底,任意 `num_experts` 都能跑。

---

## 1. CUDA 必备概念速览

读后续内容前先掌握这些基础概念(如果你已熟悉可跳过)。

### 1.1 GPU 的线程层级

CUDA 的并行度组织成 **三级嵌套**:

```
Grid (整个 kernel 启动的线程总集)
 └── Block (一组能共享 shared memory 的线程, 上限 1024)
      └── Warp (32 个线程为一组同步执行, 硬件固定)
           └── Thread (最小单位)
```

CPU 角度: kernel 启动时指定 `<<<gridDim, blockDim>>>`,对应 `blockDim.x * blockDim.y * blockDim.z` 个线程每 block,`gridDim.x * ...` 个 block。

Kernel 内部,每个线程通过 `blockIdx`、`threadIdx` 标识自己。常见模式是:
- `blockIdx.x = 行号`:一个 block 处理一行数据
- `threadIdx.x = 行内偏移`:block 内 TPB(Threads Per Block) 个线程并行处理一行

### 1.2 Warp 与 Shuffle (洗牌) 指令

- **Warp** 是硬件调度的最小单位(NVIDIA GPU 固定 32 线程;AMD MI 系列 64)。Warp 内部天然**步调一致**(SIMT),不需要 `__syncthreads` 就能互相读数据。
- **Shuffle 指令**(`__shfl_xor_sync` 等)允许 warp 内的线程**直接互相读寄存器**,不经过 shared memory,延迟非常低。本文件的融合 kernel 完全依赖 shuffle 做归约,**整个 kernel 不用 shared memory**。

典型的 butterfly reduce (蝴蝶归约):
```
mask = 16, 8, 4, 2, 1:  每次把自己和另一个线程的值做 max/sum
                        经过 log2(THREADS_PER_ROW) 轮, 所有线程拿到全局结果
```

### 1.3 Shared Memory 与 Block Reduce

- **Shared Memory** 是 block 内所有线程共享的片上存储(几十到上百 KB)。Block 内跨 warp 通信必须走它,并且需要 `__syncthreads()` 同步。
- **`cub::BlockReduce`** 是 NVIDIA CUB 库的 block 级归约模板,内部用 shared memory 实现 `max/sum/argmax` 等操作。通用路径 `moeSoftmax`/`moeTopK` 使用 CUB,因为一行(= num_experts 个元素)可能比一个 warp 大。

### 1.4 Global Memory 的访问模式

- **Coalesced Access(合并访问)**:warp 内相邻线程读相邻地址时,硬件把多个加载合并成一次 memory transaction。**这是高性能 kernel 的第一要求**。
- **向量化加载**:一条指令加载 4/8/16 字节,减少指令数。本文件用 `AlignedArray<T, N>` + 指针 reinterpret 做向量化加载。

### 1.5 CUDA 里的几个常见标注

| 语法 | 含义 |
|-----|-----|
| `__global__` | GPU 上运行的 kernel 函数,由 CPU 启动 |
| `__device__` | GPU 上运行的普通函数(不能从 CPU 直接调用) |
| `__forceinline__` | 强制内联,避免函数调用开销 |
| `__shared__` | 声明 shared memory 变量 |
| `__syncthreads()` | block 内屏障,所有线程到此处才能继续 |
| `__launch_bounds__(N)` | 告诉编译器"每个 block 最多 N 线程",用于寄存器分配优化 |
| `#pragma unroll` | 强制展开循环,编译期固定迭代次数才生效 |

### 1.6 模板与编译期常量

CUDA kernel 的模板参数 `<int TPB, typename T, ...>` 在**编译期**确定,因此编译器可以把 TPB 当成常量做激进优化(循环展开、寄存器分配、死代码消除)。本文件大量用模板,一个源代码文件最终会被编译出 **几十到上百个特化版本**(`num_experts=1/2/4/.../512`、input dtype=fp32/fp16/bf16、topk_indices type=int/uint32/int64、scoring_func=softmax/sigmoid 组合)。

---

## 2. 文件整体结构总览

```
┌─────────────────────────────────────────────────────────────┐
│                  文件:topk_softmax_kernels.cu                 │
├─────────────────────────────────────────────────────────────┤
│ § 辅助工具                                                    │
│   - AlignedArray<T, N>       向量化加载的对齐数组            │
│   - toFloat<T>()             bf16/fp16/fp32 → float 转换     │
│   - enum ScoringFunc         SOFTMAX=0 / SIGMOID=1           │
├─────────────────────────────────────────────────────────────┤
│ § 通用路径 kernel(任意 num_experts)                          │
│   - moeSoftmax <TPB, InputType>  一行一个 block              │
│   - moeSigmoid <TPB, InputType>  一行一个 block              │
│   - moeTopK    <TPB, IndType>     一行一个 block,选 k 次    │
├─────────────────────────────────────────────────────────────┤
│ § 融合快速路径 kernel                                         │
│   - topkGating <VPT, NUM_EXPERTS, WARPS_PER_CTA,             │
│                  BYTES_PER_LDG, WARP_SIZE, IndType,          │
│                  InputType, SF>                              │
│       一个 thread-group 处理一行,softmax+sigmoid+topk 全融合 │
│   - TopkConstants<...>         编译期计算每线程元素数        │
├─────────────────────────────────────────────────────────────┤
│ § Launcher 层                                                │
│   - topkGatingLauncherHelper<...>  计算 grid/block 并启动    │
│   - LAUNCH_TOPK(N, W, B)           switch-case 的宏          │
│   - topkGatingKernelLauncher<...>  根据 num_experts 分发     │
├─────────────────────────────────────────────────────────────┤
│ § Host 端入口                                                 │
│   - dispatch_topk_launch<CT, SF>   根据 IndType 分发          │
│   - topk_softmax(...)              Python 可见的入口         │
│   - topk_sigmoid(...)              Python 可见的入口         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 辅助工具(非算子)

### 3.1 `AlignedArray<T, N>` —— 向量化加载的载体

```cpp
// L43-52
template <typename T, int N, int Alignment = sizeof(T) * N>
struct alignas(Alignment) AlignedArray {
    T data[N];
};
```

**作用**: 创建一个 "**N 个 T 类型元素、对齐到 Alignment 字节**" 的结构体。

**为什么需要它**:GPU 的向量化 load 指令(如 `LDG.128`,一次读 16 字节)要求目标地址必须按 16 字节对齐。把指针 reinterpret 成 `AlignedArray<T, N>*` 后,编译器就会发射向量化指令,而不是 N 条标量 load。

**使用示例**(topkGating 里):
```cpp
using VecType = AlignedArray<float, 4>;       // 4×4=16 字节
VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(&row_chunk);
// 现在一次 copy 16 字节 = 一条 128-bit 向量化 load
```

### 3.2 `toFloat<T>()` —— 类型转 float

```cpp
// L54-63
template <typename T>
__device__ __forceinline__ float toFloat(T value) {
    if constexpr (std::is_same_v<T, float>)         return value;
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
                                                     return __bfloat162float(value);
    else if constexpr (std::is_same_v<T, __half>)   return __half2float(value);
}
```

**作用**: 把 bf16/fp16 转成 fp32 计算。`if constexpr` 是 C++17 的**编译期 if**,只有匹配的分支会进入最终二进制。

**为什么要转 fp32**: softmax 里有 `expf()` 和除法,fp16 范围/精度容易溢出;fp32 稳定得多。代价是寄存器多用一点。

### 3.3 `enum ScoringFunc`

```cpp
// L66-69
enum ScoringFunc {
    SCORING_SOFTMAX = 0,
    SCORING_SIGMOID = 1
};
```

作为**非类型模板参数**传给 `topkGating`,让编译器为两种评分函数生成两份特化代码(无运行时分支)。

---

## 4. 通用路径:3 个独立算子

当 `num_experts` 不在支持列表里(非 2 的幂、又不是 64 的倍数),vLLM 走这条通用但略慢的路径:**先 softmax/sigmoid 写到一块 workspace,再 topK**。

### 4.1 `moeSoftmax` kernel

> **源码位置**: L74-132

#### 4.1.1 Kernel 启动参数

```cpp
moeSoftmax<TPB=256, InputType><<<num_tokens, 256>>>(
    gating_output,   // 输入 [num_tokens, num_cols]
    nullptr,         // finished 数组(推理中总为 nullptr)
    workspace,       // 输出 [num_tokens, num_cols], fp32
    num_cols         // = num_experts
);
```

- **Grid**: `num_tokens` 个 block,每个 block 处理**一行**(一个 token 的 logits)。
- **Block**: `TPB=256` 线程。

#### 4.1.2 算法:三遍扫描的 Block-wide Softmax

数值稳定的 softmax 公式:
```
y_i = exp(x_i - max) / Σ_j exp(x_j - max)
```
需要**两次 reduce(max、sum) + 一次 elementwise**,共三遍扫描整行。

#### 4.1.3 逐行拆解

```cpp
// ①初始化 shared memory
__shared__ float normalizing_factor;  // 全 block 共享 1/Σ
__shared__ float float_max;           // 全 block 共享 max
const int thread_row_offset = blockIdx.x * num_cols;  // 本 block 负责的行起点
float threadData(-FLT_MAX);  // 每线程本地的 max 初值

// ②finished 早退出
if ((finished != nullptr) && finished[blockIdx.x]) return;

// ③第一次扫描:每个线程找自己负责列的 max
for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const float val = toFloat(input[thread_row_offset + ii]);
    threadData = max(val, threadData);
}

// ④Block 级归约:256 个线程 → 1 个 max 值
const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, CubMaxOp());
if (threadIdx.x == 0) float_max = maxElem;   // 只让线程 0 写 shared
__syncthreads();                              // 确保写入可见

// ⑤第二次扫描:每线程累加 exp(x - max)
threadData = 0;
for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    threadData += expf(toFloat(input[...]) - float_max);
}

// ⑥Block 级归约:256 个线程 → 1 个 Σ
const auto Z = BlockReduce(tmpStorage).Reduce(threadData, CubAddOp());
if (threadIdx.x == 0) normalizing_factor = 1.f / Z;
__syncthreads();

// ⑦第三次扫描:写回 softmax 值
for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    output[thread_row_offset + ii] = expf(toFloat(input[...]) - float_max)
                                      * normalizing_factor;
}
```

#### 4.1.4 具体实例分析（num_experts=96, bf16, num_tokens=4）

以一个非 2 的幂的真实场景来展示 `moeSoftmax` 的完整执行过程。

##### 启动参数

```
Grid:  4 个 block（每个 block 处理 1 个 token 的 96 个 logits）
Block: 256 个线程
num_cols = 96
```

##### 线程到列的映射

stride 循环 `for (int ii = threadIdx.x; ii < 96; ii += 256)`：

| threadIdx.x | ii 的值序列 | 循环次数 | 说明 |
|---|---|---|---|
| 0 | 0 | 1 | 读 col 0 |
| 1 | 1 | 1 | 读 col 1 |
| ... | ... | 1 | ... |
| 95 | 95 | 1 | 读 col 95 |
| 96 | _(96 ≥ 96, 跳过)_ | **0** | **空转** |
| 97~255 | _(均 ≥ 96)_ | **0** | **全部空转** |

**关键结论：256 个线程中只有 96 个干活，160 个空转。** 线程利用率仅 96/256 = 37.5%。

##### 三遍扫描的具体数据流

假设 token 0 的 96 个 logits（bf16）为 `[1.2, 0.5, 3.1, ..., -0.8]`：

**第一遍：求 max**
```
线程 0:  threadData = max(-FLT_MAX, toFloat(bf16_1.2)) = 1.2
线程 1:  threadData = max(-FLT_MAX, toFloat(bf16_0.5)) = 0.5
...
线程 42: threadData = max(-FLT_MAX, toFloat(bf16_3.1)) = 3.1  ← 全局最大
...
线程 95: threadData = max(-FLT_MAX, toFloat(bf16_-0.8)) = -0.8
线程 96~255: threadData = -FLT_MAX（没进循环，保持初值）

BlockReduce(256 线程) → maxElem = 3.1
线程 0 写: float_max = 3.1
__syncthreads()
```

> **空转线程的影响**：线程 96~255 的 `threadData = -FLT_MAX` 参与 BlockReduce，但 `max(x, -FLT_MAX) = x`，不影响结果。

**第二遍：求 Σexp(x - max)**
```
线程 0:  threadData = expf(1.2 - 3.1) = expf(-1.9) ≈ 0.1496
线程 42: threadData = expf(3.1 - 3.1) = expf(0) = 1.0      ← 最大贡献
...
线程 96~255: threadData = 0（没进循环，保持初值 0）

BlockReduce(256 线程) → Z = Σ ≈ 5.23（假设值）
线程 0 写: normalizing_factor = 1/5.23 ≈ 0.1912
__syncthreads()
```

> **空转线程的影响**：线程 96~255 的 `threadData = 0` 参与 sum，`sum + 0 = sum`，不影响结果。

**第三遍：写回**
```
线程 0:  output[0] = expf(1.2 - 3.1) * 0.1912 ≈ 0.0286
线程 42: output[42] = expf(0) * 0.1912 ≈ 0.1912
...
线程 96~255: 不进循环，不写任何输出
```

##### 如果 num_experts=1024 会怎样？

此时每个线程循环 `⌈1024/256⌉ = 4` 次：

| threadIdx.x | ii 的值序列 | 循环次数 |
|---|---|---|
| 0 | 0, 256, 512, 768 | 4 |
| 1 | 1, 257, 513, 769 | 4 |
| 255 | 255, 511, 767, 1023 | 4 |

所有 256 个线程满载，利用率 100%，每个线程本地先做 4 个值的 max/sum，再 BlockReduce。

#### 4.1.5 关键点

- **stride 循环的线程利用率取决于 num_cols vs TPB 的比值**。当 num_cols < TPB 时部分线程空转,但 BlockReduce 设计保证空转线程不影响正确性（max 的初值 `-FLT_MAX`、sum 的初值 `0` 都是对应运算的单位元）。
- **两次 `__syncthreads()`**: 因为 `float_max`/`normalizing_factor` 是 block 共享变量,写入的线程和读取的线程可能不同 warp,必须显式同步。
- **`cub::BlockReduce`**: CUB 内部用 shared memory + 对数树归约。256 线程需要 log₂(256)=8 级归约,约 8 条指令完成。
- **此路径只在 default 分支触发**：64 experts 这样的 2 的幂会走 `topkGating` 融合路径,不会调用 `moeSoftmax`。

#### 4.1.6 性能特征

| 指标 | num_experts=96 | num_experts=1024 |
|---|---|---|
| 活跃线程 | 96/256 = 37.5% | 256/256 = 100% |
| 循环次数/线程 | 1 | 4 |
| Global Memory 读 | 3 × 96 × sizeof(InputType) | 3 × 1024 × sizeof(InputType) |
| BlockReduce 次数 | 2（max + sum） | 2 |
| 适用场景 | 小奇数专家数 | 大专家数 |

---

### 4.2 `moeSigmoid` kernel

> **源码位置**: L134-153

这是最简单的 kernel 之一:没有跨线程归约,纯 elementwise。

```cpp
for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const float val = toFloat(input[thread_row_offset + ii]);
    output[thread_row_offset + ii] = 1.0f / (1.0f + __expf(-val));
}
```

#### 4.2.1 具体实例（num_experts=96, bf16）

启动参数与 `moeSoftmax` 完全一致：`<<<num_tokens, 256>>>`。

##### 执行过程

与 softmax 不同，sigmoid 是**纯 map 操作**,不需要跨线程通信：

```
线程 0:  output[0] = sigmoid(1.2) = 1/(1+exp(-1.2)) ≈ 0.769
线程 42: output[42] = sigmoid(3.1) = 1/(1+exp(-3.1)) ≈ 0.957
线程 95: output[95] = sigmoid(-0.8) = 1/(1+exp(0.8)) ≈ 0.310
线程 96~255: 不进循环，空转
```

##### 与 moeSoftmax 的关键差异

| 对比项 | moeSoftmax | moeSigmoid |
|---|---|---|
| 扫描次数 | 3 遍 | 1 遍 |
| 需要 BlockReduce | 是（max + sum） | 否 |
| 需要 shared memory | 是（float_max, normalizing_factor, tmpStorage） | 否 |
| 需要 `__syncthreads` | 2 次 | 0 次 |
| 输出总和 | = 1.0（概率归一化） | 每个值独立，总和通常 ≠ 1 |
| `expf` vs `__expf` | `expf`（高精度） | `__expf`（快速近似，约 2-4x 快） |

> `__expf` 是 CUDA 内建近似版本,精度略低但速度更快。softmax 用 `expf` 因为 max/sum reduce 需要精度一致;sigmoid 逐元素独立,近似误差不会累积。

**典型用途**: DeepSeek-V3 等用 sigmoid scoring + noisy top-k gating 的模型。

---

### 4.3 `moeTopK` kernel

> **源码位置**: L155-245

这是通用路径的主干:**给定 softmax/sigmoid 后的概率矩阵,选出每行最大的 k 个**。

#### 4.3.1 关键数据结构:`cub::KeyValuePair`

```cpp
using cub_kvp = cub::KeyValuePair<int, float>;
```
一个结构体,`key = expert_id`, `value = 概率值`。`cub::ArgMax` 会比较 value,保留 key。

#### 4.3.2 算法:朴素的 K 次扫描

为选出 top-k,最直接的办法是:**扫 k 遍,每遍求 argmax,并屏蔽之前已选中的**。

```cpp
for (int k_idx = 0; k_idx < k; ++k_idx) {
    // ①本线程扫自己负责的列,找本地 argmax
    thread_kvp.key = 0;
    thread_kvp.value = -1.f;

    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
        inp_kvp.key = expert;
        inp_kvp.value = inputs_after_softmax[...] + (bias ? bias[expert] : 0);

        // ②屏蔽之前已选中的专家:把它们的 value 替换成当前 best,等价于跳过
        for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
            if (indices[k * block_row + prior_k] == expert)
                inp_kvp = thread_kvp;
        }

        // ③累积本地 argmax
        thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    // ④Block 级 argmax 归约
    const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);

    // ⑤线程 0 写出结果
    if (threadIdx.x == 0) {
        const int expert = result_kvp.key;
        output[k * block_row + k_idx] = inputs_after_softmax[...expert...];  // unbiased
        indices[k * block_row + k_idx] = (should_process_row) ? (expert - start_expert)
                                                               : num_experts;
        source_rows[k * block_row + k_idx] = k_idx * num_rows + block_row;
        if (renormalize) selected_sum += inputs_after_softmax[...];
    }
    __syncthreads();
}

// ⑥renormalize 分支
if (renormalize && threadIdx.x == 0) {
    const float denom = selected_sum > 0.f ? selected_sum : 1.f;
    for (int k_idx = 0; k_idx < k; ++k_idx)
        output[k * block_row + k_idx] /= denom;
}
```

#### 4.3.3 具体实例分析（num_experts=96, topk=6, renormalize=true）

假设 token 0 经过 softmax 后的概率分布（96 个专家,只列出非零显著值）:

```
expert  0: 0.01   expert 11: 0.25   expert 23: 0.08
expert 42: 0.18   expert 55: 0.12   expert 67: 0.06
expert 80: 0.04   其余: 各约 0.003
```

##### 启动参数

```
Grid:  num_tokens 个 block
Block: TPB=256 线程
每个 block 处理 1 行 → blockIdx.x = token 编号
```

##### 第 1 轮（k_idx=0）：找全局 argmax

**Step 1: 线程内扫描**

stride 循环 `for (expert = threadIdx.x; expert < 96; expert += 256)`：

```
线程 0:   扫描 expert 0  → inp_kvp = {0, 0.01}   → thread_kvp = {0, 0.01}
线程 11:  扫描 expert 11 → inp_kvp = {11, 0.25}  → thread_kvp = {11, 0.25}
线程 42:  扫描 expert 42 → inp_kvp = {42, 0.18}  → thread_kvp = {42, 0.18}
线程 96~255: 不进循环 → thread_kvp = {0, -1.0}（初值）
```

> 每个活跃线程只扫描 1 个专家（96 < 256），空转线程的 `thread_kvp.value = -1.0` 在 ArgMax 归约中不会胜出。

**Step 2: BlockReduce argmax（256 线程 → 1 个 winner）**

```
256 个 thread_kvp 经过 CUB BlockReduce:
  max({0,0.01}, {11,0.25}, {42,0.18}, ..., {0,-1.0}, {0,-1.0}, ...)
  → result_kvp = {11, 0.25}  ✓
```

**Step 3: 线程 0 写出结果**

```cpp
output[6*0 + 0]       = 0.25          // topk_weights[token=0, k=0]
indices[6*0 + 0]      = 11            // topk_indices[token=0, k=0] = expert 11
source_rows[6*0 + 0]  = 0*num_rows+0  // = 0
selected_sum          += 0.25          // = 0.25
```

**`__syncthreads()`** — 确保 `indices[0] = 11` 对下一轮所有线程可见。

##### 第 2 轮（k_idx=1）：找第 2 大（屏蔽 expert 11）

**屏蔽机制的核心**：

```cpp
for (int prior_k = 0; prior_k < 1; ++prior_k) {
    const int prior_winning_expert = indices[6 * 0 + 0]; // = 11
    if (prior_winning_expert == expert)
        inp_kvp = thread_kvp;  // 用当前 best 覆盖，等价于跳过
}
```

线程 11 扫描 expert 11 时：
```
inp_kvp = {11, 0.25}
发现 11 == indices[0](=11) → inp_kvp = thread_kvp = {0, -1.0}
→ expert 11 被有效屏蔽
```

BlockReduce 后 → `result_kvp = {42, 0.18}`

```
output[1]  = 0.18,  indices[1] = 42
selected_sum = 0.25 + 0.18 = 0.43
```

##### 6 轮汇总

| k_idx | 屏蔽列表 | winner | weight | 屏蔽检查次数/线程 |
|---|---|---|---|---|
| 0 | (无) | expert 11 | 0.25 | 0 |
| 1 | {11} | expert 42 | 0.18 | 1 |
| 2 | {11, 42} | expert 55 | 0.12 | 2 |
| 3 | {11, 42, 55} | expert 23 | 0.08 | 3 |
| 4 | {11, 42, 55, 23} | expert 67 | 0.06 | 4 |
| 5 | {11, 42, 55, 23, 67} | expert 80 | 0.04 | 5 |

##### Renormalize

```cpp
selected_sum = 0.25 + 0.18 + 0.12 + 0.08 + 0.06 + 0.04 = 0.73
denom = 0.73

output[0] = 0.25 / 0.73 ≈ 0.342
output[1] = 0.18 / 0.73 ≈ 0.247
output[2] = 0.12 / 0.73 ≈ 0.164
output[3] = 0.08 / 0.73 ≈ 0.110
output[4] = 0.06 / 0.73 ≈ 0.082
output[5] = 0.04 / 0.73 ≈ 0.055
总和 = 1.000 ✓
```

##### 屏蔽机制的开销分析

屏蔽通过**读取 Global Memory 中已写入的 indices**,复杂度为 O(k²)：

| k | 每轮屏蔽检查次数 | 总检查次数 |
|---|---|---|
| 2 | 0+1 = 1 | 1 |
| 6 | 0+1+2+3+4+5 = 15 | 15 |
| 8 | 0+1+...+7 = 28 | 28 |

对比 `topkGating` 的屏蔽方式：直接在寄存器中置 -10000（O(1) 每轮），无 Global Memory 读取。这也是为什么 `topkGating` 更快的原因之一。

#### 4.3.4 三个值得注意的细节

**(1) Bias-aware selection, unbiased weight**:
如果传入 `bias` (常见于 DeepSeek 的 correction bias),**选择时用 `prob + bias`** 决定哪个专家胜出,但**写出的权重是不含 bias 的原始概率**(L224: `output[idx] = inputs_after_softmax[...expert]`)。bias 仅影响"路由决策",不影响"权重计算"。

**(2) Expert Parallel 的 start/end 过滤**:
```cpp
const bool node_uses_expert = expert >= start_expert && expert < end_expert;
indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
```
在 EP 场景下,每个 GPU 只负责部分专家。如果选中的专家**不在本 rank 负责范围**,就写成 `num_experts`(一个哨兵值),下游 kernel 会跳过。如果在范围内,写成**本地相对编号** `expert - start_expert`。

**(3) `source_rows` 的作用**:
```cpp
source_rows[idx] = k_idx * num_rows + block_row;
```
后续 kernel 用这个索引把"(token, k 次选择)"二元组排序,按专家分组进行 A2A 或分组 GEMM。

#### 4.3.5 通用路径 vs 融合路径性能对比

| 对比项 | 通用路径 (moeSoftmax + moeTopK) | 融合路径 (topkGating) |
|---|---|---|
| Kernel 数量 | 2 个（两次启动开销） | 1 个 |
| Global Memory 访问 | softmax: 3 遍读 + 1 遍写; topK: k 遍读 | 1 遍读（到寄存器后全在寄存器/shuffle） |
| 归约方式 | shared memory + BlockReduce | 寄存器 + warp shuffle |
| 屏蔽已选 expert | 读 Global Memory indices[prior_k], O(k²) | 寄存器置 -10000, O(1) |
| 线程利用率(96 experts) | 96/256 = 37.5% | 不适用（96 走通用路径） |
| 线程利用率(64 experts) | N/A（走融合路径） | 128/128 = 100% |
| Shared memory | 需要（BlockReduce tmpStorage） | 不需要 |
| 适用条件 | 任意 num_experts | 2 的幂 或 64 的倍数 |

---

## 5. 融合快速路径:`topkGating` kernel

这是整个文件的**重头戏** —— 当 `num_experts` 是 2 的幂(≤512)或 64 的倍数时,**把 softmax/sigmoid + top-k + renormalize 全部融合进一个 kernel**,且**不使用 shared memory**,完全靠 warp shuffle 完成归约。

> **源码位置**: L263-563

### 5.1 模板参数

```cpp
template <int VPT,              // Values Per Thread, 每线程处理几个 expert
          int NUM_EXPERTS,      // 编译期专家数 (1/2/4/.../512 或 192/320/...)
          int WARPS_PER_CTA,    // 每 block 的 warp 数 (固定 4)
          int BYTES_PER_LDG,    // 每次向量化 load 多少字节 (4/8/16)
          int WARP_SIZE_PARAM,  // warp 大小 (32 NV / 64 ROCm)
          typename IndType,     // indices 数据类型
          typename InputType,   // logits 数据类型
          ScoringFunc SF>       // softmax 或 sigmoid
__global__ void topkGating(...)
```

### 5.2 核心设计思想:一个 thread-group 处理一行

不同于通用路径"一个 block 处理一行",融合路径**更激进**:

```
专家数 NUM_EXPERTS = 64 (DeepSeek)
每线程处理 VPT 个专家,假设 VPT=2 → THREADS_PER_ROW = 32
一个 warp (32 线程) 正好处理一行,无需 shared memory
每 block 4 warp 同时处理 4 行,每 block 4 行
```

当 NUM_EXPERTS 很小(如 8),可以有多行共享一个 warp;当 NUM_EXPERTS 大(如 256),需要分更多 LDG(多次向量化加载)。

### 5.3 编译期常量推导

```cpp
// L279-298
ELTS_PER_LDG   = BYTES_PER_LDG / sizeof(InputType)      // 每次加载元素数
ELTS_PER_ROW   = NUM_EXPERTS                             // 一行元素数
THREADS_PER_ROW = ELTS_PER_ROW / VPT                     // 一行几个线程
LDG_PER_THREAD = VPT / ELTS_PER_LDG                      // 每线程几次加载
ELTS_PER_WARP  = WARP_SIZE * VPT                         // 一个 warp 处理元素总数
ROWS_PER_WARP  = ELTS_PER_WARP / ELTS_PER_ROW            // 一个 warp 处理行数
ROWS_PER_CTA   = WARPS_PER_CTA * ROWS_PER_WARP           // 一个 block 处理行数
```

一系列 `static_assert` 保证所有除法都能整除、THREADS_PER_ROW 是 2 的幂(为了 butterfly reduce)。

**DeepSeek 举例**(NUM_EXPERTS=64, BYTES_PER_LDG=16, sizeof(bf16)=2):
- ELTS_PER_LDG = 16/2 = 8
- VPT 需满足 VPT%8 == 0 且 NUM_EXPERTS%VPT == 0, 最小选 8 → THREADS_PER_ROW = 64/8 = 8
- ROWS_PER_WARP = 32 × 8 / 64 = 4 (每 warp 4 行)
- ROWS_PER_CTA = 4 × 4 = 16 (每 block 16 行)
- 若有 1024 tokens → 启动 1024/16 = 64 个 block

### 5.4 五阶段执行流程

```
① Row 分配           → 计算本线程处理第几行
② 向量化加载         → 把 VPT 个 logits 搬到寄存器(类型转换)
③ Softmax/Sigmoid    → warp shuffle butterfly reduce (max + sum)
④ Top-K 循环         → k 次 argmax + mask,全在寄存器做
⑤ Renormalize        → 重归一化 k 个权重(可选)
```

#### 5.4.1 阶段①:行分配 (L307-322)

```cpp
const int cta_base_row = blockIdx.x * ROWS_PER_CTA;         // 本 block 的起始行
const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;  // 本 warp 起始行
const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;  // warp 内第几行
const int thread_row = warp_base_row + thread_row_in_warp;     // 全局行号

if (thread_row >= num_rows) return;  // 尾部 block 多余线程早退
```

**Block 的 `blockDim = (WARP_SIZE, WARPS_PER_CTA)` = `(32, 4)`**: `threadIdx.x` 是 warp 内偏移(0-31),`threadIdx.y` 是第几个 warp(0-3)。

#### 5.4.2 阶段②:向量化加载 + 类型转换 (L326-391)

```cpp
// 计算本线程组负责的列起点
const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;

float row_chunk[VPT];   // 每线程本地寄存器数组,存 VPT 个 fp32

// Float 路径:直接向量化 copy (无类型转换)
if constexpr (std::is_same_v<InputType, float>) {
    using VecType = AlignedArray<float, ELTS_PER_LDG>;
    // ... 一次加载 ELTS_PER_LDG 个 float 到 row_chunk
}
// BF16 路径:加载后用 __bfloat1622float2 成对转换
else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
    // 用 __nv_bfloat162 封装两个 bf16, 一次 __bfloat1622float2 转成 float2
    // 利用硬件的 bf16x2 → fp32x2 转换指令, 比单个转换快一倍
}
```

**为什么要 `THREADS_PER_ROW` 个线程交叉式读**:相邻线程读相邻的 BYTES_PER_LDG 字节块,硬件合并成一次 memory transaction。每线程的 VPT 个元素来自 `LDG_PER_THREAD` 次加载,每次跳 `THREADS_PER_ROW * ELTS_PER_LDG` 列。

#### 5.4.2a 向量化加载过程详解（以 DeepSeek-MoE-16B bf16 为例）

本小节以 **DeepSeek-MoE-16B** 的具体参数，完整推演向量化加载过程中每个线程读取的地址和数据布局。

##### 编译期常量推导

```
模型参数:  NUM_EXPERTS = 64, InputType = bf16 (2 bytes), BYTES_PER_LDG = 16
固定参数:  WARPS_PER_CTA = 4, WARP_SIZE = 32

推导过程:
  ELTS_PER_LDG    = BYTES_PER_LDG / sizeof(bf16) = 16 / 2 = 8     ← 每次加载 8 个 bf16
  VECs_PER_THREAD = MAX(1, 64 / (8 × 32)) = MAX(1, 0.25) = 1      ← 每线程 1 次向量化加载
  VPT             = VECs_PER_THREAD × ELTS_PER_LDG = 1 × 8 = 8    ← 每线程处理 8 个专家
  THREADS_PER_ROW = NUM_EXPERTS / VPT = 64 / 8 = 8                 ← 8 个线程协作处理一行
  LDG_PER_THREAD  = VPT / ELTS_PER_LDG = 8 / 8 = 1                ← 每线程只需 1 次 load
  ROWS_PER_WARP   = WARP_SIZE / THREADS_PER_ROW = 32 / 8 = 4      ← 一个 warp 处理 4 行
  ROWS_PER_CTA    = WARPS_PER_CTA × ROWS_PER_WARP = 4 × 4 = 16   ← 一个 block 处理 16 行
```

##### Grid / Block 维度

```
blockDim = dim3(WARP_SIZE, WARPS_PER_CTA) = dim3(32, 4)
         → 每 block 共 32 × 4 = 128 个线程

gridDim.x = ceil(num_tokens / ROWS_PER_CTA)
         → 若 num_tokens = 1024, gridDim.x = 1024 / 16 = 64 个 block
```

##### Warp 内线程到行/列的映射

以 **warp 0** (`threadIdx.y = 0`) 为例，32 个线程分成 4 组，每组 8 个线程处理 1 行：

```
threadIdx.x │ thread_row_in_warp │ thread_group_idx │ 负责的行 │ 负责的列起点
────────────┼────────────────────┼──────────────────┼──────────┼──────────────
  0         │ 0/8 = 0            │ 0%8 = 0          │ row 0    │ col 0~7
  1         │ 1/8 = 0            │ 1%8 = 1          │ row 0    │ col 8~15
  2         │ 2/8 = 0            │ 2%8 = 2          │ row 0    │ col 16~23
  3         │ 3/8 = 0            │ 3%8 = 3          │ row 0    │ col 24~31
  4         │ 4/8 = 0            │ 4%8 = 4          │ row 0    │ col 32~39
  5         │ 5/8 = 0            │ 5%8 = 5          │ row 0    │ col 40~47
  6         │ 6/8 = 0            │ 6%8 = 6          │ row 0    │ col 48~55
  7         │ 7/8 = 0            │ 7%8 = 7          │ row 0    │ col 56~63
────────────┼────────────────────┼──────────────────┼──────────┼──────────────
  8         │ 8/8 = 1            │ 8%8 = 0          │ row 1    │ col 0~7
  9         │ 9/8 = 1            │ 9%8 = 1          │ row 1    │ col 8~15
  ...       │ ...                │ ...              │ ...      │ ...
  15        │ 15/8 = 1           │ 15%8 = 7         │ row 1    │ col 56~63
────────────┼────────────────────┼──────────────────┼──────────┼──────────────
  16~23     │ 2                  │ 0~7              │ row 2    │ col 0~63
  24~31     │ 3                  │ 0~7              │ row 3    │ col 0~63
```

##### 完整 Block 内数据加载架构图

```
                          Global Memory: gating_output [num_tokens × 64] (bf16)
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │  row 0:  [e0 e1 e2 e3 e4 e5 e6 e7 | e8 ... e15 | e16...e23 | ... | e56...e63] │
    │  row 1:  [e0 e1 e2 e3 e4 e5 e6 e7 | e8 ... e15 | e16...e23 | ... | e56...e63] │
    │  ...                                                                            │
    │  row 15: [e0 e1 e2 e3 e4 e5 e6 e7 | e8 ... e15 | e16...e23 | ... | e56...e63] │
    └──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────── Block (blockIdx.x = B) ───────────────────────────┐
    │                                                                           │
    │  blockDim = (32, 4)  →  128 threads  →  4 warps × 32 threads/warp       │
    │                                                                           │
    │  ┌─── Warp 0 (threadIdx.y=0): rows B*16+0 ~ B*16+3 ───────────────────┐ │
    │  │                                                                      │ │
    │  │  Thread Group 0 (T0~T7)  → row B*16+0                               │ │
    │  │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                  │ │
    │  │  │ T0  │ T1  │ T2  │ T3  │ T4  │ T5  │ T6  │ T7  │                  │ │
    │  │  │0~7  │8~15 │16~23│24~31│32~39│40~47│48~55│56~63│  ← expert cols   │ │
    │  │  │16B  │16B  │16B  │16B  │16B  │16B  │16B  │16B  │  ← 每次 LDG.128  │ │
    │  │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘                  │ │
    │  │       ↓ 8个线程并行读 → 合并为 1 次 128B memory transaction            │ │
    │  │                                                                      │ │
    │  │  Thread Group 1 (T8~T15) → row B*16+1  (同上布局)                    │ │
    │  │  Thread Group 2 (T16~T23)→ row B*16+2  (同上布局)                    │ │
    │  │  Thread Group 3 (T24~T31)→ row B*16+3  (同上布局)                    │ │
    │  └──────────────────────────────────────────────────────────────────────┘ │
    │                                                                           │
    │  ┌─── Warp 1 (threadIdx.y=1): rows B*16+4 ~ B*16+7 ──────────────────┐  │
    │  │  (结构同 Warp 0, 处理第 4~7 行)                                     │  │
    │  └──────────────────────────────────────────────────────────────────────┘ │
    │                                                                           │
    │  ┌─── Warp 2 (threadIdx.y=2): rows B*16+8 ~ B*16+11 ─────────────────┐  │
    │  │  (结构同 Warp 0, 处理第 8~11 行)                                    │  │
    │  └──────────────────────────────────────────────────────────────────────┘ │
    │                                                                           │
    │  ┌─── Warp 3 (threadIdx.y=3): rows B*16+12 ~ B*16+15 ────────────────┐  │
    │  │  (结构同 Warp 0, 处理第 12~15 行)                                   │  │
    │  └──────────────────────────────────────────────────────────────────────┘ │
    └───────────────────────────────────────────────────────────────────────────┘
```

##### 单线程加载的代码路径（bf16, ELTS_PER_LDG=8）

```cpp
// 以 Thread 3 (threadIdx.x=3, warp 0) 为例:
//   thread_row_in_warp = 3 / 8 = 0  → 处理 row 0
//   thread_group_idx   = 3 % 8 = 3  → 该行第 3 个 thread
//   first_elt_read_by_thread = 3 * 8 = 24  → 从 col 24 开始

// bf16 路径, ELTS_PER_LDG=8 ≥ 2:
using VecType = AlignedArray<__nv_bfloat16, 8>;  // 8×2=16 bytes, 对齐到 16B

// LDG_PER_THREAD = 1, 所以只循环一次 (ii=0):
VecType vec = vec_thread_read_ptr[0 * THREADS_PER_ROW];   // 0 * 8 = 0 → 就读自己的 16B
//   → 发射一条 LDG.128 指令, 从 &input[row*64 + 24] 读取 16 字节 (bf16 × 8)
//   → vec.data[0..7] = {e24, e25, e26, e27, e28, e29, e30, e31}

// 然后逐对转换: ELTS_PER_LDG/2 = 4 次
// jj=0: row_chunk_f2[0] = __bfloat1622float2(*(bf162*)(vec.data+0))  → {f(e24), f(e25)}
// jj=1: row_chunk_f2[1] = __bfloat1622float2(*(bf162*)(vec.data+2))  → {f(e26), f(e27)}
// jj=2: row_chunk_f2[2] = __bfloat1622float2(*(bf162*)(vec.data+4))  → {f(e28), f(e29)}
// jj=3: row_chunk_f2[3] = __bfloat1622float2(*(bf162*)(vec.data+6))  → {f(e30), f(e31)}
//
// 最终: row_chunk[0..7] = {f(e24), f(e25), ..., f(e31)} (float32)
```

##### 三种 InputType 路径对比

| 路径 | InputType | sizeof | ELTS_PER_LDG (16B) | VPT | THREADS_PER_ROW | LDG_PER_THREAD | 加载方式 | 转换方式 |
|------|-----------|--------|---------------------|-----|-----------------|----------------|----------|----------|
| float | float | 4B | 4 | 4 | 16 | 1 | `AlignedArray<float,4>` 直接 copy | 无需转换 |
| bf16 | `__nv_bfloat16` | 2B | 8 | 8 | 8 | 1 | `AlignedArray<bf16,8>` | `__bfloat1622float2` 成对转 |
| fp16 | `__half` | 2B | 8 | 8 | 8 | 1 | `AlignedArray<half,8>` | `__half22float2` 成对转 |

> **注意**: float 路径下 THREADS_PER_ROW=16，一个 warp 处理 32/16=2 行；bf16/fp16 路径下 THREADS_PER_ROW=8，一个 warp 处理 32/8=4 行。因此 **bf16/fp16 的吞吐量是 float 的 2 倍**（每 block 处理 16 行 vs 8 行）。

##### 多次加载场景（NUM_EXPERTS=256, bf16）

当专家数更大时，每线程需要多次加载：

```
NUM_EXPERTS=256, bf16, BYTES_PER_LDG=16:
  ELTS_PER_LDG    = 8
  VECs_PER_THREAD = MAX(1, 256/(8×32)) = 1  → 但实际 256/8=32 > WARP_SIZE...
  
  实际走 TopkConstants 推导:
  VECs_PER_THREAD = MAX(1, 256/(8×32)) = 1
  VPT = 1 × 8 = 8
  THREADS_PER_ROW = 256/8 = 32  → 整个 warp 处理一行！
  LDG_PER_THREAD = 8/8 = 1
  ROWS_PER_WARP = 32/32 = 1    → 一个 warp 只处理 1 行
  ROWS_PER_CTA = 4 × 1 = 4    → 一个 block 处理 4 行
```

```
NUM_EXPERTS=512, bf16, BYTES_PER_LDG=16:
  VECs_PER_THREAD = MAX(1, 512/(8×32)) = 2
  VPT = 2 × 8 = 16              → 每线程处理 16 个专家
  THREADS_PER_ROW = 512/16 = 32  → 整个 warp 处理一行
  LDG_PER_THREAD = 16/8 = 2     → 每线程 2 次 LDG.128
  ROWS_PER_WARP = 1
  ROWS_PER_CTA = 4
```

此时每线程的 2 次加载**跨步读取**：

```cpp
// ii=0: vec_thread_read_ptr[0 * 32] → 读 col first_elt 处的 8 个 bf16
// ii=1: vec_thread_read_ptr[1 * 32] → 读 col first_elt + 32*8=256 处？
//       不对！实际是 first_elt + THREADS_PER_ROW*ELTS_PER_LDG = 32*8 = 256
//       但 NUM_EXPERTS=512, 所以 col = first_elt + 256
//       → 每线程的第 2 次 load 读的是行内后半部分
```

##### 内存合并访问（Coalesced Access）示意

```
                     Global Memory 中 row 0 的 64 个 bf16 元素 (128 bytes)
    地址:  0     16    32    48    64    80    96    112
           ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
           │ T0  │ T1  │ T2  │ T3  │ T4  │ T5  │ T6  │ T7  │  ← 8 threads
           │16B  │16B  │16B  │16B  │16B  │16B  │16B  │16B  │
           └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                              │
                              ▼
    GPU memory controller 看到: 8 个线程请求连续的 128 字节
    → 合并为 1 次 128B transaction (L1 cache line = 128B)
    → 带宽利用率 100%, 零浪费！

    对比非合并访问（假设每线程读不连续的位置）:
    → 需要 8 次独立 transaction, 每次 128B 但只用 16B
    → 带宽利用率仅 12.5%！
```

##### 加载后的寄存器布局

加载完成后，每个线程的 `row_chunk[VPT]` 数组存储在**寄存器**中（不在 shared memory），后续 softmax/topk 全在寄存器上操作：

```
Thread Group 处理 row 0 (THREADS_PER_ROW=8, VPT=8):

  T0.row_chunk[0..7] = {f(e0),  f(e1),  ..., f(e7)}    ← 8 个 float 寄存器
  T1.row_chunk[0..7] = {f(e8),  f(e9),  ..., f(e15)}
  T2.row_chunk[0..7] = {f(e16), f(e17), ..., f(e23)}
  T3.row_chunk[0..7] = {f(e24), f(e25), ..., f(e31)}
  T4.row_chunk[0..7] = {f(e32), f(e33), ..., f(e39)}
  T5.row_chunk[0..7] = {f(e40), f(e41), ..., f(e47)}
  T6.row_chunk[0..7] = {f(e48), f(e49), ..., f(e55)}
  T7.row_chunk[0..7] = {f(e56), f(e57), ..., f(e63)}

  → 8 个线程 × 8 个寄存器 = 64 个 float = 完整一行 64 个专家的 softmax 概率
  → 后续 butterfly reduce 用 shuffle 跨线程交换, 完全不需要 shared memory
  → 每线程占用 8 × 4B = 32B 寄存器 (很少, 不会造成 register pressure)
```

#### 5.4.3 阶段③:Softmax/Sigmoid via Warp Shuffle (L393-443)

**Softmax 分支** — butterfly reduce max:

```cpp
// a. 线程内 max
float thread_max = row_chunk[0];
for (int ii = 1; ii < VPT; ++ii)
    thread_max = max(thread_max, row_chunk[ii]);

// b. 线程组内 max (butterfly reduce)
for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    thread_max = max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
}
// 经过 log2(THREADS_PER_ROW) 轮, 所有线程拿到全局 max
```

**Butterfly Reduce 图示**(THREADS_PER_ROW=8):
```
mask=4:   T0 ↔ T4    T1 ↔ T5    T2 ↔ T6    T3 ↔ T7
mask=2:   T0 ↔ T2    T1 ↔ T3    T4 ↔ T6    T5 ↔ T7
mask=1:   T0 ↔ T1    T2 ↔ T3    T4 ↔ T5    T6 ↔ T7
结果: 8 个线程都拿到同一个 max 值, 仅用 3 次 shuffle
```

**`VLLM_SHFL_XOR_SYNC_WIDTH(val, mask, width)`**: 本线程和 `lane_id XOR mask` 的线程交换 val。`width` 参数限定只在当前 `width` 个线程子组内洗牌(跨组隔离)。

接下来同样的 butterfly 做 row_sum,然后:
```cpp
const float reciprocal_row_sum = 1.f / row_sum;
for (int ii = 0; ii < VPT; ++ii)
    row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
```

**Sigmoid 分支**更简单,纯 elementwise,无归约:
```cpp
for (int ii = 0; ii < VPT; ++ii)
    row_chunk[ii] = 1.0f / (1.0f + __expf(-row_chunk[ii]));
```

#### 5.4.4 阶段④:Top-K 循环 (L445-549)

这是最精细的一段。与通用路径不同,它不能直接用 `cub::BlockReduce`,而是自己实现 **warp 级的 argmax butterfly reduce**。

```cpp
// 如果有 bias, 计算 "用于选择"的值 = prob + bias
float row_chunk_for_choice[VPT];   // 寄存器数组
if (bias != nullptr) {
    for (每个 expert) row_chunk_for_choice[ii] = row_chunk[ii] + bias[expert];
} else {
    // 直接复制
}

for (int k_idx = 0; k_idx < k; ++k_idx) {
    // ① 线程内 argmax
    float max_val_for_choice = row_chunk_for_choice[0];
    float max_val            = row_chunk[0];
    int expert               = start_col;
    for (遍历本线程 VPT 个元素) {
        if (val_for_choice > max_val_for_choice) {
            max_val_for_choice = val_for_choice;
            max_val            = val;     // 不含 bias 的原始概率
            expert             = col + ii;
        }
    }

    // ② 线程组内 argmax via butterfly shuffle
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_mfc  = VLLM_SHFL_XOR_SYNC_WIDTH(max_val_for_choice, mask, TPR);
        float other_max  = VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, TPR);
        int   other_exp  = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, TPR);

        // 并列时选较小 expert 编号(和 PyTorch argmax 一致)
        if (other_mfc > max_val_for_choice ||
            (other_mfc == max_val_for_choice && other_exp < expert)) {
            max_val_for_choice = other_mfc;
            max_val = other_max;
            expert = other_exp;
        }
    }

    // ③ 线程组的 leader 写回结果
    if (thread_group_idx == 0) {
        const int idx = k * thread_row + k_idx;
        output[idx]  = max_val;            // ← 写的是 unbiased 概率
        indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
        source_rows[idx] = k_idx * num_rows + thread_row;
        if (renormalize) selected_sum += max_val;
    }

    // ④ 屏蔽已选中的 expert,准备下一轮
    if (k_idx + 1 < k) {
        const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
        const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
        if (thread_group_idx == thread_to_clear_in_group) {
            const int offset = expert % ELTS_PER_LDG;
            row_chunk_for_choice[ldg_group_for_expert * ELTS_PER_LDG + offset] = -10000.f;
        }
    }
}
```

**关键观察**:

1. **两套 max**: `max_val_for_choice` 用于**选择排序**(可能带 bias),`max_val` 是**真实权重**(不带 bias)。两者在 butterfly 中一起移动,保证最终 leader 拿到的 `max_val` 对应正确的 expert。
2. **屏蔽不需要同步**: 因为下一轮 argmax 扫描的是同一个 `row_chunk_for_choice` 数组,而这个数组在**寄存器里**,每个线程只看自己的部分。被屏蔽的值由对应线程在自己的寄存器里改成 `-10000.f`,天然可见。
3. **Tie-breaking**: `other_exp < expert` 保证相同概率时选较小编号,和 PyTorch 一致,避免跨实现的不确定性。

#### 5.4.4a Top-K 线程内 argmax 详解（以 DeepSeek-MoE-16B bf16, topk=6 为例）

本节结合 DeepSeek-MoE-16B（64 experts, bf16, topk=6）的具体数值，逐步演示 Top-K 循环中 **线程内 local argmax** 和 **线程组间 butterfly argmax reduce** 的完整过程。

##### 编译期常量回顾

| 常量 | 值 | 含义 |
|---|---|---|
| NUM_EXPERTS | 64 | 一行 64 个专家 |
| VPT | 8 | 每线程处理 8 个专家 |
| ELTS_PER_LDG | 8 | 每次向量化加载 8 个 bf16 |
| THREADS_PER_ROW | 8 | 8 个线程协作处理一行 |
| LDG_PER_THREAD | 1 | 每线程只需 1 次加载 |
| COLS_PER_GROUP_LDG | 64 | = 8 × 8，一个 ldg group 覆盖的列数 |

##### 初始状态：softmax 完成后

假设某 token 的 softmax 概率分布（64 个专家，只列出部分非零值）：

```
row_chunk 在 8 个线程中的分布（无 bias 场景，row_chunk_for_choice == row_chunk）：

线程0 (expert 0-7):  [0.01, 0.02, 0.15, 0.03, 0.01, 0.02, 0.01, 0.01]
线程1 (expert 8-15): [0.02, 0.01, 0.03, 0.20, 0.01, 0.01, 0.02, 0.01]
线程2 (expert 16-23):[0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.01, 0.01]
线程3 (expert 24-31):[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
线程4 (expert 32-39):[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.10, 0.01]
线程5 (expert 40-47):[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
线程6 (expert 48-55):[0.01, 0.08, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
线程7 (expert 56-63):[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
```

> 注：总和 ≈ 1.0（softmax 保证）。全局 Top-6 应为：expert 11(0.20), expert 2(0.15), expert 38(0.10), expert 49(0.08), expert 21(0.05), expert 10(0.03)。

##### 第 1 轮 Top-K（k_idx=0）：找全局最大值

**Step 1：线程内 local argmax**

每个线程扫描自己的 `row_chunk_for_choice[0..7]`，找出本地最大值：

```cpp
float max_val_for_choice = row_chunk_for_choice[0];  // 初始化为第 0 个元素
float max_val = row_chunk[0];
int expert = start_col;  // = first_elt_read_by_thread = thread_group_idx * 8
```

代码中的双层循环在本例中展开为（LDG_PER_THREAD=1, ELTS_PER_LDG=8）：

```cpp
// ldg=0, col=start_col
for (int ii = 0; ii < 8; ++ii) {  // ELTS_PER_LDG=8
    float val_for_choice = row_chunk_for_choice[ii];  // ldg*8+ii = ii
    float val = row_chunk[ii];
    if (val_for_choice > max_val_for_choice) {
        max_val_for_choice = val_for_choice;
        max_val = val;
        expert = start_col + ii;  // col + ii
    }
}
```

各线程扫描结果：

| 线程 | start_col | 扫描范围 | local max | expert |
|---|---|---|---|---|
| T0 | 0 | expert 0-7 | **0.15** | **2** |
| T1 | 8 | expert 8-15 | **0.20** | **11** |
| T2 | 16 | expert 16-23 | **0.05** | **21** |
| T3 | 24 | expert 24-31 | **0.01** | **24** |
| T4 | 32 | expert 32-39 | **0.10** | **38** |
| T5 | 40 | expert 40-47 | **0.01** | **40** |
| T6 | 48 | expert 48-55 | **0.08** | **49** |
| T7 | 56 | expert 56-63 | **0.01** | **56** |

**Step 2：线程组内 butterfly argmax reduce**

THREADS_PER_ROW=8，需要 log₂(8)=3 轮 shuffle：

```
轮1 (mask=4): T0↔T4, T1↔T5, T2↔T6, T3↔T7
  T0: max(0.15, 0.10) → 保留 0.15, expert=2
  T1: max(0.20, 0.01) → 保留 0.20, expert=11
  T2: max(0.05, 0.08) → 取 0.08, expert=49
  T3: max(0.01, 0.01) → 保留 0.01, expert=24 (较小编号胜出)
  T4: max(0.10, 0.15) → 取 0.15, expert=2
  T5: max(0.01, 0.20) → 取 0.20, expert=11
  T6: max(0.08, 0.05) → 保留 0.08, expert=49
  T7: max(0.01, 0.01) → 保留 0.01, expert=24

轮2 (mask=2): T0↔T2, T1↔T3, T4↔T6, T5↔T7
  T0: max(0.15, 0.08) → 保留 0.15, expert=2
  T1: max(0.20, 0.01) → 保留 0.20, expert=11
  T2: max(0.08, 0.15) → 取 0.15, expert=2
  T3: max(0.01, 0.20) → 取 0.20, expert=11
  ...

轮3 (mask=1): T0↔T1, T2↔T3, T4↔T5, T6↔T7
  T0: max(0.15, 0.20) → 取 0.20, expert=11  ✓
  T1: max(0.20, 0.15) → 保留 0.20, expert=11  ✓
  所有线程达成共识: max_val=0.20, expert=11
```

**Step 3：leader 写回** (thread_group_idx==0，即 T0)

```cpp
output[k * thread_row + 0] = 0.20;     // 不含 bias 的真实概率
indices[k * thread_row + 0] = 11;       // 选中 expert 11
source_rows[...] = 0 * num_rows + thread_row;
```

**Step 4：屏蔽 expert 11**

expert=11 属于线程 T1（expert 8-15），在 T1 的寄存器中：

```cpp
ldg_group_for_expert = 11 / 64 = 0
thread_to_clear_in_group = (11 / 8) % 8 = 1  // 即 T1
offset_for_expert = 11 % 8 = 3

// T1 执行：
row_chunk_for_choice[0 * 8 + 3] = -10000.f;
// 即 row_chunk_for_choice[3] = -10000.f（原来是 0.20）
```

> 注意：只修改 `row_chunk_for_choice`，**不修改** `row_chunk`。这样下一轮 expert 11 不会被再次选中，但如果需要读取其真实概率（renormalize 等场景），`row_chunk` 中仍保留原值。

##### 第 2 轮 Top-K（k_idx=1）：找第二大

屏蔽 expert 11 后，T1 的 `row_chunk_for_choice` 变为：

```
T1: [0.02, 0.01, 0.03, -10000, 0.01, 0.01, 0.02, 0.01]
                       ^^^^^^^ expert 11 被屏蔽
```

各线程 local argmax 结果：

| 线程 | local max | expert |
|---|---|---|
| T0 | 0.15 | 2 |
| T1 | 0.03 | 10 |  ← expert 11 被跳过
| T2 | 0.05 | 21 |
| T3 | 0.01 | 24 |
| T4 | 0.10 | 38 |
| T5 | 0.01 | 40 |
| T6 | 0.08 | 49 |
| T7 | 0.01 | 56 |

经过 3 轮 butterfly reduce → 共识：**max_val=0.15, expert=2**

写回后屏蔽 expert 2（T0 的 `row_chunk_for_choice[2] = -10000.f`）。

##### 后续轮次汇总

| k_idx | 选中 expert | max_val | 屏蔽操作 |
|---|---|---|---|
| 0 | expert 11 | 0.20 | T1: for_choice[3] = -10000 |
| 1 | expert 2 | 0.15 | T0: for_choice[2] = -10000 |
| 2 | expert 38 | 0.10 | T4: for_choice[6] = -10000 |
| 3 | expert 49 | 0.08 | T6: for_choice[1] = -10000 |
| 4 | expert 21 | 0.05 | T2: for_choice[5] = -10000 |
| 5 | expert 10 | 0.03 | T1: for_choice[2] = -10000 |

##### col 变量的寻址作用

双层循环中 `col` 的递进方式值得注意：

```cpp
for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
```

- `start_col = first_elt_read_by_thread = thread_group_idx × ELTS_PER_LDG`
- `COLS_PER_GROUP_LDG = ELTS_PER_LDG × THREADS_PER_ROW = 64`

**本例 LDG_PER_THREAD=1**，所以只有一轮外循环，`col` 始终等于 `start_col`。但对于 **NUM_EXPERTS=256** 的场景（LDG_PER_THREAD=2），每个线程的数据在内存中是**不连续**的（stride = COLS_PER_GROUP_LDG = 64），`col` 的跳跃确保 `expert = col + ii` 能正确映射到全局专家编号：

```
256 experts, bf16: VPT=16, THREADS_PER_ROW=16, ELTS_PER_LDG=8, LDG_PER_THREAD=2

线程 T0 (thread_group_idx=0):
  ldg=0: col=0,   读 expert [0..7]    → row_chunk[0..7]
  ldg=1: col=128, 读 expert [128..135] → row_chunk[8..15]

线程 T1 (thread_group_idx=1):
  ldg=0: col=8,   读 expert [8..15]   → row_chunk[0..7]
  ldg=1: col=136, 读 expert [136..143] → row_chunk[8..15]
```

`row_chunk` 数组下标是连续的 [0..VPT-1]，但对应的全局 expert 编号不连续。`col + ii` 的计算将寄存器下标正确映射回全局 expert 编号，这是 argmax 能返回正确 expert ID 的关键。

##### 设计要点总结

| 设计点 | 实现方式 | 目的 |
|---|---|---|
| 双数组分离 | `row_chunk`(真实概率) vs `row_chunk_for_choice`(可能含 bias) | bias 只影响选择，不影响输出权重 |
| 屏蔽已选 expert | 对应线程将 `for_choice[offset]` 置为 -10000 | 避免重复选择，无需线程间通信 |
| Butterfly reduce | log₂(THREADS_PER_ROW) 轮 `__shfl_xor` | 纯寄存器+shuffle，零 shared memory |
| Tie-breaking | `other_exp < expert` | 与 PyTorch argmax 语义一致 |
| col 跳跃寻址 | `col += COLS_PER_GROUP_LDG` | 正确映射非连续寄存器到全局 expert 编号 |

#### 5.4.5 阶段⑤:Renormalize (L551-562)

```cpp
if (renormalize && thread_group_idx == 0) {
    const float denom = selected_sum > 0.f ? selected_sum : 1.f;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        const int idx = k * thread_row + k_idx;
        output[idx] = output[idx] / denom;
    }
}
```

注意: 这里**再次写回 global memory**,把之前写入的 output[idx] 再读出来除一次。相比阶段④在寄存器里做完再写,这样牺牲一点点带宽换取代码清晰。

---

### 5.5 `TopkConstants` —— 编译期自动计算 VPT

```cpp
// L567-577
template <int EXPERTS, int BYTES_PER_LDG, int WARP_SIZE, typename InputType>
struct TopkConstants {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static const int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
```

**作用**: 给定 EXPERTS 和 BYTES_PER_LDG,自动推导 VPT、THREADS_PER_ROW、ROWS_PER_WARP。这些全在编译期完成,运行时零开销。

### 5.6 `topkGatingLauncherHelper`(L580-595)

```cpp
template <int EXPERTS, int WARPS_PER_TB, int WARP_SIZE, int MAX_BYTES_PER_LDG,
          typename IndType, typename InputType, ScoringFunc SF>
void topkGatingLauncherHelper(...) {
    static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(InputType) * EXPERTS);
    using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE, InputType>;

    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);   // (32, 4)

    topkGating<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG, WARP_SIZE,
               IndType, InputType, SF>
        <<<num_blocks, block_dim, 0, stream>>>(...);
}
```

**职责**:
1. 用 `TopkConstants` 算出 VPT / ROWS_PER_WARP;
2. 根据 num_rows 算出 grid size;
3. 启动 kernel。

### 5.7 `LAUNCH_TOPK` 宏(L597-624)

```cpp
#define LAUNCH_TOPK(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)   \
    topkGatingLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, WARP_SIZE, MAX_BYTES, \
                             IndType, InputType, SF>(      \
        gating_output, nullptr, topk_weights, topk_indices, \
        token_expert_indices, num_tokens, topk,             \
        0, num_experts, renormalize, bias, stream);
```

ROCm 分支多一层 WARP_SIZE 运行时判断(32 或 64)。这个宏存在的唯一目的是**减少下面 switch-case 的重复代码**。

---

## 6. Host 端调度层

### 6.1 `topkGatingKernelLauncher` —— 按 num_experts 分发(L626-716)

```cpp
template <typename IndType, typename InputType, ScoringFunc SF>
void topkGatingKernelLauncher(...) {
    static constexpr int WARPS_PER_TB = 4;
    static constexpr int BYTES_PER_LDG_POWER_OF_2 = 16;
    static constexpr int BYTES_PER_LDG_MULTIPLE_64 =
        (bf16/fp16) ? 4 : 8;   // bf16/fp16 下用 4 字节, fp32 下用 8 字节

    switch (num_experts) {
        case 1:   LAUNCH_TOPK(1,   WARPS_PER_TB, 16); break;
        case 2:   LAUNCH_TOPK(2,   WARPS_PER_TB, 16); break;
        ...
        case 256: LAUNCH_TOPK(256, WARPS_PER_TB, 16); break;
        case 512: LAUNCH_TOPK(512, WARPS_PER_TB, 16); break;
        // 非 2 的幂但 64 的倍数
        case 192: LAUNCH_TOPK(192, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;
        case 320: LAUNCH_TOPK(320, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;
        case 384: LAUNCH_TOPK(384, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;
        case 448: LAUNCH_TOPK(448, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;
        case 576: LAUNCH_TOPK(576, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;

        default: {
            // 兜底: 走通用路径
            moeSoftmax / moeSigmoid → workspace
            moeTopK        → workspace → 输出
        }
    }
}
```

**观察**: 每个 `case` 会在编译期实例化一份完整特化。整个文件最终编译出的 kernel 数量约为:

```
num_experts 选项数 (15) × input dtype (3) × IndType (3) × SF (2) ≈ 270 个特化
```

编译时间与二进制膨胀的代价,换来运行时零分支的性能。

### 6.2 `dispatch_topk_launch` —— 按 IndType 分发(L722-772)

```cpp
template<typename ComputeType, ScoringFunc SF>
void dispatch_topk_launch(...) {
    // 检查 bias tensor
    if (bias.has_value()) {
        TORCH_CHECK(bias.scalar_type() == Float);  // 必须 fp32
        TORCH_CHECK(bias.dim() == 1);
        TORCH_CHECK(bias.size(0) == num_experts);
        bias_ptr = bias.data_ptr<float>();
    }

    // 按 topk_indices 的 dtype 分发
    if (topk_indices.scalar_type() == Int)
        topkGatingKernelLauncher<int, ComputeType, SF>(...);
    else if (topk_indices.scalar_type() == UInt32)
        topkGatingKernelLauncher<uint32_t, ComputeType, SF>(...);
    else if (topk_indices.scalar_type() == Long)
        topkGatingKernelLauncher<int64_t, ComputeType, SF>(...);
}
```

**为什么支持三种 IndType**: vLLM 不同后端对索引 dtype 有不同要求,例如 cutlass grouped GEMM 要 int32,CUTLASS sm90 要 int64,通用 PyTorch 路径要 int64。提前做类型分发可让下游 kernel 直接用匹配类型,避免额外 cast。

### 6.3 `topk_softmax` / `topk_sigmoid` —— Python 入口(L774-848)

```cpp
void topk_softmax(
    torch::Tensor& topk_weights,          // [num_tokens, topk], fp32
    torch::Tensor& topk_indices,          // [num_tokens, topk], int/int64
    torch::Tensor& token_expert_indices,  // [num_tokens, topk], int
    torch::Tensor& gating_output,         // [num_tokens, num_experts], bf16/fp16/fp32
    bool renormalize,
    std::optional<torch::Tensor> bias)
{
    const int num_experts = gating_output.size(-1);
    const auto num_tokens = gating_output.numel() / num_experts;
    const int topk = topk_weights.size(-1);

    // 判断是否需要 workspace (通用路径才需要)
    const bool is_pow_2 = (num_experts & (num_experts - 1)) == 0;
    const bool needs_workspace = !is_pow_2 || num_experts > 256;
    const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

    // 绑定 CUDA device + stream
    const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 分配 workspace (仅非融合路径实际使用)
    torch::Tensor softmax_workspace = torch::empty({workspace_size}, ...);

    // 按 gating_output 的 dtype 分发到对应 ComputeType
    if (dtype == Float)
        dispatch_topk_launch<float,         SCORING_SOFTMAX>(...);
    else if (dtype == Half)
        dispatch_topk_launch<__half,        SCORING_SOFTMAX>(...);
    else if (dtype == BFloat16)
        dispatch_topk_launch<__nv_bfloat16, SCORING_SOFTMAX>(...);
}
```

**`topk_sigmoid` 的结构完全一样**,只有模板参数换成 `SCORING_SIGMOID`。

**`needs_workspace` 的判断细节**:
- `is_pow_2` 但 `num_experts > 256`: workspace 仍然分配(编译时 512 也有特化,但保险起见给兜底)
- `is_pow_2` 且 `num_experts ≤ 256`: 不分配,走融合路径
- 非 2 的幂: 判断是否 64 的倍数在 switch-case 里处理;如果是 192/320/384/448/576,实际不用 workspace,但这里 workspace 还是被分配了(一点点显存浪费,但逻辑简单)

**TORCH 绑定**(`csrc/moe/torch_bindings.cpp`):
```cpp
m.def("topk_softmax(Tensor! topk_weights, ...) -> ()");
m.impl("topk_softmax", torch::kCUDA, &topk_softmax);
```

在 Python 里可通过 `torch.ops._moe_C.topk_softmax(...)` 调用。

---

## 7. Python 端调用链

### 7.1 用户入口

```python
# vllm/model_executor/layers/fused_moe/router/fused_topk_router.py

def fused_topk(hidden_states, gating_output, topk, renormalize,
               indices_type=None, scoring_func="softmax"):
    M = hidden_states.size(0)
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=...)
    topk_ids     = torch.empty(M, topk,
                               dtype=torch.int32 if indices_type is None else indices_type,
                               device=...)
    token_expert_indices = torch.empty(M, topk, dtype=torch.int32, device=...)

    if scoring_func == "softmax":
        topk_func = dispatch_topk_softmax_func(use_rocm_aiter=...)
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize)
    elif scoring_func == "sigmoid":
        topk_func = dispatch_topk_sigmoid_func(use_rocm_aiter=...)
        topk_weights, topk_ids = topk_func(...)
    return topk_weights, topk_ids, token_expert_indices


def vllm_topk_softmax(topk_weights, topk_indices, token_expert_indices,
                      gating_output, renormalize=False):
    ops.topk_softmax(topk_weights, topk_indices, token_expert_indices,
                     gating_output, renormalize)
    return topk_weights, topk_indices
```

### 7.2 `FusedTopKRouter._compute_routing`

```python
class FusedTopKRouter(BaseRouter):
    def _compute_routing(self, hidden_states, router_logits, indices_type):
        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            scoring_func=self.scoring_func,
        )
        return topk_weights, topk_ids
```

该 router 最终被 `FusedMoE.forward` 调用,拿到 topk_weights 和 topk_ids 后,配合 `topk_indices` 的 `num_experts` 哨兵值(EP 场景)进入 dispatch / expert GEMM 阶段。

### 7.3 ROCm aiter 分支

如果检测到 ROCm + aiter 后端可用(`rocm_aiter_ops.is_fused_moe_enabled()`),会走 `rocm_aiter_ops.topk_softmax` —— 这是另一个 ROCm 专用实现,本文件不涉及。

### 7.4 `_custom_ops.py` 对外层

```python
# vllm/_custom_ops.py L2363
def topk_softmax(topk_weights, topk_indices, token_expert_indices,
                 gating_output, renormalize=False):
    torch.ops._moe_C.topk_softmax(
        topk_weights, topk_indices, token_expert_indices,
        gating_output, renormalize, None)
```

这一层只是做一次类型守护,然后调用 C++ 注册的算子。

---

## 8. 完整调用链路图

```
┌─────────────────────────────────────────────────────────────────────┐
│  Python 侧                                                            │
│                                                                      │
│  FusedMoE.forward                                                    │
│    └─ FusedTopKRouter._compute_routing                              │
│         └─ fused_topk(hidden, router_logits, topk, renormalize, ..) │
│              ├─ 分配 topk_weights, topk_ids, token_expert_indices   │
│              └─ vllm_topk_softmax(...)                              │
│                   └─ ops.topk_softmax(...)                          │
│                        └─ torch.ops._moe_C.topk_softmax(...)        │
└─────────────────────────────────────────┬───────────────────────────┘
                                          │ (pybind / torch 算子调用边界)
┌─────────────────────────────────────────▼───────────────────────────┐
│  C++ 侧 (topk_softmax_kernels.cu)                                    │
│                                                                      │
│  topk_softmax(...)              // L774   Python 可见入口            │
│    │                                                                 │
│    ├─ 判断 num_experts 是否 pow2、是否需要 workspace                  │
│    ├─ 分配 workspace                                                 │
│    │                                                                 │
│    └─ dispatch_topk_launch<ComputeType, SF>(...)   // L722  按 dtype │
│         │                                                            │
│         └─ topkGatingKernelLauncher<IndType, CT, SF>(...)  // L626  │
│              │                                                       │
│              ├─ switch (num_experts):                                │
│              │     case 2^k or 64×: LAUNCH_TOPK(N, 4, 16)          │
│              │        └─ topkGatingLauncherHelper<...>(...)  //L580 │
│              │             └─ topkGating<<<grid, (32,4)>>>(...)      │
│              │                  ├─ 向量化加载 bf16/fp16 → fp32      │
│              │                  ├─ Softmax/Sigmoid via butterfly    │
│              │                  ├─ k 次 argmax via butterfly        │
│              │                  └─ (renormalize)                    │
│              │                                                       │
│              └─ default:                                             │
│                   ├─ moeSoftmax/moeSigmoid<<<num_tokens,256>>>      │
│                   │    (输出到 workspace)                            │
│                   └─ moeTopK<<<num_tokens,256>>>(workspace, ...)    │
└──────────────────────────────────────────────────────────────────────┘
```

**分支决策表**:

| `num_experts` | 路径 | Kernel 数 | workspace |
|---|---|---|---|
| 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 (2 的幂) | 融合 `topkGating` | 1 | 不用 |
| 192, 320, 384, 448, 576 (64 的倍数, CUDA only) | 融合 `topkGating` | 1 | 分配但不用 |
| 其他(如 96, 100, 1024) | 通用路径 | 2(softmax + topK) | **必需** |

---

## 9. DeepSeek-MoE-16B 端到端实战

用 DeepSeek-MoE-16B(`num_experts=64`, `topk=6`, bf16)为例,走完一次 routing。

### 9.1 输入规格

假设 batch size 下 1024 tokens:
```
hidden_states        形状 [1024, 2048], bf16
router_logits = gate(hidden_states)
                     形状 [1024, 64],  bf16
```

### 9.2 Python 调用

```python
# FusedTopKRouter._compute_routing
topk_weights, topk_ids, tei = fused_topk(
    hidden_states=hidden_states,
    gating_output=router_logits,   # [1024, 64]
    topk=6,
    renormalize=False,              # DeepSeek-MoE-16B 用 norm_topk_prob=False
    indices_type=torch.int32,
    scoring_func="softmax",
)
```

### 9.3 进入 C++

1. `topk_softmax` 入口:
   - `num_experts = 64`, `num_tokens = 1024`, `topk = 6`
   - `is_pow_2 = True`, `needs_workspace = False`, `workspace_size = 0`
   - dtype = bf16 → `dispatch_topk_launch<__nv_bfloat16, SCORING_SOFTMAX>`

2. `dispatch_topk_launch`:
   - `topk_indices.scalar_type() == Int` → `topkGatingKernelLauncher<int, __nv_bfloat16, SCORING_SOFTMAX>`

3. `topkGatingKernelLauncher`:
   - `switch(64): LAUNCH_TOPK(64, 4, 16)`
   - 展开后 → `topkGatingLauncherHelper<64, 4, 32, 16, int, __nv_bfloat16, SCORING_SOFTMAX>`

4. `topkGatingLauncherHelper` 编译期常量:
   ```
   BYTES_PER_LDG = min(16, 2*64) = 16
   ELTS_PER_LDG  = 16 / 2 = 8
   VPT           = max(1, 64 / (8*32)) * 8 = 1 * 8 = 8
   THREADS_PER_ROW = 64 / 8 = 8
   ROWS_PER_WARP = 32 / 8 = 4
   ROWS_PER_CTA  = 4 * 4 = 16
   num_warps     = (1024 + 4 - 1) / 4 = 256
   num_blocks    = (256 + 4 - 1) / 4 = 64
   block_dim     = (32, 4)   即每 block 128 线程
   ```

5. `topkGating<<<64, (32,4)>>>`:
   - 64 个 block,每 block 16 行,共 1024 行 ✓
   - 每 warp 4 行,每行 8 个线程
   - 每线程持有 8 个 bf16 → fp32 转成 8 个 fp32 寄存器
   - softmax via butterfly reduce(3 轮 shuffle)
   - 6 次 top-k argmax(每次 3 轮 butterfly + 写 global)
   - 无 renormalize 分支

### 9.4 输出

```
topk_weights         [1024, 6]  fp32     选中的概率(未 renormalize)
topk_ids             [1024, 6]  int32    选中的专家编号
token_expert_indices [1024, 6]  int32    = k_idx * 1024 + token_idx 映射
```

### 9.5 DeepSeek 的后续流程

`topk_ids` 会与 `expert_map`(EP 场景下哪些专家本 rank 持有) 一起进入 All2All dispatch,把 token 发到对应专家的 rank;`topk_weights` 最后在 combine 阶段做加权求和:

```
final_hidden = Σ_{k=0..5}  topk_weights[t, k] * expert[topk_ids[t, k]](hidden[t])
```

---

## 10. 性能设计要点总结

| 设计 | 目的 | 代价 |
|-----|-----|-----|
| 融合 softmax+topk+renorm 到一个 kernel | 避免多次读写 global memory | 代码复杂,二进制膨胀 |
| 用 warp shuffle 替代 shared memory | 延迟低、SM 占用率高、无 bank conflict | 要求 row 能塞进一个线程组的寄存器 |
| 向量化加载(16 字节一次) | 降低 memory transaction 数 | 要求对齐 + num_experts 特定倍数 |
| bf16×2 / fp16×2 硬件转换 | 单指令 2 个元素 | 要求 ELTS_PER_LDG 偶数 |
| `__expf`(sigmoid) vs `expf`(softmax) | sigmoid 不涉及归约,精度损失不累积 | 小的 ulp 误差 |
| 模板特化 num_experts 为 switch-case | 编译期确定常量,循环展开彻底 | 270+ kernel 特化,编译慢 |
| Bias-aware 选择 + unbiased 权重 | 支持 DeepSeek correction bias,又不污染 logits | 维护两套 max |
| `indices[idx] = num_experts` 哨兵 | EP 场景下简洁表示"不在本 rank" | 下游需识别此值 |
| Tie-breaking 选小 expert 编号 | 与 PyTorch 一致,跨后端可复现 | 多两次 shuffle 比较 |
| 32 步一次 AllReduce (DP wave,不在本文件) | 与本 kernel 无关 | — |

### 10.1 两条路径的取舍

| 指标 | 融合 `topkGating` | 通用 `moeSoftmax + moeTopK` |
|-----|------|------|
| Kernel 启动次数 | 1 | 2 |
| Global memory 读写 | 1 读 + 1 写 | 1 读 + 2 写 + 2 读 |
| Shared memory 用量 | 0 | 数 KB |
| 适用 num_experts | 1/2/.../512, 或 192/320/384/448/576 | 任意 |
| 二进制大小 | 大(270+ 特化) | 小 |

### 10.2 潜在改进方向(仅分析,非建议 PR)

- **renormalize 在阶段④寄存器里完成**:当前在阶段⑤从 global 再读回来除,多一次 L2 命中。
- **Sigmoid 路径也接入 bias**:当前 `row_chunk_for_choice` 的 bias 相加在 sigmoid 后做,若能融合到 `1/(1+exp(-x))` 中可省一轮加法。
- **num_experts > 512**:目前直接 fallback 到 workspace 路径;新兴大 MoE (如 Qwen2.5-MoE-A14B 用 60 expert、Llama-4 Scout 用 16 × 17B)暂不触达,但未来可能需要 1024 特化。

---

## 附录 A:关键源码行号索引

| 实体 | 文件 | 行号 |
|-----|------|-----|
| `AlignedArray` | `topk_softmax_kernels.cu` | L43-52 |
| `toFloat` | `topk_softmax_kernels.cu` | L54-63 |
| `enum ScoringFunc` | `topk_softmax_kernels.cu` | L66-69 |
| `moeSoftmax` kernel | `topk_softmax_kernels.cu` | L74-132 |
| `moeSigmoid` kernel | `topk_softmax_kernels.cu` | L134-153 |
| `moeTopK` kernel | `topk_softmax_kernels.cu` | L155-245 |
| `topkGating` kernel | `topk_softmax_kernels.cu` | L263-563 |
| `TopkConstants` | `topk_softmax_kernels.cu` | L567-577 |
| `topkGatingLauncherHelper` | `topk_softmax_kernels.cu` | L580-595 |
| `LAUNCH_TOPK` 宏 | `topk_softmax_kernels.cu` | L597-624 |
| `topkGatingKernelLauncher` | `topk_softmax_kernels.cu` | L626-716 |
| `dispatch_topk_launch` | `topk_softmax_kernels.cu` | L722-772 |
| `topk_softmax` 入口 | `topk_softmax_kernels.cu` | L774-810 |
| `topk_sigmoid` 入口 | `topk_softmax_kernels.cu` | L812-848 |
| Torch 算子注册 | `csrc/moe/torch_bindings.cpp` | L7, L10, L14, L17 |
| `fused_topk` Python | `fused_moe/router/fused_topk_router.py` | L69-113 |
| `vllm_topk_softmax` Python | `fused_moe/router/fused_topk_router.py` | L17-32 |
| `FusedTopKRouter` | `fused_moe/router/fused_topk_router.py` | L116-165 |

## 附录 B:关键 CUDA 术语速查

| 术语 | 含义 |
|-----|-----|
| SM (Streaming Multiprocessor) | GPU 上的一个计算单元,承载多个 block |
| Warp | 32 线程为一组步调一致执行(NVIDIA);ROCm 64 |
| Lane | Warp 内线程的编号(0-31) |
| SIMT | Single Instruction Multiple Threads,warp 内同指令多数据 |
| Coalesced memory access | 相邻线程读相邻地址,硬件合并成一次 transaction |
| Bank conflict | 多个线程同时访问 shared memory 同一 bank,串行化 |
| Occupancy | SM 上同时驻留的 warp 数 / 最大可能数 |
| ILP (Instruction Level Parallelism) | 单线程内无依赖指令并行 |
| Butterfly reduce | 用 XOR 模式在 log2(N) 轮内完成 N 线程归约 |
| Register pressure | 寄存器使用过多导致 occupancy 下降或 spill 到 local memory |
| LDG | Load from Global memory 指令 |

---

> **报告结束**
> 本文档基于 `csrc/moe/topk_softmax_kernels.cu` 全文(848 行)及其调用方 `fused_topk_router.py` 编写。
> 所有行号与实际源码对齐,可直接点击定位。
