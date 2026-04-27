# `moe_align_sum_kernels.cu` 调用链与实现逻辑分析

> 入口脚本：`examples/offline_inference/data_parallel.py`  
> 目标算子：`csrc/moe/moe_align_sum_kernels.cu`  
> 面向读者：MoE / CUDA kernel 初学者  
> 参考模型：默认 `/data/models/deepseek/deepseek-moe-16b-base`
> （`num_routed_experts = 64`，`num_experts_per_tok (topk) = 6`，`hidden_size = 2048`）

---

## 0. 总览：这个 `.cu` 文件里到底有什么？

文件里一共实现了 **3 组 Kernel 对应 3 个对外 C++ 函数**，都会被 Python 端通过 `vllm._custom_ops.ops.xxx` 调到：

| 对外函数 | Kernel（核函数） | 用途 |
|---|---|---|
| `moe_align_block_size` | `moe_align_block_size_kernel` + `count_and_sort_expert_tokens_kernel`（或它们的 small-batch 版本） | **MoE GEMM 前的排序/分桶/对齐**——把每个 token 按被分派到的 expert 分组，填充成 `block_size` 的整数倍，准备好 GroupGEMM 要消费的 `sorted_token_ids / expert_ids / num_tokens_post_pad` |
| `batched_moe_align_block_size` | `batched_moe_align_block_size_kernel` | "已经分桶好"的场景（batched expert format，常用于 All-to-All 的 EP 后端）做 block padding |
| `moe_sum` | `moe_sum_kernel<scalar_t, TOPK>` | MoE 最后一步：把 `top_k` 路 experts 的输出在 k 维做求和，得到最终的 `[num_tokens, hidden]` |

另外还有两组 **LoRA 变体**（`moe_lora_align_block_size_kernel` 等），逻辑就是把上面带一个 `lora_id` 的循环，本文不展开。

---

## 1. 从 `data_parallel.py` 到 CUDA Kernel 的完整调用链

```
examples/offline_inference/data_parallel.py
└── LLM(model="deepseek-moe-16b-base").generate(prompts, sampling_params)
    └── [V1 Engine] ModelRunner.execute_model
        └── DeepseekModel.forward → DeepseekDecoderLayer.forward
            └── DeepseekMoE.forward          # vllm/model_executor/models/deepseek.py
                └── FusedMoE.forward          # layers/fused_moe/layer.py
                    └── fused_experts(...)    # layers/fused_moe/fused_moe.py
                        ├── moe_align_block_size(topk_ids, BLOCK_SIZE_M=64, num_experts=64, ...)
                        │   └── vllm/_custom_ops.py  : torch.ops._moe_C.moe_align_block_size
                        │       └── csrc/moe/moe_align_sum_kernels.cu
                        │           : void moe_align_block_size(...)
                        │             ├── moe_align_block_size_kernel<scalar_t><<<2, 1024>>>
                        │             └── count_and_sort_expert_tokens_kernel<scalar_t><<<grid, 256>>>
                        │
                        ├── (两次 GroupGEMM：w1 up/gate + w2 down)
                        │
                        └── ops.moe_sum(intermediate_cache3, out_hidden_states)
                            └── csrc/moe/moe_align_sum_kernels.cu
                                : void moe_sum(...)
                                  └── moe_sum_kernel<scalar_t, TOPK=6→default branch>
                                      实际 DeepSeek topk=6 不在 {2,3,4}，
                                      走 default 分支 at::sum_out(output, input, 1)
```

> 重要：DeepSeek-MoE-16B 的 `topk=6`，落入 `moe_sum` 的 `default` 分支，**使用 ATen `sum_out`**；只有当模型是 `topk ∈ {2,3,4}` 时才会启动 `moe_sum_kernel`。

---

## 2. 背景：为什么要 "align block size"？

MoE 的核心计算是 GroupGEMM：**每个 expert 是一组权重 `W_e`**，而一个 batch 里不同 token 被路由到不同的 expert。
Triton/CUTLASS 里的 GEMM 要求一个 block（`BLOCK_M` 行）里所有 token 必须属于**同一个 expert**。所以在算 GEMM 之前要：

1. 统计每个 expert 被分到多少 token；
2. 把同一 expert 的 token 聚在一起（其实不搬数据，只搬**索引**）；
3. 每个 expert 末尾补齐到 `block_size` 的整数倍（padding）；
4. 输出三个张量喂给 GEMM：
   - `sorted_token_ids[M_padded]`：token 全局索引，按 expert 分桶 & 末尾填充 `numel`（哨兵）；
   - `expert_ids[M_padded / block_size]`：每个 block 对应哪个 expert（`-1` 表示空 block）；
   - `num_tokens_post_pad[1]`：padding 后的总 token 数。

### 2.1 一个跑得动的小例子

先用官方 docstring 里的小例子建立直觉（`num_experts=4, block_size=4, topk=3`）：

```
topk_ids (shape=[4,3]) =
  [[2, 3, 4],     # token0 被送去 expert 2/3/4
   [1, 2, 4],
   [1, 3, 4],
   [1, 2, 3]]
topk_ids.numel() = 12   # 这就是 sorted_token_ids 里的"哨兵"值
```

每个 expert 实际 token 数：`exp1=3, exp2=3, exp3=3, exp4=3`，`block_size=4` → 每个 expert pad 到 4。

```
sorted_token_ids = [3, 6, 9, 12,   # expert 1, 索引 3/6/9 = (1,0)/(2,0)/(3,0), 12 是 pad
                    0, 4, 10,12,   # expert 2
                    1, 7, 11,12,   # expert 3
                    2, 5, 8, 12]   # expert 4
expert_ids       = [1, 2, 3, 4]
num_tokens_post_pad = 16
```

注意 `sorted_token_ids[i]` 存的是**展平后 topk_ids 里的位置**（不是 token 下标），GEMM 里再通过 `idx / topk` 取对应 `hidden_state` 行。

---

## 3. `moe_align_block_size` 的 Host 端调度（line 477-569）

```cpp
void moe_align_block_size(topk_ids, num_experts, block_size,
                          sorted_token_ids, experts_ids,
                          num_tokens_post_pad, maybe_expert_map) {
    int64_t padded_num_experts = ceil(num_experts, WARP_SIZE) * WARP_SIZE;
    int experts_per_warp       = WARP_SIZE;  // 32
    int threads                = 1024;
    TORCH_CHECK(padded_num_experts < 1024);   // ① 硬性限制
    ...
    bool small_batch_expert_mode =
        (topk_ids.numel() < 1024) && (num_experts <= 64);   // ② 分流
```

### ① 为什么 `padded_num_experts < 1024`？
因为后面 prefix-sum 用的是 `cub::BlockScan<int32_t, 1024>`，一个 block 1024 线程，**一个线程负责一个 expert**。

### ② 两条 code path
- **small-batch** 路径（tokens 少 `<1024` 且 expert 少 `≤64`）：一个 kernel 干完所有事；
- **通用路径**：分两步 kernel：`moe_align_block_size_kernel`（统计+cumsum+填 expert_ids）+ `count_and_sort_expert_tokens_kernel`（用原子加写 sorted_token_ids）。

### 以 DeepSeek-MoE-16B 做参数推演

假设一次 forward `num_tokens = 2048, topk = 6, num_experts = 64, block_size = 64`：

```
topk_ids.numel()     = 2048 * 6     = 12288
padded_num_experts   = ceil(64, 32) * 32 = 64
experts_per_warp     = 32
threads              = 1024
num_warps_per_block  = 1024 / 32   = 32
small_batch_expert_mode = false    # (12288 < 1024) 不成立
```

走 **通用路径**。两个 kernel launch 参数：

```cpp
align_kernel<<<2, 1024, shared_mem = 32*32*4 = 4096 B, stream>>>(...);
//  ^grid=2：一个 block 干 counting+cumsum+填 expert_ids，另一个 block 负责把 sorted_token_ids 初始化成哨兵
//  ^block=1024：1 thread 对应 1 expert（最多 1023 个 expert）

sort_kernel<<<dim3(1, actual_blocks), 256, 0, stream>>>(...);
//  block=256, grid.y = ceil(12288/256) = 48
```

---

## 4. `moe_align_block_size_kernel` 逐步拆解（line 323→81）

```cpp
template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* topk_ids,       // [num_tokens, topk]，DeepSeek: int32
    int32_t*        sorted_token_ids,
    int32_t*        expert_ids,
    int32_t*        total_tokens_post_pad,
    int32_t*        expert_map,     // EP 场景：全局 expert_id -> 本 rank 索引；-1 表示不归我
    int32_t num_experts,            // 64
    int32_t padded_num_experts,     // 64
    int32_t experts_per_warp,       // 32
    int32_t block_size,             // 64
    size_t  numel,                  // 12288
    int32_t* cumsum,                // [num_experts+1] 工作区
    int32_t  max_num_tokens_padded, // = numel + num_experts*(block-1) = 12288+64*63=16320
    int32_t  topk_num,              // 6
    bool     has_expert_map);
```

模板参数 `scalar_t` 由 `VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES` 决定——`topk_ids` 多是 `int32`，也可能 `uint32`。

### Step 0：两个 block 分工（`blockIdx.x % 2`）

```cpp
if (blockIdx.x % 2) {                                    // blockIdx.x == 1
    for (it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x)
        sorted_token_ids[it] = numel;   // 把整个数组写成哨兵（12288）
    return;
}
```

- `blockIdx.x==1` 专职把 16320 个 int 写成 `numel=12288`（后面 block 0 填不到的就保持哨兵）。
- `blockIdx.x==0` 做真正的统计/排序。

> 之所以拆两个 block：GPU 不能跨 block 同步，把"初始化"这种只写不读的工作单独给一个 block 并行覆盖，能和 block 0 的计算重叠。

### Step 1：按 expert 计数（block 0）

```cpp
extern __shared__ int32_t shared_counts[];   // 大小 = num_warps * experts_per_warp = 32*32 ints
const int warp_id          = threadIdx.x / WARP_SIZE;  // 0..31
const int my_expert_start  = warp_id * experts_per_warp;

for (int i = 0; i < experts_per_warp; ++i)
    shared_counts[warp_id*experts_per_warp + i] = 0;   // warp-local 清零
__syncthreads();

for (size_t i = tid; i < numel; i += blockDim.x) {     // 12288 个 topk_id
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts) continue;            // 越界（无效）跳过
    if (has_expert_map) {                              // EP 场景
        expert_id = expert_map[expert_id];
        if (expert_id == -1) continue;                 // 不归本 rank
    }
    int warp_idx       = expert_id / experts_per_warp; // 哪个 warp 负责
    int expert_offset  = expert_id % experts_per_warp; // warp 内偏移
    atomicAdd(&shared_counts[warp_idx*experts_per_warp + expert_offset], 1);
}
__syncthreads();
```

**设计精髓**：`shared_counts` 被**按 warp 切成 32 段**（warp-striped layout），访问
`shared_counts[warp_idx*32 + off]` 时，同 warp 内不同 lane 访问**不同 bank**，有效降低 shared-memory bank conflict 和 atomicAdd 竞争。

数值直觉：`64` 个 expert 均匀分布 → 每个 expert 大约 `12288/64 = 192` tokens，即 shared_counts 每格约 192。

### Step 2：按 expert 前缀和（BlockScan，1 thread = 1 expert）

```cpp
using BlockScan = cub::BlockScan<int32_t, 1024>;
__shared__ typename BlockScan::TempStorage temp_storage;

int expert_count = 0;
int expert_id    = threadIdx.x;              // tid 就是 expert_id
if (expert_id < num_experts) {
    expert_count = shared_counts[warp_idx*experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;  // 先 pad 到 64 的倍数！
}

int cumsum_val;
BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);     // 独占前缀和

if (expert_id <= num_experts) cumsum[expert_id] = cumsum_val;
if (expert_id == num_experts) total_tokens_post_pad[0] = cumsum_val;
```

**要点**：先把每个 expert 的计数 **向上对齐到 `block_size`**，再做前缀和，这样每个 expert 的起点一定落在 block 边界上，GEMM 才能干净地按 block 取数。

例：若 expert 0..3 实际数 = `192, 192, 192, 192`（本来就是 64 的倍数），则
`cumsum = [0, 192, 384, 576, 768, …, 12288]`，`num_tokens_post_pad = 12288`。
若某个 expert 数 = `190`，pad 后是 `192`（ceil(190,64)*64 = 3*64），下一 expert 的 offset 不变，但 `sorted_token_ids` 里最后 2 格会留下哨兵。

### Step 3：写 `expert_ids`（告诉 GEMM 每个 block 用哪个 expert）

```cpp
if (threadIdx.x < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size)
        expert_ids[i / block_size] = tid;    // 每个 block 都标记归属的 expert
}
// 尾部没用到的 block 用 -1 填
for (i = cumsum[num_experts]/block_size + tid; i < max_num_m_blocks; i += blockDim.x)
    expert_ids[i] = inactive_expert_id;      // moe_align: -1
```

DeepSeek 例：`expert_ids` 大小 `=16320/64 = 255`，前 `12288/64 = 192` 个依次是 0..63 各占 3 个 block，剩下 63 个为 `-1`。

> 注意：**`sorted_token_ids` 的真实值并不在这个 kernel 里写**，此 kernel 只写 `cumsum / expert_ids / num_tokens_post_pad`。真正往 `sorted_token_ids` 里填 token 索引是下面的 `sort_kernel`。

---

## 5. `count_and_sort_expert_tokens_kernel`（line 339/292）

```cpp
template <typename scalar_t>
__device__ void _count_and_sort_expert_tokens(...) {
    const size_t tid    = blockIdx.y * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.y;

    for (size_t i = tid; i < numel; i += stride) {     // 再扫一次 topk_ids
        int32_t expert_id = topk_ids[i];
        if (expert_id >= num_experts) continue;
        if (has_expert_map) {
            expert_id = expert_map[expert_id];
            if (expert_id == -1) continue;
        }
        int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;           // 写入该 expert 桶内下一个槽位
    }
}
```

- **Launch 维度**：`grid=(1, 48), block=256`，总共 `48*256 = 12288` 个线程恰好一人处理一个 topk_id（接近 1:1）。
- **算法**：把 `cumsum[e]`（expert e 的桶起始偏移）当 `atomicAdd` 的"下一个空位"指针，线程 i 读到 expert e 就把自己的下标 `i` 写进去，然后 `cumsum[e]++`。
- **结果**：`sorted_token_ids[…]` 被同 expert 的 token 索引填满；**pad 的槽保留 block 0（`blockIdx.x==1` 那个 block）写进去的 `numel=12288` 哨兵不动**。

> 思考："写完 atomicAdd 后 `cumsum` 就被改得乱七八糟了"——对，这里的 `cumsum` 是一次性工作区（`cumsum_buffer`），用完即弃。

### 为什么比直接 bucket-sort 高效？
- 只需要 2 次全量扫 `topk_ids`；
- 不需要在 host 端或 shared 里建 `num_experts` 大的 bucket 指针；
- 绝大部分冲突被分散在 64 个 atomic 地址上，性能足够。

---

## 6. Small-batch 分支（line 183-289）

当 token 很少（`numel < 1024`）且 expert ≤64 时，上面的两段式 launch 反而不划算（kernel launch 开销 + 分散工作）。于是 small-batch 分支把所有工作塞进一个 kernel：

- `fill_threads=256` 个线程负责把 `sorted_token_ids` 写成 `numel` 哨兵；
- 剩下 `max(num_experts, WARP_SIZE)` 个线程做 counting + cumsum + 写 expert_ids + 写 sorted_token_ids；
- counting 用 `tokens_cnts[(tid+1)*num_experts + e]` 的 **per-thread 计数**，再做 per-expert 横向累加，避免 atomic。

因为这时线程数不够做 BlockScan，用普通的串行 prefix sum（`if (tid == 0)` 单线程循环 64 次）即可。

---

## 7. `batched_moe_align_block_size_kernel`（line 22-78）

和上面完全不同的场景：**输入已经是 "每个 expert 一个 batch 的平铺结构"**（典型：DeepEP / pplx-kernels all-to-all 之后），只需要根据 `batch_num_tokens[e]` 做 padding，不需要排序。

```cpp
__global__ void batched_moe_align_block_size_kernel(
    int32_t num_batches,               // = num_local_experts
    int32_t max_tokens_per_batch,      // 每个 expert 预留的槽位
    int32_t block_size,
    const int32_t* batch_num_tokens,   // [num_batches]
    int32_t* sorted_ids,
    int32_t* block_ids,                // 每个 block 对应哪个 batch/expert
    int32_t* num_tokens_post_pad);
```

- 1 个 block，1024 线程，**1 thread = 1 batch**（因此 `B ≤ 1024`）；
- 用 `cub::BlockScan<int32_t, 1024>` 做 exclusive prefix sum 得到每个 batch 的起始 offset；
- 没有 expert_id 排序，只是把第 b 个 batch 的 `b_num_tokens` 个 token 顺序塞进 `sorted_ids[cumsum_val..]`，pad 位保留 `SENTINEL = num_batches * max_tokens_per_batch`。

docstring 里给过例子（`num_batches=5, max_tokens_per_batch=8, block_size=4`），可直接对照 line 106-168。

---

## 8. `moe_sum_kernel`（line 350-363）

最简单的一个，但模板参数很关键：

```cpp
template <typename scalar_t, int TOPK>          // TOPK 是编译期常量！
__global__ void moe_sum_kernel(
    scalar_t* out,        // [num_tokens, d]
    const scalar_t* input,// [num_tokens, topk, d]
    const int d) {
    const int64_t token_idx = blockIdx.x;       // 一个 block 处理一个 token
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
        scalar_t x = 0.0;
#pragma unroll                                   // 因为 TOPK 是 constexpr，可以完全展开
        for (int k = 0; k < TOPK; ++k)
            x += VLLM_LDG(&input[token_idx*TOPK*d + k*d + idx]);
        out[token_idx*d + idx] = x;
    }
}
```

### Host dispatch：
```cpp
dim3 grid(num_tokens);
dim3 block(min(hidden_size, 1024));
switch (topk) {
    case 2: moe_sum_kernel<scalar_t, 2>...
    case 3: moe_sum_kernel<scalar_t, 3>...
    case 4: moe_sum_kernel<scalar_t, 4>...
    default: at::sum_out(output, input, 1);      // ← DeepSeek topk=6 走这里
}
```

### 为什么只特化 2/3/4？
- 这些是 Mixtral / Qwen-MoE 等主流模型的常见 topk；
- `TOPK` 是 constexpr，`#pragma unroll` 后 inner-loop 完全展开，ld.global 可被合并/重排，性能显著高于 ATen 的通用 sum；
- topk 很大时（如 DeepSeek 6、DeepSeek-V3 topk=8），展开反而寄存器压力过大，直接回退到 `at::sum_out`。

### 数值直觉（DeepSeek 假设 `num_tokens=2048, hidden=2048, topk=6`）：
- 实际走 `at::sum_out`；
- 若改成 topk=4：grid=2048，block=1024，每个 block 处理 `d=2048` 个 float，每线程做 2 次 `d` 的遍历，每次加 4 项 → 8 次读、1 次写。

---

## 9. 一步一图：以 `num_tokens=4, topk=3, num_experts=4, block_size=4` 走一遍

（同 `moe_align_block_size.py` docstring 的例子）

```
topk_ids flatten:
 idx:    0  1  2  3  4  5  6  7  8  9 10 11
 expert: 2  3  4  1  2  4  1  3  4  1  2  3
numel = 12   (也是 sorted_token_ids 的哨兵值)

── Kernel 1 (align)：
 shared_counts(统计):  e1=3, e2=3, e3=3, e4=3
 pad 到 block_size=4:  e1→4, e2→4, e3→4, e4→4
 BlockScan 独占前缀和:
     cumsum = [0, 4, 8, 12, 16]
     total_tokens_post_pad = 16
 写 expert_ids:
     blocks [0..3] 对应 expert [1, 2, 3, 4]
 blockIdx.x==1 同时在写 sorted_token_ids = [12, 12, ..., 12] (16 份)

── Kernel 2 (sort)：
 用原子加把每个索引放进对应 expert 的桶里
   i=3 (e1) → cumsum[1]++ → slot=0 → sorted[0]=3
   i=6 (e1) → slot=1       → sorted[1]=6
   i=9 (e1) → slot=2       → sorted[2]=9
   (slot 3 保持哨兵 12)
   i=0 (e2) → slot=4       → sorted[4]=0
   ... etc.
 最终 sorted_token_ids:
   [3, 6, 9,12,  0, 4,10,12,  1, 7,11,12,  2, 5, 8,12]
        expert1     expert2     expert3     expert4
 expert_ids = [1, 2, 3, 4]
 num_tokens_post_pad = 16
```

这三样东西正是下游 Triton GEMM `fused_moe_kernel` 要消费的东西。

---

## 10. 一张"cheat sheet"：模板 / 运行时参数一览

| 参数 | 来源 | 典型值 (DeepSeek-MoE-16B) | 作用 |
|---|---|---|---|
| `scalar_t` (模板) | `topk_ids.dtype` | `int32_t` | topk_ids 的元素类型（也可 uint32） |
| `TOPK` (模板, `moe_sum`) | Host `switch(topk)` | 2/3/4 特化；其他回退 ATen | 编译期 unroll 累加 |
| `fill_threads` (模板, small-batch) | 固定 `256` | 256 | 初始化 sorted_ids 的线程数 |
| `num_experts` | FusedMoE 层 | 64 | 路由 expert 数 |
| `padded_num_experts` | `ceil(num_experts, 32)*32` | 64 | 对齐到 WARP_SIZE |
| `experts_per_warp` | `WARP_SIZE` | 32 | shared_counts 的 warp 切分 |
| `block_size` | Triton config `BLOCK_SIZE_M` | 32 / 64 / 128 | GEMM 的 M 方向 block |
| `numel` | `topk_ids.numel()` | `num_tokens * topk` | 要处理的 (token,slot) 数 |
| `max_num_tokens_padded` | `numel + num_experts*(block-1)` | e.g. 16320 | sorted_token_ids 预分配大小 |
| `max_num_m_blocks` | `ceil(max_num_tokens_padded, block)` | 255 | expert_ids 大小 |
| `expert_map` | EP shard 映射 | len=64，元素 ∈ {-1, 0..local-1} | EP 过滤不归本 rank 的 expert |
| `has_expert_map` | `expert_map is not None` | `True` when `-tp=2 -dp=2 --enable-expert-parallel` | 控制是否跳过非本 rank expert |
| `ignore_invalid_experts` | Python 端参数 | `True` in fused_experts | 决定 expert_ids 中 `-1` 是否先行过滤 |

---

## 11. 给初学者的 3 条"为什么这样写"

1. **"shared_counts 按 warp 切 32 段"**：降低 shared-memory 的 bank conflict 和 atomic 冲突——同一时刻不同 warp 大概率写不同 bank。
2. **"先 ceil 到 block_size 再做 cumsum"**：保证每个 expert 的起点是 block 边界，`expert_ids[i] = sorted_token_ids[i*block_size : (i+1)*block_size]` 内的所有 token 一定归同一个 expert。
3. **"两段 kernel 代替一段"**：第一段只需要 1024 线程做 scan（不能更多，BlockScan 上限），第二段需要 `O(numel)` 并行度分散 atomic；把两件事绑到一起会导致资源浪费，分开反而总时间短。

---

## 12. 参考文件

- 入口：`examples/offline_inference/data_parallel.py`
- Python 包装：`vllm/model_executor/layers/fused_moe/moe_align_block_size.py`
- Python → C++：`vllm/_custom_ops.py`（`moe_align_block_size`, `batched_moe_align_block_size`, `moe_sum`）
- Kernel 源码：`csrc/moe/moe_align_sum_kernels.cu`
- 上游使用者：`vllm/model_executor/layers/fused_moe/fused_moe.py::fused_experts`
