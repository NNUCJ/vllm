# dsv3_router_gemm_bf16_out.cu 算子分析报告

本文分析 `csrc/moe/dsv3_router_gemm_bf16_out.cu` 中的 DeepSeek V3 router GEMM bf16 输出算子。该算子用于小 batch 的 MoE router 线性层，计算：

```text
output = mat_a @ mat_b.T

mat_a:  [num_tokens, hidden_dim]  bf16
mat_b:  [num_experts, hidden_dim] bf16
output: [num_tokens, num_experts] bf16
```

在当前实现中，入口 `dsv3_router_gemm_entry.cu` 约束为：

```text
num_tokens: 1..16
hidden_dim: 7168
num_experts: 256 或 384
dtype: mat_a/mat_b 为 bf16
output: bf16 或 fp32
GPU: SM90 到 SM103
```

`dsv3_router_gemm_bf16_out.cu` 是 `output.dtype == torch.bfloat16` 时使用的版本。若输出是 fp32，则走相邻的 `dsv3_router_gemm_float_out.cu`。

## 一、真实 DeepSeek V3 场景

DeepSeek V3 的 router 层本质是一个小 M、大 K、中等 N 的 GEMM：

```text
M = num_tokens <= 16
K = hidden_dim = 7168
N = num_experts = 256
```

计算公式为：

```text
output[m, n] = sum_{k=0}^{7167} mat_a[m, k] * mat_b[n, k]
```

其中：

- `mat_a[m, :]` 是当前 token 的 hidden state。
- `mat_b[n, :]` 是第 `n` 个 expert 的 router 权重。
- `output[m, n]` 是 token `m` 对 expert `n` 的 router logit。

在 vLLM Python 层，`gate_linear.py` 中当满足以下条件时会优先调用该 specialized kernel：

```text
input_size == 7168
output_size in {256, 384}
x.shape[0] <= 16
运行在 Hopper/Blackwell 等支持平台
```

这说明该 kernel 不是通用 GEMM，而是针对 DeepSeek V3/Kimi-K2 这类 router 形状的小 batch 低延迟路径。

## 二、整体并行设计

### 1. gridDim 设计

launch 处：

```cpp
config.gridDim = kNumExperts;
```

也就是说：

```text
gridDim.x = num_experts
```

每个 CUDA block 负责一个 expert，也就是一个输出列 `n_idx`：

```cpp
int const n_idx = blockIdx.x;
```

对于 DeepSeek V3：

```text
num_experts = 256
gridDim.x = 256
```

所以一次 kernel launch 有 256 个 block。每个 block 计算该 expert 对所有 token 的输出：

```text
output[0, n_idx]
output[1, n_idx]
...
output[num_tokens-1, n_idx]
```

如果是 Kimi-K2 路径：

```text
num_experts = 384
gridDim.x = 384
```

### 2. blockDim 设计

launch 处：

```cpp
constexpr int kBlockSize = 128;
config.blockDim = kBlockSize;
```

也就是说：

```text
blockDim.x = 128
```

一个 block 内有 128 个线程，也就是 4 个 warp：

```cpp
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kBlockSize / kWarpSize; // 128 / 32 = 4
```

每个 block 内 128 个线程沿 hidden 维 `K=7168` 做并行切分。每个线程一次读取 8 个 bf16 元素：

```cpp
constexpr int VPT = 16 / sizeof(T); // T=bf16, sizeof(T)=2, VPT=8
```

因此一个 block 在一轮 K 迭代中处理：

```text
128 threads * 8 bf16/thread = 1024 个 K 元素
```

hidden_dim 为 7168：

```text
7168 / 1024 = 7
```

所以每个 block 做 7 轮 K 迭代，刚好覆盖完整 hidden 维。

### 3. 每个线程负责哪些 K

代码中：

```cpp
k_base = ki * k_elems_per_k_iteration + tid * VPT;
```

实际代入：

```text
k_base = ki * 1024 + tid * 8
```

对于线程 `tid=5`：

```text
ki=0: k = 40..47
ki=1: k = 1064..1071
ki=2: k = 2088..2095
ki=3: k = 3112..3119
ki=4: k = 4136..4143
ki=5: k = 5160..5167
ki=6: k = 6184..6191
```

因此每个线程总共处理：

```text
7 iterations * 8 elements = 56 个 hidden 元素
```

每个线程会对每个 token 维护一个局部累加器：

```cpp
float acc[kNumTokens] = {};
```

如果 `num_tokens=16`，每个线程有 16 个 float 累加器。`acc[m]` 表示该线程负责的 56 个 K 元素对 `output[m, n_idx]` 的局部点积贡献。

## 三、文件逐段逐行解释

### 1. 头文件和依赖

```cpp
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "dsv3_router_gemm_utils.h"
```

- `ATen/ATen.h` 和 `ATen/cuda/CUDAContext.h` 提供 PyTorch/CUDA 侧的上下文能力。
- `cuda_bf16.h` 提供 `__nv_bfloat16`、`__bfloat162float`、`__float2bfloat16`。
- `cuda_runtime.h` 提供 CUDA runtime API。
- `dsv3_router_gemm_utils.h` 提供 `getEnvEnablePDL()`，用于 PDL 相关 launch 属性。

### 2. 自定义 fma 函数

```cpp
__device__ __forceinline__ void fma(float2& d, float2 const& a, float2 const& b,
                                    float2 const& c) {
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
               : "=l"(reinterpret_cast<uint64_t&>(d))
               : "l"(reinterpret_cast<uint64_t const&>(a)),
                 "l"(reinterpret_cast<uint64_t const&>(b)),
                 "l"(reinterpret_cast<uint64_t const&>(c)));
}
```

这是一个用 PTX 写的 `float2` FMA 辅助函数。它使用 `fma.rn.f32x2` 对两个 fp32 lane 做 fused multiply-add。

不过在当前 `router_gemm_kernel_bf16_output` 主体中，这个 `fma` 函数没有被调用。实际乘加使用的是：

```cpp
acc[m_idx] += a * b;
```

所以这段更像是从同源实现保留下来的工具函数，当前 bf16 输出 kernel 并未使用它。

### 3. uint4 到 8 个 float 的转换

```cpp
template <int VPT>
__device__ __forceinline__ void bf16_uint4_to_float8(uint4 const& vec,
                                                     float* dst) {
  __nv_bfloat16* bf16_ptr =
      reinterpret_cast<__nv_bfloat16*>(const_cast<uint4*>(&vec));
```

`uint4` 是 128 bit，即 16 字节。bf16 是 2 字节，所以一个 `uint4` 正好装下 8 个 bf16。

这里并不是把 `uint4` 当整数运算，而是把它作为“16 字节原始数据容器”。`reinterpret_cast` 将这 16 字节重新解释为 `__nv_bfloat16[8]`。

```cpp
#pragma unroll
  for (int i = 0; i < VPT; i++) {
    dst[i] = __bfloat162float(bf16_ptr[i]);
  }
}
```

`VPT=8` 时，这个循环把 8 个 bf16 转成 8 个 float。这样做的原因是：输入和权重用 bf16 存储以节省带宽，但累加用 fp32，以减少点积误差。

### 4. kernel 模板参数

```cpp
template <typename T, int kBlockSize, int VPT, int kNumTokens, int kNumExperts,
          int kHiddenDim>
__global__ __launch_bounds__(128, 1) void router_gemm_kernel_bf16_output(
    __nv_bfloat16* out, T const* mat_a, T const* mat_b) {
```

模板参数含义：

```text
T: 输入和权重类型，实际为 __nv_bfloat16
kBlockSize: block 内线程数，实际为 128
VPT: values per thread，实际为 8
kNumTokens: token 数，编译期常量，1..16
kNumExperts: expert 数，256 或 384
kHiddenDim: hidden 维，7168
```

`__launch_bounds__(128, 1)` 告诉编译器该 kernel 每个 block 最多 128 个线程，并且期望至少 1 个 block resident。这里有助于编译器做寄存器/occupancy 权衡。

输出指针类型是 `__nv_bfloat16*`，说明该文件对应 bf16 输出版本。

### 5. block 和 thread 索引

```cpp
  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;
```

- `n_idx` 是当前 block 负责的 expert 下标。
- `tid` 是当前线程在 block 内的线程号，范围 `[0, 127]`。

真实 DeepSeek V3 中：

```text
n_idx: 0..255
tid:   0..127
```

### 6. warp 常量和 K 迭代常量

```cpp
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
```

实际为：

```text
kNumWarps = 128 / 32 = 4
```

```cpp
  constexpr int k_elems_per_k_iteration = VPT * kBlockSize;
```

实际为：

```text
k_elems_per_k_iteration = 8 * 128 = 1024
```

```cpp
  constexpr int k_iterations =
      kHiddenDim / k_elems_per_k_iteration;
```

实际为：

```text
k_iterations = 7168 / 1024 = 7
```

这个设计利用了 DeepSeek V3 的 `hidden_dim=7168` 能被 `128*8=1024` 整除的特点，避免了边界判断。

### 7. 每线程局部累加器

```cpp
  float acc[kNumTokens] = {};
```

每个线程有自己的 `acc` 数组。`kNumTokens` 是编译期常量，因此编译器可以展开 token 循环并把这些累加器尽量放入寄存器。

若 `num_tokens=1`：

```text
每个线程 1 个 fp32 acc
```

若 `num_tokens=16`：

```text
每个线程 16 个 fp32 acc
```

`acc[m_idx]` 表示当前线程对 `output[m_idx, n_idx]` 的局部部分和。完整点积需要把 block 内 128 个线程的 `acc[m_idx]` 全部加起来。

### 8. shared memory 归约缓存

```cpp
  __shared__ float sm_reduction[kNumTokens][kNumWarps];
```

实际最多：

```text
kNumTokens = 16
kNumWarps = 4
shared memory = 16 * 4 * 4 bytes = 256 bytes
```

这块 shared memory 只保存每个 warp 的归约结果，不保存输入矩阵 tile。因此 shared memory 占用很小。

### 9. 定位当前 expert 的权重行

```cpp
  T const* b_col = mat_b + n_idx * kHiddenDim;
```

虽然注释说 “B matrix is in column-major order”，但从入口定义看，`mat_b` 形状是：

```text
mat_b: [num_experts, hidden_dim]
```

且计算是 `mat_a @ mat_b.T`。因此内存访问上，`mat_b[n_idx, :]` 是当前 expert 的一整行权重，长度 7168。

这里 `b_col` 实际指向：

```text
&mat_b[n_idx, 0]
```

后续 block 内所有线程都会从该 expert 的权重行中读取不同 K 片段。

### 10. 预计算每轮 k_base

```cpp
  int k_bases[k_iterations];
#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    k_bases[ki] = ki * k_elems_per_k_iteration + tid * VPT;
  }
```

实际为：

```text
k_bases[ki] = ki * 1024 + tid * 8
```

由于 `k_iterations=7` 是编译期常量，该循环会展开。预计算 `k_bases` 可以减少主计算循环里的重复整数计算。

### 11. Programmatic Dependent Launch 等待

```cpp
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
```

这是 SM90+ 的 programmatic dependent launch 相关控制指令。它配合 launch 侧的：

```cpp
cudaLaunchAttributeProgrammaticStreamSerialization
```

使用。`getEnvEnablePDL()` 读取环境变量 `TRTLLM_ENABLE_PDL` 决定是否允许该机制。直观理解是：在支持的架构上，让依赖 kernel 的调度序列更细粒度可控，面向低延迟 pipeline。

### 12. 主 K 循环

```cpp
  for (int ki = 0; ki < k_iterations; ki++) {
    int const k_base = k_bases[ki];
```

主循环一共 7 次。每一轮覆盖 1024 个 K 元素，每个线程负责其中连续 8 个。

### 13. 向量化读取 B

```cpp
    uint4 b_vec = *reinterpret_cast<uint4 const*>(b_col + k_base);
```

实际读取：

```text
mat_b[n_idx, k_base : k_base + 7]
```

因为 `uint4=16 bytes`，刚好等于 8 个 bf16，所以这是一条 128-bit vector load。

例如 `n_idx=10, tid=5, ki=2`：

```text
k_base = 2 * 1024 + 5 * 8 = 2088
读取 mat_b[10, 2088..2095]
```

这种设计的好处：

- 每个线程一次读 16 字节，访存指令更少。
- 相邻线程读取连续地址，整个 warp 的访问连续。
- `k_base` 总是 8 的倍数，16 字节对齐更自然。

### 14. B 从 bf16 转 float

```cpp
    float b_float[VPT];
    bf16_uint4_to_float8<VPT>(b_vec, b_float);
```

`b_float` 是当前线程读取到的 8 个权重值的 fp32 表示。后续它会被所有 token 复用。

这点很关键：一个 block 固定一个 expert，`b_float` 与 token 无关，所以在同一轮 `ki` 内，对 `m_idx=0..kNumTokens-1` 都使用同一个 `b_float`。

### 15. 遍历 token

```cpp
#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
```

`kNumTokens` 是编译期常量，范围 1..16。入口通过 `LoopUnroller` 选择具体模板实例，因此这里可以完全展开。

这也是该 kernel 限制 `num_tokens<=16` 的重要原因：token 循环展开后，每个线程需要 `kNumTokens` 个累加器和临时数组，token 太大会导致寄存器压力过高。

### 16. 向量化读取 A

```cpp
      uint4 a_vec = *reinterpret_cast<uint4 const*>(
          mat_a + (m_idx * kHiddenDim) + k_base);
```

实际读取：

```text
mat_a[m_idx, k_base : k_base + 7]
```

例如 `m_idx=3, tid=5, ki=2`：

```text
k_base = 2088
读取 mat_a[3, 2088..2095]
一维偏移 = 3 * 7168 + 2088
```

同样，这里通过 `uint4` 做 16 字节向量化读取。它要求 `mat_a` 在 hidden 维上连续存储，实际调用中 router 输入通常是 contiguous 的 `[num_tokens, 7168]`。

### 17. A 从 bf16 转 float

```cpp
      float a_float[VPT];
      bf16_uint4_to_float8<VPT>(a_vec, a_float);
```

当前 token 的 8 个 hidden 值被转换成 fp32，准备和 `b_float` 做乘加。

### 18. 每线程局部乘加

```cpp
#pragma unroll
      for (int k = 0; k < VPT; k++) {
        float a = a_float[k];
        float b = b_float[k];
        acc[m_idx] += a * b;
      }
```

实际为 8 次乘加：

```text
acc[m_idx] += mat_a[m_idx, k_base + 0] * mat_b[n_idx, k_base + 0]
...
acc[m_idx] += mat_a[m_idx, k_base + 7] * mat_b[n_idx, k_base + 7]
```

完成 7 轮后，每个线程的 `acc[m_idx]` 包含该线程负责的 56 个 K 元素的部分和。

以 `tid=5` 为例：

```text
acc[m_idx] =
  sum_{k in {40..47,1064..1071,...,6184..6191}}
    mat_a[m_idx,k] * mat_b[n_idx,k]
```

完整结果还需要跨 128 个线程归约。

### 19. warp 信息

```cpp
  int const warpSize = 32;
  int const warpId = tid / warpSize;
  int const laneId = tid % warpSize;
```

实际：

```text
warpId: 0..3
laneId: 0..31
```

这里 `warpSize` 重新定义为运行时 `int`，语义上与前面的 `kWarpSize=32` 一致。

### 20. 拷贝到 warp_result

```cpp
  float warp_result[kNumTokens];

#pragma unroll
  for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
    warp_result[m_idx] = acc[m_idx];
  }
```

`warp_result` 是归约阶段的临时寄存器数组。每个线程把自己的局部 `acc` 拷贝进去，随后用 shuffle 做 warp 内归约。

### 21. warp 内 butterfly 归约

```cpp
#pragma unroll
  for (int m = 0; m < kNumTokens; m++) {
    float sum = warp_result[m];

    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
```

这是 warp 内 32 个 lane 的规约。对每个 token `m`，一个 warp 内的 32 个线程各自有一个局部和，经过 5 次 `shfl_xor` 后，每个 lane 都得到该 warp 的总和。

归约范围：

```text
每个 warp 覆盖 32 threads * 8 elements/thread * 7 iterations = 1792 个 K 元素
```

4 个 warp 正好覆盖：

```text
4 * 1792 = 7168 个 K 元素
```

### 22. 每个 warp 写 shared memory

```cpp
    if (laneId == 0) {
      sm_reduction[m][warpId] = sum;
    }
  }
```

由于每个 lane 都有相同的 warp 归约结果，只需要 lane 0 写出即可。

写入布局：

```text
sm_reduction[m][0] = token m 在 warp 0 的部分和
sm_reduction[m][1] = token m 在 warp 1 的部分和
sm_reduction[m][2] = token m 在 warp 2 的部分和
sm_reduction[m][3] = token m 在 warp 3 的部分和
```

### 23. block 内同步

```cpp
  __syncthreads();
```

确保 4 个 warp 的 lane 0 都已经把结果写入 shared memory。否则 `tid==0` 读取时可能读到未写完的数据。

### 24. 跨 warp 最终归约

```cpp
  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      float final_sum = 0.0f;
```

最终只由 `tid==0` 完成。它对每个 token 收集 4 个 warp 的部分和。

```cpp
#pragma unroll
      for (int w = 0; w < kNumWarps; w++) {
        final_sum += sm_reduction[m][w];
      }
```

实际就是：

```text
final_sum =
  sm_reduction[m][0] +
  sm_reduction[m][1] +
  sm_reduction[m][2] +
  sm_reduction[m][3]
```

这就是完整的：

```text
sum_{k=0}^{7167} mat_a[m,k] * mat_b[n_idx,k]
```

### 25. 写回 bf16 输出

```cpp
      out[m * kNumExperts + n_idx] = __float2bfloat16(final_sum);
```

`out` 逻辑形状为 `[kNumTokens, kNumExperts]`，row-major。因此：

```text
out[m, n_idx] = out[m * kNumExperts + n_idx]
```

该 bf16 输出版本会把 fp32 累加结果转换为 bf16 再写出。

对于 DeepSeek V3：

```text
kNumExperts = 256
out[m, n_idx] 地址 = out + m * 256 + n_idx
```

### 26. Programmatic Dependent Launch 完成通知

```cpp
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
```

与前面的 `griddepcontrol.wait` 配套，用于 SM90+ 的 PDL 控制。当环境变量允许 PDL 时，依赖 kernel 可以更早被调度。

## 四、host launch 逻辑

```cpp
template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemmBf16Output(__nv_bfloat16* output, T const* mat_a,
                                T const* mat_b, cudaStream_t stream) {
```

这是 host 侧模板封装。`kNumTokens`、`kNumExperts`、`kHiddenDim` 都是编译期常量。

```cpp
  constexpr int VPT = 16 / sizeof(T);
```

实际：

```text
T=__nv_bfloat16
sizeof(T)=2
VPT=8
```

它保证每个线程每次加载 16 字节。

```cpp
  constexpr int kBlockSize = 128;
```

固定 block 线程数为 128。

```cpp
  cudaLaunchConfig_t config;
  config.gridDim = kNumExperts;
  config.blockDim = kBlockSize;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
```

launch 配置：

```text
gridDim = num_experts
blockDim = 128
dynamic shared memory = 0
stream = 当前 PyTorch CUDA stream
```

kernel 使用的是静态 shared memory：

```cpp
__shared__ float sm_reduction[kNumTokens][kNumWarps];
```

所以 dynamic shared memory 为 0。

```cpp
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;
```

设置 PDL 相关 launch attribute。`getEnvEnablePDL()` 只有在 SM90+ 且环境变量 `TRTLLM_ENABLE_PDL=1` 时返回 true。

```cpp
  cudaLaunchKernelEx(
      &config,
      router_gemm_kernel_bf16_output<T, kBlockSize, VPT, kNumTokens,
                                     kNumExperts, kHiddenDim>,
      output, mat_a, mat_b);
```

用 `cudaLaunchKernelEx` 启动具体模板实例。

## 五、模板实例化设计

文件底部显式实例化：

```cpp
template void invokeRouterGemmBf16Output<__nv_bfloat16, 1, 256, 7168>(...);
...
template void invokeRouterGemmBf16Output<__nv_bfloat16, 16, 256, 7168>(...);

template void invokeRouterGemmBf16Output<__nv_bfloat16, 1, 384, 7168>(...);
...
template void invokeRouterGemmBf16Output<__nv_bfloat16, 16, 384, 7168>(...);
```

也就是说它提前编译了：

```text
num_tokens = 1..16
num_experts = 256
hidden_dim = 7168
```

和：

```text
num_tokens = 1..16
num_experts = 384
hidden_dim = 7168
```

两组形状。

这样做的好处：

- `kNumTokens` 是编译期常量，token 循环可展开。
- `kNumExperts` 是编译期常量，输出地址计算可优化。
- `kHiddenDim=7168` 是编译期常量，K 迭代数固定为 7。
- 没有通用 GEMM 的动态 shape 分支和边界判断。

代价是编译产物更多，且只能覆盖预定义 shape。

## 六、入口调用链

Python 层：

```text
vllm/model_executor/layers/fused_moe/router/gate_linear.py
```

满足 specialized 条件且 `x.shape[0] <= 16` 时调用：

```python
ops.dsv3_router_gemm(
    hidden_states=x,
    router_weight=self.weight,
    output_dtype=self.out_dtype,
)
```

C++ 注册入口：

```text
csrc/moe/dsv3_router_gemm_entry.cu
```

入口检查：

```text
mat_a.dim() == 2
mat_b.dim() == 2
output.dim() == 2
mat_a.size(1) == mat_b.size(1)
hidden_dim == 7168
num_experts == 256 或 384
num_tokens in [1,16]
mat_a/mat_b dtype == bf16
output dtype == fp32 或 bf16
SM in [90,103]
```

如果输出 dtype 是 bf16：

```cpp
LoopUnroller<1, 16, kNumExperts, 7168>::unroll_bf16_output(...)
```

`LoopUnroller` 根据运行时 `num_tokens` 选择编译期模板：

```text
num_tokens=1  -> kNumTokens=1
num_tokens=8  -> kNumTokens=8
num_tokens=16 -> kNumTokens=16
```

最后进入：

```cpp
invokeRouterGemmBf16Output<__nv_bfloat16, kNumTokens, kNumExperts, 7168>
```

## 七、访存模式分析

### 1. B 权重读取

每个 block 对应一个 expert，因此每个 block 读取：

```text
mat_b[n_idx, 0..7167]
```

每个线程读取 56 个 bf16：

```text
56 * 2 bytes = 112 bytes/thread
```

每个 block 读取 B：

```text
128 threads * 112 bytes = 14336 bytes = 7168 bf16
```

正好是一行 expert 权重。

### 2. A 输入读取

每个 block 对每个 token 都会读取完整 hidden 向量：

```text
mat_a[m, 0..7167]
```

如果 `num_tokens=16`，每个 block 读取 A：

```text
16 * 7168 bf16 * 2 bytes = 229376 bytes
```

注意 A 会被所有 expert block 重复读取。例如 DeepSeek V3 256 experts 时，同一批 token 的 A 会被 256 个 block 重复读取。这是该设计为了低延迟和简单并行做出的取舍：不做跨 expert 的 A tile 复用，而是让每个 expert 独立 block 计算。

### 3. 输出写入

每个 block 写 `num_tokens` 个 bf16：

```text
out[0..num_tokens-1, n_idx]
```

对于 DeepSeek V3，整个 grid 写：

```text
num_tokens * 256 个 bf16
```

输出带宽不是瓶颈。

## 八、为什么不用普通 GEMM tile

这个问题的 shape 是：

```text
M <= 16
N = 256 或 384
K = 7168
```

普通 GEMM kernel 通常会设计二维 tile，例如 block 同时覆盖多个 M 和 N，并使用 shared memory 复用 A/B tile。但这里 M 很小，目标是 router logits 的低延迟，而不是大矩阵吞吐最大化。

当前设计选择：

```text
一个 block = 一个 expert
block 内 128 线程并行归约 K
一次算完该 expert 对所有 token 的输出
```

优点：

- 逻辑简单，launch 后没有复杂 tile 调度。
- K 维完全由 128 线程并行归约，单个 output 的延迟低。
- `num_tokens<=16` 时，token 维直接放在寄存器数组中处理。
- B 权重一轮读取后可被多个 token 复用。
- hidden_dim=7168 刚好能被 1024 整除，无需边界分支。

缺点：

- A 在不同 expert block 之间重复读取。
- 每个 block 只负责一个 expert，block 数等于 expert 数，N 很小时并行度受限。
- 该实现高度绑定 DeepSeek V3/Kimi-K2 的固定 shape，不适合作通用 GEMM。

## 九、blockDim=128 的设计原因

`blockDim=128` 结合 `VPT=8` 后，每轮处理 1024 个 K 元素。对 `K=7168` 来说正好 7 轮。

如果 blockDim 更小，例如 64：

```text
64 * 8 = 512
7168 / 512 = 14 轮
```

每个线程仍读更多轮，归约线程数少，单 output 的 K 并行度下降。

如果 blockDim 更大，例如 256：

```text
256 * 8 = 2048
7168 / 2048 = 3.5
```

不能整除，需要边界处理或改变 VPT。同时 256 线程会带来更多 warp 间归约开销，并可能降低 occupancy。

128 的好处：

- 4 个 warp，warp 间归约只需 4 个 partial sum。
- 每轮 1024 个元素，7168 正好 7 轮。
- 每线程总共 56 个 K 元素，工作量适中。
- shared memory 很小，主要压力在寄存器。

## 十、gridDim=num_experts 的设计原因

router GEMM 的 N 维是 expert 数。每个 expert 的输出互相独立，因此按 expert 切 block 是最直接的并行方式：

```text
block 0 -> expert 0
block 1 -> expert 1
...
block 255 -> expert 255
```

这样每个 block 都读取一行权重 `mat_b[n_idx, :]`，并计算该 expert 对所有 token 的 logits。

对于小 batch router，这种方式降低了调度和同步复杂度：

- block 之间没有通信。
- 每个 block 内只做 K 维归约。
- 输出地址自然是 `out[m * kNumExperts + n_idx]`。

## 十一、一次完整计算示例

假设 DeepSeek V3：

```text
num_tokens = 4
num_experts = 256
hidden_dim = 7168
output dtype = bf16
```

launch：

```text
gridDim.x = 256
blockDim.x = 128
```

看 `blockIdx.x=10`：

```text
该 block 负责 expert 10
计算 output[0,10], output[1,10], output[2,10], output[3,10]
```

看其中 `threadIdx.x=5`：

```text
该线程每轮读取 8 个 K
总共 7 轮，读取 56 个 K
```

它会维护：

```text
acc[0], acc[1], acc[2], acc[3]
```

每个 `acc[m]` 是该线程对 token `m` 和 expert 10 点积的局部贡献。

完成主循环后：

```text
128 个线程各自持有 acc[0..3]
```

然后：

```text
warp 内 shfl 归约 -> 每个 warp 得到 4 个 token 的 partial sum
lane 0 写 sm_reduction[m][warpId]
tid 0 汇总 4 个 warp
写 output[m,10]
```

最终 256 个 block 各自写一列 expert logits，得到完整 `[4,256]` 输出。

## 十二、数值精度路径

该 kernel 的数值路径是：

```text
bf16 input
bf16 weight
uint4 vector load
bf16 -> fp32
fp32 multiply
fp32 accumulate
fp32 final_sum
fp32 -> bf16 output
```

因此相比直接 bf16 累加，它保留了 fp32 accumulator 的精度；但最终输出是 bf16，会发生一次舍入。DeepSeek V3 routing 在部分量化路径中可能要求 fp32 router logits，此时应走 `float_out` 版本或上层选择 fp32 输出。

## 十三、适用边界和隐含假设

该 kernel 有几个重要边界：

```text
hidden_dim 必须是 7168
num_tokens 必须是 1..16
num_experts 必须是 256 或 384
输入权重必须是 bf16
GPU 必须是 SM90 到 SM103
```

实现中没有处理 K 尾部，因为：

```text
7168 % (128 * 8) == 0
```

向量化读取隐含要求 hidden 维连续布局，并且地址满足 16 字节读取的对齐需求。入口当前检查了 shape 和 dtype，但没有显式检查 stride；实际使用路径通常传入 contiguous 的 hidden states 和 router weight。

## 十四、总结

`dsv3_router_gemm_bf16_out.cu` 是一个面向 DeepSeek V3 router 形状的专用低延迟 GEMM：

```text
M <= 16, K = 7168, N = 256/384
```

核心策略是：

```text
一个 block 负责一个 expert
128 个线程并行切分 K
每线程每轮用 uint4 读取 8 个 bf16
7 轮覆盖完整 7168 hidden
每线程用 fp32 acc[kNumTokens] 累加
warp shuffle + shared memory 完成 block 归约
最终写出 bf16 logits
```

这个设计牺牲了一部分 A 矩阵跨 expert 的复用，但换来了固定 shape 下非常直接的并行归约路径、少量 shared memory、无 K 边界分支和较低的调度复杂度。它适合 decode 或小 batch router logits 计算，不适合作为通用大矩阵 GEMM。
