/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc2/cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh
 * Copyright (c) 2026, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

namespace vllm {
namespace moe {
namespace reduce_topk {
namespace cg = cooperative_groups;
static constexpr int kWARP_SIZE = 32;

template <typename T_>
struct TopKRedType {
  using T = T_;
  static_assert(
      std::is_same_v<T, float> || std::is_same_v<T, half> ||
          std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, int>,
      "Top K reduction only implemented for int, float, float16 and bfloat16");

  using TypeCmp = std::conditional_t<sizeof(T) == 4, uint64_t, uint32_t>;
  using IdxT = std::conditional_t<sizeof(T) == 4, int32_t, int16_t>;

  static constexpr int kMoveBits = (sizeof(T) == 4) ? 32 : 16;
  static constexpr int kMaxIdx = 65535;
  TypeCmp compValIdx;

  static __host__ __device__ inline TypeCmp makeCmpVal(T val, int32_t idx = 0) {
    auto valueBits = cub::Traits<T>::TwiddleIn(
        reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(val));
    TypeCmp compactTmp = valueBits;
    compactTmp = (compactTmp << kMoveBits) | (0xFFFF & (kMaxIdx - idx));
    // Use 65535 minus idx to give higher priority to elements with smaller
    // indices.
    return compactTmp;
  }

  static __host__ __device__ void unpack(T& value, int32_t& index,
                                         TypeCmp cmp) {
    // Since “65535-idx” is always smaller than 65536 and positive, we can
    // directly use it as the lower 16 bits
    index = kMaxIdx - static_cast<int32_t>((cmp & 0xFFFF));

    auto compactTmp = cmp >> kMoveBits;
    auto valueBits = cub::Traits<T>::TwiddleOut(
        reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(compactTmp));
    value = reinterpret_cast<T&>(valueBits);
  }

  __host__ __device__ TopKRedType() = default;

  __host__ __device__ TopKRedType(T val, int32_t idx)
      : compValIdx(makeCmpVal(val, idx)) {}

  __host__ __device__ operator TypeCmp() const noexcept { return compValIdx; }

  __device__ inline TypeCmp reduce(
      cg::thread_block_tile<kWARP_SIZE> const& warp) {
    return cg::reduce(warp, compValIdx, cg::greater<TypeCmp>{});
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int K_, bool Enable_>
struct TopKIdx {
  // by default, empty
};

template <int K_>
struct TopKIdx<K_, true> {
  static constexpr int K = K_;
  int32_t val[K];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#define TOPK_SWAP(I, J)                                         \
  {                                                             \
    auto pairMin = min(topK[I].compValIdx, topK[J].compValIdx); \
    auto pairMax = max(topK[I].compValIdx, topK[J].compValIdx); \
    topK[I].compValIdx = pairMax;                               \
    topK[J].compValIdx = pairMin;                               \
  }

template <int N, typename RedType>
struct Sort;

template <typename RedType>
struct Sort<1, RedType> {
  static __device__ void run(RedType* topK) {}
};

template <typename RedType>
struct Sort<2, RedType> {
  static __device__ void run(RedType* topK) { TOPK_SWAP(0, 1); }
};

template <typename RedType>
struct Sort<3, RedType> {
  static __device__ void run(RedType* topK) {
    TOPK_SWAP(0, 1);
    TOPK_SWAP(1, 2);
    TOPK_SWAP(0, 1);
  }
};

template <typename RedType>
struct Sort<4, RedType> {
  static __device__ void run(RedType* topK) {
    TOPK_SWAP(0, 2);
    TOPK_SWAP(1, 3);
    TOPK_SWAP(0, 1);
    TOPK_SWAP(2, 3);
    TOPK_SWAP(1, 2);
  }
};

template <int K, typename Type>
__forceinline__ __device__ void reduceTopK(
    cg::thread_block_tile<kWARP_SIZE> const& warp, Type (&out)[K],
    int32_t (&outIdx)[K], Type value, int32_t idx, Type const minValue,
    int actualK = K) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K < kWARP_SIZE, "Top K must have K < kWARP_SIZE");
  using RedType = TopKRedType<Type>;
  RedType topK{value, idx};
  typename RedType::TypeCmp packedMax{};
#pragma unroll
  for (int kk = 0; kk < actualK; ++kk) {
    topK =
        kk > 0 && packedMax == topK.compValIdx ? RedType{minValue, idx} : topK;
    // get the next largest value
    packedMax = topK.reduce(warp);
    RedType::unpack(out[kk], outIdx[kk], packedMax);
  }
};

/*
数组输入版本的warp 级 topk-K 规约函数
每个lane 持有N 个候选值（value 和 idx），最终从整个warp 的N * 32个元素中选出top-k 
*/
template <int K, typename Type, int N, bool IsSorted = false>
__device__ void reduceTopKFunc(cg::thread_block_tile<kWARP_SIZE> const& warp,
                               Type (&out)[K],                // 输出: K个最大值及其索引
                               int32_t (&outIdx)[K],
                               Type (&value)[N],              // 输入：每lane 的N 个候选 
                               int32_t (&idx)[N],
                               Type minValue,                 // 哨兵值（通常 -INF）
                               int actualK = K                // 实际需要的K (≤K)
  ) {     
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K < kWARP_SIZE, "Top K must have K < kWARP_SIZE");
  static_assert(N > 0, "Top K must have N > 0");
  static_assert(N < 5,
                "Only support candidates number less than or equal to 128");
  using RedType = TopKRedType<Type>;
  RedType topK[N];
#pragma unroll
  // 打包输入
  for (int nn = 0; nn < N; ++nn) {
    topK[nn] = RedType{value[nn], idx[nn]};   // 每队 (value, idx) 打包成一个 RedType 结构，内部通过 makeCmpVal 转换成一个可比较的 compValIdx
  }
  // Sort<N> 寄存器内排序网络
  if constexpr (!IsSorted) {
    Sort<N, RedType>::run(topK);  // 本地按打包值降序排列，即topK[0] 是本lane 当前的最大值
  }
  typename RedType::TypeCmp packedMax{};
#pragma unroll
  // 外层循环kk, 连续抽取 K 个最大值
  for (int kk = 0; kk < actualK; ++kk) {
    // 下一轮reduce 前(kk 必须大于1)， 必须把上一轮选出的最大值，lane 的topk[0] 拿掉，否则会被重复选中
    bool update = kk > 0 && packedMax == topK[0].compValIdx;
#pragma unroll
    // 内层循环nn: 移除已选出的元素（左移）
    for (int nn = 0; nn < N; ++nn) {
      /* 
      非冠军 lane(update = false) topk[nn] 保持不变
      冠军lane(update = True)，topk[0] <- topK[1], topk[1] <- topK[2], ...topk[N-1]= minvalue d
      */
      topK[nn] = update && nn == N - 1 ? RedType{minValue, idx[nn]}
                 : update              ? topK[nn + 1]
                                       : topK[nn];
    }
    // get the next largest value
    // topK[0].reduce 实际将会调用 cg::reduce(warp, compValIdx, cg::greater<TypeCmp>{}); 
    packedMax = topK[0].reduce(warp);
    RedType::unpack(out[kk], outIdx[kk], packedMax);
  }
};

// reduceTopK 的作用: 在一个warp内，把每个线程本地N 个候选（value 和 idx）进行规约，选出全 warp 范围内的最大 actualK 个值，并写入 out/outIdx
template <int K, typename Type, int N>
__forceinline__ __device__ void reduceTopK(
    cg::thread_block_tile<kWARP_SIZE> const& warp,  // 一个cooperative groups 的 warp tile， 通常大小即为 KWARP_SIZE, 用于在一个warp 内进行并行规约、
    Type (&out)[K],                                 // 输出数组，长度为K
    int32_t (&outIdx)[K],                           // 输出的 top K 的值和对应的索引 
    Type (&value)[N],                               // 输入数组，长度为N，包含了需要进行 top K 计算的候选值
    int32_t (&idx)[N],                      // 输入数组，长度为N，包含了对应候选值的索引    
    Type const minValue,               // 最小值，用于初始化 top K 的比较，通常设置为一个非常小的数，以确保任何候选值都能被正确比较和更新 
    int actualK = K                     // 实际需要输出的 Topk 数量，默认为K，可以小于K以输出更少的结果
  ) 
  {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K < kWARP_SIZE, "Top K must have K < kWARP_SIZE");
  static_assert(N > 0, "Top K must have N > 0");
  static_assert(
      N <= 16,
      "Only support candidates number less than or equal to 16*32=512");
  static_assert(N <= 4 || N % 4 == 0,
                "Only support candidates number is a multiple of 4*32=128 or "
                "less than or equal to 4");
  using RedType = TopKRedType<Type>;

  if constexpr (N <= 4) {
    reduceTopKFunc<K, Type, N>(warp, out, outIdx, value, idx, minValue,
                               actualK);
  } else {
    /*
    每个 lane 的N 个候选被拆分成多个组，每个组4个。
    N =16 时，每个 lane 有 16 个候选，拆成 4 轮，每轮处理 4 个。
    */ 
    constexpr int numLoops = N / 4; 
    constexpr int numResults = (numLoops * K - 1) / kWARP_SIZE + 1;

    Type topKBufferValue[numResults];
    int32_t topKBufferIdx[numResults];
    int32_t laneIdx = threadIdx.x % kWARP_SIZE;

    for (int ii = 0; ii < numResults; ++ii) {
      topKBufferValue[ii] = minValue;
      topKBufferIdx[ii] = ii * kWARP_SIZE - 1;
    }
    for (int loop = 0; loop < numLoops; ++loop) {
      int start = loop * 4;
      Type topKValue[K];
      int32_t topKIdx[K];
      Type inValue[4];
      int32_t inIdx[4];
      for (int i = 0; i < 4; ++i) {
        inValue[i] = value[start + i];
        inIdx[i] = idx[start + i];
      }
      reduceTopKFunc<K, Type, 4>(warp, topKValue, topKIdx, inValue, inIdx,
                                 minValue, actualK);
      int inOffset = laneIdx % K;
      if (laneIdx >= loop * K && laneIdx < (loop + 1) * K) {
        topKBufferValue[0] = topKValue[inOffset];
        topKBufferIdx[0] = topKIdx[inOffset];
      }
      if (loop == numLoops - 1 && (laneIdx < (numLoops * K - kWARP_SIZE))) {
        topKBufferValue[1] = topKValue[inOffset];
        topKBufferIdx[1] = topKIdx[inOffset];
      }
    }

    reduceTopKFunc<K, Type, numResults>(warp, out, outIdx, topKBufferValue,
                                        topKBufferIdx, minValue, actualK);
  }
};

#undef TOPK_SWAP

}  // namespace reduce_topk
}  // namespace moe
}  // namespace vllm
