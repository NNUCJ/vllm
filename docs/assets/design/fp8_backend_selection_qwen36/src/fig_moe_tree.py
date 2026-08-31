from fig_common import *

fig, ax = new_fig(17, 26.2)
title(ax, "MoE FP8 的后端选择：select_fp8_moe_backend",
      "和 dense 完全是两套代码；oracle/fp8.py:271")

y = 91.5
h0 = box(ax, 12, 0, 76, None, top=y, title=None, lines=[
    "Fp8MoEMethod.__init__ -> select_fp8_moe_backend()      quantization/fp8.py:527 -> oracle/fp8.py:271",
], color="gray", ls=9.4, align="center", ts=12)

y1 = h0[1] - 3.4
s0 = box(ax, 3, 0, 94, None, top=y1,
         title="S0  _get_priority_backends：只重排，不删除      oracle/fp8.py:69", lines=[
    "默认顺序 _AVAILABLE_BACKENDS  :80",
    "  AITER  ->  FLASHINFER_TRTLLM  ->  FLASHINFER_CUTLASS  ->  DEEPGEMM  ->  VLLM_CUTLASS  ->  TRITON",
    "  ->  MARLIN  ->  HUMMING  ->  BATCHED_DEEPGEMM  ->  BATCHED_VLLM_CUTLASS  ->  BATCHED_TRITON  ->  XPU  ->  CPU  ->  HPC",
    "",
    "四条重排分支（按代码顺序执行，后面的会盖住前面的）：",
    "  :103  SM100(family) + DeepEP v2 + block-fp8       -> FLASHINFER_TRTLLM 提到最前",
    "  :113  SM90(严格) + block-fp8 + ep_size > 1        -> FLASHINFER_CUTLASS 提到最前",
    "  :113  SM90(严格) + block-fp8 + ep_size <= 1       -> TRITON 提到最前   <- Hopper 单机 TP 落这里",
    "  :124  XPU -> XPU 提前     :129  CPU -> CPU 提前",
], color="orange", ls=9.0, align="left", ts=12)
arrow(ax, (50, h0[1]), (50, y1 + 0.3))

y2 = s0[1] - 3.4
o0 = box(ax, 3, 0, 94, None, top=y2,
         title="S2-S6  四个硬覆盖开关，先到先得；命中即 _return_or_raise（不支持直接抛错，不回退）", lines=[
    "  S2  --moe-backend != auto              :330   map_fp8_backend；batched 格式下 DEEPGEMM->BATCHED_DEEPGEMM 等",
    "  S3  VLLM_USE_DEEP_GEMM / VLLM_MOE_USE_DEEP_GEMM  :359",
    "        注意判据是 envs.is_set()——必须显式设过才生效",
    "        设为真 -> 强制 DEEPGEMM ；设为假 -> 从候选表里 remove(DEEPGEMM) 和 remove(BATCHED_DEEPGEMM)",
    "  S4  VLLM_TEST_FORCE_FP8_MARLIN         :374   强制 MARLIN",
    "  S5  VLLM_ROCM_USE_AITER(_MOE)          :381   同 S3 的双向语义",
    "  S6  allow_vllm_cutlass=False           :390   remove(VLLM_CUTLASS) + remove(BATCHED_VLLM_CUTLASS)",
    "        Fp8MoEMethod 传的就是 False（quantization/fp8.py:531）-> vLLM 自带的 CUTLASS MoE 默认不参选",
], color="purple", ls=9.0, align="left", ts=12)
arrow(ax, (50, s0[1]), (50, y2 + 0.3))

y3 = o0[1] - 3.4
m0 = box(ax, 3, 0, 94, None, top=y3,
         title="S7  主循环：按候选表顺序，每个 backend 取一组 experts 类，逐个试 is_supported_config()   :395", lines=[
    "  is_supported_config 的 11 道检查，短路 if/elif 链   modular_kernel.py:536",
    "   1 _supports_current_device        2 is_act_and_mul / _supports_no_act_and_mul   3 _supports_activation",
    "   4 _supports_quant_scheme(weight_key, activation_key)                            5 _supports_parallel_config",
    "   6 _supports_routing_method        7 _supports_router_logits_dtype               8 _supports_shape",
    "   9 activation_format 必须与 prepare_finalize 一致   10 batch-invariant   11 LoRA",
    "",
    "  一个 backend 可能对应多个类，按序试：",
    "    FLASHINFER_TRTLLM -> [TrtLlmFp8ExpertsMonolithic, TrtLlmFp8ExpertsModular]     :145",
    "    DEEPGEMM          -> [TritonOrDeepGemmExperts]                                 :159",
    "    HUMMING           -> [BatchedHummingGrouped, HummingGrouped, HummingIndexed]   :175",
    "  全部落选 -> CUDA/ROCm 上 raise NotImplementedError   :414",
], color="teal", ls=9.0, align="left", ts=12)
arrow(ax, (50, o0[1]), (50, y3 + 0.3))

y4 = m0[1] - 3.4
r0 = box(ax, 3, 0, 94, None, top=y4,
         title="运行期还有第二次判定（只有 Fallback 系列有）", lines=[
    "  TritonOrDeepGemmExperts._select_experts_impl        triton_deep_gemm_moe.py:83",
    "      is_deep_gemm_e8m0_used() or _valid_deep_gemm(...)  ->  DeepGemmExperts，否则 TritonExperts",
    "      注意前半句默认为真（Hopper/Blackwell + 装了 deep_gemm），会把后面整串形状检查短路掉",
    "      _valid_deep_gemm 的五条：has_deep_gemm / M>=128 且 N,K 对齐 128 / N<=512 直接否 / 权重必须 e4m3 / 必须 contiguous",
    "  TritonOrCutlassExperts._select_experts_impl         triton_cutlass_moe.py:75",
    "      SM100 且 M<=8  ->  TritonExperts，否则 CutlassExpertsFp8",
], color="blue", ls=9.0, align="left", ts=12)
arrow(ax, (50, m0[1]), (50, y4 + 0.3))

y5 = r0[1] - 3.4
c0 = box(ax, 3, 0, 94, None, top=y5,
         title="默认配置（单机 TP、无 EP、--moe-backend auto、不设任何 env）下的实际结果", lines=[
    "  Hopper SM90 + block-fp8   ->  TRITON + TritonExperts",
    "      :113 那条重排就是为此显式加的。即使装了 deep_gemm 也不会选 DeepGEMM，",
    "      必须显式 --moe-backend deep_gemm 或显式 export VLLM_USE_DEEP_GEMM=1。",
    "",
    "  Blackwell SM100 + block-fp8  ->  FLASHINFER_TRTLLM + TrtLlmFp8ExpertsMonolithic（走 Monolithic 路！）",
    "      没装 flashinfer 或路由方法不在白名单 -> 顺延；FLASHINFER_CUTLASS 的 block-fp8 限死 SM90 也过不了",
    "      -> 最终落到 DEEPGEMM + TritonOrDeepGemmExperts",
    "",
    "  SM120（RTX 50 系）           ->  FI-TRTLLM 只认 family 100，够不着；落到 DEEPGEMM",
], color="green", ls=9.0, align="left", ts=12)
arrow(ax, (50, r0[1]), (50, y5 + 0.3))

import sys
print("bottom =", c0[1], file=sys.stderr)
save(fig, "moe_backend_tree.png")
