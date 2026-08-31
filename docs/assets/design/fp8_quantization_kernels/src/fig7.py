from fig_common import *

fig, ax = new_fig(17, 21.8)
title(ax, "DeepGEMM 与 csrc FP8 算子的分工",
      "同一批量化 kernel，两条 GEMM 路线：量化在 csrc，GEMM 分叉")

a0 = box(ax, 30, 0, 40, None, top=91.5, title=None, lines=[
    "Fp8LinearMethod：checkpoint 带 weight_block_size=[128,128]",
    "-> block_quant，激活 GroupShape(1,128)",
], color="gray", ls=9.4, align="center", ts=12)

# 共用量化层
q0 = box(ax, 6, 0, 88, None, top=a0[1] - 3.5,
         title="共用的量化层：两条路线都从这里拿 FP8 张量（csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu）", lines=[
    "QuantFP8(group_shape=(1,128), use_ue8m0=?, column_major_scales=?, tma_aligned_scales=?)     input_quant_fp8.py:84",
    "     |",
    "     +-- use_ue8m0 且 oracle == UE8M0  -> torch.ops._C.per_token_group_fp8_quant_packed   scale 打包成 int32（每个装 4 个 UE8M0 指数）",
    "     +-- 其余情况                       -> torch.ops._C.per_token_group_fp8_quant          scale 是 fp32，按传入的 stride 写成列主序 / TMA 对齐",
    "     +-- 非 contiguous / 非 CUDA        -> Triton fallback（fp8_utils.py:566 的 _per_token_group_quant_fp8[_colmajor]）",
    "",
    "关键：scale 张量的形状和 stride 由 Python 侧 torch.empty_strided 事先分配，csrc kernel 只负责按 stride 填数——",
    "所以「列主序」「TMA 对齐」「UE8M0 打包」这些 DeepGEMM 的排布要求，落到 kernel 里只是 IS_COLUMN_MAJOR 模板参数和几个 stride。",
], color="green", ls=9.2, align="left", ts=12.5)

gy = q0[1] - 4.0
l0 = box(ax, 4, 0, 45, None, top=gy, title="路线 A：CUTLASS blockwise（仓库内）", lines=[
    "QuantFP8(use_ue8m0=False, column_major_scales=True)",
    "                              cutlass.py:279",
    "  -> a_fp8 [M,K]，a_scale fp32 [M, K/128] 列主序",
    "",
    "ops.cutlass_scaled_mm(...)        scaled_mm_entry.cu:197",
    "  -> dispatch_scaled_mm 看到 2D scale",
    "  -> blockwise_func",
    "  -> scaled_mm_blockwise_sm{90,100,120}_fp8.cu",
    "",
    "· CUTLASS 3.x C++ 模板，随 vLLM 一起 AOT 编译",
    "· 架构由 CMakeLists 门禁决定（SM90 起才有 blockwise）",
    "· scale 一律 fp32，不做 2 的幂对齐",
    "· 编译期定型：形状变化只换 tile 配置，不重新编译",
], color="blue", ls=9.2, align="left", ts=12.5)

r0 = box(ax, 51, 0, 45, None, top=gy, title="路线 B：DeepGEMM（外部库 / vendored）", lines=[
    "QuantFP8(use_ue8m0=oracle, tma_aligned_scales=env,",
    "         column_major_scales=True)   deep_gemm.py:38",
    "  -> a_fp8 [M,K]，a_scale 按 oracle 选三种排布之一",
    "",
    "torch.ops.vllm.fp8_gemm_nt_op(...)   deep_gemm.py:126",
    "  -> vllm/utils/deep_gemm.py:444 fp8_gemm_nt",
    "  -> deep_gemm.fp8_gemm_nt（site-packages 或",
    "     vllm.third_party.deep_gemm）",
    "",
    "· DeepSeek 开源的 JIT GEMM 库，运行时按形状编译 kernel",
    "· 只支持 Hopper / Blackwell（support_deep_gemm()）",
    "· 吃 UE8M0 scale，为此要求权重也重量化成 2 的幂",
    "· 首次遇到新形状要 JIT，所以有 warmup（deep_gemm_warmup.py）",
], color="purple", ls=9.2, align="left", ts=12.5)

arrow(ax, (30, q0[1]), (22, gy + 0.3), rad=0.1)
arrow(ax, (70, q0[1]), (78, gy + 0.3), rad=-0.1)

# scale 格式 oracle
o0 = box(ax, 4, 0, 92, None, top=min(l0[1], r0[1]) - 3.5,
         title="DeepGemmQuantScaleFMT：scale 用什么格式，由设备决定（vllm/utils/deep_gemm.py:49）", lines=[
    "FLOAT32              VLLM_USE_DEEP_GEMM_E8M0=0 或 DeepGEMM 不可用      fp32 scale，和 CUTLASS 路线拿到的东西一样",
    "FLOAT32_CEIL_UE8M0   Hopper（SM90）                                   数值上 ceil 成 2 的幂，但仍存成 fp32 张量",
    "UE8M0                Blackwell（capability family 100 / 120）          真的打包成 int32，走 per_token_group_fp8_quant_packed",
    "",
    "权重侧对应地要做一次性重排：deepgemm_post_process_fp8_weight_block（fp8_utils.py:1089）",
    "  · checkpoint 的 scale 已是 e8m0 -> _upcast_e8m0_to_fp32，跳过重量化",
    "  · 否则 use_e8m0 时 -> requant_weight_ue8m0_inplace：反量化回 fp32，再用 2 的幂 scale 重新量化一遍",
    "  · 最后 transform_sf_into_required_layout 交给 DeepGEMM 自己把 scale 摆成目标架构要的布局",
], color="orange", ls=9.2, align="left", ts=12.5)

# MoE
m0 = box(ax, 4, 0, 92, None, top=o0[1] - 3.5,
         title="MoE 里两者是交替出现的（batched_deep_gemm_moe.py）", lines=[
    "a1q, a1q_scale  <- 上游 prepare 阶段量化",
    "   |",
    "   +-> fp8_m_grouped_gemm_nt_masked((a1q, a1q_scale), (w1, w1_scale), workspace1, ...)      DeepGEMM  :425",
    "   |",
    "   +-> torch.ops._C.persistent_masked_m_silu_mul_quant(y, tokens_per_expert, y_q, y_s, ceil_ue8m0)   csrc  activation_kernels.cu",
    "   |     SwiGLU + 1x128 分块量化 + 按 DeepGemmQuantScaleFMT 决定是否 ceil 成 UE8M0，一个 persistent kernel 做完",
    "   |     ROCm 上没有这个 C++ kernel（#ifndef USE_ROCM），退回 Triton 的 _silu_mul_fp8_quant_deep_gemm",
    "   |",
    "   +-> fp8_m_grouped_gemm_nt_masked((a2q, a2q_scale), (w2, w2_scale), output, ...)          DeepGEMM  :440",
    "",
    "也就是说：DeepGEMM 负责所有 GEMM，csrc 负责两个 GEMM 之间的「激活函数 + 重新量化」。谁也替代不了谁。",
], color="teal", ls=9.2, align="left", ts=12.5)

import sys
print("bottom =", m0[1], file=sys.stderr)
save(fig, "deepgemm_vs_cutlass.png")
