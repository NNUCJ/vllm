from fig_common import *

fig, ax = new_fig(17, 23.0)
title(ax, "一层的真实算子序列（TP=1，Hopper + DeepGEMM）",
      "40 层 = linear_attention x30 + full_attention x10，每层后面都跟同一个 MoE 块")

hy = 91.0
h0 = box(ax, 26, 0, 48, None, top=hy, title=None, lines=[
    "hidden_states  bf16  [M, 2048]      M = 本 step 的 token 数",
], color="gray", ls=9.6, align="center", ts=12)

ay = h0[1] - 3.5
l0 = box(ax, 3, 0, 45, None, top=ay,
         title="A. linear_attention 层（30 个，Gated DeltaNet）", lines=[
    "input_layernorm            RMSNorm bf16，未量化",
    "  fusion pass 可把它和下一步的量化合成",
    "  rms_norm_per_block_quant（csrc）",
    "",
    "in_proj_qkv  [2048 -> 8192]   FP8  DeepGEMM",
    "in_proj_z    [2048 -> 4096]   FP8  DeepGEMM",
    "in_proj_a / in_proj_b  [2048 -> 32]  BF16 普通 GEMM",
    "                              （太小，不量化）",
    "",
    "conv1d(kernel=4) + A_log / dt_bias",
    "  GDN 递归，bf16 / fp32 状态，Triton kernel",
    "linear_attn.norm  [128]       BF16",
    "",
    "out_proj     [4096 -> 2048]   FP8  DeepGEMM",
], color="teal", ls=9.0, align="left", ts=12)

r0 = box(ax, 51, 0, 45, None, top=ay,
         title="B. full_attention 层（10 个）", lines=[
    "input_layernorm            RMSNorm bf16，未量化",
    "",
    "q_proj  [2048 -> 8192]        FP8  DeepGEMM",
    "        attn_output_gate=true，8192 = q 4096 + gate 4096",
    "k_proj  [2048 ->  512]        FP8  DeepGEMM",
    "v_proj  [2048 ->  512]        FP8  DeepGEMM",
    "        2 kv-head x head_dim 256",
    "",
    "q_norm / k_norm  [256]        BF16，未量化",
    "rotary_embedding（partial_rotary_factor 0.25）",
    "attention                     FlashAttention / FlashInfer",
    "        kv cache dtype 由 --kv-cache-dtype 决定，",
    "        与权重 FP8 无关",
    "",
    "o_proj  [4096 -> 2048]        FP8  DeepGEMM",
], color="blue", ls=9.0, align="left", ts=12)

arrow(ax, (40, h0[1]), (25, ay + 0.3), rad=0.12)
arrow(ax, (60, h0[1]), (74, ay + 0.3), rad=-0.12)

# 每个 FP8 Linear 的展开
dy = min(l0[1], r0[1]) - 3.5
d0 = box(ax, 3, 0, 93, None, top=dy,
         title="上面每一个「FP8 DeepGEMM」展开都是这三步（Hopper）", lines=[
    "  1) QuantFP8(group_shape=(1,128), use_ue8m0=is_deep_gemm_e8m0_used(), column_major_scales=True)      input_quant_fp8.py:84",
    "       -> torch.ops._C.per_token_group_fp8_quant   激活 [M,K] fp8 + scale fp32[M, K/128] 列主序，数值 ceil 成 2 的幂",
    "  2) torch.ops.vllm.fp8_gemm_nt_op((a, a_s), (w, w_s), out)         deep_gemm.py:126 -> vllm/utils/deep_gemm.py:444",
    "       -> deep_gemm.fp8_gemm_nt   JIT 出来的 warp-specialized kernel，权重 scale 已在加载期摆好",
    "  3) 输出直接是 bf16，反量化在 DeepGEMM 的 epilogue 里做掉",
], color="purple", ls=9.0, align="left", ts=12.5)

# Blackwell 上这份权重走不到 DeepGEMM
by = d0[1] - 2.6
b0 = box(ax, 3, 0, 93, None, top=by,
         title="Blackwell 上这份权重的 dense Linear 走不到 DeepGEMM", lines=[
    "  model_type = qwen3_5_moe_text 命中 _DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES（utils/deep_gemm.py:27）",
    "  -> DeepGemmFp8BlockScaledMMKernel.can_implement 返回 False -> 顺延到 CutlassFp8BlockScaledMMKernel",
    "  1) 同上，但 use_ue8m0 恒为 False（cutlass.py:283），scale 一直是 fp32；权重也不会被重量化",
    "  2) ops.cutlass_scaled_mm(A, B.T, scale_a=As, scale_b=Bs.T)        3) epilogue 同样出 bf16",
    "  原因：DeepGEMM 的 E8M0 scale 对这个架构有精度退化（#38083）。routed MoE 不受此限制。",
], color="orange", ls=9.0, align="left", ts=12.5)

my = b0[1] - 3.5
m0 = box(ax, 3, 0, 93, None, top=my,
         title="C. MoE 块（两种层后面都有；256 routed experts + 1 shared expert）", lines=[
    "post_attention_layernorm                       bf16，未量化",
    "     |",
    "     +-- mlp.gate  [2048 -> 256]   BF16 普通 GEMM（router 不量化）-> topk 选 8 个专家",
    "     |",
    "     +-- shared_expert（走的是普通 Linear 路径，也就是上面那三步）",
    "     |      gate_proj / up_proj  [2048 -> 512]   FP8 DeepGEMM",
    "     |      down_proj            [ 512 -> 2048]  FP8 DeepGEMM",
    "     |      shared_expert_gate   [2048 -> 1]     BF16",
    "     |",
    "     +-- routed experts（走 FusedMoE / modular kernel，和普通 Linear 是两套代码）",
    "            prepare：moe_kernel_quantize_input -> _fp8_quantize -> per_token_group_quant_fp8(A, 128)   fused_moe/utils.py:147",
    "                     用的还是 csrc 的 per_token_group_fp8_quant",
    "            permute：deepgemm_moe_permute 把 token 按专家排好，凑成 M_sum",
    "            GEMM1  ：m_grouped_fp8_gemm_nt_contiguous((a1q,a1q_s), (w1,w1_s))    DeepGEMM  deep_gemm_moe.py:358",
    "            激活+再量化：_act_mul_quant                                          deep_gemm_moe.py:225",
    "                     Blackwell(UE8M0)+SiLU -> silu_mul_quant_fp8_packed_triton      Triton",
    "                     Hopper 非 UE8M0 +SiLU -> silu_mul_per_token_group_quant_fp8_colmajor  Triton",
    "                     非 SiLU 激活          -> activation + per_token_group_quant_fp8       csrc",
    "                     （DP/EP 的 batched 布局走另一条：torch.ops._C.persistent_masked_m_silu_mul_quant  csrc）",
    "            GEMM2  ：m_grouped_fp8_gemm_nt_contiguous((a2q,a2q_s), (w2,w2_s))    DeepGEMM  deep_gemm_moe.py:375",
    "            finalize：deepgemm_unpermute_and_reduce 按 topk_weights 加权合并",
], color="green", ls=9.0, align="left", ts=12.5)

import sys
print("bottom =", m0[1], file=sys.stderr)
save(fig, "qwen35_deepgemm_runtime.png")
