from fig_common import *

fig, ax = new_fig(17, 19.5)
title(ax, "谁真的会走 DeepGEMM：Linear 会，MoE 不一定",
      "这份 checkpoint 的 moe_intermediate_size = 512，正好卡在 DeepGEMM 的形状门槛上")

a0 = box(ax, 4, 0, 92, None, top=91.0,
         title="① Linear 层：默认就是 DeepGEMM（只要设备支持）", lines=[
    "_POSSIBLE_FP8_BLOCK_KERNELS[CUDA] = [FlashInfer+DeepGEMM, DeepGemm, Cutlass, Marlin, Triton, Humming]      kernels/linear/__init__.py:355",
    "按顺序取第一个 is_supported() 且 can_implement() 的：",
    "  · is_supported  -> is_deep_gemm_supported()：Hopper / Blackwell，且 deep_gemm 能 import",
    "  · can_implement -> out_dtype==bf16 且 N%64==0 且 K%128==0（should_use_deepgemm_for_fp8_linear，utils/deep_gemm.py:700）",
    "",
    "这份权重的每个量化 Linear 都满足：8192/4096/2048/512 都是 64 的倍数，2048/4096/512 都是 128 的倍数。",
    "所以在 H20/H100/H800/B200 上，attention 的 q/k/v/o、GDN 的 in_proj_qkv/z/out_proj、shared_expert 的三个投影，全部走 DeepGEMM。",
], color="green", ls=9.0, align="left", ts=12.5)

b0 = box(ax, 4, 0, 92, None, top=a0[1] - 3.0,
         title="② MoE 层：要过两道关，默认很可能不是 DeepGEMM", lines=[
    "第一关  select_fp8_moe_backend（fused_moe/oracle/fp8.py:271）按这个顺序决定：",
    "   1. --moe-backend 显式指定           -> 直接用（deep_gemm / triton / flashinfer_cutlass ...）",
    "   2. 显式 set 了 VLLM_USE_DEEP_GEMM 或 VLLM_MOE_USE_DEEP_GEMM   -> 值为真则直接选 DEEPGEMM，为假则把它从候选里删掉",
    "   3. VLLM_TEST_FORCE_FP8_MARLIN / AITER 的特判",
    "   4. 都没有，才走 _get_priority_backends 的优先级列表，而这个列表会被设备和并行方式改写：",
    "        Hopper(SM90) + block fp8 + ep_size==1（纯 TP）   -> TRITON 被提到最前            <- 默认不走 DeepGEMM",
    "        Hopper(SM90) + block fp8 + ep_size>1（EP）        -> FLASHINFER_CUTLASS 被提到最前",
    "        Blackwell(SM100) + DeepEP v2 + block fp8          -> FLASHINFER_TRTLLM 被提到最前",
    "        其余情况的默认序：AITER, FI_TRTLLM, FI_CUTLASS, DEEPGEMM, VLLM_CUTLASS, TRITON, ...",
    "",
    "第二关  就算选中了 DEEPGEMM，拿到的也是 TritonOrDeepGemmExperts，每次 forward 还会再判一次：",
    "        _select_experts_impl:  is_deep_gemm_e8m0_used() or _valid_deep_gemm(hidden_states, w1, w2)",
    "        _valid_deep_gemm 里有一条：N <= 512 直接返回 False（deep_gemm_moe.py:88）",
    "        本模型 w2 = [256, 2048, 512] -> K=2048, N=512，正好命中 N<=512",
    "        -> VLLM_USE_DEEP_GEMM_E8M0=1（默认）时被前半句短路，仍走 DeepGEMM",
    "        -> VLLM_USE_DEEP_GEMM_E8M0=0 时，MoE 每次都回退 TritonExperts",
], color="orange", ls=9.0, align="left", ts=12.5)

c0 = box(ax, 4, 0, 92, None, top=b0[1] - 3.0,
         title="③ 结论矩阵（这份 checkpoint，默认参数）", lines=[
    "  设备 / 并行                        Linear（q/k/v/o、in_proj、shared_expert）    routed MoE（256 experts）",
    "  ---------------------------------+------------------------------------------+----------------------------------------",
    "  H20 / H100 / H800，TP only        DeepGEMM  fp8_gemm_nt                       TritonExperts（oracle 把 Triton 提前）",
    "  H20 / H100 / H800，EP             DeepGEMM                                    FlashInfer CUTLASS（不可用则往下顺延）",
    "  B200，DeepEP v2                   DeepGEMM（UE8M0 packed scale）               FlashInfer TRTLLM",
    "  B200，普通 TP                      DeepGEMM                                    DeepGEMM m_grouped_fp8_gemm_nt_contiguous",
    "  任意 H/B + VLLM_USE_DEEP_GEMM=1    DeepGEMM                                    DeepGEMM（显式 set 会跳过优先级表）",
    "  RTX 4090 / L40S（SM89）            Triton block scaled                         TritonExperts",
    "",
    "想让 MoE 也确定性地走 DeepGEMM，最直接的两个办法：VLLM_USE_DEEP_GEMM=1 环境变量，或 --moe-backend deep_gemm。",
], color="blue", ls=9.0, align="left", ts=12.5)

d0 = box(ax, 4, 0, 92, None, top=c0[1] - 3.0,
         title="④ 本机（4 x RTX 4090，SM 8.9）：这份权重跑不了 DeepGEMM", lines=[
    "is_deep_gemm_supported() -> current_platform.support_deep_gemm() -> False（只认 Hopper / Blackwell）",
    "CUTLASS blockwise 也要 SM90+（CMakeLists 里 scaled_mm_blockwise_sm90 起编），SM89 编不出来",
    "  -> Linear 顺延到 TritonFp8BlockScaledMMKernel，MoE 顺延到 TritonExperts",
    "  -> 激活量化仍然是 csrc 的 per_token_group_fp8_quant（它不挑架构，SM80 起都能跑）",
    "",
    "另外这台机器上的 torch 是 2.12.0+cpu，torch.cuda.is_available() == False，现在这个环境本身也起不了推理进程。",
    "",
    "要确认线上到底走了哪条路，看这三行日志（都是 info_once）：",
    "  \"Using DEEPGEMM Fp8 MoE backend out of potential backends: [...]\"     oracle/fp8.py 的 _make_log_backend",
    "  \"DeepGEMM E8M0 enabled on current platform.\"                          utils/deep_gemm.py:118",
    "  \"DeepGemm disabled for N <= 512 ...\"（debug 级）                       deep_gemm_moe.py:89",
], color="red", ls=9.0, align="left", ts=12.5)

import sys
print("bottom =", d0[1], file=sys.stderr)
save(fig, "qwen35_deepgemm_backend_matrix.png")
