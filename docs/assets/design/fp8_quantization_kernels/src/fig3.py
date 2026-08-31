from fig_common import *

fig, ax = new_fig(17, 21.2)
title(ax, "csrc 下的 FP8 算子地图",
      "文件 -> 它注册出的算子（Python 侧 torch.ops._C.<name>）")

b0 = box(ax, 4, 0, 92, None, top=93.0, title="公共 device 头：不注册算子，被下面所有 kernel include", lines=[
    "csrc/quantization/utils.cuh                       quant_type_max<T> / min_scaling_factor<T>",
    "csrc/quantization/w8a8/fp8/common.cuh             scaled_fp8_conversion / atomicMaxFloat / is_fp8_ocp",
    "csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh vec_conversion / scaled_convert：fp8 <-> half/bf16/float 的向量化转换",
    "csrc/quantization/w8a8/fp8/amd/quant_utils.cuh    同上的 ROCm 版（e4m3fnuz / cvt_c10）",
    "csrc/attention/dtype_fp8.cuh                      Fp8KVCacheDataType 枚举 + kv_cache_dtype 字符串解析",
    "csrc/libtorch_stable/quantization/fused_kernels/quant_conversions.cuh   融合 kernel 用的 float->fp8/int8 封装",
    "csrc/libtorch_stable/quantization/vectorization{,_utils}.cuh            128-bit 对齐向量化读写",
], color="gray", ls=9.2, align="left")

b1 = box(ax, 4, 0, 45, None, top=b0[1] - 2.0, title="① 独立的激活量化 kernel", lines=[
    "w8a8/fp8/common.cu",
    "  static_scaled_fp8_quant       任意 group_shape",
    "  dynamic_scaled_fp8_quant      per-tensor 两趟",
    "  dynamic_per_token_scaled_fp8_quant",
    "w8a8/fp8/per_token_group_quant.cu",
    "  per_token_group_fp8_quant        1x128 分块",
    "  per_token_group_fp8_quant_packed UE8M0 打包",
], color="green", ls=9.2, align="left")

b2 = box(ax, 4, 0, 45, None, top=b1[1] - 2.0, title="② 融合：量化被塞进上一个算子的尾巴", lines=[
    "layernorm_quant_kernels.cu",
    "  rms_norm_static_fp8_quant",
    "  fused_add_rms_norm_static_fp8_quant",
    "fused_kernels/fused_layernorm_dynamic_per_token_quant.cu",
    "  rms_norm_dynamic_per_token_quant",
    "  rms_norm_per_block_quant",
    "fused_kernels/fused_silu_mul_block_quant.cu",
    "  silu_and_mul_per_block_quant",
    "quantization/activation_kernels.cu",
    "  silu_and_mul_quant",
    "  persistent_masked_m_silu_mul_quant   (MoE, DeepGEMM)",
    "fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu",
    "  ..._full_cache_fp8_insert  norm+rope+量化+写 KV 一趟做完",
], color="purple", ls=9.2, align="left")

b3 = box(ax, 51, 0, 45, None, top=b0[1] - 2.0, title="③ CUTLASS W8A8 GEMM（消费 FP8 张量）", lines=[
    "w8a8/cutlass/scaled_mm_entry.cu        <- 唯一入口",
    "  cutlass_scaled_mm / cutlass_scaled_mm_azp",
    "  cutlass_scaled_mm_supports_fp8 / _block_fp8",
    "  cutlass_moe_mm / get_cutlass_moe_mm_data",
    "",
    "c3x/scaled_mm_helper.hpp   按 scale 维度选实现",
    "",
    "cutlass/scaled_mm_c2x.cu                   SM75/80/89",
    "  + scaled_mm_c2x_sm89_fp8_dispatch.cuh    (CUTLASS 2.x)",
    "cutlass/scaled_mm_c3x_sm90.cu  + c3x/scaled_mm_sm90_fp8.cu",
    "cutlass/scaled_mm_c3x_sm100.cu + c3x/scaled_mm_sm100_fp8.cu",
    "cutlass/scaled_mm_c3x_sm120.cu + c3x/scaled_mm_sm120_fp8.cu",
    "c3x/scaled_mm_blockwise_sm{90,100,120}_fp8.cu",
    "                        1x128 激活 x 128x128 权重",
    "",
    "cutlass/moe/grouped_mm_c3x_sm{90,100}.cu",
    "                        MoE grouped GEMM，逐专家 problem",
    "",
    "cutlass_extensions/epilogue/scaled_mm_epilogues_c{2,3}x.hpp",
    "                        反量化 epilogue（EVT）",
], color="blue", ls=9.2, align="left")

b4 = box(ax, 4, 0, 45, None, top=b2[1] - 2.0, title="④ KV cache 的 FP8 写入与读出", lines=[
    "libtorch_stable/cache_kernels.cu",
    "  reshape_and_cache / reshape_and_cache_flash",
    "      写 KV 时顺手 scaled_convert 成 fp8",
    "  concat_and_cache_mla        含 fp8_ds_mla 特殊布局",
    "  convert_fp8                 离线整块转换",
    "  gather_and_maybe_dequant_cache  取出时反量化",
    "  cp_gather_and_upconvert_fp8_kv_cache",
    "  indexer_k_quant_and_cache   DeepSeek 稀疏 indexer",
    "cache_kernels_fused.cu        rope + 写 cache 融合版",
], color="teal", ls=9.2, align="left")

b5 = box(ax, 51, 0, 45, None, top=b3[1] - 2.0, title="⑤ 非 CUDA 平台，以及形近但不同的邻居", lines=[
    "csrc/cpu/sgl-kernels/gemm_fp8.cpp / moe_fp8.cpp",
    "csrc/cpu/cpu_attn_fp8.hpp        CPU 上的 fp8 KV attention",
    "csrc/rocm/skinny_gemms.cu        wvSplitKQ：小 batch fp8 GEMM",
    "csrc/rocm/attention.cu           ROCm paged attention + fp8 KV",
    "",
    "以下不是 W8A8-FP8，别混：",
    "  marlin/marlin_int4_fp8_preprocess.cu   W4A8（权重 int4）",
    "  quantization/cutlass_w4a8/             W4A8 GEMM",
    "  quantization/fp4/*.cu                  NVFP4 / MXFP4",
], color="orange", ls=9.2, align="left")


b7 = box(ax, 51, 0, 45, None, top=b5[1] - 2.0, title="Python 侧入口（都在 vllm/ 下）", lines=[
    "_custom_ops.py:1797  scaled_fp8_quant   ①②③ 的统一入口",
    "fp8_utils.py:566     per_token_group_quant_fp8  CUDA/Triton",
    "w8a8_utils.py:11     cutlass_fp8_supported()",
    "kernels/linear/scaled_mm/cutlass.py  调用 cutlass_scaled_mm",
], color="gray", ls=9.2, align="left")

b6 = box(ax, 4, 0, 92, None, top=min(b4[1], b7[1]) - 2.5, title="编译门禁：文件存在 ≠ 算子可用（CMakeLists.txt）", lines=[
    "SM89（Ada）     走 c2x，8.9+PTX；没有 blockwise fp8 GEMM，DeepSeek 风格模型只能退回 Triton / DeepGEMM 之外的路径",
    "SM90（Hopper）  9.0a + CUDA>=12.0  -> -DENABLE_SCALED_MM_SM90，含 blockwise",
    "SM100（Blackwell）10.0a/10.1a/10.3a（CUDA13 起用 10.0f/11.0f）+ CUDA>=12.8 -> -DENABLE_SCALED_MM_SM100",
    "SM120（RTX 50） 12.0a/12.1a（CUDA13 起 12.0f）+ CUDA>=12.8 -> -DENABLE_SCALED_MM_SM120",
    "c2x 只为「3x 没覆盖到的」架构编译：CMakeLists.txt:841 先取 7.5;8.0;8.7;8.9+PTX 再减去 SCALED_MM_3X_ARCHS",
], color="red", ls=9.2, align="left")

import sys; print("bottom =", b6[1], file=sys.stderr)
save(fig, "csrc_fp8_map.png")
