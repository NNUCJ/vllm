from fig_common import *

fig, ax = new_fig(17, 16.6)
title(ax, "DeepGEMM 与 CUTLASS：同一件事各调哪个函数",
      "DeepGEMM 侧唯一封装层是 vllm/utils/deep_gemm.py；CUTLASS 侧唯一入口是 scaled_mm_entry.cu")

y = 91.5
d0 = box(ax, 2, 0, 46, None, top=y,
         title="DeepGEMM：vllm/utils/deep_gemm.py", lines=[
    "_lazy_init() 绑定 17 个 _*_impl 符号  :211",
    "  外部包优先，退到 vllm/third_party/deep_gemm",
    "  SM90+ 顺手开 PDL :262",
    "  末尾 init_oracle_cache() 定 scale 格式 :296",
    "",
    "GEMM 入口",
    "  fp8_gemm_nt                        :444",
    "  m_grouped_fp8_gemm_nt_contiguous   :463",
    "  fp8_m_grouped_gemm_nt_masked       :481",
    "  m_grouped_fp8_fp4_gemm_nt_contiguous :472",
    "  fp8_einsum                         :456",
    "  这几个都会注入 disable_ue8m0_cast",
    "",
    "scale 布局",
    "  transform_sf_into_required_layout  :490",
    "  pack_ue8m0_to_int                  :395",
    "  get_mn_major_tma_aligned_*         :408 :419",
    "",
    "查询 / 开关",
    "  is_deep_gemm_supported             :93",
    "  is_deep_gemm_e8m0_used             :102",
    "  should_use_deepgemm_for_fp8_linear :700",
    "  should_auto_disable_deep_gemm      :33",
    "  get_mk_alignment_for_contiguous_layout :315",
    "  mk_alignment_scope                 :371",
    "  set_num_sms / get_num_sms          :307 :299",
], color="teal", ls=8.8, align="left", ts=12)

c0 = box(ax, 52, 0, 46, None, top=y,
         title="CUTLASS：csrc .../cutlass/scaled_mm_entry.cu", lines=[
    "按 SM 分发  :197",
    "  >=120 sm120 / [100,120) sm100",
    "  [90,100) sm90 / ==89 sm89 / >=80 sm80 / >=75 sm75",
    "再按 scale 维度二级分发",
    "  scaled_mm_helper.hpp:6  dispatch_scaled_mm",
    "  numel 为 1 或 M/N -> fp8_func / int8_func",
    "  2D scale -> blockwise_func（不支持 bias）",
    "",
    "导出算子（Python 封装在 _custom_ops.py）",
    "  cutlass_scaled_mm                  :725",
    "  cutlass_scaled_mm_azp              :776",
    "  cutlass_moe_mm                     :938",
    "  get_cutlass_moe_mm_data            :816",
    "  get_cutlass_moe_mm_problem_sizes_  :869",
    "     from_expert_offsets",
    "  get_cutlass_batched_moe_mm_data    :903",
    "  cutlass_scaled_mm_supports_fp8     :717",
    "  cutlass_scaled_mm_supports_block_fp8 :721",
    "  cutlass_group_gemm_supported       :806",
    "",
    "epilogue（EVT）scaled_mm_epilogues_c3x.hpp",
    "  ScaledEpilogue          无 bias  :151",
    "  ScaledEpilogueBias      行向量 bias :195",
    "  ScaledEpilogueColumnBias swap-AB :238",
    "  ScaledEpilogueArray     grouped :417",
    "  ScaledEpilogueBiasAzp(Token)  非对称 :284",
], color="blue", ls=8.8, align="left", ts=12)

y1 = min(d0[1], c0[1]) - 3.6
t0 = box(ax, 2, 0, 96, None, top=y1,
         title="逐项对照", lines=[
    "  任务                          DeepGEMM 路线                              CUTLASS 路线",
    "  ---------------------------   ----------------------------------------   ----------------------------------------",
    "  per-tensor / per-token GEMM   不支持（DG 只做 blockwise 和 grouped）      ops.cutlass_scaled_mm -> fp8_func",
    "  非对称量化（zero-point）      无                                          ops.cutlass_scaled_mm_azp",
    "  blockwise 1x128 x 128x128     fp8_gemm_nt                                ops.cutlass_scaled_mm -> blockwise_func",
    "  grouped MoE（contiguous）     m_grouped_fp8_gemm_nt_contiguous           ops.cutlass_moe_mm + ScaledEpilogueArray",
    "  grouped MoE（masked/batched） fp8_m_grouped_gemm_nt_masked               ops.cutlass_moe_mm + get_cutlass_batched_moe_mm_data",
    "  MoE 分组元数据                Python 侧算 M_sum / expert_ids（无 C++ op） ops.get_cutlass_moe_mm_data 一次算全",
    "  激活量化                      两边共用 csrc 的 per_token_group_fp8_quant / scaled_fp8_quant",
    "  scale 布局变换                transform_sf_into_required_layout 等一组     无对应物，直接吃 fp32 row/col-major",
    "  能力探测                      is_deep_gemm_supported / _e8m0_used         cutlass_scaled_mm_supports_(block_)fp8",
    "  SM 数切分（ubatch）           set_num_sms                                无",
    "",
    "  DeepGEMM 是 JIT 的，首次遇到新形状要在 hot path 上编译，所以有 warmup/deep_gemm_warmup.py 提前跑一遍；",
    "  CUTLASS 是 AOT 的，没有这个问题，但架构覆盖由 CMakeLists 的 CUDA_ARCHS 决定，编译时缺了运行期就没有。",
], color="purple", ls=8.8, align="left", ts=12.5)

import sys
print("bottom =", t0[1], file=sys.stderr)
save(fig, "api_surface.png")
