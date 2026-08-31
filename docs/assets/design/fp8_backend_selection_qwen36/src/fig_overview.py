from fig_common import *

fig, ax = new_fig(17, 17.0)
title(ax, "Qwen3.6-35B-A3B-FP8 的 FP8 推理总流程",
      "40 层 = linear_attention x30 + full_attention x10；每层后跟一个 MoE 块；256 routed + 1 shared expert")

y = 92.0
h0 = box(ax, 22, 0, 56, None, top=y, title=None, lines=[
    "quantization_config: quant_method=fp8, activation_scheme=dynamic,",
    "fmt=e4m3, weight_block_size=[128,128]   ->   block 量化，粒度 1x128 / 128x128",
], color="gray", ls=9.4, align="center", ts=12)

# 加载期
y1 = h0[1] - 3.6
l0 = box(ax, 3, 0, 94, None, top=y1,
         title="加载期（每层各跑一次）", lines=[
    "  ① Fp8LinearMethod.__init__ -> init_fp8_linear_kernel   选定 dense kernel（图 2）",
    "     Fp8MoEMethod.__init__   -> select_fp8_moe_backend   选定 MoE backend（图 3）",
    "  ② weight_loader 把 F8_E4M3 权重和 BF16 的 weight_scale_inv 读进来，scale 自动 upcast 成 fp32",
    "  ③ process_weights_after_loading",
    "        DeepGEMM 路: deepgemm_post_process_fp8_weight_block  fp8_utils.py:1089",
    "            requant_weight_ue8m0_inplace   把权重反量化再用 2 的幂重量化（纯 torch）",
    "            transform_sf_into_required_layout  SM90 出 fp32 / SM100,120 出 int32 打包 UE8M0",
    "        CUTLASS 路: 只调 process_fp8_weight_block_strategy 调布局，权重一个字节都不动",
], color="orange", ls=9.0, align="left", ts=12)
arrow(ax, (50, h0[1]), (50, y1 + 0.3))

# 运行期两条路
y2 = l0[1] - 4.0
lb = box(ax, 2, 0, 46, None, top=y2,
         title="dense 路（每 token 每层都走）", lines=[
    "覆盖：q/k/v/o_proj、GDN 的 in_proj_*/out_proj、",
    "      shared_expert 的 gate/up/down、",
    "      routed 之外的一切量化 Linear",
    "",
    "Fp8LinearMethod.apply            fp8.py:446",
    " -> Fp8BlockScaledMMLinearKernel.apply_weights",
    "    ① QuantFP8(...)  1x128 动态量化   [csrc]",
    "    ② apply_block_scaled_mm",
    "         DeepGEMM: fp8_gemm_nt",
    "         CUTLASS : ops.cutlass_scaled_mm",
    "    ③ epilogue 里反量化，直接出 bf16",
], color="teal", ls=9.0, align="left", ts=12)

rb = box(ax, 52, 0, 46, None, top=y2,
         title="MoE 路（routed experts，另一套代码）", lines=[
    "覆盖：256 个 routed expert 的 gate/up/down",
    "      w13 [256,1024,2048]  w2 [256,2048,512]",
    "",
    "Fp8MoEMethod.apply               fp8.py:833",
    " -> FusedMoEKernel.apply       modular_kernel.py",
    "    prepare  -> 量化           [csrc]",
    "    permute  -> 按专家排好      [Triton]",
    "    GEMM1 / 激活+再量化 / GEMM2",
    "    finalize -> 按 topk 加权合并 [Triton]",
    "",
    "router(mlp.gate) 不量化，恒 BF16",
], color="blue", ls=9.0, align="left", ts=12)

arrow(ax, (40, l0[1]), (25, y2 + 0.3), rad=0.12)
arrow(ax, (60, l0[1]), (75, y2 + 0.3), rad=-0.12)

# 后端落点
y3 = min(lb[1], rb[1]) - 3.8
b0 = box(ax, 3, 0, 94, None, top=y3,
         title="两条路各自落到哪个后端（这份权重，默认配置）", lines=[
    "  设备              dense Linear                              routed MoE",
    "  ---------------   ---------------------------------------   ---------------------------------------",
    "  Hopper H100/H800  FlashInfer+DeepGEMM（没装则 DeepGEMM）      TritonExperts（oracle 把 TRITON 提前）",
    "  Blackwell B200    CUTLASS blockwise（model_type 黑名单）      FlashInfer TRTLLM（Monolithic）",
    "  RTX 50 系 SM120   CUTLASS blockwise（黑名单同样命中）          DeepGEMM（FI-TRTLLM 只认 SM100）",
    "  Ada  4090/L40S    Triton block scaled（无 SM90+ 的 kernel）    TritonExperts",
    "",
    "  两条路唯一的共用件：激活量化算子 per_token_group_fp8_quant（csrc），它不挑架构，SM80 起都能跑。",
], color="purple", ls=9.0, align="left", ts=12)
arrow(ax, (25, lb[1]), (35, y3 + 0.3), rad=-0.10)
arrow(ax, (75, rb[1]), (65, y3 + 0.3), rad=0.10)

y4 = b0[1] - 3.6
n0 = box(ax, 3, 0, 94, None, top=y4, title=None, lines=[
    "没有被量化的部分（modules_to_not_convert 共 648 项）：整个 vision 塔、所有 layernorm、router(mlp.gate)、",
    "shared_expert_gate、q_norm/k_norm、GDN 的 A_log / conv1d / dt_bias / in_proj_a / in_proj_b / in_proj_ba / norm。",
    "规律：router、门控、归一化、卷积、状态参数一律留 BF16——要么太小不值得，要么对误差敏感。",
], color="red", ls=9.0, align="center", ts=12)
arrow(ax, (50, b0[1]), (50, y4 + 0.3))

import sys
print("bottom =", n0[1], file=sys.stderr)
save(fig, "overview.png")
