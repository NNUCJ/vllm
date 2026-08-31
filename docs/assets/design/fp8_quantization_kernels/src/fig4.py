from fig_common import *

fig, ax = new_fig(17, 22.5)
title(ax, "一次 FP8 Linear 前向：从 bf16 激活到 bf16 输出",
      "vllm/model_executor/layers/quantization/fp8.py -> _custom_ops.py -> csrc/.../scaled_mm_entry.cu")

CX = 50.0

a0 = box(ax, 30, 0, 40, None, top=90.0, title=None, lines=[
    "hidden_states  bf16/fp16  [M, K]",
], color="gray", ls=10.5, align="center", ts=12)

# 三条量化分支
qy = a0[1] - 5.0
q1 = box(ax, 3, 0, 30, None, top=qy, title="per-tensor（静态）", lines=[
    "ops.scaled_fp8_quant(x, scale)",
    "-> static_scaled_fp8_quant",
    "   common.cu:183",
    "",
    "scale 来自 checkpoint 的",
    "input_scale，前向不再求 amax",
], color="blue", ls=9.4, align="left", ts=12)

q2 = box(ax, 35, 0, 30, None, top=qy, title="per-token（动态）", lines=[
    "ops.scaled_fp8_quant(x, None,",
    "     use_per_token_if_dynamic=True)",
    "-> dynamic_per_token_scaled_fp8_quant",
    "   common.cu:383",
    "",
    "每个 token 现算 amax，可带 scale_ub",
], color="green", ls=9.4, align="left", ts=12)

q3 = box(ax, 67, 0, 30, None, top=qy, title="1x128 group（block）", lines=[
    "per_token_group_quant_fp8(x, 128)",
    "-> per_token_group_fp8_quant",
    "   per_token_group_quant.cu:193",
    "",
    "DeepSeek 风格；配 128x128 权重 scale",
    "UE8M0 变体喂给 DeepGEMM",
], color="purple", ls=9.4, align="left", ts=12)

arrow(ax, (CX, a0[1]), (18, qy + 0.3), rad=0.12)
arrow(ax, (CX, a0[1]), (CX, qy + 0.3))
arrow(ax, (CX, a0[1]), (82, qy + 0.3), rad=-0.12)

# 融合旁注
fy = min(q1[1], q2[1], q3[1]) - 3.0
f1 = box(ax, 3, 0, 94, None, top=fy,
         title="torch.compile 的 fusion pass 会把「上一个算子 + 量化」合成一个 kernel（vllm/compilation/passes/fusion）", lines=[
    "rms_norm + static_scaled_fp8_quant        -> rms_norm_static_fp8_quant / fused_add_rms_norm_static_fp8_quant   layernorm_quant_kernels.cu",
    "rms_norm + dynamic_per_token_quant        -> rms_norm_dynamic_per_token_quant      fused_layernorm_dynamic_per_token_quant.cu:173",
    "rms_norm + per_token_group_quant          -> rms_norm_per_block_quant              fused_layernorm_dynamic_per_token_quant.cu:272",
    "silu_and_mul + quant                      -> silu_and_mul_quant / silu_and_mul_per_block_quant                 activation_kernels.cu",
    "融合掉的是一次完整的 HBM 往返：不融合时归一化结果要先写回显存，再被量化 kernel 读一遍。",
], color="orange", ls=9.2, align="left", ts=11.5)

# GEMM 入口
gy = f1[1] - 4.0
g0 = box(ax, 18, 0, 64, None, top=gy, title="ops.cutlass_scaled_mm(out, a_fp8, b_fp8, a_scales, b_scales, bias)", lines=[
    "scaled_mm_entry.cu:197   先查布局：a 行主序、b 列主序、c 行主序且 c.stride(0) % 16 == 0",
], color="teal", ls=9.4, align="center", ts=12.5)
arrow(ax, (CX, f1[1]), (CX, gy + 0.3))

# 两个正交的分档
dy = g0[1] - 4.5
d1 = box(ax, 3, 0, 45, None, top=dy, title="① 按 SM 版本选实现（编译期已定）", lines=[
    "get_sm_version_num()",
    "  >= 120        -> cutlass_scaled_mm_sm120   c3x",
    "  [100, 120)    -> cutlass_scaled_mm_sm100   c3x",
    "  [90, 100)     -> cutlass_scaled_mm_sm90    c3x",
    "  == 89         -> cutlass_scaled_mm_sm89    c2x",
    "  >= 80 / >= 75 -> cutlass_scaled_mm_sm80/75 c2x（int8 为主）",
    "都没编进来就 STD_TORCH_CHECK_NOT_IMPLEMENTED",
], color="blue", ls=9.4, align="left", ts=12)

d2 = box(ax, 52, 0, 45, None, top=dy, title="② 按 scale 形状选 kernel（运行期）", lines=[
    "dispatch_scaled_mm   scaled_mm_helper.hpp:6",
    "",
    "a_scales.numel() ∈ {1, M} 且",
    "b_scales.numel() ∈ {1, N}   -> fp8_func  普通 scaled_mm",
    "",
    "否则要求 2D scale，并校验",
    "  a_scales == [M, ceil(K/128)]",
    "  b_scales == [ceil(K/128), ceil(N/128)]  -> blockwise_func",
    "blockwise 目前不支持 bias",
], color="purple", ls=9.4, align="left", ts=12)
arrow(ax, (35, g0[1]), (25, dy + 0.3), rad=0.1)
arrow(ax, (65, g0[1]), (75, dy + 0.3), rad=-0.1)

# 输出
oy = min(d1[1], d2[1]) - 4.5
o0 = box(ax, 22, 0, 56, None, top=oy, title=None, lines=[
    "epilogue（EVT）：acc_fp32 * a_scale * b_scale (+ bias)  ->  out bf16/fp16 [M, N]",
], color="green", ls=10.0, align="center", ts=12)
arrow(ax, (25, d1[1]), (40, oy + 0.3), rad=-0.1)
arrow(ax, (75, d2[1]), (60, oy + 0.3), rad=0.1)

# 旁路
b0 = box(ax, 3, 0, 94, None, top=o0[1] - 3.5,
         title="并非所有 FP8 GEMM 都走 CUTLASS：后端在 Python 侧选定", lines=[
    "vllm/model_executor/kernels/linear/scaled_mm/",
    "  cutlass.py · deep_gemm.py · triton.py · flashinfer.py · aiter.py · marlin.py · pytorch.py(torch._scaled_mm) · cpu.py · xpu.py",
    "选中哪个后端，决定了激活该用哪种粒度量化——所以「量化 kernel」和「GEMM kernel」必须成对出现，不能各选各的。",
], color="red", ls=9.2, align="left", ts=12)

import sys
print("bottom =", b0[1], file=sys.stderr)
save(fig, "runtime_path.png")
