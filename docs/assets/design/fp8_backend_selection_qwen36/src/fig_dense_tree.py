from fig_common import *

fig, ax = new_fig(16, 15.4)
title(ax, "dense FP8 Linear 的后端选择",
      "构造期决定一次；优先级是静态列表顺序，第一个过三道门的胜出")

y = 91.0
h0 = box(ax, 20, 0, 60, None, top=y, title=None, lines=[
    "Fp8LinearMethod.__init__   ->   init_fp8_linear_kernel()",
    "quantization/fp8.py:387          kernels/linear/__init__.py:576",
], color="gray", ls=9.6, align="center", ts=12)

# 第一层分流
y1 = h0[1] - 4.0
s0 = box(ax, 14, 0, 72, None, top=y1, title=None, lines=[
    "activation_quant_key.scale.group_shape.is_per_group()  ?      __init__.py:594",
], color="orange", ls=10.0, align="center", ts=12)

y2 = s0[1] - 4.2
lb = box(ax, 2, 0, 46, None, top=y2,
         title="是：block 量化（weight_block_size=[128,128]）", lines=[
    "_POSSIBLE_FP8_BLOCK_KERNELS[CUDA]        :352",
    "",
    "1. FlashInferFp8DeepGEMMDynamicBlockScaled   仅 SM90",
    "2. DeepGemmFp8BlockScaledMMKernel            SM90 / SM100 / SM120",
    "3. CutlassFp8BlockScaledMMKernel             SM90+（SM100+ 需 CUDA>=12.8）",
    "4. MarlinFP8ScaledMMLinearKernel             需 FORCE_FP8_MARLIN",
    "5. TritonFp8BlockScaledMMKernel              恒可用，兜底",
    "6. HummingFP8ScaledMMLinearKernel",
], color="teal", ls=9.0, align="left", ts=12)

rb = box(ax, 52, 0, 46, None, top=y2,
         title="否：per-tensor / per-token", lines=[
    "_POSSIBLE_FP8_KERNELS[CUDA]              :322",
    "",
    "1. MarlinFP8ScaledMMLinearKernel             需 FORCE_FP8_MARLIN",
    "2. FlashInferFP8ScaledMMLinearKernel         仅 cc>=100（Blackwell）",
    "3. CutlassFP8ScaledMMLinearKernel            主力",
    "4. PerTensorTorchFP8ScaledMMLinearKernel     torch._scaled_mm",
    "5. ChannelWiseTorchFP8ScaledMMLinearKernel",
    "6. HummingFP8ScaledMMLinearKernel",
    "",
    "DeepGEMM 不在这张表里 —— 它只做 block 量化",
], color="blue", ls=9.0, align="left", ts=12)

arrow(ax, (50, h0[1]), (50, y1 + 0.3))
arrow(ax, (40, s0[1]), (25, y2 + 0.3), rad=0.12)
arrow(ax, (60, s0[1]), (75, y2 + 0.3), rad=-0.12)

# 三道门
y3 = min(lb[1], rb[1]) - 4.0
g0 = box(ax, 8, 0, 84, None, top=y3,
         title="按列表顺序逐个试，第一个同时过这三道门的胜出   choose_scaled_mm_linear_kernel :504", lines=[
    "  ① kernel.__name__ not in VLLM_DISABLED_KERNELS                        :482   环境变量黑名单",
    "  ② kernel.is_supported(compute_capability)                             :490   设备 + 编译产物是否有这个 kernel",
    "  ③ kernel.can_implement(config)                                        :494   这一层的 dtype / 形状 / 粒度是否匹配",
    "",
    "  三道全过 -> return kernel（构造期定死，forward 不再判）",
    "  全部落选 -> ValueError，把每个候选的失败原因拼起来抛出                    :569",
], color="purple", ls=9.0, align="left", ts=12.5)

arrow(ax, (25, lb[1]), (35, y3 + 0.3), rad=-0.10)
arrow(ax, (75, rb[1]), (65, y3 + 0.3), rad=0.10)

# 三个覆盖开关
y4 = g0[1] - 3.6
o0 = box(ax, 8, 0, 84, None, top=y4,
         title="三个可以覆盖优先级的开关", lines=[
    "  --linear-backend <name>    _filter_kernels_by_backend :301  先把候选表按名字过滤，过滤完为空直接报错",
    "                             可选值含 cutlass / deep_gemm / triton / marlin / flashinfer_* / torch ...",
    "  force_kernel 参数          :536  调用方指定，过门禁则直接返回，跳过整张表",
    "  VLLM_DISABLED_KERNELS      :482  按类名逐个禁用",
], color="green", ls=9.0, align="left", ts=12.5)

arrow(ax, (50, g0[1]), (50, y4 + 0.3))

# 结论
y5 = o0[1] - 3.6
c0 = box(ax, 8, 0, 84, None, top=y5, title=None, lines=[
    "结论：CUTLASS 排在 DeepGEMM 之后，是 DeepGEMM 的兜底而不是并列选项。",
    "两者能对调是因为输入输出约定一致：都吃 per_token_group_fp8_quant 的 (fp8, scale)，都在 epilogue 反量化成 bf16。",
], color="red", ls=9.6, align="center", ts=12)
arrow(ax, (50, o0[1]), (50, y5 + 0.3))

import sys
print("bottom =", c0[1], file=sys.stderr)
save(fig, "dense_backend_tree.png")
