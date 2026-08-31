from fig_common import *
import sys

fig, ax = new_fig(17.6, 17.6)
title(ax, "Blackwell 上 Qwen3.6 dense 层的候选逐个落选过程",
      "SM100 / SM120 + block-FP8 [128,128]；同一份权重在 SM90 上第 1 名就命中")

G = 2.6
y = 90.0
a = box(ax, 3, 0, 94, None, top=y,
        title="输入：一层 dense block-FP8 Linear，例如 model.layers.0.linear_attn.in_proj_qkvz", lines=[
    'weight [12288, 2048] float8_e4m3fn   weight_scale_inv [96, 16] fp32   out_dtype = bf16',
    'activation_quant_key = fp8 dynamic GroupShape(1,128)   ->  查 _POSSIBLE_FP8_BLOCK_KERNELS[CUDA]',
], color="gray", ls=9.2, align="left", ts=12)

y = a[1] - G
b = box(ax, 3, 0, 94, None, top=y,
        title="候选 1  FlashInferFp8DeepGEMMDynamicBlockScaledKernel  ——  ✗ 门 ② is_supported", lines=[
    'has_flashinfer_fp8_blockscale_gemm()   utils/flashinfer.py:934',
    '    current_platform.is_device_capability(90)  and  hasattr(..., "fp8_blockscale_gemm_sm90")',
    'SM100/SM120 上第一个条件就 False —— 这个候选是 Hopper 专属，与模型无关',
], color="red", ls=9.2, align="left", ts=12)
arrow(ax, (50, a[1]), (50, y + 0.3))

y = b[1] - G
c = box(ax, 3, 0, 94, None, top=y,
        title="候选 2  DeepGemmFp8BlockScaledMMKernel  ——  ✗ 只挂在最后一条", lines=[
    'is_supported:  is_deep_gemm_supported()  -> SM90/100/120 都过                          ✓',
    'can_implement  scaled_mm/deep_gemm.py:55',
    '   基类：激活必须 dynamic          BlockScaledMMLinearKernel.py:62                    ✓',
    '   out_dtype == torch.bfloat16                                                          ✓',
    '   activation group_shape == GroupShape(1,128)                                          ✓',
    '   should_use_deepgemm_for_fp8_linear: N % 64 == 0 and K % 128 == 0   utils/deep_gemm.py:700',
    '        12288 % 64 = 0，2048 % 128 = 0  —— 这层过，而且它几乎不可能不过（见正文）  ✓',
    '   should_auto_disable_deep_gemm(model_type)          utils/deep_gemm.py:33            ✗',
    '        model_type = hf_text_config.model_type = "qwen3_5_moe_text"',
    '        ∈ _DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES = {"qwen3_5_text","qwen3_5_moe_text"}  :27',
    '        且 capability family ∈ {100, 120}',
    '   -> return False, "Should not use deepgemm for model qwen3_5_moe_text"',
], color="orange", ls=9.2, align="left", ts=12, lw=2.4)
arrow(ax, (50, b[1]), (50, y + 0.3))

y = c[1] - G
d = box(ax, 3, 0, 94, None, top=y,
        title="候选 3  CutlassFp8BlockScaledMMKernel  ——  ✓ 选中", lines=[
    'SM100 有 scaled_mm_blockwise_sm100_fp8.cu，SM120 有 _sm120_fp8.cu（CUDA >= 12.8 起编）',
    'can_implement 只要求 group_shape == (1,128)；use_ue8m0 恒为 False，权重一个字节都不动',
    '日志："Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod"   kernels/linear/__init__.py:600',
], color="green", ls=9.2, align="left", ts=12)
arrow(ax, (50, c[1]), (50, y + 0.3))

y = d[1] - G
e = box(ax, 3, 0, 94, None, top=y,
        title="为什么名单里有它：E8M0 的精度代价", lines=[
    'Blackwell 上 DeepGemmQuantScaleFMT = UE8M0，scale 被约束成 2 的幂：amax 只能映到 256 而非 448，',
    '上端最多丢 1 bit 动态范围；加载期还会 requant_weight_ue8m0_inplace 把权重按 2 的幂重量化一次。',
    '多数模型扛得住，Qwen3.5/3.6 这个架构扛不住 —— 是 gsm8k 评测抓出来的实测回归，不是理论推导。',
    '出处 52069012f (#38083)，SM120 由 44d95069e (#43477) 补入判定。',
], color="purple", ls=9.2, align="left", ts=12)
arrow(ax, (50, d[1]), (50, y + 0.3))

y = e[1] - G
f = box(ax, 3, 0, 94, None, top=y,
        title="三条以为能绕开、其实不能的路", lines=[
    'VLLM_USE_DEEP_GEMM=1        黑名单不读这个变量 -> 无效',
    'VLLM_USE_DEEP_GEMM_E8M0=0   能消掉精度问题的根，但黑名单只看 model_type -> 仍然无效',
    '--linear-backend deep_gemm  候选表被过滤成只剩它，can_implement 仍失败 -> 直接 raise，不是回退',
], color="pink", ls=9.2, align="left", ts=12)
arrow(ax, (50, e[1]), (50, y + 0.3))

print("bottom =", f[1], file=sys.stderr)
save(fig, "qwen36_dense_deepgemm_gate.png")
