from fig_common import *
from matplotlib.patches import FancyBboxPatch
import sys

# ------- 装配第 ⑥ 步内部：从候选表到一个具体算子（横版，按执行阶段分带） -------
fig, ax = new_fig(22, 13.2)
title(ax, "装配第 ⑥ 步内部：create_weights 如何定出「这一层调哪个算子」",
      "实线框 = 初始化阶段 · 构造期（A–E，只跑一次）    虚线框 = forward 阶段（F，每次前向）"
      "    ★ = 选后端最终唯一的落点")

OR, PU, TEAL = C["orange"][1], C["purple"][1], C["teal"][1]


def band(x, y, w, h, label, sub, color, dashed=False):
    fc, ec = C[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=1,
                                linestyle=(0, (5, 3)) if dashed else "solid", alpha=0.5))
    ax.text(x + 1.4, y + h - 1.8, label, ha="left", va="center", fontsize=12,
            color=ec, family=SANS, zorder=6)
    ax.text(x + 1.4, y + h - 4.0, sub, ha="left", va="center", fontsize=9,
            color=ec, family=MONO, zorder=6, alpha=0.9)


def step(x, y, w, h, tag, name, loc, lines, color="gray", star=False, fs=8.2):
    fc, ec = C[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.7",
                                facecolor=fc, edgecolor=ec,
                                linewidth=2.6 if star else 1.5, zorder=4))
    ax.text(x + w / 2, y + h - 1.9, f"{tag}  {name}", ha="center", va="center",
            fontsize=10.2, color=ec, family=SANS, zorder=6)
    ax.text(x + w / 2, y + h - 3.9, loc, ha="center", va="center",
            fontsize=8.0, color=ec, family=MONO, zorder=6, alpha=0.85)
    ax.text(x + 1.4, y + h - 5.6, "\n".join(lines), ha="left", va="top",
            fontsize=fs, color="#20242b", family=MONO, zorder=6, linespacing=1.75)
    if star:
        ax.text(x + w - 1.4, y + h - 1.6, "★", ha="center", va="center",
                fontsize=12, color=OR, zorder=7)
    return (x, y, w, h)


# ═══════════ 带 1：构造期 A–E ═══════════
B1Y, B1H = 56.0, 36.0
band(2, B1Y, 96, B1H, "初始化阶段 · 构造期",
     "init_fp8_linear_kernel  kernels/linear/__init__.py:576 —— 建层时跑一次，运行期不再判定", "blue")

SW, SG, SX0 = 18.0, 1.5, 3.5
SH, SY = 24.4, B1Y + 2.0
segs = [
    ("A", "入口", "fp8.py:322 → :387",
     ["权重 register_parameter", "之后才调用，因为要", "layer.weight.shape",
      "",
      "__init__ 已算出两个", "QuantKey，唯独缺 shape", "—— 这就是选择被推迟", "到第 ⑥ 步的全部原因"], "gray", False),
    ("B", "选表", "__init__.py:594",
     ["is_per_group() ?", "  真 → BLOCK 表  :352", "  假 → 非 block 表 :322",
      "",
      "两表互不相交地划分", "后端能力：", "DeepGEMM 只在 block 表", "FlashInfer 两表架构相反"], "green", False),
    ("C", "两条旁路", "__init__.py:504",
     ["force_kernel  点名一类", "--linear-backend 按名", "  过滤表，空则 raise",
      "",
      "语义是「缩小候选范围」", "不是「强制使用」——", "留下的仍要过三道门", "（5.3.5 启动失败之因）"], "purple", False),
    ("D", "主循环", "__init__.py:562",
     ["列表顺序即优先级，", "无打分 / benchmark", "", "gate1 DISABLED_KERNELS",
      "gate2 is_supported 机器", "gate3 can_implement 该层", "",
      "命中即停；全落选把", "reasons 拼成一段 raise"], "orange", True),
    ("E", "实例化 + 日志", "__init__.py:596 / :617",
     ["Selected <kernel> for", "  Fp8LinearMethod", "scope=\"global\" ⇒ 全进程",
      "  只打一条，非每层一条", "",
      "Marlin 那一支需额外传", "layer_param_names，", "其余只吃 config"], "teal", False),
]
for i, (tag, name, loc, lines, col, star) in enumerate(segs):
    x = SX0 + i * (SW + SG)
    step(x, SY, SW, SH, tag, name, loc, lines, col, star)
    if i:
        arrow(ax, (x - SG + 0.15, SY + SH / 2), (x - 0.15, SY + SH / 2), lw=1.6)

# ═══════════ 带 2：forward 阶段 F ═══════════
B2Y, B2H = 4.0, 49.0
band(2, B2Y, 96, B2H, "forward 阶段",
     "F —— 前向骨架写在基类 Fp8BlockScaledMMLinearKernel.apply_weights"
     "（BlockScaledMMLinearKernel.py:97），CUDA block 表候选共用", "purple", dashed=True)

FY, FH = B2Y + 29.5, 13.6
fw = 29.0
f1 = step(4.0, FY, fw, FH, "1", "激活量化", "self.quant_fp8(input_2d, ...)",
          ["受类级开关 apply_input_quant 控制（:119）", "DeepGEMM / CUTLASS / Triton  → 走",
           "FlashInfer 混合体设 False → 跳过，", "  直接把 BF16 送进 kernel 内部转换"], "gray", fs=8.0)
f2 = step(35.5, FY, fw, FH, "2", "block GEMM", "self.apply_block_scaled_mm(A,B,As,Bs)",
          ["基类唯一的抽象方法（@abstractmethod）", "",
           "「选后端」这件事，最终只落在", "这一个方法的实现上"], "orange", star=True, fs=8.0)
f3 = step(67.0, FY, fw, FH, "3", "epilogue", "(out+bias).to(out_dtype).view(...)",
          ["+bias → 类型转换 → 恢复形状", "",
           "各后端完全共用，无差别"], "gray", fs=8.0)
arrow(ax, (33.2, FY + FH / 2), (35.3, FY + FH / 2), lw=1.6)
arrow(ax, (64.7, FY + FH / 2), (66.8, FY + FH / 2), lw=1.6)

# 第 2 步向下扇出到五个候选
LY = B2Y + 2.5
ax.add_patch(FancyBboxPatch((14, LY), 72, 25.0, boxstyle="round,pad=0.3,rounding_size=0.8",
                            facecolor="#fffdf8", edgecolor=OR, linewidth=1.4, zorder=3))
ax.text(50, LY + 23.0, "block 表五个候选各自的 apply_block_scaled_mm 落到哪个算子",
        ha="center", va="center", fontsize=10.4, color=OR, family=SANS, zorder=6)
arrow(ax, (50, FY), (50, LY + 25.2), lw=1.8, ls=(0, (4, 2)), color=OR)

rows = [
    ("FlashInferFp8DeepGEMMDynamicBlockScaled", "torch.cond：M < 32 → FlashInfer swapAB，否则 → DeepGEMM", "flashinfer.py:149"),
    ("DeepGemmFp8BlockScaledMMKernel", "torch.ops.vllm.fp8_gemm_nt_op → fp8_gemm_nt", "deep_gemm.py:122"),
    ("CutlassFp8BlockScaledMMKernel", "ops.cutlass_scaled_mm(A, B.T, scale_a=As, scale_b=Bs.T)", "cutlass.py:320"),
    ("TritonFp8BlockScaledMMKernel", "torch.ops.vllm.w8a8_triton_block_scaled_mm_func", "triton.py:173"),
    ("MarlinFP8ScaledMMLinearKernel", "属 FP8ScaledMMLinearKernel 一支，自行重写 apply_weights", "marlin.py:86"),
]
for i, (cls, op, loc) in enumerate(rows):
    y = LY + 19.0 - i * 3.3
    ax.text(16, y, cls, ha="left", va="center", fontsize=8.2, color=PU, family=MONO, zorder=6)
    ax.text(48, y, "→", ha="center", va="center", fontsize=8.6, color="#8a9099", zorder=6)
    ax.text(50, y, op, ha="left", va="center", fontsize=8.2, color="#20242b", family=MONO, zorder=6)
    ax.text(84.5, y, loc, ha="right", va="center", fontsize=7.6, color="#8a9099", family=MONO, zorder=6)

ax.text(50, LY + 2.0,
        "第 1 步的 quant_fp8 由各 kernel 在自己的 __init__ 里建、参数各不相同（use_ue8m0 / column_major_scales / "
        "tma_aligned_scales），\n但底层调的是同一批 csrc 算子 —— 这就是「更换 GEMM 后端不会加速激活量化」的实现层依据。",
        ha="center", va="center", fontsize=8.4, color="#3c4149", family=MONO, zorder=6, linespacing=1.9)

print("ok", file=sys.stderr)
save(fig, "dense_kernel_pick_workflow.png")
