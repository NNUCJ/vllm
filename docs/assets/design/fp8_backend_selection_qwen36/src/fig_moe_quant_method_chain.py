from fig_common import *
from matplotlib.patches import FancyBboxPatch
import sys

# ---------------- MoE：mlp.experts 的 quant_method 装配链（横版，按执行阶段分带） ----------------
fig, ax = new_fig(22, 13.4)
title(ax, "MoE：mlp.experts 从模型代码走到 Fp8MoEMethod",
      "实线框 = 初始化阶段（构造期 → 加载期 → 通信器准备期）    虚线框 = forward 阶段"
      "    ★ = 与 dense 链不同的三处")

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


def step(x, y, w, h, tag, name, loc, lines, color="gray", star=False, fs=8.0):
    fc, ec = C[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.7",
                                facecolor=fc, edgecolor=ec,
                                linewidth=2.6 if star else 1.5, zorder=4))
    ax.text(x + w / 2, y + h - 1.9, f"{tag}  {name}", ha="center", va="center",
            fontsize=10.0, color=ec, family=SANS, zorder=6)
    ax.text(x + w / 2, y + h - 3.8, loc, ha="center", va="center",
            fontsize=7.8, color=ec, family=MONO, zorder=6, alpha=0.85)
    ax.text(x + 1.2, y + h - 5.4, "\n".join(lines), ha="left", va="top",
            fontsize=fs, color="#20242b", family=MONO, zorder=6, linespacing=1.72)
    if star:
        ax.text(x + w - 1.3, y + h - 1.6, "★", ha="center", va="center",
                fontsize=11, color=OR, zorder=7)
    return (x, y, w, h)


# ═══════════ 带 1：构造期 ═══════════
B1Y, B1H = 55.0, 37.0
band(2, B1Y, 96, B1H, "初始化阶段 · 构造期",
     "建层时执行一次；与 dense 链共用同一个 Fp8Config，只是命中另一个 isinstance 分支", "blue")

SW, SG, SX0 = 14.8, 1.4, 3.6
SH, SY = 25.0, B1Y + 2.0
segs = [
    ("①", "MoE block 建层", "qwen3_next.py:102",
     ["gate / shared_expert_gate", "  传 quant_config=None", "  → 压根不问 Fp8Config",
      "shared_expert 走 dense 链", "", "self.experts = FusedMoE(", "  quant_config=...)  :176"],
     "gray", False),
    ("②", "工厂函数，非类", "fused_moe/layer.py:100",
     ["建 FusedMoEConfig      :344", "建 RoutedExperts       :367", "返回 MoERunner         :400",
      "", "权重与 quant_method 都", "挂在 RoutedExperts 上，", "MoERunner 只是调度外壳"],
     "gray", False),
    ("③", "提问点", "routed_experts.py:186",
     ["get_quant_method(", "     self, prefix)", "",
      "返回 None 时兜底成", "UnquantizedFusedMoEMethod", "", "dense 侧是 raise —— 行为", "不对称"],
     "purple", True),
    ("④", "另一个 isinstance 分支", "fp8.py:197",
     ["isinstance(layer,", "          RoutedExperts)", "  跳过 → Unquantized",
      "  mxfp4 → Mxfp4MoEMethod", "  fp8   → Fp8MoEMethod(", "            self, layer)",
      "", "多传了 layer 本身"], "orange", True),
    ("⑤", "选后端", "fp8.py:505 → :527",
     ["select_fp8_moe_backend(", "    config=self.moe, ...)", "",
      "入参是 moe_config，不是", "quant_config —— 同一份", "配置换个并行规模就会", "选出不同后端"],
     "green", False),
    ("⑥", "create_weights", "routed_experts.py:172",
     ["w13 [E, 2N, K] fp8 + scale", "w2  [E,  K, N] fp8 + scale", "E = num_local_experts",
      "", "Qwen3.6 单卡 TP：", "  [256, 1024, 2048]", "  [256, 2048,  512]"], "teal", False),
]
for i, (tag, name, loc, lines, col, star) in enumerate(segs):
    x = SX0 + i * (SW + SG)
    step(x, SY, SW, SH, tag, name, loc, lines, col, star)
    if i:
        arrow(ax, (x - SG + 0.15, SY + SH / 2), (x - 0.15, SY + SH / 2), lw=1.6)

# ═══════════ 带 2：加载期 + 通信器准备期 ═══════════
B2Y, B2H = 27.0, 25.0
band(2, B2Y, 96, B2H, "初始化阶段 · 加载期 → 通信器准备期",
     "权重读完之后仍属初始化；MoE 比 dense 多出右侧这一步", "teal")

step(3.6, B2Y + 1.0, 44, 15.6, "⑦", "process_weights_after_loading", "fp8.py:720",
     ["按 ⑤ 选中的 backend 重排权重布局：",
      "convert_to_fp8_moe_kernel_format（oracle/fp8.py:457）",
      "DeepGEMM 重排 / FlashInfer shuffle / Marlin repack 各走各的"],
     "teal", fs=8.2)
step(52, B2Y + 1.0, 44, 15.6, "⑧", "maybe_init_modular_kernel", "moe_runner.py:858",
     ["由 prepare_communication_buffer_for_model 调用，",
      "时机在「所有权重加载与后处理完成之后」（源码注释）",
      "若需要 prepare/finalize → _replace_quant_method(",
      "     FusedMoEModularMethod.make(...))    :884",
      "quant_method 对象在此被替换一次，原对象存 old_quant_method"],
     "orange", star=True, fs=8.2)
arrow(ax, (25.5, B1Y + 1.0), (25.5, B2Y + 16.8), lw=1.8)
arrow(ax, (47.8, B2Y + 8.8), (51.8, B2Y + 8.8), lw=1.8)
ax.text(74, B2Y + 20.0, "dense 侧没有这一步：其 quant_method 构造完就不再变化",
        ha="center", va="center", fontsize=9.2, color=OR, family=SANS, zorder=6)

# ═══════════ 带 3：forward ═══════════
B3Y, B3H = 3.0, 22.0
band(2, B3Y, 96, B3H, "forward 阶段",
     "每次前向都执行；此时 quant_method 已是 ⑧ 之后的最终形态", "purple", dashed=True)
step(3.6, B3Y + 1.0, 44, 14.0, "⑨", "quant_method.apply", "fp8.py:833 / :809",
     ["MoERunner → routed_experts.quant_method",
      "Modular    apply(topk_weights, topk_ids)   路由在外部完成",
      "Monolithic apply_monolithic(router_logits) 路由在 kernel 内",
      "→ 六步流水线（prepare → permute → GEMM1 → … ）见 3.2"],
     "purple", fs=8.2)
arrow(ax, (25.5, B2Y + 1.0), (25.5, B3Y + 15.2), lw=1.8, ls=(0, (5, 3)))
ax.text(74, B3Y + 9.0,
        "走 Modular 还是 Monolithic，由 ⑤ 选中的 experts 类是否继承\n"
        "FusedMoEExpertsMonolithic 决定（见 2.4.3）；FP8 里只有\n"
        "TrtLlmFp8ExpertsMonolithic 与 CPUExpertsFp8 属 Monolithic。",
        ha="center", va="center", fontsize=9.0, color=PU, family=SANS, zorder=6, linespacing=1.9)

print("ok", file=sys.stderr)
save(fig, "moe_quant_method_chain.png")
