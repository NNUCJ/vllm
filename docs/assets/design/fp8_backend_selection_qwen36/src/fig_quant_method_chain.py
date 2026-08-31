from fig_common import *
from matplotlib.patches import FancyBboxPatch
import sys

# ---------------- dense：一层 Linear 的 quant_method 装配链（横版） ----------------
fig, ax = new_fig(22, 11.6)
title(ax, "dense：in_proj_qkvz 从模型代码走到 Fp8LinearMethod",
      "实线框 = 初始化阶段（构造期一次 + 加载期一次）    虚线框 = forward 阶段（每次前向）"
      "    ★ = 整条链上唯一真正查 quant_config 的一步")

GY, PU, OR = "#6b7079", C["purple"][1], C["orange"][1]
LINE = "#5a6270"


def band(x, y, w, h, label, sub, color, dashed=False):
    """阶段泳道容器。"""
    fc, ec = C[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=1,
                                linestyle=(0, (5, 3)) if dashed else "solid", alpha=0.5))
    ax.text(x + 1.4, y + h - 1.9, label, ha="left", va="center", fontsize=12,
            color=ec, family=SANS, zorder=6)
    ax.text(x + 1.4, y + h - 4.4, sub, ha="left", va="center", fontsize=9,
            color=ec, family=MONO, zorder=6, alpha=0.9)


def step(x, y, w, h, num, name, loc, lines, color="gray", star=False):
    fc, ec = C[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.7",
                                facecolor=fc, edgecolor=ec,
                                linewidth=2.6 if star else 1.5, zorder=4))
    ax.text(x + w / 2, y + h - 2.0, f"{num}  {name}", ha="center", va="center",
            fontsize=10.2, color=ec, family=SANS, zorder=6)
    ax.text(x + w / 2, y + h - 4.3, loc, ha="center", va="center",
            fontsize=8.2, color=ec, family=MONO, zorder=6, alpha=0.85)
    ax.text(x + w / 2, y + h - 6.2, "\n".join(lines), ha="center", va="top",
            fontsize=8.4, color="#20242b", family=MONO, zorder=6, linespacing=1.7)
    if star:
        ax.text(x + w - 1.6, y + h - 1.6, "★", ha="center", va="center",
                fontsize=12, color=OR, zorder=7)
    return (x, y, w, h)


# ═══════════ 带 1：构造期 ═══════════
B1Y, B1H = 52.0, 38.0
band(2, B1Y, 96, B1H, "初始化阶段 · 构造期", "建层时执行一次，之后 quant_method 不再变化", "blue")

SW, SH, SG = 14.4, 25.0, 1.6
SX0, SY = 3.6, B1Y + 2.4
steps = [
    ("①", "模型侧透传", "qwen3_5.py:130\ngdn/base.py:41",
     ["取 vllm_config 上的", "quant_config 单例", "", "不复制、不修改"], "gray", False),
    ("②", "算形状与分片", "qwen_gdn_linear_attn.py:566\nlinear.py:606 → :423",
     ["定 output_sizes", "算 TP 分片尺寸", "", "配置原样透传，", "一个字段都不看"], "gray", False),
    ("③", "唯一的提问点", "linear.py:274\nLinearBase.__init__",
     ["quant_config", "  .get_quant_method(", "     self, prefix)", "",
      "传层对象 + 层名"], "purple", True),
    ("④", "两次分流", "fp8.py:175\nFp8Config.get_quant_method",
     ["按层类型 isinstance", "按层名 is_layer_skipped", "  （拆融合名查 648 项）", "",
      "→ Fp8LinearMethod"], "orange", False),
    ("⑤", "配置译成 QuantKey", "fp8.py:280\nFp8LinearMethod.__init__",
     ["weight_block_size", "  [128,128]  译为", "act  (1,128) dynamic", "wgt (128,128) static"],
     "green", False),
    ("⑥", "注册权重 + 选 kernel", "fp8.py:322 → :387\ncreate_weights",
     ["register_parameter", "  weight / scale_inv", "", "init_fp8_linear_kernel", "→ kernel 类定死"],
     "teal", False),
]
boxes = []
for i, (num, name, loc, lines, col, star) in enumerate(steps):
    x = SX0 + i * (SW + SG)
    b = step(x, SY, SW, SH, num, name, loc, lines, col, star)
    boxes.append(b)
    if i:
        arrow(ax, (x - SG + 0.2, SY + SH / 2), (x - 0.2, SY + SH / 2), lw=1.6)

# ═══════════ 带 2：加载期 ═══════════
B2Y, B2H = 30.0, 19.0
band(2, B2Y, 96, B2H, "初始化阶段 · 加载期", "权重读完之后、首次 forward 之前，每层执行一次", "teal")
step(SX0, B2Y + 0.8, 44, 13.0, "⑦", "process_weights_after_loading", "fp8.py:398",
     ["按 ⑥ 选中的 kernel 重排权重布局：",
      "DeepGEMM 路 requant + 打包 UE8M0 scale（见 4.2 节）",
      "CUTLASS 路只调整布局，权重字节不变"], "teal")
arrow(ax, (25.5, B1Y + 1.0), (25.5, B2Y + 14.0), lw=1.8)
ax.text(50.5, B2Y + 7.0,
        "此后权重形态定型。以上 ①–⑦ 全部只发生一次——\n"
        "「选后端」这件事到此结束，运行期不再有任何判定。",
        ha="left", va="center", fontsize=9.6, color=C["teal"][1], family=SANS, zorder=6,
        linespacing=1.9)

# ═══════════ 带 3：forward 期 ═══════════
B3Y, B3H = 6.0, 21.0
band(2, B3Y, 96, B3H, "forward 阶段", "每次前向都执行；此时只是调用构造期定下的那个 kernel", "purple",
     dashed=True)
step(SX0, B3Y + 1.0, 44, 13.6, "⑧", "quant_method.apply", "linear.py:558 → fp8.py:446",
     ["ColumnParallelLinear.forward 调 quant_method.apply(self, x, bias)",
      "→ Fp8LinearMethod.apply → self.fp8_linear.apply_weights",
      "→ 三步骨架：激活量化 → block GEMM → epilogue（见 2.3.2）"], "purple")
arrow(ax, (25.5, B2Y + 1.0), (25.5, B3Y + 14.8), lw=1.8, ls=(0, (5, 3)))
ax.text(50.5, B3Y + 8.2,
        "同一个 quant_method 对象被反复调用，\n"
        "self.fp8_linear 指向的 kernel 类始终是 ⑥ 选定的那一个。",
        ha="left", va="center", fontsize=9.6, color=PU, family=SANS, zorder=6,
        linespacing=1.9)

print("ok", file=sys.stderr)
save(fig, "dense_quant_method_chain.png")
