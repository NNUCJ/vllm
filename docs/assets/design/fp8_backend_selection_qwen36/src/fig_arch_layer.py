from fig_common import *
from fig_common import _meas
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Rectangle
import sys

fig, ax = new_fig(19, 21.5)
ax.text(50, 99.2, "Qwen3.6-35B-A3B-FP8  一层的数据流与权重位置",
        ha="center", va="top", fontsize=21, family=SANS, color="#12151a")
ax.text(50, 96.6, "层名 / shape / dtype 实测自 checkpoint   ·   hidden_size 2048   ·   40 层 = [linear_attn ×3 + full_attn] ×10",
        ha="center", va="top", fontsize=10.2, family=MONO, color="#666c76")

GDN, ATT, MOE, NRM = C["green"][1], C["orange"][1], C["blue"][1], C["blue"][1]
PU = C["purple"][1]
FP8D, FP8M, BF16 = "#7b3fb5", "#c46a00", "#8a9099"
LINE = "#3c4149"


# ───────────────────────── 基础图元 ─────────────────────────
def rrect(x, y, w, h, txt, col="blue", fs=8.6, fc=None, ec=None, lw=1.4, tc=None, z=4):
    f, e = C[col]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.5",
                                facecolor=fc or f, edgecolor=ec or e, linewidth=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            color=tc or e, family=SANS, zorder=z + 2)
    return (x, y, w, h)


def trap(x, y, w, h, txt, quant=None, fs=8.4, col="green", narrow=0.62):
    """Linear 层：梯形。quant = 'dense' | 'moe' | None(BF16)"""
    f, e = C[col]
    dx = w * (1 - narrow) / 2
    pts = [(x, y), (x + w, y), (x + w - dx, y + h), (x + dx, y + h)]
    ec = {"dense": FP8D, "moe": FP8M}.get(quant, BF16)
    lw = 2.6 if quant else 1.2
    ax.add_patch(Polygon(pts, closed=True, facecolor=f, edgecolor=ec, linewidth=lw, zorder=4))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            color=e, family=MONO, zorder=6)
    if quant:
        ax.text(x - 0.8, y + h / 2, "FP8", ha="right", va="center", fontsize=6.6,
                color=ec, family=MONO, zorder=7)
    return (x, y, w, h)


def circ(x, y, sym, col="#3c4149", r=1.15):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=col, linewidth=1.5, zorder=5))
    ax.text(x, y, sym, ha="center", va="center", fontsize=10, color=col, zorder=6)


def dbox(x, y, w, h, label, col, fs=10.5, lx=None, fc="none", z=2, chip=False, tab=False):
    e = col
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.9",
                                facecolor=fc, edgecolor=e, linewidth=1.7,
                                linestyle=(0, (5, 3)), zorder=z))
    tx = lx if lx is not None else (x + 3.2 if tab else x + 1.8)
    ty = (y + h) if tab else (y + h - 1.6)
    if chip:
        tw = _meas(ax, label, fs, SANS)[1] + 2.4
        ax.add_patch(FancyBboxPatch((tx - 0.9, ty - 1.05), tw, 2.1,
                                    boxstyle="round,pad=0.12,rounding_size=0.5",
                                    facecolor=e, edgecolor="none", zorder=z + 3))
        ax.text(tx, ty, label, ha="left", va="center", fontsize=fs,
                color="white", family=SANS, zorder=z + 4)
    else:
        ax.text(tx, ty, label, ha="left", va="center", fontsize=fs,
                color=e, family=SANS, zorder=z + 4)


def ar(p0, p1, col=LINE, lw=1.2, rad=0.0):
    arrow(ax, p0, p1, color=col, lw=lw, rad=rad, fs=7)


def ln(pts, col=LINE, lw=1.2, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=col, lw=lw, ls=ls, zorder=2, solid_capstyle="round")


def wlab(x, y, t, col=BF16, fs=7.0, ha="left"):
    ax.text(x, y, t, ha=ha, va="center", fontsize=fs, color=col, family=MONO, zorder=7)


def slab(x, y, t, fs=7.2, ha="center"):
    ax.text(x, y, t, ha=ha, va="center", fontsize=fs, color="#5a6270", family=MONO, zorder=7)


# ═════════════════ 左：模型骨架 ═════════════════
SX, SW = 3.0, 19.0
cx = SX + SW / 2

ax.text(cx, 2.0, "Text", ha="center", va="center", fontsize=9.5, family=SANS, color="#12151a")
ar((cx, 2.9), (cx, 4.0))
rrect(SX, 4.2, SW, 3.0, "Tokenizer", "blue")
ar((cx, 7.2), (cx, 8.4))
rrect(SX, 8.6, SW, 3.0, "Embedding", "blue")
slab(cx, 12.6, "[seq_len, 2048]")
ar((cx, 11.6), (cx, 13.6))

# —— 层堆叠 ——
dbox(SX - 1.6, 14.0, SW + 3.2, 60.0, "10 ×   3 layer + 1 layer", "#8a9099", fs=10, fc="#fbfbfc", z=1)

# 3 × LinearAttention
dbox(SX - 0.4, 15.4, SW + 0.8, 27.5, "3 × LinearAttention", GDN, fs=9.2, fc="#eef9f1", z=1, chip=True)
y = 17.2
rrect(SX, y, SW, 2.6, "RMSNorm", "blue", fs=8.2)
ar((cx, y + 2.6), (cx, y + 3.8))
gdn_box = (SX, y + 4.0, SW, 3.4)
ax.add_patch(FancyBboxPatch((SX, y + 4.0), SW, 3.4, boxstyle="round,pad=0.2,rounding_size=0.6",
                            facecolor="#dcf0e3", edgecolor=GDN, linewidth=1.8,
                            linestyle=(0, (4, 2.5)), zorder=4))
ax.text(cx, y + 5.7, "Gated DeltaNet", ha="center", va="center", fontsize=9.2,
        color=GDN, family=SANS, zorder=6)
ln([(SX - 0.2, y + 1.3), (SX - 1.05, y + 1.3), (SX - 1.05, y + 9.6), (cx - 1.3, y + 9.6)])
circ(cx, y + 9.6, "+")
ar((cx, y + 7.4), (cx, y + 8.4))
ar((cx, y + 10.8), (cx, y + 12.0))
rrect(SX, y + 12.2, SW, 2.6, "RMSNorm", "blue", fs=8.2)
ax.add_patch(FancyBboxPatch((SX, y + 16.0), SW, 3.4, boxstyle="round,pad=0.2,rounding_size=0.6",
                            facecolor="#dbe8fb", edgecolor=MOE, linewidth=1.8,
                            linestyle=(0, (4, 2.5)), zorder=4))
ax.text(cx, y + 17.7, "MoE", ha="center", va="center", fontsize=9.2,
        color=MOE, family=SANS, zorder=6)
ar((cx, y + 14.8), (cx, y + 15.9))
ln([(SX - 0.2, y + 13.5), (SX - 1.05, y + 13.5), (SX - 1.05, y + 21.6), (cx - 1.3, y + 21.6)])
circ(cx, y + 21.6, "+")
ar((cx, y + 19.4), (cx, y + 20.4))
ar((cx, y + 22.8), (cx, y + 24.0))

# 1 × FullAttention
dbox(SX - 0.4, 44.0, SW + 0.8, 27.5, "1 × FullAttention", ATT, fs=9.2, fc="#fdf6ea", z=1, chip=True)
y2 = 45.8
rrect(SX, y2, SW, 2.6, "RMSNorm", "blue", fs=8.2)
ar((cx, y2 + 2.6), (cx, y2 + 3.8))
ax.add_patch(FancyBboxPatch((SX, y2 + 4.0), SW, 3.4, boxstyle="round,pad=0.2,rounding_size=0.6",
                            facecolor="#fbe7c9", edgecolor=ATT, linewidth=1.8,
                            linestyle=(0, (4, 2.5)), zorder=4))
ax.text(cx, y2 + 5.7, "Gated Attention", ha="center", va="center", fontsize=9.2,
        color=ATT, family=SANS, zorder=6)
ln([(SX - 0.2, y2 + 1.3), (SX - 1.05, y2 + 1.3), (SX - 1.05, y2 + 9.6), (cx - 1.3, y2 + 9.6)])
circ(cx, y2 + 9.6, "+")
ar((cx, y2 + 7.4), (cx, y2 + 8.4))
ar((cx, y2 + 10.8), (cx, y2 + 12.0))
rrect(SX, y2 + 12.2, SW, 2.6, "RMSNorm", "blue", fs=8.2)
ax.add_patch(FancyBboxPatch((SX, y2 + 16.0), SW, 3.4, boxstyle="round,pad=0.2,rounding_size=0.6",
                            facecolor="#dbe8fb", edgecolor=MOE, linewidth=1.8,
                            linestyle=(0, (4, 2.5)), zorder=4))
ax.text(cx, y2 + 17.7, "MoE", ha="center", va="center", fontsize=9.2,
        color=MOE, family=SANS, zorder=6)
ar((cx, y2 + 14.8), (cx, y2 + 15.9))
ln([(SX - 0.2, y2 + 13.5), (SX - 1.05, y2 + 13.5), (SX - 1.05, y2 + 21.6), (cx - 1.3, y2 + 21.6)])
circ(cx, y2 + 21.6, "+")
ar((cx, y2 + 19.4), (cx, y2 + 20.4))
ar((cx, y2 + 22.8), (cx, y2 + 24.4))

rrect(SX, 75.0, SW, 2.8, "RMSNorm", "blue", fs=8.2)
ar((cx, 77.8), (cx, 79.0))
rrect(SX, 79.2, SW, 2.8, "LM-Head", "blue", fs=8.6)
ar((cx, 82.0), (cx, 83.0))

# ═════════════════ 右下：Gated DeltaNet ═════════════════
PX, PW = 26.0, 71.0
dbox(PX, 3.0, PW, 36.0, "Gated DeltaNet   —— linear_attention 层 ×30", GDN, fc="#f5fcf8", z=1, chip=True, tab=True)
ln([(SX + SW + 0.4, 22.9), (PX, 30.0)], col=GDN, ls=(0, (4, 3)))
ln([(SX + SW + 0.4, 21.2), (PX, 8.0)], col=GDN, ls=(0, (4, 3)))

slab(PX + 34, 5.0, "[seq_len, 2048]")
ln([(PX + 34, 5.6), (PX + 34, 6.6)], col=LINE)
ln([(PX + 3.5, 6.6), (PX + 64, 6.6)], col=LINE)

cols = [(PX + 3.0, "q  16×128"), (PX + 15.0, "k  16×128"), (PX + 27.0, "v  32×128")]
for x0, _ in cols + [(PX + 40.0, ""), (PX + 52.0, ""), (PX + 61.0, "")]:
    pass

# 四个输入投影
trap(PX + 2.0, 7.6, 26.0, 3.0, "in_proj_qkv    [8192, 2048]", "dense", fs=8.0, col="green")
trap(PX + 31.0, 7.6, 9.5, 3.0, "in_proj_b", None, fs=7.6, col="green")
trap(PX + 42.0, 7.6, 9.5, 3.0, "in_proj_a", None, fs=7.6, col="green")
trap(PX + 53.5, 7.6, 12.5, 3.0, "in_proj_z", "dense", fs=7.8, col="green")
wlab(PX + 31.0, 6.0, "[32,2048] BF16", BF16, 6.6)
wlab(PX + 42.0, 6.0, "[32,2048] BF16", BF16, 6.6)
wlab(PX + 53.5, 6.0, "[4096,2048]", FP8D, 6.6)
for x0 in (PX + 15.0, PX + 35.7, PX + 46.7, PX + 59.7):
    ar((x0, 6.6), (x0, 7.5))

# conv1d + 激活
rrect(PX + 2.0, 12.4, 26.0, 2.6, "conv1d（depthwise, k=4）  [8192,1,4] BF16", "green", fs=7.6)
ar((PX + 15.0, 10.6), (PX + 15.0, 12.3))
for x0, lb in ((PX + 5.0, "q"), (PX + 13.0, "k"), (PX + 21.0, "v")):
    ar((x0, 15.0), (x0, 16.4))
rrect(PX + 2.0, 16.6, 14.0, 2.4, "L2 norm（q, k）", "green", fs=7.4)
rrect(PX + 17.5, 16.6, 10.5, 2.4, "SiLU（v）", "green", fs=7.4)

rrect(PX + 31.0, 12.4, 9.5, 2.4, "sigmoid", "green", fs=7.6)
ax.text(PX + 35.7, 15.6, "β", ha="center", va="center", fontsize=10, color=GDN, zorder=6)
ar((PX + 35.7, 10.6), (PX + 35.7, 12.3))
rrect(PX + 42.0, 12.4, 9.5, 2.4, "softplus", "green", fs=7.6)
ax.text(PX + 46.7, 15.6, "g", ha="center", va="center", fontsize=10, color=GDN, zorder=6)
ar((PX + 46.7, 10.6), (PX + 46.7, 12.3))
wlab(PX + 42.0, 11.0, "A_log[32] dt_bias[32] BF16", BF16, 6.4)

# 递推核
rrect(PX + 2.0, 20.6, 49.5, 3.4, "Gated Delta Rule（chunk / fused_recurrent）", "green", fs=9.0)
for x0 in (PX + 9.0, PX + 22.0, PX + 35.7, PX + 46.7):
    ar((x0, 19.0 if x0 < PX + 30 else 16.2), (x0, 20.5))
slab(PX + 26.0, 24.8, "[seq_len, 32, 128]")
ar((PX + 26.0, 24.0), (PX + 26.0, 26.0))

# 门控 + 归一 + 输出
circ(PX + 26.0, 27.2, "×", GDN)
ln([(PX + 59.7, 10.6), (PX + 59.7, 27.2), (PX + 27.2, 27.2)], col=PU)
ax.text(PX + 61.5, 22.0, "z（输出门）", ha="left", va="center", fontsize=7.6, color=PU,
        family=SANS, zorder=6)
ar((PX + 26.0, 28.4), (PX + 26.0, 29.4))
rrect(PX + 14.0, 29.6, 24.0, 2.6, "RMSNorm   norm [128] BF16", "blue", fs=7.8)
ar((PX + 26.0, 32.2), (PX + 26.0, 33.4))
trap(PX + 16.0, 33.6, 20.0, 3.0, "out_proj   [2048, 4096]", "dense", fs=8.0, col="green")
ar((PX + 26.0, 36.6), (PX + 26.0, 37.8))
slab(PX + 34.0, 37.9, "[seq_len, 2048]")

# ═════════════════ 右中：Gated Attention ═════════════════
dbox(PX, 41.0, PW, 25.5, "Gated Attention   —— full_attention 层 ×10", ATT, fc="#fefaf3", z=1, chip=True, tab=True)
ln([(SX + SW + 0.4, 51.5), (PX, 60.0)], col=ATT, ls=(0, (4, 3)))
ln([(SX + SW + 0.4, 49.8), (PX, 45.0)], col=ATT, ls=(0, (4, 3)))

slab(PX + 34, 43.0, "[seq_len, 2048]")
ln([(PX + 34, 43.6), (PX + 34, 44.4)], col=LINE)
ln([(PX + 8, 44.4), (PX + 58, 44.4)], col=LINE)
for x0, nm, sh in ((PX + 8.0, "q_proj", "[8192,2048]"), (PX + 24.0, "k_proj", "[512,2048]"),
                   (PX + 40.0, "v_proj", "[512,2048]"), (PX + 56.0, "gate_proj*", "")):
    if nm == "gate_proj*":
        continue
    trap(x0 - 6.0, 45.4, 12.0, 3.0, f"{nm}  {sh}", "dense", fs=7.6, col="orange")
    ar((x0, 44.4), (x0, 45.3))
rrect(PX + 2.0, 50.0, 12.0, 2.4, "q_norm [256]", "blue", fs=7.4)
rrect(PX + 18.0, 50.0, 12.0, 2.4, "k_norm [256]", "blue", fs=7.4)
ar((PX + 8.0, 48.4), (PX + 8.0, 49.9))
ar((PX + 24.0, 48.4), (PX + 24.0, 49.9))
rrect(PX + 2.0, 53.6, 28.0, 2.4, "RoPE", "orange", fs=8.0)
ar((PX + 8.0, 52.4), (PX + 8.0, 53.5))
ar((PX + 24.0, 52.4), (PX + 24.0, 53.5))
rrect(PX + 2.0, 57.4, 44.0, 3.2, "GQA（16 q-head / 2 kv-head, head_dim 256）", "orange", fs=8.6)
ar((PX + 16.0, 56.0), (PX + 16.0, 57.3))
ar((PX + 40.0, 48.4), (PX + 40.0, 57.3))
slab(PX + 50.0, 59.0, "[seq_len, 16, 256]")
ar((PX + 24.0, 60.6), (PX + 24.0, 61.6))
trap(PX + 12.0, 61.8, 24.0, 3.0, "o_proj   [2048, 4096]", "dense", fs=8.0, col="orange")
ar((PX + 24.0, 64.8), (PX + 24.0, 65.8))
slab(PX + 32.0, 65.9, "[seq_len, 2048]")

# ═════════════════ 右上：MoE ═════════════════
dbox(PX, 68.6, PW, 24.9, "MoE   —— 两种层共用", MOE, fc="#f6f9fe", z=1, chip=True, tab=True)
ln([(SX + SW + 0.4, 39.6), (PX, 72.0)], col=MOE, ls=(0, (4, 3)))
ln([(SX + SW + 0.4, 68.0), (PX, 78.0)], col=MOE, ls=(0, (4, 3)))

slab(PX + 34, 70.4, "[seq_len, 2048]")
ln([(PX + 34, 71.0), (PX + 34, 71.8)], col=LINE)
ln([(PX + 8, 71.8), (PX + 60, 71.8)], col=LINE)

trap(PX + 2.0, 72.8, 12.0, 2.8, "gate（router）", None, fs=7.4, col="blue")
wlab(PX + 2.0, 71.3, "[256,2048] BF16", BF16, 6.6)
ar((PX + 8.0, 71.8), (PX + 8.0, 72.7))
ax.text(PX + 15.5, 74.2, "top-8 / 256", ha="left", va="center", fontsize=7.6,
        color=MOE, family=MONO, zorder=6)

dbox(PX + 2.0, 78.0, 40.0, 12.0, "256 routed experts  →  MoE 路径", FP8M, fs=9.0, fc="#fdf3e6", z=3)
for i, x0 in enumerate((PX + 5.0, PX + 28.0)):
    trap(x0, 79.6, 11.0, 2.6, "gate/up_proj", "moe", fs=7.0, col="orange")
    trap(x0, 83.6, 11.0, 2.6, "down_proj", "moe", fs=7.0, col="orange")
    ar((x0 + 5.5, 82.2), (x0 + 5.5, 83.5))
ax.text(PX + 22.0, 82.8, "· · ·", ha="center", va="center", fontsize=13, color=FP8M, zorder=6)
wlab(PX + 4.0, 87.6, "gate/up [512,2048] ×2   down [2048,512]     每种 ×256", FP8M, 6.8)
ar((PX + 8.0, 75.6), (PX + 10.5, 79.5))
ln([(PX + 8.0, 71.8), (PX + 33.5, 71.8)], col=LINE)
ar((PX + 33.5, 71.8), (PX + 33.5, 79.5))

dbox(PX + 45.0, 78.0, 24.0, 12.0, "shared expert ×1  →  dense 路径", FP8D, fs=9.0, fc="#f6eefc", z=3)
trap(PX + 48.0, 79.6, 18.0, 2.6, "gate_proj / up_proj", "dense", fs=7.2, col="purple")
trap(PX + 48.0, 83.6, 18.0, 2.6, "down_proj", "dense", fs=7.2, col="purple")
ar((PX + 57.0, 82.2), (PX + 57.0, 83.5))
wlab(PX + 47.0, 87.6, "形状与 routed 完全相同，却走另一套后端选择", FP8D, 6.8)
ln([(PX + 60.0, 71.8), (PX + 60.0, 76.4)], col=LINE)
ar((PX + 60.0, 76.4), (PX + 57.0, 79.5))

trap(PX + 45.0, 72.8, 12.0, 2.8, "shared_expert_gate", None, fs=6.8, col="blue")
wlab(PX + 45.0, 71.3, "[1,2048] BF16 → sigmoid", BF16, 6.6)

circ(PX + 34.0, 92.6, "+", MOE)
ln([(PX + 22.0, 90.0), (PX + 22.0, 92.6), (PX + 32.8, 92.6)], col=FP8M)
ln([(PX + 57.0, 90.0), (PX + 57.0, 92.6), (PX + 35.2, 92.6)], col=FP8D)
ar((PX + 34.0, 93.8), (PX + 34.0, 94.8))
slab(PX + 42.0, 94.9, "[seq_len, 2048]")

# ═════════════════ 图例 ═════════════════
lx, ly = 3.0, 84.8
ax.add_patch(FancyBboxPatch((lx - 0.6, ly - 0.6), 20.5, 9.6,
                            boxstyle="round,pad=0.3,rounding_size=0.8",
                            facecolor="#fbfbfd", edgecolor="#c8ccd2", linewidth=1.2, zorder=3))
ax.text(lx + 9.5, ly + 8.0, "量化标注", ha="center", va="center", fontsize=9.6,
        family=SANS, color="#12151a", zorder=6)
for i, (col, t) in enumerate(((FP8D, "紫框 FP8 —— dense 路径"),
                              (FP8M, "橙框 FP8 —— MoE 路径"),
                              (BF16, "细灰框 —— 保持 BF16"))):
    yy = ly + 6.0 - i * 2.0
    dx = 2.2 * (1 - 0.62) / 2
    ax.add_patch(Polygon([(lx + 0.4, yy - 0.7), (lx + 2.6, yy - 0.7),
                          (lx + 2.6 - dx, yy + 0.7), (lx + 0.4 + dx, yy + 0.7)],
                         closed=True, facecolor="#f0f0f4", edgecolor=col,
                         linewidth=2.4 if col != BF16 else 1.2, zorder=6))
    ax.text(lx + 3.4, yy, t, ha="left", va="center", fontsize=7.8,
            color="#20242b", family=SANS, zorder=6)
ax.text(lx + 0.4, ly + 0.2, "梯形 = Linear（唯一可量化的算子）", ha="left", va="center",
        fontsize=7.2, color="#5a6270", family=SANS, zorder=6)

print("ok", file=sys.stderr)
save(fig, "qwen36_quantized_layer_map.png")
