from fig_common import *
from fig_common import _meas
from matplotlib.patches import FancyBboxPatch, Rectangle
import sys

fig, ax = new_fig(17.5, 15.5)
ax.text(50, 98.8, "激活为什么是 [1, 128]，权重为什么是 [128, 128]",
        ha="center", va="top", fontsize=18, family=SANS, color="#12151a")
ax.text(50, 94.6, "fp8.py:305-311  ·  BlockScaledMMLinearKernel.py:53 / :120  ·  input_quant_fp8.py:84"
                  "        以 Qwen3.6 的 in_proj_qkv 为例：K = 2048 = 16 × 128",
        ha="center", va="top", fontsize=9.2, family=MONO, color="#666c76")

BL, PU, OR, GR, TE = C["blue"][1], C["purple"][1], C["orange"][1], C["green"][1], C["teal"][1]
GY, LINE = "#6b7079", "#5a6270"
HL_A, HL_W = "#cfe3fb", "#e6d6f7"


def sect(x, y, w, h, label, col, fc="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.9",
                                facecolor=fc, edgecolor=col, linewidth=1.6, zorder=1))
    tw = _meas(ax, label, 10.5, SANS)[1] + 2.6
    ax.add_patch(FancyBboxPatch((x + 2.6, y + h - 1.1), tw, 2.2,
                                boxstyle="round,pad=0.12,rounding_size=0.5",
                                facecolor=col, edgecolor="none", zorder=5))
    ax.text(x + 3.9, y + h, label, ha="left", va="center", fontsize=10.5,
            color="white", family=SANS, zorder=6)


def grid(x, y, w, h, nc, nr, fc, ec, lwv=0.8, lwh=0.8):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=3))
    for i in range(1, nc):
        ax.plot([x + i * w / nc] * 2, [y, y + h], color=ec, lw=lwv, alpha=0.75, zorder=4)
    for j in range(1, nr):
        ax.plot([x, x + w], [y + j * h / nr] * 2, color=ec, lw=lwh, alpha=0.45, zorder=4)
    return w / nc, h / nr


def txt(x, y, t, fs=8.4, col="#20242b", ha="center", fam=MONO, z=7, rot=0, va="center"):
    ax.text(x, y, t, ha=ha, va=va, fontsize=fs, color=col, family=fam, zorder=z, rotation=rot)


def ar(p0, p1, col=LINE, lw=1.4, ls="-"):
    arrow(ax, p0, p1, color=col, lw=lw, ls=ls, fs=7)


def elbow(p0, p1, col=LINE, lw=1.4, first="v"):
    """正交折线：先竖后横（v）或先横后竖（h），末端带箭头。"""
    (x0, y0), (x1, y1) = p0, p1
    mid = (x0, y1) if first == "v" else (x1, y0)
    ax.plot([x0, mid[0]], [y0, mid[1]], color=col, lw=lw, zorder=2, solid_capstyle="round")
    arrow(ax, mid, p1, color=col, lw=lw, fs=7)


# ══════════ ① 粒度示意（主体） ══════════
S1Y, S1H = 50.0, 41.0
sect(3, S1Y, 94, S1H, "① 两侧的 K 方向切法完全相同", BL, fc="#fafcff")

GX, GW = 10.0, 40.0        # 矩阵区
SX, SW_ = 55.0, 10.0       # scale 区
NC = 4                     # 画 4 组代表 16 组

# —— 激活 A ——
AY, AH = S1Y + 24.5, 10.5
txt(GX + GW / 2, AY + AH + 2.6, "激活 A   [M, 2048]   BF16", 9.6, BL, fam=SANS)
cw, ch = grid(GX, AY, GW, AH, NC, 6, "#eef5fe", BL, lwv=1.4, lwh=0.7)
ax.add_patch(Rectangle((GX, AY + AH - ch), cw, ch, facecolor=HL_A,
                       edgecolor=BL, linewidth=2.2, zorder=5))
txt(GX + cw / 2, AY + AH - ch / 2, "1×128", 7.2, BL)
txt(GX - 1.4, AY + AH / 2, "M = 本步 token 数", 8.0, GY, rot=90)
txt(GX + GW / 2, AY - 2.0, "每 128 列一组 · 共 16 组 · 行方向不合并", 7.8, GY)

asw, ash = grid(SX, AY, SW_, AH, NC, 6, "#dcebfd", BL, lwv=1.4, lwh=0.7)
ax.add_patch(Rectangle((SX, AY + AH - ash), asw, ash, facecolor=HL_A,
                       edgecolor=BL, linewidth=2.2, zorder=5))
txt(SX + SW_ / 2, AY + AH + 2.6, "As  [M, 16]", 9.0, BL, fam=SANS)
ar((GX + GW + 1.0, AY + AH / 2), (SX - 1.0, AY + AH / 2), col=BL)

# —— 权重 W ——
WY, WH = S1Y + 3.5, 10.5
txt(GX + GW / 2, WY + WH + 2.2, "权重 W   [N, 2048]   FP8", 9.6, PU, fam=SANS)
cw2, ch2 = grid(GX, WY, GW, WH, NC, 4, "#f6f0fd", PU, lwv=1.4, lwh=1.4)
ax.add_patch(Rectangle((GX, WY + WH - ch2), cw2, ch2, facecolor=HL_W,
                       edgecolor=PU, linewidth=2.2, zorder=5))
txt(GX + cw2 / 2, WY + WH - ch2 / 2, "128×128", 7.2, PU)
txt(GX - 1.4, WY + WH / 2, "N = 输出维", 8.0, GY, rot=90)
txt(GX + GW / 2, WY - 2.0, "每 128 列一组 · 共 16 组 · 行方向 128 行合并成一块", 7.8, GY)

bsw, bsh = grid(SX, WY, SW_, WH, NC, 4, "#ece2fa", PU, lwv=1.4, lwh=1.4)
ax.add_patch(Rectangle((SX, WY + WH - bsh), bsw, bsh, facecolor=HL_W,
                       edgecolor=PU, linewidth=2.2, zorder=5))
txt(SX + SW_ / 2, WY + WH + 2.6, "Bs  [N/128, 16]", 9.0, PU, fam=SANS)
ar((GX + GW + 1.0, WY + WH / 2), (SX - 1.0, WY + WH / 2), col=PU)

# —— K 方向对齐的竖虚线 ——
for i in range(NC + 1):
    xx = GX + i * cw
    ax.plot([xx, xx], [WY + WH + 0.6, AY - 0.6], color=OR, lw=1.1,
            ls=(0, (3, 2.5)), zorder=2, alpha=0.85)
_kl = "K 方向：同一套 128 划分（2048 = 16 × 128）"
_kw = _meas(ax, _kl, 8.6, SANS)[1] + 3.0
_ky = (AY + WY + WH) / 2
ax.add_patch(Rectangle((GX + GW / 2 - _kw / 2, _ky - 1.5), _kw, 3.0,
                       facecolor="#fafcff", edgecolor="none", zorder=5))
txt(GX + GW / 2, _ky, _kl, 8.6, OR, fam=SANS, z=6)

# —— 右侧说明 ——
RX = 68.0
txt(RX, S1Y + S1H - 5.0, "读图要点", 10.0, "#12151a", ha="left", fam=SANS)
txt(RX, S1Y + S1H - 9.0,
    "· 高亮的那一格 = 一个 scale\n"
    "  A 侧覆盖 1×128 = 128 个元素\n"
    "  W 侧覆盖 128×128 = 16384 个",
    8.2, "#20242b", ha="left", va="top")
txt(RX, S1Y + S1H - 17.5,
    "· 两侧列数都是 16，边界重合\n"
    "  ⇒ 这是 GEMM 能整段累加的前提",
    8.2, OR, ha="left", va="top")
txt(RX, S1Y + S1H - 24.0,
    "· 差别只在行方向：\n"
    "  A 逐行独立，W 每 128 行合并\n"
    "  ⇒ scale 数量差 128 倍",
    8.2, "#20242b", ha="left", va="top")
txt(RX, S1Y + S1H - 32.5,
    "组宽不是独立配置项：\n"
    "fp8.py:307 直接取\n"
    "weight_block_size[0]",
    8.0, GY, ha="left", va="top")

# ══════════ ② K 方向为什么必须对齐 ══════════
S2Y, S2H = 26.0, 21.0
sect(3, S2Y, 45.5, S2H, "② K 方向：数学前提", OR, fc="#fffaf3")
txt(5.6, S2Y + S2H - 5.0,
    "GEMM 按 K 切 128 一片，片内先点积再乘 scale：\n"
    "\n"
    "  acc += (Aq[m,kt] · Wq[n,kt]ᵀ) × As[m,kt] × Bs[n,kt]\n"
    "\n"
    "成立的前提是片内 As、Bs 都是常数，\n"
    "反量化因子才能提到求和号外：\n"
    "  Σₖ (aq·s_a)(bq·s_b) = s_a·s_b·Σₖ aq·bq\n"
    "\n"
    "组宽 ≠ 128 → scale 片内不再是常数 → 只能逐元素\n"
    "反量化，FP8 TensorCore 的整段点积就白费了。",
    8.0, "#20242b", ha="left", va="top")

# ══════════ ③ M 方向为什么逐行 ══════════
sect(51.5, S2Y, 45.5, S2H, "③ M 方向：精度选择", GR, fc="#f6fcf8")
txt(54.1, S2Y + S2H - 5.0,
    "激活：逐 token 幅值差异大、常有离群值。\n"
    "  [1,·] 让离群 token 的 absmax 只污染自己那一组；\n"
    "  代价极小 —— scale 只有 M × 16 个。\n"
    "\n"
    "权重：分布平稳，且能离线慢慢挑最优 scale。\n"
    "  128 行共享一个 scale 精度损失可控，\n"
    "  换来 128 倍的 scale 存储节省，\n"
    "  GEMM 内层也少一次逐列乘法。\n"
    "\n"
    "一句话：细在需要的地方，粗在不吃亏的地方。",
    8.0, "#20242b", ha="left", va="top")

# ══════════ ④ 谁在运行期做这件事 ══════════
S4Y, S4H = 4.0, 19.0
sect(3, S4Y, 94, S4H, "④ 这两个粒度分别由谁产生", TE, fc="#f6fcfc")

txt(6.0, S4Y + S4H - 5.2, "激活侧 As —— 每次 forward 现算", 9.4, BL, ha="left", fam=SANS)
txt(6.0, S4Y + S4H - 8.4,
    "构造期定死  QuantFP8(static=False, group_shape=(1,128))\n"
    "运行期调用  self.quant_fp8(input_2d)  →  csrc per_token_group_fp8_quant(_packed)\n"
    "            amax = max(|x[m, 128g:128g+128]|, eps);  s = amax/448;  q = clamp(x/s, ±448)",
    7.8, "#20242b", ha="left", va="top")

txt(53.0, S4Y + S4H - 5.2, "权重侧 Bs —— 离线量化，加载后不变", 9.4, PU, ha="left", fam=SANS)
txt(53.0, S4Y + S4H - 8.4,
    "checkpoint 里就是 weight_scale_inv [N/128, K/128]\n"
    "加载期只按后端调布局（第 4 章）；DeepGEMM on Blackwell\n"
    "会再做一次 UE8M0 重量化 + 打包，其余后端权重字节不动。",
    7.8, "#20242b", ha="left", va="top")

ax.plot([50.0, 50.0], [S4Y + 1.6, S4Y + S4H - 3.4], color=GY, lw=1.0,
        ls=(0, (3, 3)), zorder=3)

print("ok", file=sys.stderr)
save(fig, "quant_fp8_block_workflow.png")
