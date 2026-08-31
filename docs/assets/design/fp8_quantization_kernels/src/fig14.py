"""16.2 的三张配图：权重侧、激活侧、K 向对齐，各自独立成图。

一张画布塞下三件事会让每一件都看不清，所以拆成三张，正文各自就近讲解：
  block_quant_weight.png      权重侧 128×128 块 ↔ weight_scale_inv
  block_quant_activation.png  激活侧 1×128 组 ↔ As，以及动态量化怎么构建
  block_quant_k_align.png     两侧 K 向为什么必须是同一套 128 切法
"""

import sys

from fig_common import *
from matplotlib.patches import Rectangle


def tools(fig, ax):
    """返回绑定到这一张图的绘图辅助函数。"""

    def ratio():
        """1 个 x 单位 : 1 个 y 单位 的物理长度比，用来把方格画成正方形。"""
        bb = ax.get_window_extent(fig.canvas.get_renderer())
        return bb.width / bb.height

    def ppu():
        """每个 y 单位折合多少 pt，用于按字号推行高。"""
        return 0.5544 * fig.get_size_inches()[1]

    def grid(x, y, w, h, nc, nr, color, lw=0.7):
        fc, ec = C[color]
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                               linewidth=1.6, zorder=2))
        cw, ch = w / nc, h / nr
        for i in range(1, nc):
            ax.plot([x + i * cw] * 2, [y, y + h], color=ec, lw=lw,
                    zorder=3, alpha=0.55)
        for j in range(1, nr):
            ax.plot([x, x + w], [y + j * ch] * 2, color=ec, lw=lw,
                    zorder=3, alpha=0.55)
        return cw, ch

    def lab(x, y, t, fs=10, color="#20242b", ha="center", fam=MONO, rot=0):
        ax.text(x, y, t, fontsize=fs, color=color, ha=ha, va="center",
                family=fam, rotation=rot, zorder=6)

    def cell(x, y, w, h, color="orange", lw=2.0, z=5):
        fc, ec = C[color]
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                               linewidth=lw, zorder=z))

    def zoom(xa, xb, y0, x1, ytop, ybot):
        """高亮格先沿本行引到矩阵右缘，再扇形连到放大子图，避免斜穿矩阵。"""
        ax.plot([xa, xb], [y0, y0], color=C["orange"][1], lw=1.0,
                ls=(0, (4, 3)), alpha=0.9, zorder=6)
        for yb in (ytop, ybot):
            ax.plot([xb, x1], [y0, yb], color=C["orange"][1], lw=1.0,
                    ls=(0, (4, 3)), alpha=0.85, zorder=3)

    def head(main, sub, y=98, ts=16, ss=9.5):
        u = ppu()
        ax.text(50, y, main, ha="center", va="top", fontsize=ts,
                family=SANS, color="#12151a")
        y2 = y - ts * 1.45 / u
        ax.text(50, y2, sub, ha="center", va="top", fontsize=ss,
                family=MONO, color="#666c76")
        return y2 - ss * 2.2 / u

    return ratio, grid, lab, cell, zoom, head


def check(name, bottom):
    print(f"  {name}: bottom y = {bottom:.1f}", file=sys.stderr)
    assert 0.5 < bottom < 14, f"{name} 版面留白失衡（bottom={bottom:.1f}）"


# ══════════════════════════════════════════════════════════════════
# 图 A：权重侧 —— 真实块划分 + 块 (0,0) 放大 + weight_scale_inv
# ══════════════════════════════════════════════════════════════════
def fig_weight():
    fig, ax = new_fig(16, 5.6)
    ratio, grid, lab, cell, zoom, head = tools(fig, ax)
    top = head("权重侧：每 128×128 一个 scale，量化在 checkpoint 里已经做完",
               "实测 layers.0.linear_attn.in_proj_qkv · "
               "weight F8_E4M3 [8192,2048] · weight_scale_inv BF16 [64,16]")
    R = ratio()
    gt = top - 8.5                       # 让出两行列标题

    # —— 左：weight 的真实块划分，64 × 16 = 1024 块，一格即一个 128×128 块 ——
    gx, gw = 4.0, 5.5
    gh = (8192 / 2048) * gw * R          # 真实长宽比 4:1，单块恰是正方形
    gy = gt - gh
    cw, ch = grid(gx, gy, gw, gh, 16, 64, "purple", lw=0.3)
    cell(gx, gt - ch, cw, ch, lw=1.6)
    lab(gx + gw / 2, top - 1.5, "weight  F8_E4M3", fs=10.5)
    lab(gx + gw / 2, top - 4.3, "[8192, 2048]", fs=9.5, color="#555")
    lab(gx + gw / 2, gy - 3.0, "2048 列 = 16 块", fs=9, color="#555")
    lab(gx + gw / 2, gy - 6.0, "共 1024 个块", fs=9, color="#7a6a95")
    lab(gx - 1.7, gy + gh / 2, "8192 行 = 64 块", fs=9, color="#555", rot=90)

    # —— 中：块 (0,0) 放大，实测数据 ——
    b = box(ax, 14, 0, 43, None, top=gt, color="orange",
            title="块 (0,0) 放大：16384 个元素共用 1 个 scale", ts=12, lines=[
        "scale = weight_scale_inv[0,0] = 1.745224e-04",
        "",
        "前 4 个 fp8 值  -104.0    +72.0    -1.625   +20.0",
        "× scale ⇒ 权重  -0.018150 +0.012566 -0.000284 +0.003490",
        "",
        "块内 |wq|.max = 448（E4M3 满量程）",
        "  ⇒ scale = 块内原始 amax / 448 = 0.078186 / 448",
    ], ls=10.0, align="left")
    zoom(gx + cw, gx + gw, gt - ch / 2, b[0], b[1] + b[3], b[1])

    # —— 右：weight_scale_inv，形状与左侧块划分逐格对应 ——
    sx, sw = 62.0, 5.5
    sh = (64 / 16) * sw * R
    sy = gt - sh
    scw, sch = grid(sx, sy, sw, sh, 16, 64, "teal", lw=0.3)
    cell(sx, gt - sch, scw, sch, lw=1.6)
    lab(sx + sw / 2, top - 1.5, "weight_scale_inv", fs=10.5)
    lab(sx + sw / 2, top - 4.3, "BF16  [64, 16]", fs=9.5, color="#555")
    lab(sx + sw / 2, sy - 3.0, "共 1024 个 scale", fs=9, color="#555")
    lab(sx + sw / 2, sy - 6.0, "与左侧的块一一对应", fs=9, color="#0f6f75")
    arrow(ax, (b[0] + b[2] + 0.8, b[1] + b[3] * 0.74),
          (sx - 1.2, gt - sch / 2))

    # —— 最右：scale 左上 3×3 实测值 ——
    zx, zw = 72.0, 22.0
    zh = zw * R
    zy = gt - zh
    zcw, zch = grid(zx, zy, zw, zh, 3, 3, "teal")
    cell(zx, gt - zch, zcw, zch, lw=2.2, z=4)
    vals = [["1.745", "1.974", "1.850"],
            ["2.880", "2.089", "1.993"],
            ["2.089", "2.079", "1.621"]]
    for r in range(3):
        for c in range(3):
            col = C["orange"][1] if (r == 0 and c == 0) else "#20242b"
            lab(zx + (c + 0.5) * zcw, gt - (r + 0.5) * zch,
                vals[r][c], fs=11, color=col)
    lab(zx + zw / 2, top - 1.5, "左上 3×3 实测值（单位 1e-4）", fs=10)
    lab(zx + zw / 2, zy - 3.0, "最大 / 最小相差 6.9 倍", fs=9, color="#555")
    zoom(sx + scw, sx + sw, gt - sch / 2, zx, gt, zy)

    check("weight", min(gy - 6.0, b[1], zy - 3.0))
    save(fig, "block_quant_weight.png")


# ══════════════════════════════════════════════════════════════════
# 图 B：激活侧 —— 1×128 分组，以及动态量化怎么构建
# ══════════════════════════════════════════════════════════════════
def fig_activation():
    fig, ax = new_fig(16, 4.6)
    ratio, grid, lab, cell, zoom, head = tools(fig, ax)
    top = head("激活侧：每 1×128 一个 scale，每次 forward 现算",
               "激活 x BF16 [M, 2048] · As [M, 16] · "
               "M = 本步 token 数，组宽 128 取自 weight_block_size[0]")
    gt = top - 4.5

    # —— 左：激活矩阵，一行被横向切成 16 组（示意 8 组）——
    ax_, aw_, ah_ = 5.0, 40.0, 20.0
    ay_ = gt - ah_
    acw, ach = grid(ax_, ay_, aw_, ah_, 8, 5, "blue")
    cell(ax_, gt - ach, aw_, ach, color="green", lw=1.6, z=4)   # 整行 = 一个 token
    cell(ax_, gt - ach, acw, ach, lw=2.2, z=5)                  # 行内一组 = [1,128]
    lab(ax_ + acw / 2, gt - ach / 2, "1×128", fs=8, color=C["orange"][1])
    lab(ax_ + aw_ / 2, top - 1.8, "激活 x  BF16  [M, 2048]", fs=10.5)
    lab(ax_ + aw_ / 2, ay_ - 3.5, "2048 列 = 16 组 × 128（示意 8 组）",
        fs=9, color="#555")
    lab(ax_ + aw_ / 2, ay_ - 7.0, "行方向绝不合并：一个 token 一套 scale",
        fs=9, color="#5a7f92")
    lab(ax_ - 1.7, ay_ + ah_ / 2, "M = token 数", fs=9, color="#555", rot=90)

    # —— 中：As ——
    sx2, sw2 = 54.0, 16.0
    s2cw, s2ch = grid(sx2, ay_, sw2, ah_, 8, 5, "green")
    cell(sx2, gt - s2ch, s2cw, s2ch, lw=2.2, z=5)
    lab(sx2 + sw2 / 2, top - 1.8, "As  [M, 16]", fs=10.5)
    lab(sx2 + sw2 / 2, ay_ - 3.5, "每 token 16 个 scale", fs=9, color="#555")
    arrow(ax, (ax_ + aw_ + 1.0, gt - ach / 2), (sx2 - 1.2, gt - s2ch / 2),
          label="每组一个 scale", lx=0, ly=2.6, fs=9.5)

    # —— 右：为什么行方向不合并 ——
    blue = box(ax, 74, 0, 23, None, top=gt, color="blue",
               title="为何行方向逐 token", ts=12, lines=[
        "离群 token 的 absmax 只",
        "污染自己那一组；scale",
        "仅 M×16 个，开销可忽略",
    ], ls=9.5, align="left")

    # —— 下：动态量化的两段式构建 ——
    y2 = min(ay_ - 10.5, blue[1] - 3.5)
    box(ax, 4, 0, 45, None, top=y2, color="teal",
        title="构造期：组宽在 kernel __init__ 定死", ts=12, lines=[
        "self.quant_fp8 = QuantFP8(",
        "    static=False,          # 无预存 scale",
        "    group_shape=(1, 128))  # = weight_block_size[0]",
        "BlockScaledMMLinearKernel.py:53，运行期不再变",
    ], ls=9.5, align="left")

    d = box(ax, 52, 0, 45, None, top=y2, color="green",
            title="运行期：每次 forward 现算", ts=12, lines=[
        "对每行 m、每组 g：",
        "  amax   = max(|x[m, 128g : 128g+128]|, eps)",
        "  s[m,g] = amax / 448     # 与权重侧同一个公式",
        "  q      = clamp(x / s, ±448).to(fp8)",
    ], ls=9.5, align="left")

    check("activation", d[1])
    save(fig, "block_quant_activation.png")


# ══════════════════════════════════════════════════════════════════
# 图 C：两侧 K 向必须是同一套 128 切法
# ══════════════════════════════════════════════════════════════════
def fig_k_align():
    fig, ax = new_fig(16, 4.4)
    ratio, grid, lab, cell, zoom, head = tools(fig, ax)
    top = head("K 向对齐：两侧的 128 切法必须重合，否则 scale 提不出求和号",
               "acc += (Aq[m,kt] · Wq[n,kt]ᵀ) × As[m,kt] × Bs[n,kt]"
               "    ·    kt = 0 … 15")

    KT = 5                                # 高亮第几片，任取
    gx, gw, h = 11.0, 72.0, 10.0
    y_a, y_w = top - 14, top - 30

    def strip(y, color, name, shape, sname):
        cw, _ = grid(gx, y, gw, h, 16, 3, color)
        cell(gx + KT * cw, y, cw, h, lw=2.0, z=4)
        lab(gx - 1.5, y + h / 2 + 1.6, name, fs=11, ha="right")
        lab(gx - 1.5, y + h / 2 - 2.2, shape, fs=8.5, ha="right", color="#555")
        lab(gx + gw + 1.5, y + h / 2, f"{sname}[·, kt]", fs=9.5, ha="left",
            color=C["orange"][1])
        return cw

    cw = strip(y_a, "blue", "Aq", "[M, 2048]", "As")
    strip(y_w, "purple", "Wq", "[N, 2048]", "Bs")

    for dx in (KT * cw, (KT + 1) * cw):   # 两条竖虚线标出「同一片」
        ax.plot([gx + dx] * 2, [y_w, y_a + h], color=C["orange"][1],
                lw=1.0, ls=(0, (4, 3)), alpha=0.85, zorder=6)
    lab(gx + (KT + 0.5) * cw, y_a + h + 2.4, "第 kt 片：两侧各 128 列",
        fs=9, color=C["orange"][1])
    lab(gx + gw * 0.74, y_a - 3.2, "同一套 K 划分：2048 = 16 × 128",
        fs=9.5, color="#555")

    b = box(ax, 6, 0, 88, None, top=y_w - 5.0, color="green",
            title="片内 As、Bs 是常数，反量化因子才能提到求和号外面", ts=12,
            lines=[
        "Σₖ (aq·s_a)(bq·s_b) = s_a · s_b · Σₖ aq·bq"
        "      ⇒ FP8 TensorCore 整段点积，每片只乘一次 scale",
        "",
        "组宽不等 ⇒ scale 在片内不再是常数 ⇒ 只能逐元素反量化，FP8 点积失去意义",
        "所以激活组宽不是独立配置项：fp8.py:307 直接取 weight_block_size[0]",
    ], ls=10.0, align="left")

    check("k_align", b[1])
    save(fig, "block_quant_k_align.png")


fig_weight()
fig_activation()
fig_k_align()
