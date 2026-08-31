import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

CJK = "Noto Sans CJK SC"
MONO = ["DejaVu Sans Mono", CJK]
SANS = [CJK, "DejaVu Sans"]

plt.rcParams["font.family"] = SANS
plt.rcParams["axes.unicode_minus"] = False

# 调色板（与仓库既有设计图一致的柔和风格）
C = {
    "gray":   ("#f5f5f5", "#4a4a4a"),
    "blue":   ("#e8f0fe", "#1a5fb4"),
    "green":  ("#e6f4ea", "#1e7d3a"),
    "purple": ("#f0e8fb", "#6b3fa0"),
    "orange": ("#fdf2e0", "#b06d00"),
    "red":    ("#fdeaea", "#b32020"),
    "teal":   ("#e2f4f4", "#0f6f75"),
    "pink":   ("#fdeaf4", "#a3286e"),
}


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def _upp(ax):
    return 100.0 / (ax.figure.get_size_inches()[1] * 72.0)


def _meas(ax, text, fs, family, linespacing=1.55):
    """真实测量一段文本的高度（axes 数据单位）。"""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    t = ax.text(0, 0, text, fontsize=fs, family=family,
                linespacing=linespacing, va="top", ha="left")
    bb = t.get_window_extent(r)
    t.remove()
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return abs(y1 - y0), abs(x1 - x0)


def box(ax, x, y, w, h=None, title=None, lines=None, color="gray", ts=13,
        ls=10.5, align="center", pad=1.6, lw=1.6, title_color=None,
        alpha=1.0, top=None, gap=1.0):
    """左下角 (x, y)；给 top 则按顶部定位。h=None 时按内容自动算高。"""
    th = _meas(ax, title, ts, SANS)[0] if title else 0.0
    bh = _meas(ax, "\n".join(lines), ls, MONO)[0] if lines else 0.0
    need = pad * 2 + th + (gap if (title and lines) else 0) + bh
    if h is None:
        h = need
    elif need > h + 0.05:
        import sys
        print(f"  [tight] {title!r}: need {need:.1f} given {h:.1f}",
              file=sys.stderr)
    if top is not None:
        y = top - h
    fc, ec = C[color]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha, zorder=2))
    cy = y + h - pad
    if title:
        ax.text(x + w / 2, cy, title, ha="center", va="top", fontsize=ts,
                color=title_color or ec, family=SANS, zorder=3)
        cy -= th + gap
    if lines:
        tx = x + w / 2 if align == "center" else x + pad
        ha = "center" if align == "center" else "left"
        ax.text(tx, cy, "\n".join(lines), ha=ha, va="top", fontsize=ls,
                color="#20242b", family=MONO, linespacing=1.55, zorder=3)
    return (x, y, w, h)


def arrow(ax, p0, p1, color="#5a6270", label=None, ls="-", lw=1.8,
          rad=0.0, fs=9.5, lx=0, ly=0, lcolor=None, ha="center"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=17, linewidth=lw,
        linestyle=ls, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=4))
    if label:
        mx, my = (p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly
        ax.text(mx, my, label, ha=ha, va="center", fontsize=fs,
                color=lcolor or color, family=MONO, zorder=5)


def title(ax, main, sub=None):
    ax.text(50, 99, main, ha="center", va="top", fontsize=19, family=SANS,
            color="#12151a")
    if sub:
        ax.text(50, 95.4, sub, ha="center", va="top", fontsize=11,
                family=MONO, color="#666c76")


def save(fig, name):
    import os
    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), name)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
