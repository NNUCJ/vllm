from fig_common import *
from matplotlib.patches import Rectangle
import math

fig, ax = new_fig(17, 24.6)
title(ax, "一个具体的数字：6 个值走完量化与反量化",
      "x = [512, 37, -2.5, 0.125, 0.0195, -0.0007]   一个 token 的 6 个激活值（数值由 torch.float8_e4m3fn 实算）")

a0 = box(ax, 4, 0, 92, None, top=91.0,
         title="第一步：求 amax，定 scale", lines=[
    "amax = max(|x|) = 512.0",
    "scale        = amax / FP8_MAX = 512 / 448 = 1.142857...      per-token 动态量化实际用的",
    "scale_ue8m0  = 2^ceil(log2(1.142857)) = 2.0                  UE8M0 变体：scale 只保留指数",
    "scale_tensor = 4096 / 448 = 9.142857...                      假设同一 batch 里另一行的 amax 是 4096（per-tensor 会被它拖着走）",
], color="gray", ls=9.4, align="left", ts=12.5)

b0 = box(ax, 4, 0, 92, None, top=a0[1] - 3.0,
         title="第二步：q = cvt_e4m3(clamp(x / 1.142857, ±448))，第三步：deq = q * 1.142857", lines=[
    "        x  |        x / scale  |     q (e4m3)  |  bits  |         deq  |   相对误差  |  发生了什么",
    "  ---------+-------------------+---------------+--------+--------------+-------------+------------------------------------------",
    "     512.0 |          448.0    |       448.0   |  0x7E  |      512.0   |     0.00 %  |  正好顶到 E4M3 上限，无损",
    "      37.0 |           32.375  |        32.0   |  0x60  |       36.571 |     1.16 %  |  这一档格点间距是 4（32, 36, 40 ...），落回 32",
    "      -2.5 |           -2.1875 |        -2.25  |  0xC1  |       -2.571 |     2.86 %  |  -2.1875 被舍到最近格点 -2.25",
    "     0.125 |            0.1094 |        0.1094 |  0x1E  |        0.125 |     0.00 %  |  2 的幂，精确可表示",
    "    0.0195 |            0.0171 |        0.0176 |  0x09  |        0.0201 |     3.02 %  |  normal 的最低一档（2^-6 档，格点间距 2^-9）",
    "   -0.0007 |           -0.0006 |        -0.0   |  0x80  |       -0.0   |   100.00 %  |  不到最小 subnormal 2^-9 的一半，舍成 0",
], color="blue", ls=9.2, align="left", ts=12.5)

# ---- 对数数轴：scale 的作用是挪窗口 ----
c0 = box(ax, 4, 0, 92, 22.0, top=b0[1] - 3.0,
         title="scale 干的事：把数据的量级窗口对准 E4M3 的可表示区间", lines=None,
         color="gray", ts=12.5)

X0, X1 = 16.0, 92.0
LO, HI = -4.2, 3.2          # log10 范围


def px(v):
    return X0 + (math.log10(v) - LO) / (HI - LO) * (X1 - X0)


base = c0[1] + 3.0          # 数轴带的底
bandh = 1.6
BANDH = 12.0
# E4M3 可表示区间
sub_lo, norm_lo, top = 2 ** -9, 2 ** -6, 448.0
ax.add_patch(Rectangle((px(sub_lo), base), px(norm_lo) - px(sub_lo), BANDH,
                       facecolor="#f8d9a8", edgecolor="none", zorder=2))
ax.add_patch(Rectangle((px(norm_lo), base), px(top) - px(norm_lo), BANDH,
                       facecolor="#cfeed8", edgecolor="none", zorder=2))
for v, lab, col in [(sub_lo, "2⁻⁹\nsubnormal 下限", "#b06d00"),
                    (norm_lo, "2⁻⁶\nnormal 下限", "#1e7d3a"),
                    (top, "448\nE4M3 上限", "#1e7d3a")]:
    ax.plot([px(v), px(v)], [base, base + BANDH], color=col, lw=1.3,
            ls="--", zorder=3)
    ax.text(px(v), base + BANDH + 0.5, lab, ha="center", va="bottom",
            fontsize=8.8, family=MONO, color=col, linespacing=1.4, zorder=4)

vals = [512.0, 37.0, 2.5, 0.125, 0.0195, 0.0007]
rows = [("原始 |x|", 1.0, "#4a4a4a"),
        ("÷ 1.1429 (per-token)", 1.1428571428571428, "#1a5fb4"),
        ("÷ 9.1429 (per-tensor)", 9.142857142857142, "#a3286e")]
for k, (name, s, col) in enumerate(rows):
    yy = base + 2.8 + (2 - k) * 3.1
    ax.plot([X0 - 2, X1 + 1], [yy, yy], color="#c8ccd2", lw=0.8, zorder=3)
    ax.text(X0 - 3.0, yy, name, ha="right", va="center", fontsize=9.0,
            family=MONO, color=col, zorder=4)
    for v in vals:
        vs = v / s
        if vs < sub_lo:                      # 下溢
            ax.plot(px(vs), yy, marker="x", ms=8, mew=2.0, color="#b32020",
                    zorder=5)
        else:
            ax.plot(px(vs), yy, marker="o", ms=7, color=col, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.8)

for e in range(-4, 4):
    v = 10.0 ** e
    ax.plot([px(v), px(v)], [base - 0.4, base], color="#8b919a", lw=0.8, zorder=3)
    ax.text(px(v), base - 0.8, f"1e{e}", ha="center", va="top", fontsize=8.4,
            family=MONO, color="#6b7178", zorder=4)

ax.text(X0 - 3.0, base + 1.0,
        "x  下溢（成 0）", ha="right", va="center", fontsize=9.0,
        family=MONO, color="#b32020", zorder=4)

# ---- 三种 scale 的结果对比 ----
d0 = box(ax, 4, 0, 55, None, top=c0[1] - 3.0,
         title="换个 scale，同样的 6 个数（反量化后的值 / 相对误差）", lines=[
    "        x |  per-token 1.1429 |    UE8M0 2.0     | per-tensor 9.1429",
    " ---------+-------------------+------------------+-------------------",
    "    512.0 |  512.0    0.00 %  |  512.0    0.00 % |  512.0    0.00 %",
    "     37.0 |   36.571  1.16 %  |   36.0    2.70 % |   36.571  1.16 %",
    "     -2.5 |   -2.571  2.86 %  |   -2.5    0.00 % |   -2.571  2.86 %",
    "    0.125 |    0.125  0.00 %  |    0.125  0.00 % |    0.125  0.00 %",
    "   0.0195 |    0.0201 3.02 %  |    0.0195 0.16 % |    0.0179 8.42 %",
    "  -0.0007 |    0      100  %  |    0      100  % |    0      100  %",
], color="purple", ls=9.0, align="left", ts=12.0)

e0 = box(ax, 62, 0, 34, None, top=c0[1] - 3.0,
         title="三个能从这张表里读出来的结论", lines=[
    "① 中间量级的值（37、-2.5），",
    "   per-token 和 per-tensor 误差完全一样。",
    "   FP8 的相对分辨率是恒定的（尾数 3 bit,",
    "   半个 ulp = 1/16 = 6.25 % 上界），",
    "   不随 scale 变——这点和 INT8 相反。",
    "",
    "② scale 只影响两端：",
    "   偏大 -> 小值往 subnormal 掉（0.0195: 3 % -> 8.4 %）",
    "   偏小 -> 大值被 clamp 截断。",
    "   所以 scale 是「挪窗口」，不是「调分辨率」。",
    "",
    "③ UE8M0 的 scale 是 2 的幂，x/s 只改指数、",
    "   不动尾数，本来就是 fp8 格点的值保持精确",
    "   （-2.5 误差 0）。代价是 amax 只映射到 256",
    "   而不是 448，浪费掉最多 1 bit 的上端范围。",
], color="orange", ls=9.0, align="left", ts=12.0)

f0 = box(ax, 4, 0, 92, None, top=min(d0[1], e0[1]) - 3.0,
         title="反量化在哪：GEMM 出口一次乘回去", lines=[
    "a  = [1.0, -2.0,  3.0, 4.0]   scale_a = 4.0  / 448 = 0.0089286      ->  a_fp8 = [112, -224, 320, 448]",
    "w  = [0.5, -1.25, 2.0, 0.75]  scale_w = 2.0  / 448 = 0.0044643      ->  w_fp8 = [112, -288, 448, 160]",
    "",
    "Tensor Core 里累加的是 fp8 x fp8 -> fp32：acc = 112*112 + (-224)*(-288) + 320*448 + 448*160 = 292096.0",
    "epilogue 出口：out = acc * scale_a * scale_w = 292096 * 0.0089286 * 0.0044643 = 11.643      精确值 = 12.0，误差 2.98 %",
    "误差全部来自 w 里的 -1.25：-1.25/0.0044643 = -280 落到格点 -288，其余三个元素都是精确的。",
], color="teal", ls=9.2, align="left", ts=12.5)

import sys
print("bottom =", f0[1], file=sys.stderr)
save(fig, "worked_example.png")
