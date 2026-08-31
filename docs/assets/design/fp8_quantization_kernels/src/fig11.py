from fig_common import *
from matplotlib.patches import Rectangle
import math

fig, ax = new_fig(17, 20.6)
title(ax, "格式谱系：FP8 站在哪，旁边都有谁",
      "同一套 S/E/M 解码规则，只是位怎么分；分块格式再在上面套一层 scale")

t0 = box(ax, 3, 0, 94, None, top=91.5, title="① 单个数的格式", lines=[
    "  格式            位分配      bias   最大值        最小 normal    半 ulp 相对误差   说明",
    "  --------------+-----------+------+-------------+--------------+----------------+-------------------------------------",
    "  FP32            1+8+23      127    3.4e38        2^-126         2^-24  ≈ 6e-8     累加器、scale 的默认容器",
    "  BF16            1+8+7       127    3.39e38       2^-126         2^-8   ≈ 0.4 %    砍尾数保范围，训练/推理的默认激活",
    "  FP16            1+5+10      15     65504         2^-14          2^-11  ≈ 0.05 %   精度高但范围小，容易溢出",
    "  E5M2            1+5+2       15     57344         2^-14          2^-3   = 12.5 %   FP16 砍到 8 bit，只用于 KV cache",
    "  E4M3            1+4+3       7      448           2^-6           2^-4   = 6.25 %   权重 / 激活 / KV cache 的主力",
    "  E4M3FNUZ        1+4+3       8      240           2^-7           2^-4   = 6.25 %   ROCm gfx94x 的硬件格式",
    "  E2M1 (FP4)      1+2+1       1      6             2^0 = 1        2^-2   = 25 %     只有 {0,±.5,±1,±1.5,±2,±3,±4,±6} 16 个值",
    "  E8M0            0+8+0       127    2^127         2^-127         —（只能表示 2 的幂）  纯指数，不表示 1.5 这种数，专职当 scale",
    "  INT8            定点，无指数  —     127           1（步长恒定）   绝对步长 amax/127  分辨率是绝对的，不随数值大小变",
], color="gray", ls=9.0, align="left", ts=12.5)

# ② 动态范围数轴
r0 = box(ax, 3, 0, 94, 30.0, top=t0[1] - 3.0,
         title="② 动态范围（对数轴）：FP8 的两个变体差了 8 个数量级", lines=None,
         color="blue", ts=12.5)

X0, X1 = 20.0, 93.0
LO, HI = -42.0, 42.0


def px(v):
    return X0 + (v - LO) / (HI - LO) * (X1 - X0)


rows = [
    ("FP32",      -126 * 0.301,  38.53, "#cfe3fb"),
    ("BF16",      -126 * 0.301,  38.53, "#cfe3fb"),
    ("E8M0 (scale)", -127 * 0.301, 38.23, "#e8e3b8"),
    ("FP16",      -14 * 0.301,   4.816, "#c9ecd4"),
    ("E5M2",      -14 * 0.301,   4.759, "#c8ecec"),
    ("E4M3",      -6 * 0.301,    2.651, "#fbe0bf"),
    ("E4M3FNUZ",  -7 * 0.301,    2.380, "#fbe0bf"),
    ("E2M1 (FP4)", 0.0,          0.778, "#f9cfd8"),
]
base = r0[1] + 4.0
bh = 1.9
for k, (name, lo, hi, col) in enumerate(rows):
    yy = base + (len(rows) - 1 - k) * 2.9
    ax.add_patch(Rectangle((px(lo), yy), px(hi) - px(lo), bh, facecolor=col,
                           edgecolor="#3a3f47", linewidth=0.9, zorder=3))
    ax.text(X0 - 1.5, yy + bh / 2, name, ha="right", va="center", fontsize=9.4,
            family=MONO, color="#20242b", zorder=4)
    ax.text(px(hi) + 0.8, yy + bh / 2, f"1e{hi:.0f}", ha="left", va="center",
            fontsize=8.4, family=MONO, color="#6b7178", zorder=4)
for e in range(-40, 41, 10):
    ax.plot([px(e), px(e)], [base - 0.6, base + len(rows) * 2.9 - 1.0],
            color="#d8dce1", lw=0.7, zorder=2)
    ax.text(px(e), base - 1.0, f"1e{e}", ha="center", va="top", fontsize=8.4,
            family=MONO, color="#6b7178")
ax.text(X0 - 1.5, base + len(rows) * 2.9 + 0.4,
        "条形 = [最小 normal, 最大值]", ha="right", va="bottom", fontsize=9.0,
        family=MONO, color="#6b7178")

p0 = box(ax, 3, 0, 94, None, top=r0[1] - 3.0,
         title="③ 分块格式：元素格式 + 每块一个 scale", lines=[
    "  方案            元素      block     scale 格式        额外的全局 scale   一个元素平均占多少 bit",
    "  --------------+---------+---------+-----------------+-----------------+------------------------------",
    "  block-FP8       E4M3      128x128   fp32 或 UE8M0     无                8 + 32/16384  ≈ 8.002",
    "  （激活 1x128）   E4M3      1x128     fp32 或 UE8M0     无                8 + 32/128    = 8.25",
    "  MXFP4           E2M1      32        E8M0（8 bit）      无                4 + 8/32      = 4.25",
    "  NVFP4           E2M1      16        E4M3（8 bit）      per-tensor fp32   4 + 8/16      = 4.5",
    "",
    "  block 越小，scale 越贴合局部分布，但每个元素摊到的 scale 开销越大——4.25 bit 和 4.5 bit 的差别就是这么来的。",
    "  MXFP4 的 scale 是 E8M0（只能是 2 的幂），NVFP4 的 scale 是 E4M3（带 3 位尾数，更贴合，但要再配一个全局 fp32 兜住范围）。",
], color="green", ls=9.0, align="left", ts=12.5)

i0 = box(ax, 3, 0, 94, None, top=p0[1] - 3.0,
         title="④ 为什么 INT8 和 FP8 不能用同一套直觉", lines=[
    "  INT8   量化后的值是 round(x/scale)，scale = amax/127。相邻可表示值的间距恒定 = scale。",
    "         -> 分辨率是绝对的。amax 被一个离群值撑大 10 倍，所有小值的相对误差就跟着涨 10 倍。",
    "         -> 所以 INT8 极度依赖细粒度（per-token / per-channel），粒度就是精度。",
    "",
    "  FP8    量化后仍是浮点，相邻可表示值的间距正比于数值本身（E4M3 在 [2^k, 2^k+1) 内间距是 2^k/8）。",
    "         -> 分辨率是相对的，半 ulp 相对误差恒定 6.25 %，和 scale 无关。",
    "         -> scale 的作用只是把数据挪进 [2^-9, 448] 这个窗口，不是提高分辨率。",
    "         -> 粒度买到的是「不被离群值挤出窗口」，收益比 INT8 小得多，但也不是没有（见第五节的实测）。",
], color="orange", ls=9.0, align="left", ts=12.5)

import sys
print("bottom =", i0[1], file=sys.stderr)
save(fig, "format_zoo.png")
