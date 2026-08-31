from fig_common import *
from matplotlib.patches import Rectangle

fig, ax = new_fig(16, 15.2)
title(ax, "FP8 的三个变体：位怎么组成一个真实值",
      "S 符号 · E 指数 · M 尾数    value = (-1)^S x 2^(E - bias) x (1 + M/2^m)")


def bitrow(x, y, w, name, seg, note, col):
    fc, ec = C[col]
    ax.text(x - 1.2, y + 1.4, name, ha="right", va="center", fontsize=12,
            family=SANS, color=ec)
    total = sum(s[1] for s in seg)
    cw = w / total
    cx = x
    for label, n, cc in seg:
        for i in range(n):
            ax.add_patch(Rectangle((cx + i * cw, y), cw, 2.8,
                                   facecolor=cc, edgecolor="#3a3f47",
                                   linewidth=1.0, zorder=3))
        ax.text(cx + n * cw / 2, y + 1.4, label, ha="center", va="center",
                fontsize=10, family=MONO, color="#12151a", zorder=4)
        cx += n * cw
    ax.text(x + w + 1.5, y + 1.4, note, ha="left", va="center", fontsize=9.6,
            family=MONO, color="#4a4a4a", linespacing=1.5)


S, E, M = "#f7d6d6", "#d6e4f7", "#d9f0dd"

ax.text(5, 92, "① 位布局", ha="left", va="top", fontsize=14, family=SANS,
        color="#12151a")

bitrow(16, 85.0, 24, "E4M3 (Float8_e4m3fn)",
       [("S", 1, S), ("E  4 bit", 4, E), ("M 3 bit", 3, M)],
       "bias = 7    max = 448.0     无 inf，只有 0x7F/0xFF 是 NaN\n"
       "权重 / 激活 / KV cache 的默认格式", "blue")

bitrow(16, 79.6, 24, "E5M2 (Float8_e5m2)",
       [("S", 1, S), ("E   5 bit", 5, E), ("M", 2, M)],
       "bias = 15   max = 57344.0   标准 IEEE 排布，有 inf\n"
       "范围换精度，只用于 kv_cache_dtype=fp8_e5m2", "teal")

bitrow(16, 74.2, 24, "E4M3FNUZ (ROCm)",
       [("S", 1, S), ("E  4 bit", 4, E), ("M 3 bit", 3, M)],
       "bias = 8    max = 240.0     没有 -0，0x80 是唯一 NaN\n"
       "gfx94x 的硬件格式", "orange")

f0 = box(ax, 5, 0, 90, None, top=72.4, title="② bias 为什么是这个值", lines=[
    "bias = 2^(指数位数 - 1) - 1        E4M3 -> 2^3-1 = 7        E5M2 -> 2^4-1 = 15",
    "  指数字段是无符号的，只能存 0..15，而浮点数要表示 2^-6 这种小数，实际指数必须能取负——所以减一个偏置。",
    "  用偏置而不是补码：位模式的无符号顺序和数值顺序保持一致，同号浮点数可以直接当整数比大小。",
    "  偏置取在中点，e 的正负范围才大致对称；E == bias 时 e = 0，这就是偏置的定义点。",
    "",
    "  normal    (E != 0):  value = (-1)^S x 2^(E - bias) x (1 + M/8)",
    "  subnormal (E == 0):  value = (-1)^S x 2^(1 - bias) x (M/8)      指数固定为 1-bias，尾数没有隐含前导 1",
], color="gray", ls=9.4, align="left", ts=12.5)

g0 = box(ax, 5, 0, 44, None, top=f0[1] - 3.0,
         title="③ 四个码位，走完整个值域", lines=[
    "0x38 = 0 0111 000  2^(7-7) x 1.0     = 1.0",
    "                   E == bias，定义点",
    "0x08 = 0 0001 000  2^(1-7) x 1.0     = 2^-6",
    "                   最小 normal",
    "0x01 = 0 0000 001  2^(1-7) x (1/8)   = 2^-9",
    "                   最小 subnormal",
    "0x7E = 0 1111 110  2^(15-7) x 1.75   = 448.0",
    "                   最大有限值",
    "0x7F = 0 1111 111                    = NaN",
], color="green", ls=9.4, align="left", ts=12.5)

h0 = box(ax, 51, 0, 44, None, top=f0[1] - 3.0,
         title="④ fn 和 uz 各改了什么", lines=[
    "IEEE 的做法：E 全 1 的一整档让给 inf / NaN",
    "  E4M3 若照办，最大只有 2^7 x 1.875 = 240",
    "",
    "fn = finite  放弃 inf，只留 S.1111.111 当 NaN",
    "  E=1111 档的其余码位全部用来表示普通数",
    "  -> 最大值前推到 2^8 x 1.75 = 448",
    "  用 16 个码位换回接近一倍的动态范围",
    "",
    "uz = unsigned zero  没有 -0，0x80 改作唯一 NaN",
    "  连 S.1111.111 都是普通数，最大码位是 0x7F",
    "  但 bias 大 1，整条数轴下移一倍 -> 240",
], color="purple", ls=9.4, align="left", ts=12.5)

t0 = box(ax, 5, 0, 90, None, top=min(g0[1], h0[1]) - 3.0,
         title="⑤ 三格式对齐着看", lines=[
    "                bias   最大值（码位）      最小 normal   最小 subnormal   inf        NaN",
    "  -------------+------+------------------+-------------+----------------+----------+---------------------------",
    "  E4M3 (fn)       7     448    (0x7E)       2^-6          2^-9            无         仅 S.1111.111",
    "  E5M2           15     57344  (0x7B)       2^-14         2^-16           S.11111.00 S.11111.{01,10,11}",
    "  E4M3FNUZ (uz)   8     240    (0x7F)       2^-7          2^-10           无         仅 0x80",
    "",
    "  E5M2 拿 1 位尾数换 1 位指数：动态范围大两个数量级，相对分辨率从 1/16 掉到 1/8。",
    "  权重和激活有 scale 把窗口对准，不缺动态范围，那 1 位尾数更值钱——所以它们用 E4M3。",
], color="blue", ls=9.4, align="left", ts=12.5)

import sys
print("bottom =", t0[1], file=sys.stderr)
save(fig, "fp8_formats.png")
