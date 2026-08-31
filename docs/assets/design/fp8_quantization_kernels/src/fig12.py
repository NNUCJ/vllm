from fig_common import *
from matplotlib.patches import Rectangle
import math

fig, ax = new_fig(16, 13.8)
title(ax, "量化的数学：一个 fp32/bf16 张量怎么变成 FP8 + scale",
      "对称、饱和、无 zero-point；反量化不是独立步骤，它长在 GEMM 的出口上")

a0 = box(ax, 4, 0, 92, None, top=91.0, title="① 量化", lines=[
    "  amax  = max(|x|)                        在一个 group 内（group 的划分方式见下一节）",
    "  scale = max(amax / FP8_MAX, s_min)      FP8_MAX = 448（E4M3）；s_min 是实现给的下限，见 ③",
    "  q     = cvt_e4m3(clamp(x / scale, -FP8_MAX, +FP8_MAX))",
    "",
    "  三个细节：",
    "    · 对称——没有 zero-point。神经网络的权重和激活基本零均值，省下 zero-point 就省掉 GEMM 里的一整项修正。",
    "    · 饱和——必须先 clamp 再转换。E4M3 的溢出结果是 NaN 而不是 inf，不夹住会直接污染整行。",
    "    · 实现上通常传 1/scale 进 kernel，用乘法代替除法（每个元素省一次除法）。",
], color="green", ls=9.2, align="left", ts=12.5)

b0 = box(ax, 4, 0, 92, None, top=a0[1] - 3.0, title="② 反量化：它不在量化 kernel 里，在 GEMM 的出口", lines=[
    "  acc[m,n] = sum_k  A_fp8[m,k] * B_fp8[k,n]        Tensor Core 里乘加，累加器是 fp32",
    "  out[m,n] = acc[m,n] * scale_a[m] * scale_b[n] (+ bias)   ->  再转回 bf16 / fp16",
    "",
    "  所以整条链上只有一次除法（量化时）和一次乘法（epilogue），中间的 GEMM 主循环对 scale 一无所知。",
    "  这也是为什么 FP8 张量在显存里永远和它的 scale 绑在一起——单看 q 是没有意义的。",
    "  分块粒度下 scale 是二维的，epilogue 要按 block 索引去取，主循环也要跟着换 tile 形状，这就是 blockwise kernel 存在的原因。",
], color="blue", ls=9.2, align="left", ts=12.5)

# ③ scale 的作用：窗口示意
c0 = box(ax, 4, 0, 92, 24.5, top=b0[1] - 3.0,
         title="③ scale 干的事是「挪窗口」，不是「调分辨率」", lines=None,
         color="purple", ts=12.5)

X0, X1 = 16.0, 76.0
LO, HI = -10.0, 4.0


def px(v):
    return X0 + (v - LO) / (HI - LO) * (X1 - X0)


import math
W_LO, W_HI = math.log10(2 ** -9), math.log10(448.0)
base = c0[1] + 14.8
bh = 2.6
ax.add_patch(Rectangle((px(W_LO), base), px(W_HI) - px(W_LO), bh,
                       facecolor="#d9f0dd", edgecolor="#1e7d3a", linewidth=1.6,
                       zorder=3))
ax.text((px(W_LO) + px(W_HI)) / 2, base + bh / 2, "E4M3 窗口  2^-9 ~ 448",
        ha="center", va="center", fontsize=9.6, family=MONO, color="#12151a",
        zorder=4)
ax.text(px(W_HI) + 1.5, base + bh / 2, "≈ 5.5 个数量级", ha="left",
        va="center", fontsize=9.0, family=MONO, color="#1e7d3a", zorder=4)
for v in (W_LO, W_HI):
    ax.plot([px(v), px(v)], [base - 12.0, base], color="#1e7d3a", lw=1.1,
            ls="--", zorder=3)

for k, (lab, lo, hi, col, note) in enumerate([
        ("原始激活分布",    -8.0, 2.0, "#cfe3fb", "跨 10 个数量级，两头都超窗"),
        ("÷ 合适的 scale", -5.3, W_HI, "#c9ecd4", "右端贴住 448，左端在窗内"),
        ("÷ 偏大的 scale", -6.3, 1.6, "#f9cfd8", "上方浪费，下方成片下溢")]):
    yy = base - 3.4 - k * 2.9
    ax.add_patch(Rectangle((px(lo), yy), px(hi) - px(lo), 2.0, facecolor=col,
                           edgecolor="#3a3f47", linewidth=1.0, zorder=4))
    ax.text(X0 - 1.5, yy + 1.0, lab, ha="right", va="center", fontsize=9.2,
            family=MONO, color="#20242b", zorder=5)
    ax.text(px(W_HI) + 1.5, yy + 1.0, note, ha="left", va="center",
            fontsize=8.8, family=MONO, color="#6b7178", zorder=5)

for e in range(-10, 5, 2):
    ax.plot([px(e), px(e)], [base - 12.4, base + bh], color="#e3e6ea",
            lw=0.7, zorder=2)
    ax.text(px(e), base - 12.8, f"1e{e}", ha="center", va="top", fontsize=8.4,
            family=MONO, color="#6b7178")

d0 = box(ax, 4, 0, 92, None, top=c0[1] - 3.0,
         title="④ 三个绕不开的数值问题（任何 FP8 实现都要处理）", lines=[
    "  scale 可能是 0        一整行全零或极小时 amax=0，除下去就是 NaN。必须给 scale 一个下限（vLLM 取 1/(FP8_MAX*512)）。",
    "  离群值吃掉窗口        一个异常大的值把 amax 撑大，整行数据被迫左移，小值成片下溢。对策是给 amax 设上界（scale_ub），",
    "                       或者干脆把粒度切细，让离群值只污染它自己那一块。",
    "  融合与非融合要对齐    「先归一化再量化」和「归一化+量化融合成一个 kernel」在 E4M3 的 tie 边界上会给出不同结果——",
    "                       融合版少了一次 round 到 bf16 的中间步骤，反而更准。要逐位对齐就得把这次 round 补回去。",
], color="orange", ls=9.2, align="left", ts=12.5)

import sys
print("bottom =", d0[1], file=sys.stderr)
save(fig, "quant_math.png")
