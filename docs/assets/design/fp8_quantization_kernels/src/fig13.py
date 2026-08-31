from fig_common import *
from matplotlib.patches import Rectangle

fig, ax = new_fig(16, 13.0)
title(ax, "缩放粒度：一个 scale 管多少个元素",
      "粒度决定 scale 张量的形状，而 scale 的形状决定 GEMM 里怎么取它")

PAL = ["#cfe3fb", "#c9ecd4", "#fbe0bf", "#e6d5f7", "#f9cfd8", "#c8ecec",
       "#e8e3b8", "#dcd6f0"]


def matrix(x, y, w, h, gr, gc, tag_rows="M", tag_cols="K", label=""):
    cw, ch = w / gc, h / gr
    k = 0
    for i in range(gr):
        for j in range(gc):
            ax.add_patch(Rectangle((x + j * cw, y + h - (i + 1) * ch), cw, ch,
                                   facecolor=PAL[k % len(PAL)],
                                   edgecolor="#7b828c", linewidth=0.9, zorder=3))
            k += 1
    for i in range(1, gr * 3):
        yy = y + h - i * h / (gr * 3)
        ax.plot([x, x + w], [yy, yy], color="#ffffff", lw=0.4, zorder=4)
    for j in range(1, gc * 3):
        xx = x + j * w / (gc * 3)
        ax.plot([xx, xx], [y, y + h], color="#ffffff", lw=0.4, zorder=4)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#3a3f47",
                           linewidth=1.4, zorder=5))
    ax.text(x - 0.9, y + h / 2, tag_rows, ha="right", va="center", fontsize=10,
            family=MONO, color="#4a4a4a", rotation=90)
    ax.text(x + w / 2, y - 1.4, tag_cols, ha="center", va="top", fontsize=10,
            family=MONO, color="#4a4a4a")
    if label:
        ax.text(x + w / 2, y + h + 1.0, label, ha="center", va="bottom",
                fontsize=10, family=MONO, color="#4a4a4a")


PW, PH = 45.5, 38.0
LX, RX = 4, 51
TY, BY = 50, 6

box(ax, LX, TY, PW, PH, "① per-tensor", None, color="blue")
matrix(LX + 3, TY + PH - 22.0, 18, 11, 1, 1, "M", "K", "激活 [M,K]")
matrix(LX + 25, TY + PH - 22.0, 15, 11, 1, 1, "K", "N", "权重 [K,N]")
ax.text(LX + 2.5, TY + 12.0, "\n".join([
    "scale 形状      [1]",
    "每元素额外开销  ≈ 0 bit",
    "",
    "整个张量共用一个 scale。任何一个离群值都会",
    "把窗口整体拉走，所有小值一起下溢。",
    "GEMM 侧最省事：epilogue 乘一个标量就行。",
]), ha="left", va="top", fontsize=9.2, family=MONO, color="#20242b",
    linespacing=1.55, zorder=6)

box(ax, RX, TY, PW, PH, "② per-token（激活）/ per-channel（权重）", None, color="green")
matrix(RX + 3, TY + PH - 22.0, 18, 11, 4, 1, "M", "K", "scale [M,1]")
matrix(RX + 25, TY + PH - 22.0, 15, 11, 1, 4, "K", "N", "scale [1,N]")
ax.text(RX + 2.5, TY + 12.0, "\n".join([
    "scale 形状      [M,1] 和 [1,N]",
    "每元素额外开销  32/K bit（K=4096 时 ≈ 0.008）",
    "",
    "一个离群 token 只污染它自己那一行。",
    "GEMM 侧仍然便宜：epilogue 里按行/列广播，",
    "主循环完全不用知道 scale 的存在。",
]), ha="left", va="top", fontsize=9.2, family=MONO, color="#20242b",
    linespacing=1.55, zorder=6)

box(ax, LX, BY, PW, PH, "③ block-wise：激活 1x128，权重 128x128", None, color="purple")
matrix(LX + 3, BY + PH - 22.0, 18, 11, 3, 4, "M", "K", "scale [M, K/128]")
matrix(LX + 25, BY + PH - 22.0, 15, 11, 4, 4, "K", "N", "scale [K/128, N/128]")
ax.text(LX + 2.5, BY + 12.0, "\n".join([
    "scale 形状      [M, K/128] 和 [K/128, N/128]",
    "每元素额外开销  32/128 = 0.25 bit",
    "",
    "离群值只污染同一个 128 元素块。",
    "代价在 GEMM 主循环：沿 K 每走 128 就要换一次",
    "scale，累加器得分段缩放——这是 blockwise kernel",
    "比普通 scaled_mm 难写、也更慢的根本原因。",
]), ha="left", va="top", fontsize=9.2, family=MONO, color="#20242b",
    linespacing=1.55, zorder=6)

box(ax, RX, BY, PW, PH, "④ 挑粒度时真正在权衡什么", None, color="orange")
ax.text(RX + 2.5, BY + PH - 6.0, "\n".join([
    "误差侧（对 FP8 而言）",
    "  粒度细 ≠ 分辨率高。FP8 的相对分辨率恒定，细粒度",
    "  买到的是「离群值不会把小值挤出窗口」。所以 FP8 从",
    "  per-tensor 换到 per-token 的收益，远小于 INT8。",
    "",
    "开销侧",
    "  存储：最多 0.25 bit/元素，可忽略。",
    "  访存：scale 要按 GEMM 的 tile 顺序读，它的内存布局",
    "        （行主序/列主序/对齐）比大小更重要。",
    "  计算：per-tensor / per-token 只动 epilogue；",
    "        block-wise 要改主循环。",
    "",
    "约束侧",
    "  激活和权重的粒度必须能在 GEMM 里对上——这决定了",
    "  哪些 kernel 能用，也决定了同一份权重在不同硬件上",
    "  会落到不同的实现。",
]), ha="left", va="top", fontsize=9.0, family=MONO, color="#20242b",
    linespacing=1.5, zorder=6)

save(fig, "quant_granularity_concept.png")
