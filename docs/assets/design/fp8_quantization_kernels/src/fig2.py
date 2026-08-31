from fig_common import *
from matplotlib.patches import Rectangle

fig, ax = new_fig(16, 12.5)
title(ax, "四种缩放粒度：scale 的形状决定走哪个 kernel",
      "csrc/libtorch_stable/quantization/w8a8/fp8/{common.cu, per_token_group_quant.cu}")

PAL = ["#cfe3fb", "#c9ecd4", "#fbe0bf", "#e6d5f7", "#f9cfd8", "#c8ecec",
       "#e8e3b8", "#dcd6f0"]


def matrix(x, y, w, h, gr, gc, tag_rows="M", tag_cols="K", label=""):
    """gr/gc = 行/列方向的分组数。"""
    cw, ch = w / gc, h / gr
    k = 0
    for i in range(gr):
        for j in range(gc):
            ax.add_patch(Rectangle((x + j * cw, y + h - (i + 1) * ch), cw, ch,
                                   facecolor=PAL[k % len(PAL)],
                                   edgecolor="#7b828c", linewidth=0.9,
                                   zorder=3))
            k += 1
    # 细网格：示意 tile 内的元素
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


def panel(px, py, pw, ph, color, head, mat, body):
    box(ax, px, py, pw, ph, head, None, color=color, ts=13)
    mat(px, py)
    ax.text(px + 2.2, py + ph - 27.0, "\n".join(body), ha="left", va="top",
            fontsize=9.6, family=MONO, color="#20242b", linespacing=1.55,
            zorder=6)


PW, PH = 45.5, 40.0
LX, RX = 4, 51
TY, BY = 50, 5

# ① per-tensor
box(ax, LX, TY, PW, PH, "① per-tensor    scale.numel() == 1", None, color="blue")
matrix(LX + 3, TY + PH - 22.0, 18, 11, 1, 1, "M", "K", "激活 [M,K]")
matrix(LX + 25, TY + PH - 22.0, 15, 11, 1, 1, "K", "N", "权重 [K,N]")
ax.text(LX + 2.5, TY + 15.5, "\n".join([
    "static_scaled_fp8_quant(out, input, scale)      common.cu:183",
    "dynamic_scaled_fp8_quant(out, input, scale!)    common.cu:337",
    "",
    "动态版分两趟：segmented_max_reduction_strided 先用",
    "atomicMaxFloat 求全张量 amax，再整体量化。",
    "两趟 + 原子操作，是四种粒度里最慢的一种。",
]), ha="left", va="top", fontsize=9.6, family=MONO, color="#20242b",
    linespacing=1.55, zorder=6)

# ② per-token / per-channel
box(ax, RX, TY, PW, PH, "② per-token（激活）+ per-channel（权重）", None, color="green")
matrix(RX + 3, TY + PH - 22.0, 18, 11, 4, 1, "M", "K", "scale [M,1]")
matrix(RX + 25, TY + PH - 22.0, 15, 11, 1, 4, "K", "N", "scale [1,N]")
ax.text(RX + 2.5, TY + 15.5, "\n".join([
    "dynamic_per_token_scaled_fp8_quant(out, in, scale!, ub)",
    "                                       common.cu:383",
    "",
    "一个 block 干一行：cub::BlockReduce 求 amax -> 定 scale",
    "-> 就地量化，单趟、无原子操作。",
    "权重的 per-channel scale 由 checkpoint 直接给出。",
]), ha="left", va="top", fontsize=9.6, family=MONO, color="#20242b",
    linespacing=1.55, zorder=6)

# ③ block 128x128
box(ax, LX, BY, PW, PH, "③ block-wise：激活 1x128 + 权重 128x128", None, color="purple")
matrix(LX + 3, BY + PH - 22.0, 18, 11, 3, 4, "M", "K", "scale [M, K/128]")
matrix(LX + 25, BY + PH - 22.0, 15, 11, 4, 4, "K", "N", "scale [K/128, N/128]")
ax.text(LX + 2.5, BY + 15.5, "\n".join([
    "per_token_group_fp8_quant(input, out_q!, out_s!, 128, ...)",
    "                              per_token_group_quant.cu:193",
    "",
    "DeepSeek-V3 / V3.1 系列的默认粒度。一个 16 线程小组处理",
    "一个 group，先搬进 shared memory 再量化，避免二次读 DRAM。",
    "scale 可按列主序写出（IS_COLUMN_MAJOR），直接喂给 GEMM。",
]), ha="left", va="top", fontsize=9.6, family=MONO, color="#20242b",
    linespacing=1.55, zorder=6)

# ④ UE8M0
box(ax, RX, BY, PW, PH, "④ UE8M0：scale 只保留指数", None, color="orange")
matrix(RX + 3, BY + PH - 22.0, 18, 11, 3, 4, "M", "K", "scale [M, K/128]")
ax.text(RX + 24, BY + PH - 12.0, "\n".join([
    "float scale",
    "  -> 2^ceil(log2(s))",
    "  -> 8-bit 指数",
    "4 个打包成 1 个 int32",
]), ha="left", va="top", fontsize=9.6, family=MONO, color="#4a4a4a",
    linespacing=1.6, zorder=6)
ax.text(RX + 2.5, BY + 15.5, "\n".join([
    "per_token_group_fp8_quant_packed(...)   torch_bindings.cpp:21",
    "  -> per_token_group_quant.cu:302 的寄存器常驻快路径",
    "",
    "为 DeepGEMM 准备：scale 用位运算取指数（与 exp2f(ceilf(",
    "log2f(x))) 逐位一致），每线程 16 个元素全程留在寄存器，",
    "不过 shared memory。SM90+ 还开了 PDL 启动重叠。",
]), ha="left", va="top", fontsize=9.6, family=MONO, color="#20242b",
    linespacing=1.55, zorder=6)

save(fig, "quant_granularity.png")
