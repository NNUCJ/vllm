from fig_common import *
from fig_common import _meas
from matplotlib.patches import FancyBboxPatch, Rectangle
import sys

fig, ax = new_fig(22, 10.0)
title(ax, "UE8M0 打包：scale 从 fp32 到 int32 的四步形态变化",
      "数值取自 layers-0.linear_attn.in_proj_qkv.weight_scale_inv 的真实前 4 个 K 向块   ·   "
      "权重侧在加载期做一次，激活侧每次 forward 现做")

TEAL, OR, PU, GY, BL = C["teal"][1], C["orange"][1], C["purple"][1], "#6b7079", C["blue"][1]


def panel(x, y, w, h, tag, name, sub, color):
    fc, ec = C[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
                                facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=2, alpha=0.5))
    ax.text(x + w / 2, y + h - 2.0, f"{tag}  {name}", ha="center", va="center",
            fontsize=11, color=ec, family=SANS, zorder=6)
    ax.text(x + w / 2, y + h - 4.2, sub, ha="center", va="center",
            fontsize=8.2, color=ec, family=MONO, zorder=6, alpha=0.9)
    return ec


def cell(x, y, w, h, txt, fc="white", ec=GY, fs=8.0, tc="#20242b", lw=1.2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=4))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            color=tc, family=MONO, zorder=6)


def note(x, y, t, col=GY, fs=8.0, ha="center"):
    ax.text(x, y, t, ha=ha, va="center", fontsize=fs, color=col, family=MONO, zorder=7)


PY_, PH = 34.0, 52.0
PW, PG = 22.5, 2.5
X0 = 3.0

# ── ① checkpoint 原始 scale ──
x = X0
ec = panel(x, PY_, PW, PH, "①", "checkpoint 里的 scale", "weight_scale_inv  BF16→fp32", "gray")
note(x + PW / 2, PY_ + PH - 7.0, "第 0 行的前 4 个 K 向块", GY, 8.0)
vals = ["1.745e-4", "1.974e-4", "1.850e-4", "2.880e-4"]
for i, v in enumerate(vals):
    cell(x + 1.6 + i * 4.9, PY_ + 30.0, 4.5, 4.0, v, fs=7.4)
    note(x + 1.6 + i * 4.9 + 2.25, PY_ + 28.2, f"g{i}", GY, 6.8)
note(x + PW / 2, PY_ + 24.0, "log2 → −12.48 / −12.31 / −12.40 / −11.76", GY, 7.6)
note(x + PW / 2, PY_ + 21.4, "全是非 2 的幂", "#b32020", 8.0)
note(x + PW / 2, PY_ + 17.0, "每块 128×128 权重共用 1 个 scale", GY, 7.4)
note(x + PW / 2, PY_ + 14.4, "存储：4 字节 / 个", GY, 7.4)
note(x + PW / 2, PY_ + 9.5, "DeepGEMM on SM100 不接受\n这种任意值 scale", "#b32020", 8.0)

# ── ② 重量化 ──
x = X0 + PW + PG
panel(x, PY_, PW, PH, "②", "重量化成 2 的幂", "requant_weight_ue8m0_inplace", "orange")
note(x + PW / 2, PY_ + PH - 7.0, "s ← 2^ceil(log2 s)", OR, 8.4)
pw2 = ["2⁻¹²", "2⁻¹²", "2⁻¹²", "2⁻¹¹"]
for i, v in enumerate(pw2):
    cell(x + 1.6 + i * 4.9, PY_ + 30.0, 4.5, 4.0, v, fc="#fdf2e0", ec=OR, fs=9.0, tc=OR)
    note(x + 1.6 + i * 4.9 + 2.25, PY_ + 28.2, f"g{i}", GY, 6.8)
note(x + PW / 2, PY_ + 24.0, "= 2.441e-4 ×3,  4.883e-4", GY, 7.6)
note(x + PW / 2, PY_ + 21.0, "scale 被抬高 1.08~1.70×", OR, 7.8)
note(x + PW / 2, PY_ + 16.6, "权重必须整段反量化再重量化：", GY, 7.2)
note(x + PW / 2, PY_ + 14.2, "只改 scale 会让数值整体偏移", GY, 7.2)
note(x + PW / 2, PY_ + 9.5, "代价：块内 amax 只能映射到\n256 而非 448（实测误差 ×1.37）", "#b32020", 7.8)

# ── ③ 取指数 → E8M0 ──
x = X0 + 2 * (PW + PG)
panel(x, PY_, PW, PH, "③", "只留指数 → E8M0", "8 bit 纯指数，bias 127", "teal")
note(x + PW / 2, PY_ + PH - 7.0, "byte = 指数 + 127", TEAL, 8.4)
bytes_ = [("0x73", "115"), ("0x73", "115"), ("0x73", "115"), ("0x74", "116")]
for i, (hx, dec) in enumerate(bytes_):
    cell(x + 1.6 + i * 4.9, PY_ + 30.0, 4.5, 4.0, hx, fc="#e2f4f4", ec=TEAL, fs=8.6, tc=TEAL)
    note(x + 1.6 + i * 4.9 + 2.25, PY_ + 28.2, dec, GY, 6.8)
note(x + PW / 2, PY_ + 24.4, "−12+127=115   −11+127=116", GY, 7.4)
note(x + PW / 2, PY_ + 20.6, "无符号位、无尾数位", TEAL, 7.8)
note(x + PW / 2, PY_ + 17.2, "存储：4 字节 → 1 字节", TEAL, 7.8)
note(x + PW / 2, PY_ + 9.5, "这正是 tcgen05 block-scaled MMA\n的 SF 操作数格式（OCP MX 规范）", TEAL, 7.6)

# ── ④ 打包 int32 ──
x = X0 + 3 * (PW + PG)
panel(x, PY_, PW, PH, "④", "4 字节合成 1 个 int32", "pack_ue8m0_to_int  deep_gemm.py:395", "purple")
bw = 4.2
for i, (hx, _) in enumerate(bytes_):
    cell(x + 1.4 + i * bw, PY_ + 36.0, bw - 0.3, 3.4, hx, fc="#e2f4f4", ec=TEAL, fs=7.8, tc=TEAL)
    note(x + 1.4 + i * bw + bw / 2 - 0.15, PY_ + 40.4, f"b{i}", GY, 6.4)
arrow(ax, (x + PW / 2, PY_ + 35.4), (x + PW / 2, PY_ + 32.2), color=PU, lw=1.6)
cell(x + 1.4, PY_ + 27.4, bw * 4 - 0.3, 4.4, "0x74737373", fc="#f0e8fb", ec=PU, fs=11, tc=PU, lw=2.0)
note(x + PW / 2, PY_ + 25.4, "一个 int32", PU, 7.4)
note(x + PW / 2, PY_ + 22.0, "b0 | b1<<8 | b2<<16 | b3<<24", PU, 7.6)
note(x + PW / 2, PY_ + 18.4, "低字节 = 低编号 K 组", GY, 7.4)
note(x + PW / 2, PY_ + 14.6, "存储：1/4（对比 fp32）", PU, 7.8)
note(x + PW / 2, PY_ + 9.5, "再按 MN-major + TMA 对齐排布\n见下方", PU, 7.6)

for i in range(3):
    arrow(ax, (X0 + (i + 1) * PW + i * PG + 0.4, PY_ + PH / 2),
          (X0 + (i + 1) * (PW + PG) - 0.4, PY_ + PH / 2), lw=2.0)

# ── 底部：最终内存布局 + 两侧对照 ──
BY = 4.0
ax.add_patch(FancyBboxPatch((3.0, BY), 94.0, 26.0, boxstyle="round,pad=0.35,rounding_size=1.0",
                            facecolor="#fbfbfd", edgecolor=GY, linewidth=1.4, zorder=2))
ax.text(50, BY + 24.0, "最终内存布局：MN-major + TMA 对齐", ha="center", va="center",
        fontsize=11, color="#20242b", family=SANS, zorder=6)

lx = 6.0
ax.text(lx, BY + 20.0,
        "形状   [mn, ⌈K_groups / 4⌉]        dtype int32\n"
        "stride (1, tma_aligned_mn)         tma_aligned_mn = round_up(mn, 4)\n"
        "        ↑ 第 0 维 stride 为 1 ⇒ 同一 K 组的不同行在内存中相邻（MN-major，即列主序）",
        ha="left", va="top", fontsize=8.4, color="#20242b", family=MONO, zorder=6, linespacing=1.9)
ax.text(lx, BY + 10.0,
        "这样排是为了 TMA：DeepGEMM 用 Tensor Memory Accelerator 整块搬 scale，\n"
        "要求 MN 维按 4 对齐、且同一 K 组的行连续，才能一次拷贝而不是逐行 gather。",
        ha="left", va="top", fontsize=8.4, color=GY, family=MONO, zorder=6, linespacing=1.9)

rx = 56.0
ax.text(rx, BY + 20.0, "权重侧（加载期一次）", ha="left", va="center", fontsize=9.4,
        color=OR, family=SANS, zorder=6)
ax.text(rx, BY + 17.4,
        "process_weights_after_loading\n"
        "  → requant_weight_ue8m0_inplace   ②\n"
        "  → transform_sf_into_required_layout  ③④",
        ha="left", va="top", fontsize=8.0, color="#20242b", family=MONO, zorder=6, linespacing=1.8)
ax.text(rx, BY + 8.4, "激活侧（每次 forward）", ha="left", va="center", fontsize=9.4,
        color=BL, family=SANS, zorder=6)
ax.text(rx, BY + 5.8,
        "torch.ops._C.per_token_group_fp8_quant_packed\n"
        "  一个 csrc kernel 内同时完成 ②③④（见第 5 章）",
        ha="left", va="top", fontsize=8.0, color="#20242b", family=MONO, zorder=6, linespacing=1.8)

print("ok", file=sys.stderr)
save(fig, "ue8m0_packing_layout.png")
