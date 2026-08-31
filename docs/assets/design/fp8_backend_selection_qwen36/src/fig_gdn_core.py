from fig_common import *
from matplotlib.patches import FancyBboxPatch, Rectangle

fig, ax = new_fig(17, 28)
title(ax, "图 1 中「Gated DeltaNet 递推核」的内部展开",
      "算法转录自 fla/ops/fused_recurrent.py:121-148   ·   维度取自 Qwen3.6 config："
      "16 k-head / 32 v-head，head_k = head_v = 128")

PU, OR, GY, BLUE, GREEN, TEAL = (C["purple"][1], C["orange"][1], "#6b7079",
                                 C["blue"][1], C["green"][1], C["teal"][1])
LINE = "#5a6270"


def nd(x, y, w, h, l1, l2=None, color="gray", fs=9.4, fs2=8.4):
    fc, ec = C[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.22,rounding_size=0.6",
                                facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=4))
    if l2 is None:
        ax.text(x + w / 2, y + h / 2, l1, ha="center", va="center",
                fontsize=fs, color=ec, family=MONO, zorder=6)
    else:
        ax.text(x + w / 2, y + h * 0.66, l1, ha="center", va="center",
                fontsize=fs, color=ec, family=MONO, zorder=6)
        ax.text(x + w / 2, y + h * 0.27, l2, ha="center", va="center",
                fontsize=fs2, color=ec, family=MONO, zorder=6, alpha=0.85)
    return (x, y, w, h)


def ar(p0, p1, color=LINE, lw=1.4, rad=0.0):
    arrow(ax, p0, p1, color=color, lw=lw, rad=rad)


def note(x, y, t, color=GY, fs=8.6, ha="left", fam=MONO):
    ax.text(x, y, t, ha=ha, va="center", fontsize=fs, color=color, family=fam, zorder=7)


# ═════════ ① 输入：来自图 1 的四条边 ═════════
yA = 90.0
ax.text(50, yA + 1.4, "① 递推核的输入 —— 全部来自图 1 的四个输入投影",
        ha="center", va="center", fontsize=11.5, color=BLUE, family=SANS, zorder=6)
items = [("q", "[T, 16, 128]", "in_proj_qkv", "purple"),
         ("k", "[T, 16, 128]", "in_proj_qkv", "purple"),
         ("v", "[T, 32, 128]", "in_proj_qkv", "purple"),
         ("β", "[T, 32]", "in_proj_b → sigmoid", "gray"),
         ("g", "[T, 32]", "in_proj_a + A_log/dt_bias", "gray")]
w = 17.4
for i, (nm, sh, src, col) in enumerate(items):
    x = 3.0 + i * (w + 1.4)
    nd(x, yA - 4.6, w, 4.0, f"{nm}   {sh}", src, col, fs=9.6, fs2=7.8)

note(50, yA - 6.6, "q、k 在 kernel 内做 L2 归一化后 q 再乘 scale；16 个 k-head 与 32 个 v-head 按 GQA 成对（2 个 v-head 共享 1 个 k-head）",
     ha="center")

# ═════════ ② 状态 S ═════════
yB = 80.0
ax.text(50, yB, "② 被递推的状态 S —— 每个 v-head 一个矩阵，跨 token 持续存在",
        ha="center", va="center", fontsize=11.5, color=TEAL, family=SANS, zorder=6)

sx, sy, sw, sh = 8.0, yB - 12.5, 20.0, 10.0
ax.add_patch(Rectangle((sx, sy), sw, sh, facecolor=C["teal"][0],
                       edgecolor=TEAL, linewidth=1.8, zorder=3))
for i in range(1, 5):
    ax.plot([sx + i * sw / 5] * 2, [sy, sy + sh], color=TEAL, lw=0.6, alpha=0.5, zorder=4)
    ax.plot([sx, sx + sw], [sy + i * sh / 5] * 2, color=TEAL, lw=0.6, alpha=0.5, zorder=4)
ax.text(sx + sw / 2, sy + sh / 2, "S", ha="center", va="center",
        fontsize=20, color=TEAL, family=MONO, zorder=6)
note(sx + sw / 2, sy + sh + 1.4, "head_k_dim = 128（列）", color=TEAL, ha="center", fs=8.4)
ax.text(sx - 1.4, sy + sh / 2, "head_v_dim = 128（行）", fontsize=8.4, color=TEAL,
        rotation=90, ha="center", va="center", family=MONO, zorder=6)

b = box(ax, 32, 0, 60, None, top=yB - 2.0, color="teal",
        title="S 的语义与规模", lines=[
    "形状      [head_v_dim, head_k_dim] = [128, 128]，每个 v-head 一份，共 32 份",
    "含义      一张「键 → 值」的关联记忆表：写入时按 k 的方向叠加 v，",
    "          读出时用 q 去查询。它替代了标准注意力的 KV cache。",
    "生命周期  跨 token 持续：prefill 逐块推进，decode 每步读旧值、写新值；",
    "          vLLM 侧存在 ssm_state 缓存里，按 state_idx 寻址（非 KV cache）",
    "代价      每层每序列 32 × 128 × 128 = 524 288 个元素，与序列长度无关",
    "          ——这正是线性注意力相对标准注意力的核心优势",
], ls=9.2, align="left")

# ═════════ ③ 单步五步骤 ═════════
yC = min(sy, b[1]) - 3.2
ax.text(50, yC, "③ 每个时间步 t 做五件事（fused_recurrent.py 逐 token 版本，公式为逐 head）",
        ha="center", va="center", fontsize=11.5, color=OR, family=SANS, zorder=6)

steps = [
    ("1  衰减", "S ← S · exp(g_t)",
     "g 来自 in_proj_a：控制旧记忆遗忘多少。g→0 则完全保留，g 越负衰减越快", "gray"),
    ("2  预测", "u ← v_t − S · k_t",
     "先用当前状态按 k_t 查一次，得到「已经记住的值」；v_t 减去它就是预测误差", "green"),
    ("3  门控", "u ← β_t · u",
     "β 来自 in_proj_b 的 sigmoid：只把误差的 β 比例写回去，控制写入强度", "gray"),
    ("4  更新", "S ← S + u ⊗ k_t",
     "把带权误差按 k_t 方向做外积累加——这是 delta rule 的核心一步", "orange"),
    ("5  读出", "o_t ← S · q_t",
     "用查询向量读出，得到该 head 的输出（送去 RMSNormGated 与 z 做门控）", "gray"),
]
yy = yC - 3.2
for i, (tag, formula, desc, col) in enumerate(steps):
    h = 4.4
    y0 = yy - i * (h + 1.0) - h
    nd(4, y0, 13, h, tag, None, col, fs=10)
    ax.add_patch(FancyBboxPatch((18.5, y0), 26, h,
                                boxstyle="round,pad=0.22,rounding_size=0.6",
                                facecolor="#fbfbfd", edgecolor=GY, linewidth=1.2, zorder=4))
    ax.text(31.5, y0 + h / 2, formula, ha="center", va="center", fontsize=11,
            color="#20242b", family=MONO, zorder=6)
    note(46.5, y0 + h / 2, desc, color="#3c4149", fs=8.8)
    if i < len(steps) - 1:
        ar((10.5, y0), (10.5, y0 - 1.0))

y_last = yy - (len(steps) - 1) * 5.4 - 4.4

# ═════════ ④ 与普通线性注意力的差别 ═════════
yD = y_last - 3.4
d0 = box(ax, 4, 0, 43, None, top=yD, color="blue",
         title="与普通线性注意力的差别", lines=[
    "普通线性注意力   S ← S + v_t ⊗ k_t",
    "                 无条件写入，键冲突时新旧值互相污染",
    "",
    "Gated DeltaNet   S ← S + β(v_t − S·k_t) ⊗ k_t",
    "                 只写入「差多少」，等价于对该键做一次修正；",
    "                 β=1 时完全覆盖旧值，β=0 时不写",
    "",
    "第 2 步的 v − S·k 就是「delta」，DeltaNet 由此得名。",
    "叠加第 1 步的 exp(g) 衰减，即「Gated」DeltaNet。",
], ls=9.2, align="left")

d1 = box(ax, 49, 0, 43, None, top=yD, color="green",
         title="两种实现：prefill 与 decode 走不同 kernel", lines=[
    "prefill  chunk_gated_delta_rule（分块并行）",
    "         把序列切块，块内用矩阵乘并行展开，",
    "         块间才串行推进 S —— 吃满 TensorCore。",
    "         后端可为 FlashInfer / CuteDSL / Triton",
    "         （_resolve_gdn_prefill_backend 选择）",
    "",
    "decode   fused_recurrent_gated_delta_rule（逐 token）",
    "         每步只有 1 个 token，无并行可言，",
    "         即上面五步的直接实现",
    "",
    "两者数学等价，差别只在并行策略。",
], ls=9.2, align="left")

# ═════════ ⑤ 回到量化主题 ═════════
yE = min(d0[1], d1[1]) - 3.0
e0 = box(ax, 4, 0, 88, None, top=yE, color="orange",
         title="⑤ 为什么图 1 里递推核不是紫色也不是橙色 —— 这一整块没有可量化的权重", lines=[
    "五个步骤里出现的全是激活与状态（q / k / v / β / g / S），没有任何权重矩阵参与：",
    "S 是运行期状态而非参数，exp(g)、β 是逐 head 的标量，外积与读出都是 [128,128] 规模的小运算。",
    "",
    "该层唯一可量化的两处大矩阵乘都在递推核之外——入口的 in_proj_qkv [8192,2048] 与出口的 out_proj [2048,4096]，",
    "它们才是图 1 中标紫（dense 路径 FP8）的部分。递推核本身按 BF16/FP32 计算，",
    "因此 FP8 后端的选择（第 2 章）完全不触及这里。",
], ls=9.2, align="left")

import sys
print(f"bottom = {e0[1]:.1f}", file=sys.stderr)
assert e0[1] > 0.5, "content overflows below canvas"

save(fig, "gdn_core_expanded.png")
