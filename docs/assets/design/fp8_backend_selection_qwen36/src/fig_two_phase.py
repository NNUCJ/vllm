from fig_common import *
from fig_common import _meas
from matplotlib.patches import FancyBboxPatch
import sys

fig, ax = new_fig(22, 7.6)

# ── 自绘标题（fig_common.title 的默认行距在这个画布尺寸下不够） ──
ax.text(50, 98.6, "vLLM FP8 量化工作流总览：初始化定型，运行时执行",
        ha="center", va="top", fontsize=17, family=SANS, color="#12151a")
ax.text(50, 93.2, "实线 = 初始化阶段（整个进程只跑一次）      虚线 = 运行时阶段（每次前向）"
                  "      ★ = 本文重点展开的环节",
        ha="center", va="top", fontsize=9.5, family=MONO, color="#666c76")

OR, PU, GY = C["orange"][1], C["purple"][1], "#6b7079"
T_OFF, L_OFF, C_OFF = 2.8, 5.6, 8.0          # 标题 / 位置行 / 正文的顶部偏移


def need_h(lines, fs, has_loc=True):
    base = C_OFF if has_loc else T_OFF + 2.4
    return base + _meas(ax, "\n".join(lines), fs, MONO, 1.7)[0] + 1.8


def band(x, y, w, h, label, sub, color, dashed=False):
    fc, ec = C[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=1,
                                linestyle=(0, (5, 3)) if dashed else "solid", alpha=0.45))
    ax.text(x + 1.6, y + h - 2.6, label, ha="left", va="center", fontsize=12,
            color=ec, family=SANS, zorder=6)
    ax.text(x + 1.6, y + h - 5.6, sub, ha="left", va="center", fontsize=8.6,
            color=ec, family=MONO, zorder=6, alpha=0.9)


def node(x, y, w, h, tag, name, loc, lines, color="gray", fs=7.6, star=False):
    fc, ec = C[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.7",
                                facecolor=fc, edgecolor=ec,
                                linewidth=2.8 if star else 1.5, zorder=4))
    ax.text(x + w / 2, y + h - T_OFF, f"{tag}  {name}" if tag else name,
            ha="center", va="center", fontsize=9.4, color=ec, family=SANS, zorder=6)
    if loc:
        ax.text(x + w / 2, y + h - L_OFF, loc, ha="center", va="center",
                fontsize=7.4, color=ec, family=MONO, zorder=6, alpha=0.85)
    ax.text(x + 1.4, y + h - (C_OFF if loc else T_OFF + 2.4), "\n".join(lines),
            ha="left", va="top", fontsize=fs, color="#20242b", family=MONO,
            zorder=6, linespacing=1.7)
    if star:
        ax.text(x + w - 1.4, y + h - 2.4, "★", ha="center", va="center",
                fontsize=11, color=OR, zorder=7)
    return (x, y, w, h)


# ═══════ 上带：初始化阶段 ═══════
SW, SG, SX0, FS = 17.6, 1.4, 3.4, 7.5
init = [
    ("①", "读 checkpoint", "config.json",
     ["quantization_config：", '  quant_method "fp8"', '  activation_scheme "dynamic"',
      "  weight_block_size [128,128]", "  modules_to_not_convert 648 项", "",
      "→ 第 1 章"], "gray", False),
    ("②", "建 Fp8Config", "config/vllm.py:925",
     ["进程级唯一的配置对象，", "只描述 checkpoint，", "不含任何硬件信息", "",
      "Blackwell 黑名单在此写入", "use_deep_gemm = False", "",
      "→ 第 2 章"], "gray", False),
    ("③", "装配 quant_method", "linear.py:274",
     ["每层拿自己的层名去问：", "get_quant_method(self, prefix)", "",
      "dense → Fp8LinearMethod", "MoE   → Fp8MoEMethod", "",
      "「ckpt 内部混合精度」全靠它", "→ 第 2 章"], "blue", False),
    ("④", "选后端", "__init__.py:576",
     ["dense：两张候选表 + 三道门禁", "MoE  ：重排 + 硬开关 + 11 检查", "",
      "选中的 kernel / experts 类", "从此定死，运行期不重判", "",
      "→ 第 3 章"], "green", False),
    ("⑤", "改权重", "fp8.py:398 / :720",
     ["process_weights_after_loading", "把 ckpt 布局转成后端要的布局", "",
      "DeepGEMM on Blackwell：", "  UE8M0 重量化 + 打包", "其他后端：只改布局或不动", "",
      "→ 第 4 章（本文重点）"], "orange", True),
]
SH = max(need_h(l, FS) for _, _, _, l, _, _ in init)
SY = 84.0 - SH
band(2, SY - 3.0, 96, (SY + SH + 7.4) - (SY - 3.0), "初始化阶段",
     "从 checkpoint 到权重定型 —— 整个进程只跑一次，之后不再有任何「选择」动作", "blue")
for i, (tag, name, loc, lines, col, star) in enumerate(init):
    x = SX0 + i * (SW + SG)
    node(x, SY, SW, SH, tag, name, loc, lines, col, fs=FS, star=star)
    if i:
        arrow(ax, (x - SG + 0.15, SY + SH / 2), (x - 0.15, SY + SH / 2), lw=1.6)

# ═══════ 定型分隔线 ═══════
YD = SY - 5.6
ax.plot([4, 96], [YD, YD], color=OR, lw=2.0, ls=(0, (7, 4)), zorder=3)
tw = _meas(ax, "权重与 kernel 至此全部定型", 9.6, SANS)[1] + 3.0
ax.add_patch(FancyBboxPatch((50 - tw / 2, YD - 1.5), tw, 3.0,
                            boxstyle="round,pad=0.12,rounding_size=0.6",
                            facecolor=OR, edgecolor="none", zorder=5))
ax.text(50, YD, "权重与 kernel 至此全部定型", ha="center", va="center",
        fontsize=9.6, color="white", family=SANS, zorder=6)
arrow(ax, (25, YD - 1.9), (25, YD - 4.2), color=OR, lw=1.8, ls=(0, (4, 2)))
arrow(ax, (75, YD - 1.9), (75, YD - 4.2), color=OR, lw=1.8, ls=(0, (4, 2)))

# ═══════ 下带：运行时阶段 ═══════
DENSE = ["① 激活量化   QuantFP8 (1,128) 分组，csrc 算子",
         "② block GEMM  ← 唯一因后端而异的一步",
         "③ epilogue    +bias → bf16，各后端共用"]
MOE = ["① prepare → ② permute → ③ GEMM1 →",
       "④ 激活+再量化 → ⑤ GEMM2 → ⑥ finalize",
       "②⑥ 是 Triton；DeepGEMM 只做 ③⑤ 两次 grouped GEMM"]
BH = max(need_h(DENSE, 7.8), need_h(MOE, 7.8))
BOXY = YD - 12.4 - BH
NOTEY = BOXY - 3.4
RB = NOTEY - 3.0
print("RB =", round(RB, 1), file=sys.stderr)

band(2, RB, 96, (YD - 5.0) - RB, "运行时阶段",
     "每次前向都执行；此时只是调用初始化阶段定下的那套 kernel   → 第二部分（5–7 章）",
     "purple", dashed=True)
node(4.0, BOXY, 44.0, BH, "", "dense Linear —— 三步骨架", "Fp8LinearMethod.apply",
     DENSE, "purple", fs=7.8)
node(52.0, BOXY, 44.0, BH, "", "routed MoE —— 六步流水线", "Fp8MoEMethod.apply",
     MOE, "purple", fs=7.8)
ax.text(50, NOTEY,
        "运行时唯一还会判定的是 MoE 的 FallbackExperts 系列（按输入 shape 二选一，见第 7 章）；dense 完全不判。",
        ha="center", va="center", fontsize=8.4, color=PU, family=SANS, zorder=6)

print("ok", file=sys.stderr)
save(fig, "overview_two_phase.png")
