from fig_common import *
from fig_common import _meas
from matplotlib.patches import FancyBboxPatch
import sys

TAGC = {"csrc": ("#e8f0fe", "#1a5fb4"), "DG": ("#e6f4ea", "#1e7d3a"),
        "CUT": ("#f0e8fb", "#6b3fa0"), "Triton": ("#fdf2e0", "#b06d00"),
        "Py": ("#f5f5f5", "#4a4a4a")}


def mk(fig, ax):
    def band(x, y, w, h, label, sub, color, dashed=True):
        fc, ec = C[color]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=1,
                                    linestyle=(0, (5, 3)) if dashed else "solid", alpha=0.45))
        ax.text(x + 1.4, y + h - 1.8, label, ha="left", va="center", fontsize=12,
                color=ec, family=SANS, zorder=6)
        ax.text(x + 1.4, y + h - 4.0, sub, ha="left", va="center", fontsize=9,
                color=ec, family=MONO, zorder=6, alpha=0.9)

    def need_h(lines, fs, has_loc=True):
        th = _meas(ax, "\n".join(lines), fs, MONO, 1.7)[0]
        return (5.8 if has_loc else 3.8) + th + 1.6

    def node(x, y, w, h, tag, name, loc, lines, color="gray", kind=None, fs=8.0, star=False):
        if h is None:
            h = need_h(lines, fs, loc is not None) + (3.2 if kind else 0.0)
        fc, ec = C[color]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.7",
                                    facecolor=fc, edgecolor=ec,
                                    linewidth=2.6 if star else 1.5, zorder=4))
        head = f"{tag}  {name}" if tag else name
        ax.text(x + w / 2, y + h - 1.9, head, ha="center", va="center",
                fontsize=9.8, color=ec, family=SANS, zorder=6)
        if loc:
            ax.text(x + w / 2, y + h - 4.1, loc, ha="center", va="center",
                    fontsize=7.6, color=ec, family=MONO, zorder=6, alpha=0.85)
        ax.text(x + 1.2, y + h - (5.8 if loc else 3.8), "\n".join(lines), ha="left", va="top",
                fontsize=fs, color="#20242b", family=MONO, zorder=6, linespacing=1.7)
        if kind:
            bf, be = TAGC[kind]
            ax.add_patch(FancyBboxPatch((x + w - 7.2, y + 0.6), 6.4, 2.2,
                                        boxstyle="round,pad=0.1,rounding_size=0.4",
                                        facecolor=bf, edgecolor=be, linewidth=1.0, zorder=6))
            ax.text(x + w - 4.0, y + 1.7, kind, ha="center", va="center",
                    fontsize=7.4, color=be, family=MONO, zorder=7)
        return (x, y, w, h)

    def ar(p0, p1, **kw):
        arrow(ax, p0, p1, **kw)

    return band, node, ar, need_h


# ════════════════════ dense 运行期（横版） ════════════════════
# 版面原则：三步骨架横排一行，每步的「二选一实现」用分叉线并排挂在它正下方，
# 纵向位置全部由内容高度推出来，最后 assert 底部留白，避免框与框重叠。
fig, ax = new_fig(22, 7.6)
band, node, ar, need_h = mk(fig, ax)

FK = "#8a6d3b"          # 分叉线的颜色，与骨架箭头的灰色区分开
UPP = 0.5544 * fig.get_size_inches()[1]      # 每个 y 单位折合多少 pt


def head(main, sub, legend, y=98.5):
    ax.text(50, y, main, ha="center", va="top", fontsize=17,
            family=SANS, color="#12151a")
    y2 = y - 17 * 1.45 / UPP
    ax.text(50, y2, sub, ha="center", va="top", fontsize=9.5,
            family=MONO, color="#666c76")
    y3 = y2 - 9.5 * 1.7 / UPP
    ax.text(50, y3, legend, ha="center", va="top", fontsize=9,
            family=MONO, color="#8a8f98")
    return y3 - 9 * 2.0 / UPP


def fork(xc, y_top, y_bot, xs, label):
    """从父框底部 (xc, y_top) 下探，横向分叉到 xs 各列，箭头落到子框顶部 y_bot。"""
    ymid = (y_top + y_bot) / 2
    ax.plot([xc, xc], [y_top, ymid], color=FK, lw=1.6, zorder=5)
    ax.plot([min(xs), max(xs)], [ymid, ymid], color=FK, lw=1.6, zorder=5)
    for x in xs:
        ar((x, ymid), (x, y_bot), lw=1.6, color=FK)
    ax.text(xc + 1.4, (y_top + ymid) / 2, label, ha="left", va="center",
            fontsize=8.2, color=FK, family=SANS, zorder=6)


BAND_TOP = head(
    "dense FP8 Linear 的运行期调用链",
    "整张图都在 forward 阶段（虚线框）—— kernel 类在构造期已定死，这里不再有任何判定",
    "角标：[csrc] 本仓库 CUDA 算子    [DG] DeepGEMM JIT    [CUT] CUTLASS AOT")

# —— 三列的横向划分：列宽 28，列间留 5 给骨架箭头 ——
C1, C2, C3, CW = 3.0, 36.0, 69.0, 28.0
SUBW = (CW - 1.4) / 2                        # 每列并排两个子框
SUB1 = [(C1, C1 + SUBW), (C1 + SUBW + 1.4, C1 + CW)]
SUB2 = [(C2, C2 + SUBW), (C2 + SUBW + 1.4, C2 + CW)]

TOP_LINES = [
    ["把输入 view 成 2D，按 (1,128) 分组量化", "受 apply_input_quant 开关控制",
     "FlashInfer 混合体设 False → 跳过本步"],
    ["基类唯一的抽象方法", "构造期选中哪个 kernel 类，", "这里就调它的实现"],
    ["out + bias → .to(out_dtype)", "→ view(output_shape)", "各后端共用，返回 bf16"],
]
SH = max(need_h(l, 7.7) for l in TOP_LINES)

SUB_LINES = [
    ["per_token_group_quant_fp8_", "     packed_for_deepgemm",
     "→ torch.ops._C.", "  per_token_group_fp8_", "  quant_packed     :753",
     "scale → int32 [M,⌈K/512⌉]", "UE8M0 ×4 打包，无 Triton"],
    ["per_token_group_quant_fp8", "→ torch.ops._C.",
     "  per_token_group_fp8_quant", "                   :637",
     "scale → fp32 [M, K/128]", "        列主序", "stride 由 Python 预分配"],
    ["torch.ops.vllm.", "  fp8_gemm_nt_op   :122", "→ deep_gemm.fp8_gemm_nt",
     "  utils/deep_gemm.py:444", "", "JIT，首遇新形状现编", ""],
    ["ops.cutlass_scaled_mm(", "  A, B.T, scale_a=As,", "  scale_b=Bs.T)",
     "  _custom_ops.py:725", "→ scaled_mm_entry.cu:197", "  按 SM 分发 → blockwise", ""],
]
SWAP_LINES = ["接口约定完全一致：",
              "  · 都吃 (fp8 张量, scale 张量) 这一对",
              "  · 都在 epilogue 里反量化成 bf16",
              "",
              "差别只有两处：",
              "  · ② 这一行调谁",
              "  · ① 产出的 scale 是 fp32 列主序，",
              "    还是打包成 int32 的 UE8M0"]
# 带角标的框要多留 3.2 给右下角那枚标签，否则会压住最后一行正文
SUBH = max([need_h(l, 7.0) + 3.2 for l in SUB_LINES]
           + [need_h(SWAP_LINES, 7.5, False)])

SY = BAND_TOP - 6.5 - SH             # 第一行（骨架）底边
FORKH = 7.0                          # 分叉区高度
R2 = SY - FORKH - SUBH               # 第二行（实现）底边
BAND_BOT = R2 - 2.5

band(2, BAND_BOT, 96, (BAND_TOP + 0.5) - BAND_BOT, "forward 阶段",
     "Fp8LinearMethod.apply  fp8.py:446 → :489 → Fp8BlockScaledMMLinearKernel.apply_weights"
     "（BlockScaledMMLinearKernel.py:97）", "purple")

# —— 第一行：三步骨架 ——
node(C1, SY, CW, SH, "①", "激活量化", "self.quant_fp8(input_2d, ...)   :120",
     TOP_LINES[0], "gray", fs=7.7)
node(C2, SY, CW, SH, "②", "block GEMM", "self.apply_block_scaled_mm(...)",
     TOP_LINES[1], "orange", star=True, fs=7.7)
node(C3, SY, CW, SH, "③", "epilogue", ":139", TOP_LINES[2], "gray", fs=7.7)

for x0, x1, lbl in ((C1 + CW, C2, "(Aq, As)"), (C2 + CW, C3, "acc → bf16")):
    ar((x0 + 0.4, SY + SH / 2), (x1 - 0.4, SY + SH / 2), lw=2.2)
    ax.text((x0 + x1) / 2, SY + SH / 2 + 2.6, lbl, ha="center", va="center",
            fontsize=7.6, color="#5a6270", family=MONO, zorder=6)

# —— 第二行：每步的两个实现，并排挂在父框下方 ——
fork(C1 + CW / 2, SY, R2 + SUBH, [(a + b) / 2 for a, b in SUB1],
     "二选一：按设备与 E8M0 开关")
node(SUB1[0][0], R2, SUBW, SUBH, "", "Blackwell + UE8M0", "fp8_utils.py:695",
     SUB_LINES[0], "blue", kind="csrc", fs=7.0)
node(SUB1[1][0], R2, SUBW, SUBH, "", "其余（Hopper / CUTLASS）", "fp8_utils.py:566",
     SUB_LINES[1], "blue", kind="csrc", fs=7.0)

fork(C2 + CW / 2, SY, R2 + SUBH, [(a + b) / 2 for a, b in SUB2],
     "二选一：按候选表定死")
node(SUB2[0][0], R2, SUBW, SUBH, "", "DeepGemmFp8BlockScaledMM", "scaled_mm/deep_gemm.py:109",
     SUB_LINES[2], "green", kind="DG", fs=7.0)
node(SUB2[1][0], R2, SUBW, SUBH, "", "CutlassFp8BlockScaledMM", "scaled_mm/cutlass.py:312",
     SUB_LINES[3], "purple", kind="CUT", fs=7.0)

# ③ 没有分支，这一格改放「为什么能互换」
node(C3, R2, CW, SUBH, "", "③ 无分支：两个后端为什么能互换", None,
     SWAP_LINES, "teal", fs=7.5)

NOTES = ["· 整条链没有任何「选择」动作 —— 走哪个分支在构造期（图 5）就定死了，forward 只是照着调。",
         "· ① 与 ② 的两处分叉互不影响：scale 格式由设备与 E8M0 开关决定，GEMM 后端由候选表决定。",
         "· 换 GEMM 后端不会加速 ①：两条分支底层都是 csrc 的同一批 per_token_group 量化算子。"]
NH = need_h(NOTES, 8.0, False)
NY = BAND_BOT - 3.0 - NH
node(2, NY, 96, NH, "", "读图要点", None, NOTES, "gray", fs=8.0)

print(f"  dense: bottom y = {NY:.1f}", file=sys.stderr)
assert 1.0 < NY < 12.0, f"dense 版面留白失衡（bottom={NY:.1f}）"
save(fig, "dense_runtime.png")


# ════════════════════ MoE 运行期（横版） ════════════════════
fig, ax = new_fig(22, 7.8)
title(ax, "routed MoE 的运行期调用链（contiguous 布局 + DeepGEMM）",
      "整张图都在 forward 阶段（虚线框）—— DeepGEMM 只负责两次 grouped GEMM，"
      "进出口重排与中间量化都是 vLLM 自己的 kernel")
band, node, ar, need_h = mk(fig, ax)


_MOE_LINES = steps = [
    ("①", "prepare", "no_dp_ep.py:57",
     ["prepare_finalize.prepare", "→ moe_kernel_quantize_input", "→ _fp8_quantize utils.py:128",
      "→ per_token_group_quant_fp8", "", "与 dense ① 同一个", "csrc 算子"], "blue", "csrc"),
    ("②", "permute", "deep_gemm_utils.py:457",
     ["deepgemm_moe_permute", "M_sum = Σ round_up(cnt,128)", "expert_ids 全填 -1，未写到",
      "  的行即 padding，DeepGEMM", "  见负数跳过整个 block", "ep_scatter 一趟搬 fp8 行 /",
      "  scale / expert_ids / inv_perm"], "orange", "Triton"),
    ("③", "GEMM1", "deep_gemm_moe.py:358",
     ["m_grouped_fp8_gemm_nt_", "        contiguous", "utils/deep_gemm.py:463", "",
      "w13 = gate+up 拼合", "[256, 1024, 2048]", "[M_sum,2048]→[M_sum,1024]"], "green", "DG"),
    ("④", "激活 + 再量化", "deep_gemm_moe.py:370 → :225",
     ["_act_mul_quant：SwiGLU 把", "前后两半合成 [M_sum,512]", "并重新量化，三种走法：",
      "UE8M0+SiLU → silu_mul_quant", "     _fp8_packed_triton  :246",
      "Hopper+SiLU → ..._colmajor :268", "非 SiLU → activation+量化 :283"], "orange", "Triton"),
    ("⑤", "GEMM2", "deep_gemm_moe.py:375",
     ["m_grouped_fp8_gemm_nt_", "        contiguous", "", "w2 = down_proj",
      "[256, 2048, 512]", "[M_sum,512]→[M_sum,2048]"], "green", "DG"),
    ("⑥", "finalize", "deep_gemm_utils.py:550",
     ["deepgemm_unpermute_and_", "        reduce", "按 inv_perm 取回每 token 的",
      "8 份结果，乘 topk_weights", "求和，还原成 [M, K]", "",
      "unpermute 与加权规约一步完成"], "blue", "Triton"),
]
MW, MG = 14.8, 1.4
MH = max(need_h(l, 7.4) for _, _, _, l, _, _ in _MOE_LINES) + 3.2
MY = 82.0 - MH

band(2, MY - 9.0, 96, 91.0 - (MY - 9.0), "forward 阶段",
     "Fp8MoEMethod.apply  fp8.py:833 → FusedMoEKernel.apply  modular_kernel.py:1640", "purple")

for i, (tag, name, loc, lines, col, kind) in enumerate(steps):
    x = 3.6 + i * (MW + MG)
    node(x, MY, MW, None, tag, name, loc, lines, col, kind=kind, fs=7.4)
    if i:
        ar((x - MG + 0.15, MY + MH / 2), (x - 0.15, MY + MH / 2), lw=1.6)

ax.text(50, MY - 4.6,
        "②⑥ 的 permute / finalize 是 Triton，既非 DeepGEMM 也非 csrc —— DeepGEMM 只做 ③⑤ 两次 grouped GEMM，"
        "进出口重排 vLLM 自己出 kernel（见 3.2.1）",
        ha="center", va="center", fontsize=9.0, color=C["purple"][1], family=SANS, zorder=6)

BOT_H = max(
    need_h(["fp8_m_grouped_gemm_nt_masked     batched_deep_gemm_moe.py:425   [DG]",
            "torch.ops._C.persistent_masked_m_silu_mul_quant                 [csrc]",
            "                                 activation_kernels.cu",
            "fp8_m_grouped_gemm_nt_masked     batched_deep_gemm_moe.py:440   [DG]", "",
            "只有这条路的中间量化用 csrc 的 persistent kernel；",
            "contiguous 布局下 SiLU 融合走的是 Triton（上图 ④）。"], 7.8),
    need_h(["· M_sum ≠ M × topk。Qwen3.6 prefill 256 个", "  token 时真实 2048 行，M_sum = 34560 行，",
            "  padding 只占显存不占算力", "", "· 三个 Triton 算子的工作量都与真实 token",
            "  数成正比，与膨胀后的 M_sum 无关", "", "· o_proj 与 shared expert 不在这条链上，",
            "  它们走 dense 路（3.1）"], 7.8, False))
BOT_Y = (MY - 9.0) - 3.2 - BOT_H
import sys; print("MoE BOT_Y =", round(BOT_Y, 1), file=sys.stderr)
node(3.6, BOT_Y, 54, BOT_H, "", "另一条路：batched（masked）布局",
     "DP/EP 场景，由 deepep_ll 或 nixl_ep 触发",
     ["fp8_m_grouped_gemm_nt_masked     batched_deep_gemm_moe.py:425   [DG]",
      "torch.ops._C.persistent_masked_m_silu_mul_quant                 [csrc]",
      "                                 activation_kernels.cu",
      "fp8_m_grouped_gemm_nt_masked     batched_deep_gemm_moe.py:440   [DG]",
      "",
      "只有这条路的中间量化用 csrc 的 persistent kernel；",
      "contiguous 布局下 SiLU 融合走的是 Triton（上图 ④）。"], "orange", fs=7.8)

node(61, BOT_Y, 35, BOT_H, "", "读图要点", None,
     ["· M_sum ≠ M × topk。Qwen3.6 prefill 256 个",
      "  token 时真实 2048 行，M_sum = 34560 行，",
      "  padding 只占显存不占算力",
      "",
      "· 三个 Triton 算子的工作量都与真实 token",
      "  数成正比，与膨胀后的 M_sum 无关",
      "",
      "· o_proj 与 shared expert 不在这条链上，",
      "  它们走 dense 路（3.1）"], "teal", fs=7.8)

print("moe ok", file=sys.stderr)
save(fig, "moe_runtime.png")
