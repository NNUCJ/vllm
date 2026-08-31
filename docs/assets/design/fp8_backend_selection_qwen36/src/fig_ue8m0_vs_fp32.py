from fig_common import *

fig, ax = new_fig(17, 20.5)
title(ax, "为什么 DeepGEMM(SM100) 要 UE8M0 scale，而 CUTLASS 能保持 fp32",
      "deep_gemm.py:49 DeepGemmQuantScaleFMT · scaled_mm_blockwise_sm100_fp8_dispatch.cuh:58 · tcgen05 block-scaled MMA")

# ── Row A：数学原理 ──────────────────────────────────────────────
ya = 91.5
a0 = box(ax, 6, 0, 88, None, top=ya, color="teal",
         title="数学原理：乘以 2 的幂 = 纯指数加法", lines=[
    "浮点数  v = ±m × 2^e        （m 尾数，e 指数）",
    "",
    "v × 2^k = ±m × 2^(e+k)      # 尾数 m 一位不动，只对指数域做整数加法",
    "  → 不需要乘法器、结果逐位精确（零舍入误差）",
    "v × s   (s 为任意 fp32)      # 需要完整 FMA：尾数相乘 + 规格化 + 一次舍入",
    "",
    "E8M0 格式：8 bit 纯指数（bias 127），无符号位、无尾数位，",
    "一个字节恰好覆盖 2^-127 … 2^127 —— 为「2 的幂 scale」量身定做的容器",
], ls=10.0, align="left")

# ── Row B：两条硬件/软件通路 ─────────────────────────────────────
yb = a0[1] - 3.0
b0 = box(ax, 2, 0, 47, None, top=yb, color="blue",
         title="DeepGEMM 路：scale 进 TensorCore（硬件）", lines=[
    "SM100 第五代 TensorCore 的 tcgen05.mma",
    "block-scale 变体：SFA/SFB 操作数按 OCP MX",
    "规范定义为 E8M0 字节，scale 在 MMA 流水",
    "内做指数加，与乘累加融合、零额外指令",
    "",
    "4 个 E8M0 打包 1 个 int32 + TMA 对齐，",
    "scale 带宽 = fp32 方案的 1/4",
    "",
    "前提：scale 必须是 2 的幂",
    "→ 加载期 requant_weight_ue8m0_inplace",
    "  把 checkpoint 的 fp32 scale 重量化（图 10）",
    "",
    "SM90 无此硬件指令 → DeepGEMM 退用",
    "CUDA core FFMA 两级累加 → fp32 scale 可用",
], ls=10.0, align="left")

b1 = box(ax, 51, 0, 47, None, top=yb, color="purple",
         title="CUTLASS 路：scale 留在 CUDA core（软件）", lines=[
    "vLLM blockwise kernel 显式声明",
    "  ElementBlockScale = float   // fp32",
    "  (…blockwise_sm100_fp8_dispatch.cuh:58)",
    "",
    "MMA 用普通 FP8 TensorCore 指令，",
    "不携带硬件 SF 操作数；每个 K-block 结束后",
    "在 CUDA core 上用 fp32 FMA 完成",
    "  acc_main += partial × s_a × s_b",
    "（软件 promotion，Sm100BlockwiseScaleConfig）",
    "",
    "任意 fp32 值皆可参与乘法 → 无格式约束",
    "→ checkpoint scale 原样使用，权重零改动",
    "",
    "代价：promotion 占用 CUDA core FMA，",
    "scale 每个 4 字节",
], ls=10.0, align="left")

arrow(ax, (30, a0[1]), (25, b0[1] + b0[3] + 0.4), label=" 硬件消费", lx=7, fs=9.5)
arrow(ax, (70, a0[1]), (75, b1[1] + b1[3] + 0.4), label=" 软件消费", lx=7, fs=9.5)

# ── Row C：同一次 GEMM 的对照 ────────────────────────────────────
yc = min(b0[1], b1[1]) - 3.0
c0 = box(ax, 2, 0, 96, None, top=yc, color="orange",
         title="同一个 K-tile(128) 的两条执行路径对照", lines=[
    "DeepGEMM SM100:  TensorCore [ Aq·Bq  +  scale 指数加(E8M0) ] ──────────────→ fp32 acc",
    "                 一条 tcgen05.mma 完成乘累加与反量化",
    "",
    "CUTLASS SM100:   TensorCore [ Aq·Bq ] → CUDA core [ partial × s_a × s_b ] → fp32 acc",
    "                 反量化是 MMA 之外的独立 fp32 FMA",
    "",
    "vLLM 的 scale 格式仲裁（deep_gemm.py:49 DeepGemmQuantScaleFMT.init_oracle_cache）：",
    "  E8M0 关闭            → FLOAT32            （fp32 张量，任意值）",
    "  E8M0 开 + SM90       → FLOAT32_CEIL_UE8M0（值取 2 的幂，仍存 fp32 张量）",
    "  E8M0 开 + SM100/120  → UE8M0             （值取 2 的幂，4 合 1 打包进 int32）",
], ls=10.0, align="left")

arrow(ax, (25, b0[1]), (35, c0[1] + c0[3] + 0.4))
arrow(ax, (75, b1[1]), (65, c0[1] + c0[3] + 0.4))

# ── Row D：得失对比 ─────────────────────────────────────────────
yd = c0[1] - 3.0
d0 = box(ax, 2, 0, 47, None, top=yd, color="green",
         title="UE8M0 换来什么", lines=[
    "反量化免费且逐位精确（指数加无舍入）",
    "scale 存储/带宽降为 1/4（1B vs 4B）",
    "kernel 主循环更短：无 promotion FMA",
    "误差被挪到加载期一次性量化，",
    "运行期数值路径反而更「干净」",
], ls=10.0, align="left")

d1 = box(ax, 51, 0, 47, None, top=yd, color="red",
         title="UE8M0 付出什么（§5.3 的根因）", lines=[
    "scale 上取整到 2^n：block amax 只能映射",
    "到 256 而非 448，上端最多损失 1 bit",
    "实测量化误差 ×1.37（附录 B）",
    "权重字节被真实改写，需加载期 requant",
    "→ Qwen3.5/3.6 精度回归 → dense 黑名单；",
    "CUTLASS fp32 路零代价保留 checkpoint 精度",
], ls=10.0, align="left")

arrow(ax, (35, c0[1]), (25, d0[1] + d0[3] + 0.4))
arrow(ax, (65, c0[1]), (75, d1[1] + d1[3] + 0.4))

import sys
bottom = min(d0[1], d1[1])
print(f"bottom y = {bottom:.1f}", file=sys.stderr)
assert bottom > 1.0, "content overflows below canvas; increase figure height"

save(fig, "ue8m0_vs_fp32_scale.png")
