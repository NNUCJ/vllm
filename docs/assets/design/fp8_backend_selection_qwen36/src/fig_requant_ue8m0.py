from fig_common import *

fig, ax = new_fig(17, 22)
title(ax, "requant_weight_ue8m0_inplace：把 fp32 block scale 重量化为 2 的幂",
      "fp8_utils.py:989 · scaled_mm/deep_gemm.py:96 · oracle/fp8.py DEEPGEMM 分支 · 加载期一次性执行")

# ── Row A：触发条件 ──────────────────────────────────────────────
ya = 91.5
a0 = box(ax, 8, 0, 84, None, top=ya, color="orange",
         title="触发条件（三者同时成立才执行）", lines=[
    "时机   process_weights_after_loading —— 权重读完、首次 forward 之前，每张量一次",
    "条件 1 该层选中 DeepGEMM 后端（dense kernel 或 MoE backend）",
    "条件 2 use_e8m0 = is_deep_gemm_e8m0_used() 为真（Blackwell 默认，VLLM_USE_DEEP_GEMM_E8M0=1）",
    "条件 3 checkpoint scale 是 fp32/bf16 —— 若已是 E8M0/uint8，只 upcast、跳过 requant",
    "       （deepgemm_post_process_fp8_weight_block :1102 的分支判断）",
], ls=10.0, align="left")

# ── Row B：两个调用方 ────────────────────────────────────────────
yb = a0[1] - 3.0
b0 = box(ax, 2, 0, 46, None, top=yb, color="blue",
         title="dense 调用方", lines=[
    "DeepGemmFp8BlockScaledMMKernel",
    "  .process_weights_after_loading",
    "  （scaled_mm/deep_gemm.py:96）",
    "→ deepgemm_post_process_fp8_weight_block",
    "",
    "Qwen3.6 on SM100：黑名单使 dense 落",
    "CUTLASS，这条链根本不会执行",
], ls=10.0, align="left")

b1 = box(ax, 52, 0, 46, None, top=yb, color="purple",
         title="MoE 调用方", lines=[
    "convert_to_fp8_moe_kernel_format",
    "  DEEPGEMM 分支（oracle/fp8.py:469）",
    "→ prepare_fp8_moe_layer_for_deepgemm",
    "  （fp8_utils.py:1155）",
    "  对 w13、w2 各调一次 post_process",
    "",
    "Qwen3.6 on SM100：无 flashinfer 或强制",
    "deep_gemm 时的实际执行路径 ←",
], ls=10.0, align="left")

arrow(ax, (30, a0[1]), (25, b0[1] + b0[3] + 0.4))
arrow(ax, (70, a0[1]), (75, b1[1] + b1[3] + 0.4))

# ── Row C：算法核心 ─────────────────────────────────────────────
yc = min(b0[1], b1[1]) - 3.0
c0 = box(ax, 2, 0, 96, None, top=yc, color="teal",
         title="算法核心（fp8_utils.py:989-1046，对每个 [M,K] 矩阵逐个执行）", lines=[
    "输入  wq: fp8 e4m3 [..., M, K]      ws: fp32 [..., M/128, K/128]（每 128×128 块一个 scale）",
    "",
    "① 旧 scale 展开   s_exp = repeat_interleave(ws, 128, 行) → 再按列 → 裁到 [M, K]",
    "② 反量化          w_dq = wq.float() × s_exp              # 还原到 fp32 数值域",
    "③ UE8M0 重量化    per_block_cast_to_fp8(w_dq, [128,128], use_ue8m0=True)",
    "                    amax = |block|.max.clamp(1e-4)",
    "                    sf   = amax / 448                     # E4M3 max",
    "                    sf   = 2^ceil(log2(sf))               # ← 关键：上取整到 2 的幂",
    "                    w_new = (w_dq / sf).to(fp8)",
    "④ 原地写回        wq.copy_(w_new)   ws.copy_(sf)          # inplace，不新分配显存",
    "",
    "为何必须整段反量化再重量化：只改 scale 不改 wq，数值会整体偏移；",
    "反量化→重量化让 fp8 尾数在新 scale 下重新取整，误差最小。",
], ls=10.0, align="left")

arrow(ax, (25, b0[1]), (25, c0[1] + c0[3] + 0.4), label=" 仅 Hopper 会走", lx=8.5, fs=9.5)
arrow(ax, (75, b1[1]), (75, c0[1] + c0[3] + 0.4), label=" w13 + w2", lx=6.5, fs=9.5)

# ── Row D：后续与动机 ───────────────────────────────────────────
yd = c0[1] - 3.0
d0 = box(ax, 2, 0, 46, None, top=yd, color="gray",
         title="紧随其后的布局变换", lines=[
    "transform_sf_into_required_layout",
    "  SM90     → fp32 scale",
    "  SM100/120 → int32 打包 UE8M0（TMA 对齐）",
    "与运行期激活侧的",
    "per_token_group_fp8_quant_packed 配套：",
    "GEMM 两侧 scale 均为纯指数字节",
], ls=10.0, align="left")

d1 = box(ax, 52, 0, 46, None, top=yd, color="red",
         title="代价（§5.3 黑名单的根因）", lines=[
    "scale 上取整到 2 的幂：",
    "  block amax 只能映射到 256 而非 448，",
    "  上端最多损失 1 bit 动态范围",
    "权重字节被真实改写（CUTLASS 路一字节不动）",
    "→ Qwen3.5/3.6 精度回归（#38083）",
    "→ dense 侧被列入 Blackwell 黑名单",
], ls=10.0, align="left")

arrow(ax, (35, c0[1]), (25, d0[1] + d0[3] + 0.4))
arrow(ax, (65, c0[1]), (75, d1[1] + d1[3] + 0.4))

# ── Row E：在 Qwen3.6 推理中的落点 ──────────────────────────────
ye = min(d0[1], d1[1]) - 3.0
e0 = box(ax, 8, 0, 84, None, top=ye, color="green",
         title="Qwen3.6-35B-A3B-FP8 on SM100 的实际落点", lines=[
    "dense  不执行 —— 黑名单让 dense 落 CUTLASS，本函数正是黑名单要避开的动作",
    "MoE    无 flashinfer / 显式 deep_gemm 时执行：256 专家 × (w13 [1024,2048] + w2 [2048,512])",
    "       共 512 个矩阵逐个 requant，加载期一次性开销，运行期零成本",
    "Hopper 对照：dense 走 DeepGEMM 但 use_e8m0=False → 本函数跳过，仅做布局变换",
], ls=10.0, align="left")

arrow(ax, (50, min(d0[1], d1[1])), (50, e0[1] + e0[3] + 0.4))

import sys
print(f"bottom y = {e0[1]:.1f}", file=sys.stderr)
assert e0[1] > 1.0, "content overflows below canvas; increase figure height"

save(fig, "requant_ue8m0_workflow.png")
