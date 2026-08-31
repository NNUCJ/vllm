from fig_common import *
from matplotlib.patches import Rectangle

fig, ax = new_fig(17, 11.6)
title(ax, "KV cache 的 FP8 通路",
      "csrc/libtorch_stable/cache_kernels.cu  ·  csrc/attention/dtype_fp8.cuh")

t0 = box(ax, 12, 0, 76, None, top=90.0,
         title="--kv-cache-dtype 字符串 -> Fp8KVCacheDataType（dtype_fp8.cuh:20）", lines=[
    "auto / float16 / bfloat16      -> kAuto     不量化，cache 里就是模型 dtype",
    "fp8 / fp8_e4m3 / fp8_ds_mla    -> kFp8E4M3  默认选择",
    "fp8_e5m2                       -> kFp8E5M2  范围换精度，一般只在调试时用",
], color="gray", ls=9.6, align="left", ts=12.5)

wy = t0[1] - 4.0
w0 = box(ax, 3, 0, 45, None, top=wy, title="写入：量化顺手做掉", lines=[
    "reshape_and_cache / reshape_and_cache_flash",
    "  cache_kernels.cu:254 / :314",
    "  CopyWithScaleOp<cache_t, scalar_t, kv_dt>   :241",
    "    kAuto  -> 直接 static_cast",
    "    否则   -> fp8::scaled_convert<...>(src, scale)",
    "",
    "concat_and_cache_mla        MLA 的 NoPE+RoPE 拼接",
    "indexer_k_quant_and_cache   DeepSeek 稀疏 indexer",
    "                            按 quant_block_size 分块，可选 UE8M0",
    "",
    "k_scale / v_scale 是标量（per-tensor），来自离线校准；",
    "没有校准就是 1.0，等于只做截断——精度全靠模型本身的动态范围。",
], color="green", ls=9.4, align="left", ts=12.5)

r0 = box(ax, 52, 0, 45, None, top=wy, title="读出：谁来反量化", lines=[
    "① attention kernel 自己吃 fp8",
    "   FlashAttention / FlashInfer / paged attention 内部",
    "   带着 k_scale、v_scale 做累加，不落中间张量",
    "",
    "② 需要 bf16 张量时显式反量化",
    "   gather_and_maybe_dequant_cache      cache_kernels.cu",
    "   cp_gather_and_upconvert_fp8_kv_cache",
    "",
    "③ 离线整块转换",
    "   convert_fp8(dst, src, scale, kv_cache_dtype)  :928",
    "",
    "转换本身都落在 fp8::scaled_convert / vec_conversion，",
    "在 nvidia/quant_utils.cuh 与 amd/quant_utils.cuh 各有一份。",
], color="blue", ls=9.4, align="left", ts=12.5)

# fp8_ds_mla 布局
ly = min(w0[1], r0[1]) - 4.0
l0 = box(ax, 3, 0, 94, 28.0, top=ly,
         title="fp8_ds_mla：一个 token 656 字节的定制布局（cache_kernels.cu:864 校验，:500 写入）",
         lines=None, color="purple", ts=12.5)

bx, bw, by, bh = 8, 84, l0[1] + 8.0, 5.0
segs = [("NoPE  512 B  =  512 x e4m3", 512, "#cfe3fb"),
        ("", 16, "#fbe0bf"),
        ("RoPE  128 B  =  64 x bf16", 128, "#c9ecd4")]
cx = bx
for name, nbytes, col in segs:
    sw = bw * nbytes / 656
    ax.add_patch(Rectangle((cx, by), sw, bh, facecolor=col,
                           edgecolor="#3a3f47", linewidth=1.2, zorder=3))
    if name:
        ax.text(cx + sw / 2, by + bh / 2, name, ha="center", va="center",
                fontsize=10, family=MONO, color="#12151a", zorder=4)
    cx += sw
# 窄段（16 B）单独引出标注
mx = bx + bw * 520 / 656
ax.plot([mx, mx + 6], [by + bh, by + bh + 2.6], color="#b06d00", lw=1.2,
        zorder=4)
ax.text(mx + 6.4, by + bh + 2.6, "16 B = 4 x fp32 tile scale",
        ha="left", va="center", fontsize=9.6, family=MONO, color="#b06d00",
        zorder=5)
for off, lab in [(0, "0"), (512, "512"), (528, "528"), (656, "656")]:
    ax.text(bx + bw * off / 656, by - 0.8, lab, ha="center", va="top",
            fontsize=9, family=MONO, color="#4a4a4a")

ax.text(bx, by - 4.0, "\n".join([
    "kv_lora_rank = 512 的 NoPE 部分量化成 e4m3；每 128 个元素一个 tile，由半个 warp（16 lane）做 shuffle 求 amax，",
    "tile_scale = max(amax/448, FLT_MIN) 写在 512 字节之后；pe_dim = 64 的 RoPE 部分保持 bf16 不量化。",
]), ha="left", va="top", fontsize=9.4, family=MONO, color="#20242b",
    linespacing=1.55, zorder=6)

import sys
print("bottom =", l0[1], file=sys.stderr)
save(fig, "kv_cache_fp8.png")
