"""用真实数据演示 requant_weight_ue8m0_inplace 的执行过程。

直接调用 vLLM 仓库里的原函数（非复刻），分两部分：
  Part 1: 4x4 矩阵 + 2x2 块，逐元素打印每一步中间量
  Part 2: Qwen3.6 专家权重真实形状 [E, 2048, 512] + 128x128 块，统计验证
"""

import math

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    requant_weight_ue8m0_inplace,
)

torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False, linewidth=120)

FP8_MAX = 448.0  # E4M3 最大正规值


def make_checkpoint(w_true: torch.Tensor, bm: int, bk: int):
    """模拟 checkpoint 的原始 block 量化：fp32 任意 scale（amax/448）。"""
    M, K = w_true.shape
    ws = torch.empty(M // bm, K // bk)
    wq = torch.empty_like(w_true)
    for i in range(M // bm):
        for j in range(K // bk):
            blk = w_true[i * bm:(i + 1) * bm, j * bk:(j + 1) * bk]
            s = blk.abs().max().clamp(1e-4) / FP8_MAX
            ws[i, j] = s
            wq[i * bm:(i + 1) * bm, j * bk:(j + 1) * bk] = blk / s
    return wq.to(torch.float8_e4m3fn), ws


def dequant(wq: torch.Tensor, ws: torch.Tensor, bm: int, bk: int):
    s_exp = torch.repeat_interleave(torch.repeat_interleave(ws, bm, 0), bk, 1)
    return wq.to(torch.float32) * s_exp[: wq.shape[0], : wq.shape[1]]


def is_pow2(x: torch.Tensor) -> bool:
    frac, _ = torch.frexp(x)
    return bool(torch.all(frac == 0.5))


# ============================================================================
print("=" * 76)
print("Part 1  4x4 矩阵，block_size=(2,2) —— 逐元素看清每一步")
print("=" * 76)

BM = BK = 2
# 构造一份“真值”权重：4 个 2x2 块，量级刻意拉开（模拟真实权重分布差异）
w_true = torch.tensor([
    [ 0.0100, -0.0080,  0.3000, -0.2500],
    [-0.0060,  0.0110, -0.3700,  0.2200],
    [ 1.8000, -1.2000,  0.0020, -0.0015],
    [-0.9000,  1.9000,  0.0018,  0.0021],
])

wq, ws = make_checkpoint(w_true, BM, BK)
ptr_w, ptr_s = wq.data_ptr(), ws.data_ptr()

print("\n[输入] checkpoint 形态")
print("wq (fp8 e4m3, 以 float 显示):")
print(wq.to(torch.float32))
print("ws (fp32, 每 2x2 块一个 scale):")
print(ws)
print("ws 的 log2（非整数 ⇒ 不是 2 的幂）:")
print(torch.log2(ws))

w_dq_before = dequant(wq, ws, BM, BK)
print("\n[requant 前反量化] wq*scale ≈ 真值:")
print(w_dq_before)

# ---- 调用真实函数（原地）----
requant_weight_ue8m0_inplace(wq, ws, block_size=(BM, BK))

print("\n[requant 后]")
print("ws (全部变成 2 的幂):")
print(ws)
print("ws 的 log2（整数 ⇒ UE8M0，纯指数可表示）:")
print(torch.log2(ws))
print("每块 scale 变化倍数 new/old ∈ [1, 2):")
old_ws = torch.tensor([[0.011, 0.37], [1.9, 0.0021]]) / FP8_MAX
print(ws / old_ws)
print("wq (fp8 尾数已按新 scale 重新取整):")
print(wq.to(torch.float32))

w_dq_after = dequant(wq, ws, BM, BK)
print("\n[requant 后反量化] 仍 ≈ 真值:")
print(w_dq_after)

print("\n[原地性] weight data_ptr 不变: %s, scale data_ptr 不变: %s"
      % (wq.data_ptr() == ptr_w, ws.data_ptr() == ptr_s))
print("[误差] |dq-真值|.max  requant 前: %.6f   requant 后: %.6f"
      % ((w_dq_before - w_true).abs().max(), (w_dq_after - w_true).abs().max()))

# ============================================================================
print()
print("=" * 76)
print("Part 2  Qwen3.6 真实形状：4 个专家的 down_proj w2 [4, 2048, 512]，块 128x128")
print("=" * 76)

BM = BK = 128
E, N, K = 4, 2048, 512          # 完整权重是 [256, 2048, 512]，这里取 4 个专家演示
w_true3 = torch.randn(E, N, K) * 0.02   # 真实权重量级 (std≈0.02)

wq3 = torch.empty(E, N, K)
ws3 = torch.empty(E, N // BM, K // BK)
for e in range(E):
    q, s = make_checkpoint(w_true3[e], BM, BK)
    wq3[e] = q.to(torch.float32)
    ws3[e] = s
wq3 = wq3.to(torch.float8_e4m3fn)
ws_old = ws3.clone()
ptr_w, ptr_s = wq3.data_ptr(), ws3.data_ptr()

err_before = (torch.stack([dequant(wq3[e], ws3[e], BM, BK) for e in range(E)])
              - w_true3)

requant_weight_ue8m0_inplace(wq3, ws3, block_size=(BM, BK))   # 3D: 内部拍平逐矩阵循环

err_after = (torch.stack([dequant(wq3[e], ws3[e], BM, BK) for e in range(E)])
             - w_true3)

ratio = ws3 / ws_old
print(f"\n矩阵数(拍平后 num_mats): {E}   每矩阵 scale 数: {N//BM}x{K//BK} = {N//BM*(K//BK)}")
print(f"旧 scale 是 2 的幂的比例: {(torch.frexp(ws_old)[0] == 0.5).float().mean():.1%}")
print(f"新 scale 是 2 的幂的比例: {(torch.frexp(ws3)[0] == 0.5).float().mean():.1%}  (is_pow2={is_pow2(ws3)})")
print(f"scale 上调倍数 new/old: min={ratio.min():.4f}  mean={ratio.mean():.4f}  max={ratio.max():.4f}  (理论范围 [1,2))")
print(f"新 scale 下 fp8 值域利用: |wq|.max = {wq3.to(torch.float32).abs().max():.1f} / 448"
      f"  (对应 448/上调倍数，上端动态范围最多让出 1 bit)")
rel_b = err_before.abs().mean() / w_true3.abs().mean()
rel_a = err_after.abs().mean() / w_true3.abs().mean()
print(f"\n量化误差(相对): requant 前 {rel_b:.5f} → requant 后 {rel_a:.5f}"
      f"  (放大 {rel_a/rel_b:.3f}x —— 即 §4.3.3 所述 UE8M0 精度代价)")
print(f"原地性: weight ptr 不变 {wq3.data_ptr() == ptr_w}, scale ptr 不变 {ws3.data_ptr() == ptr_s}")
print(f"dtype 保持: weight {wq3.dtype}, scale {ws3.dtype}")
