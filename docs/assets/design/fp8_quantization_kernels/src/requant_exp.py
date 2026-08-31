import torch
torch.manual_seed(0)
FP8_MAX = 448.0
BM = BK = 128

def ceil_ue8m0(x):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))

def block_cast(x, use_ue8m0):
    """复刻 vllm/utils/deep_gemm.py:662 per_block_cast_to_fp8"""
    m, n = x.shape
    v = x.view(-1, BM, n // BK, BK)
    amax = v.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = amax / FP8_MAX
    if use_ue8m0:
        sf = ceil_ue8m0(sf)
    q = (v * (1.0 / sf)).to(torch.float8_e4m3fn)
    return q.view_as(x).contiguous(), sf.view(v.size(0), v.size(2))

def dequant(q, sf):
    m, n = q.shape
    s = sf.repeat_interleave(BM, 0).repeat_interleave(BK, 1)[:m, :n]
    return q.float() * s

def err(a, b):
    return ((a - b).abs().mean() / b.abs().mean()).item() * 100

# 原始权重：正态 + 每块一个离群值，模拟真实权重分布
W = (torch.randn(256, 256) * 0.02).float()
W[30, 40] = 0.9; W[200, 60] = -1.4

print("路径 A：checkpoint 直接用 UE8M0 导出（一次量化）")
qA, sA = block_cast(W, use_ue8m0=True)
print(f"  误差 vs 原始权重  {err(dequant(qA, sA), W):.4f} %")

print("\n路径 B：checkpoint 是 fp32 scale，vLLM 加载时 requant（两次量化）")
q0, s0 = block_cast(W, use_ue8m0=False)          # checkpoint 里的样子
print(f"  第一次量化后（fp32 scale）      {err(dequant(q0, s0), W):.4f} %")
w_dq = dequant(q0, s0)                            # requant 第 1 步：反量化
qB, sB = block_cast(w_dq, use_ue8m0=True)         # requant 第 2 步：重新量化
print(f"  requant 之后（UE8M0 scale）     {err(dequant(qB, sB), W):.4f} %")
print(f"  requant 这一步本身引入的误差     {err(dequant(qB, sB), w_dq):.4f} %  (相对反量化出的 fp32)")

print("\n路径 C：如果偷懒，只把 scale 改成 2 的幂、不动 q")
sC = ceil_ue8m0(s0)
print(f"  误差 vs 原始权重  {err(dequant(q0, sC), W):.2f} %   <- 直接错掉")
print(f"  scale 的变化倍数  min={float((sC/s0).min()):.3f}  max={float((sC/s0).max()):.3f}")

print("\n各 block 的 scale 与 2 的幂的距离（s'/s，1.0 表示本来就是 2 的幂）:")
print(" ", [f"{v:.3f}" for v in (sC / s0).flatten().tolist()])
