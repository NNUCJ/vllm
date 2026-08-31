"""逐字节实证：读取 Qwen3.6-35B-A3B-FP8 真实权重的一层，解码 FP8 格式与 scale。"""

import torch
from safetensors import safe_open

P = "/data/chengjie/models/Qwen3.6-35B-A3B-FP8/layers-0.safetensors"
W = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
S = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv"

with safe_open(P, framework="pt") as f:
    wq = f.get_tensor(W)          # F8_E4M3
    ws = f.get_tensor(S)          # BF16

print(f"weight          : dtype={wq.dtype}, shape={list(wq.shape)}")
print(f"weight_scale_inv: dtype={ws.dtype}, shape={list(ws.shape)}")
print(f"块映射: {wq.shape[0]}/128 = {wq.shape[0]//128}, {wq.shape[1]}/128 = {wq.shape[1]//128}"
      f"  → 每个 128x128 块对应 1 个 scale，共 {ws.numel()} 个")

# ---- 1. 逐字节解码前 4 个元素 ----
raw = wq[0, :4].view(torch.uint8)
print("\n[1] weight[0, 0:4] 的原始字节 → E4M3 位域解码")
for i, b in enumerate(raw.tolist()):
    s_bit = b >> 7
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0:
        val = ((-1) ** s_bit) * (m / 8) * 2 ** (-6)          # subnormal
        kind = "subnormal"
    else:
        val = ((-1) ** s_bit) * (1 + m / 8) * 2 ** (e - 7)   # normal, bias=7
        kind = "normal"
    torch_val = wq[0, i].to(torch.float32).item()
    print(f"  byte=0x{b:02x} = 0b{b:08b}  S={s_bit} E={e:2d}({e-7:+d}) M={m}"
          f"  → {kind} 手算 {val:+.6f}  torch 解码 {torch_val:+.6f}")

# ---- 2. 块 (0,0) 的 scale 与反量化 ----
s00 = ws[0, 0].to(torch.float32).item()
blk = wq[:128, :128].to(torch.float32)
print(f"\n[2] 块(0,0) 的 scale_inv = {s00:.10f}  (bf16 原值)")
print(f"    log2(scale) = {torch.log2(torch.tensor(s00)).item():.4f}  → 非整数，不是 2 的幂")
print(f"    块内 |wq|.max = {blk.abs().max().item():.1f} / 448  (量化时 amax 映射到 E4M3 满量程)")
print(f"    反量化后块 amax = |wq|.max × scale = {blk.abs().max().item() * s00:.6f}")
print(f"    weight[0,0] 反量化 = {wq[0,0].to(torch.float32).item():+.4f} × {s00:.6f}"
      f" = {wq[0,0].to(torch.float32).item() * s00:+.8f} (bf16 权重原值的近似)")

# ---- 3. 全部 1024 个 scale 的统计 ----
wsf = ws.to(torch.float32)
frac, _ = torch.frexp(wsf)
print(f"\n[3] 全部 {ws.numel()} 个 scale 统计")
print(f"    min={wsf.min():.6e}  max={wsf.max():.6e}  mean={wsf.mean():.6e}")
print(f"    是 2 的幂的比例: {(frac == 0.5).float().mean().item():.1%}  (checkpoint 是任意 bf16 scale)")

# ---- 4. 每块 |wq|.max 分布：验证 amax→448 的量化约定 ----
blocks = wq.to(torch.float32).view(64, 128, 16, 128)
bmax = blocks.abs().amax(dim=(1, 3))                     # [64, 16]
print(f"\n[4] 每块 |wq|.max 分布（若量化按 amax/448 取 scale，应聚在 448 附近）")
print(f"    min={bmax.min():.0f}  median={bmax.median():.0f}  max={bmax.max():.0f}")
print(f"    ==448 的块占比: {(bmax == 448).float().mean().item():.1%}"
      f"   >=416 的块占比: {(bmax >= 416).float().mean().item():.1%}")

# ---- 5. fp8 数值分布 ----
wf = wq.to(torch.float32)
print(f"\n[5] 1677 万个 fp8 元素的分布")
print(f"    零值占比 {(wf == 0).float().mean().item():.2%}   饱和(|x|=448)占比 {(wf.abs() == 448).float().mean().item():.4%}")
print(f"    去量纲后权重量级: |wq×s|.mean = {(wf.abs() * ws.to(torch.float32).repeat_interleave(128,0).repeat_interleave(128,1)).mean().item():.5f}")
