import torch

FP8_MAX = 448.0
x = torch.tensor([0.1234, -2.5, 37.0, 0.0007, -128.0, 512.0, 3.3, -0.045],
                 dtype=torch.bfloat16)
xf = x.float()
print("x(bf16->f32):", [f"{v:.6g}" for v in xf.tolist()])

amax = xf.abs().max().item()
scale = amax / FP8_MAX
inv = 1.0 / scale
print(f"amax={amax}  scale={scale!r}  1/scale={inv!r}")

scaled = (xf * inv).clamp(-FP8_MAX, FP8_MAX)
q = scaled.to(torch.float8_e4m3fn)
bits = q.view(torch.uint8)
deq = q.float() * scale

print("\n idx | x            | x/scale      | q(e4m3)      | bits | deq          | rel err")
for i in range(len(xf)):
    rel = abs(deq[i].item()-xf[i].item())/max(abs(xf[i].item()),1e-30)
    print(f" {i}   | {xf[i].item():12.6g} | {scaled[i].item():12.6g} | "
          f"{q[i].float().item():12.6g} | 0x{bits[i].item():02X} | {deq[i].item():12.6g} | {rel*100:8.3f}%")

# per-tensor 对比：同一批里有个大离群行
print("\n--- per-tensor 与 per-token 对比 ---")
big = 4096.0
s_tensor = big / FP8_MAX
q_t = (xf / s_tensor).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
deq_t = q_t.float() * s_tensor
print(f"per-tensor scale={s_tensor!r} (amax={big})")
for i in range(len(xf)):
    print(f" {i}  x={xf[i].item():12.6g}  per-token={deq[i].item():12.6g}  per-tensor={deq_t[i].item():12.6g}")

def rmse(a, b):
    return (torch.sqrt(((a-b)**2).mean())).item()
print(f"\nRMSE per-token = {rmse(deq, xf):.6g}   RMSE per-tensor = {rmse(deq_t, xf):.6g}")

# 不做 scale 直接转
print("\n--- 不缩放直接转 e4m3 ---")
q_raw = xf.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
print([f"{v:.6g}" for v in q_raw.tolist()])

# GEMM 反量化示例
print("\n--- 点积的反量化 ---")
w = torch.tensor([0.5, -1.25, 2.0, 0.75], dtype=torch.bfloat16).float()
a = torch.tensor([1.0, -2.0, 3.0, 4.0], dtype=torch.bfloat16).float()
sa = a.abs().max().item()/FP8_MAX
sw = w.abs().max().item()/FP8_MAX
aq = (a/sa).to(torch.float8_e4m3fn); wq = (w/sw).to(torch.float8_e4m3fn)
acc = (aq.float()*wq.float()).sum().item()
print(f"a={a.tolist()} sa={sa!r} aq={aq.float().tolist()}")
print(f"w={w.tolist()} sw={sw!r} wq={wq.float().tolist()}")
print(f"acc(int-ish fp32)={acc}  out=acc*sa*sw={acc*sa*sw!r}   ref={(a*w).sum().item()!r}")

# min_scaling_factor
print("\nmin_scaling_factor =", 1.0/(448.0*512.0))
