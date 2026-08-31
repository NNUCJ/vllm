import math, torch
FP8_MAX = 448.0

def q_e4m3(t):
    return t.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

x = torch.tensor([512.0, 37.0, -2.5, 0.125, 0.0195, -0.0007], dtype=torch.float32)

amax = x.abs().max().item()
s_tok = amax / FP8_MAX
s_ue8 = 2.0 ** math.ceil(math.log2(s_tok))
s_ten = 4096.0 / FP8_MAX          # 同一 batch 里另有一行 amax=4096

rows = []
for name, s in [("per-token", s_tok), ("UE8M0", s_ue8), ("per-tensor", s_ten)]:
    q = q_e4m3(x / s)
    rows.append((name, s, q, q.float() * s))

print(f"amax={amax}  s_token={s_tok!r}  s_ue8m0={s_ue8}  s_tensor={s_ten!r}\n")
hdr = f"{'x':>12} | {'x/s':>12} {'q':>10} {'bits':>5} {'deq':>12} {'err%':>7}"
for name, s, q, deq in rows:
    print(f"=== {name}  scale={s!r}")
    print(hdr)
    bits = q.view(torch.uint8)
    for i in range(len(x)):
        e = abs(deq[i].item()-x[i].item())/abs(x[i].item())*100
        print(f"{x[i].item():12.6g} | {(x[i]/s).item():12.6g} {q[i].float().item():10.6g} "
              f" 0x{bits[i].item():02X} {deq[i].item():12.6g} {e:7.2f}")
    print()

print("--- 不缩放直接转 ---")
qr = q_e4m3(x).float()
for i in range(len(x)):
    print(f"{x[i].item():12.6g} -> {qr[i].item():12.6g}")

print("\n--- E4M3 在 [1,16) 的所有可表示值 ---")
vals = []
v = 1.0
for e in range(4):
    for m in range(8):
        vals.append((1 + m/8) * 2**e)
print(vals[:16])
print("相邻间距 / 值 =", 0.125, "->", "半 ulp 相对误差上界 =", 1/16)

print("\n--- subnormal 下限 ---")
print("最小 normal =", 2**-6, " 最小 subnormal =", 2**-9)
for t in [0.0195, 0.0007]:
    for name, s in [("per-token", s_tok), ("per-tensor", s_ten)]:
        r = t/s
        qq = q_e4m3(torch.tensor([r])).float().item()
        print(f"{t} / {name}({s:.4f}) = {r:.6g} -> q={qq:.6g} -> deq={qq*s:.6g}")
