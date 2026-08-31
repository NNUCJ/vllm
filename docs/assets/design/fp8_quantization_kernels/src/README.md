# 配图源码

`docs/design/fp8_quantization_kernels.md` 里全部配图的生成脚本（16 张，`fig14.py` 一个脚本产出 3 张）。

## 重新生成

脚本用 matplotlib 画图，`save()` 按脚本自身位置推导输出目录（`src/` 的上一级），
所以在本目录下直接运行即可，不需要改路径：

```bash
uv venv --python 3.12 /tmp/plotenv
/tmp/plotenv/bin/python -m pip install matplotlib   # 仅此一个依赖
cd docs/assets/design/fp8_quantization_kernels/src
for i in $(seq 1 14); do /tmp/plotenv/bin/python fig$i.py; done
```

中文渲染需要系统装有 **Noto Sans CJK SC**（`fc-list | grep "Noto Sans CJK SC"` 验证）。
缺字体不会报错，但中文会变成豆腐块。

## 文件对应关系

| 脚本 | 产出 | 文档位置 |
| --- | --- | --- |
| `fig1.py` | `fp8_formats.png` | 一、浮点数是怎么编码的 |
| `fig11.py` | `format_zoo.png` | 三、格式谱系 |
| `fig12.py` | `quant_math.png` | 四、量化的数学 |
| `fig6.py` | `worked_example.png` | 五、走一遍具体的数字 |
| `fig13.py` | `quant_granularity_concept.png` | 六、缩放粒度 |
| `fig3.py` | `csrc_fp8_map.png` | 八、csrc 下的 FP8 算子地图 |
| `fig2.py` | `quant_granularity.png` | 九、激活量化 kernel 的实现 |
| `fig4.py` | `runtime_path.png` | 十、GEMM 后端全景 |
| `fig7.py` | `deepgemm_vs_cutlass.png` | 10.2 DeepGEMM |
| `fig5.py` | `kv_cache_fp8.png` | 十二、KV cache 的 FP8 通路 |
| `fig8.py` | `qwen35_deepgemm_load.png` | 16.1 checkpoint 里有什么 |
| `fig14.py` | `block_quant_weight.png` | 16.2 权重侧：128×128 块与 weight_scale_inv |
| `fig14.py` | `block_quant_activation.png` | 16.2 激活侧：1×128 组与 As |
| `fig14.py` | `block_quant_k_align.png` | 16.2 K 向对齐 |
| `fig9.py` | `qwen35_deepgemm_runtime.png` | 16.3 一层的真实算子序列 |
| `fig10.py` | `qwen35_deepgemm_backend_matrix.png` | 16.5 谁真的会走 DeepGEMM |

另有 `inspect_real_weight.py`：16.2 的实证脚本，从
`/data/chengjie/models/Qwen3.6-35B-A3B-FP8/layers-0.safetensors` 读取
`linear_attn.in_proj_qkv` 的权重与 scale，逐字节解码 E4M3 并统计块量化特征。
它不是配图脚本，依赖 `safetensors` + CPU torch，直接运行即可复现文中所有数字。

`fig_common.py` 是公共的 `box` / `arrow` / `title` / `save` 辅助函数，各脚本 `from fig_common import *`。
布局靠 `box(..., top=y)` 自上而下堆叠，高度由文本实测（`_meas`）自动算。
加内容后如果最后一个 box 的 `bottom`（脚本末尾打到 stderr）变成负数，说明超出画布会被裁掉，
把 `new_fig(w, h)` 的 `h` 调大即可。

## 数值出处

文档里那些「实算不是手推」的数字来自这三个脚本：

| 脚本 | 算什么 |
| --- | --- |
| `calc.py` | bf16 → fp8 的 per-tensor 量化误差表 |
| `calc2.py` | 第五节那 6 个值（512.0 / 37.0 / -2.5 / 0.125 / 0.0195 / -0.0007）在 per-token、UE8M0、per-tensor 三种 scale 下的量化与反量化结果 |
| `requant_exp.py` | 复刻 `per_block_cast_to_fp8`，验证 UE8M0 重量化的误差 |

这三个需要 torch，跑在项目自己的 `.venv` 里即可。
