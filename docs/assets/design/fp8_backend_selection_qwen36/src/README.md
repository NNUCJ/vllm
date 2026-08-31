# 配图源码

`docs/design/fp8_backend_selection_qwen36.md` 里全部 16 张配图的生成脚本。

## 重新生成

```bash
uv venv --python 3.12 /tmp/plotenv
/tmp/plotenv/bin/python -m pip install matplotlib
cd docs/assets/design/fp8_backend_selection_qwen36/src
for f in fig_*.py; do /tmp/plotenv/bin/python $f; done
```

中文渲染需要系统装有 **Noto Sans CJK SC**。`save()` 按脚本自身位置推导输出目录，无需改路径。

## 文件对应关系

| 脚本 | 产出 | 图号与文档位置 |
| --- | --- | --- |
| `fig_lists.py` | `candidate_lists.png` | 已改为正文表 2（1.3 节），配图保留备用，文档不再引用 |
| `fig_arch_layer.py` | `qwen36_quantized_layer_map.png` | 图 1，1.3 一层的数据流与权重位置（架构图风格）|
| `fig_gdn_core.py` | `gdn_core_expanded.png` | 图 2，1.3 GDN 递推核的内部展开 |
| `fig_quant_config.py` | `quant_config_build.png` | 图 3，2.1 quant_config 的构造 |
| `fig_quant_method_chain.py` | `dense_quant_method_chain.png` | 图 4，2.2.1 dense 装配链 |
| `fig_kernel_pick.py` | `dense_kernel_pick_workflow.png` | 图 5，2.2.1 装配第 ⑥ 步内部的 kernel 选择 |
| `fig_moe_quant_method_chain.py` | `moe_quant_method_chain.png` | 图 6，2.2.2 MoE 装配链 |
| `fig_dense_tree.py` | `dense_backend_tree.png` | 图 7，2.3 dense FP8 后端选择 |
| `fig_moe_tree.py` | `moe_backend_tree.png` | 图 8，2.4 MoE FP8 后端选择 |
| `fig_runtime.py` | `dense_runtime.png` + `moe_runtime.png` | 图 9 / 图 11，3.1 / 3.2 运行期调用链 |
| `fig_quant_fp8_workflow.py` | `quant_fp8_block_workflow.png` | 图 10，3.1 QuantFP8 与 block GEMM 的粒度耦合 |
| `fig_ue8m0_vs_fp32.py` | `ue8m0_vs_fp32_scale.png` | 图 12，4.1 UE8M0 与 fp32 scale 两条反量化通路 |
| `fig_requant_ue8m0.py` | `requant_ue8m0_workflow.png` | 图 13，4.2 requant_weight_ue8m0_inplace 调用流程 |
| `fig_api.py` | `api_surface.png` | 图 14，4.3.1 DeepGEMM 与 CUTLASS 接口全景 |
| `fig_overview.py` | `overview.png` | 图 15，第 5 章开头 |
| `fig_qwen36_deepgemm_gate.py` | `qwen36_dense_deepgemm_gate.png` | 图 16，5.3 dense 不可用性分析 |

另有 `demo_requant_ue8m0.py`：附录 B 的数值演示脚本（直接调用仓库内的
`requant_weight_ue8m0_inplace`，CPU torch 即可运行），非配图源码。

`fig_common.py` 与 `../../fp8_quantization_kernels/src/fig_common.py` 内容一致，
两边独立各留一份，避免跨目录 import。改动其中一份时注意同步。

排版约定和调高画布的方法见
[`../../fp8_quantization_kernels/src/README.md`](../../fp8_quantization_kernels/src/README.md)。
