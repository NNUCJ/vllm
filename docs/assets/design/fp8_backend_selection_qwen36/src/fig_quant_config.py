from fig_common import *

# ---------------- quant_config 的构造 ----------------
fig, ax = new_fig(17, 12.6)
title(ax, "quant_config 是怎么被造出来的",
      "整个进程只有一个 Fp8Config 实例，40 层里所有量化层共享它")

G = 3.0
y = 90.0
a = box(ax, 4, 0, 92, None, top=y,
        title="① config.json 里的 quantization_config —— 只是一个 dict", lines=[
    '{"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128,128],',
    ' "modules_to_not_convert": [...648 项...]}',
], color="gray", ls=9.2, align="left", ts=12)

y = a[1] - G
b = box(ax, 4, 0, 92, None, top=y,
        title="② 定方法名   ModelConfig._verify_quantization   config/model.py:1016", lines=[
    'quant_method = quant_cfg["quant_method"]                     # "fp8"',
    'for name in quantization_methods:                            # 按 overrides 优先级逐个问',
    '    if method.override_quantization_method(...): 改写并 break # 如 gptq -> gptq_marlin',
    'self.quantization = "fp8"                                    # 到此只有字符串，没有对象',
], color="blue", ls=9.2, align="left", ts=12)
arrow(ax, (50, a[1]), (50, y + 0.3))

y = b[1] - G
c = box(ax, 4, 0, 92, None, top=y,
        title="③ 造对象   VllmConfig.__post_init__ config/vllm.py:925 -> _get_quantization_config :622", lines=[
    'quant_cls = get_quantization_config("fp8")     quantization/__init__.py:108   # dict 查表',
    'cfg       = quant_cls.from_config(hf_quant_config)          fp8.py:156       # 解析字段',
    '三道校验（建任何一层之前就跑，不过直接 raise）：',
    '  capability < Fp8Config.get_min_capability()=75 / dtype not in [bf16,fp16] / maybe_update_config',
], color="green", ls=9.2, align="left", ts=12)
arrow(ax, (50, b[1]), (50, y + 0.3))

y = c[1] - G
d = box(ax, 4, 0, 92, None, top=y,
        title="④ 两处全局后处理（都是往同一个对象上写字段）", lines=[
    'config/vllm.py:931   should_auto_disable_deep_gemm("qwen3_5_moe_text") -> use_deep_gemm = False',
    'model_loader/utils.py:284  configure_quant_config()  -> packed_modules_mapping = 模型类的映射表',
    '   {"qkv_proj":[q,k,v], "gate_up_proj":[...], "in_proj_qkvz":["in_proj_qkv","in_proj_z"],',
    '    "in_proj_ba":["in_proj_b","in_proj_a"]}     # 按引用传，is_layer_skipped 拆融合名靠它',
], color="red", ls=9.2, align="left", ts=12)
arrow(ax, (50, c[1]), (50, y + 0.3))

y = d[1] - G
e = box(ax, 4, 0, 92, None, top=y,
        title="⑤ 成品  vllm_config.quant_config", lines=[
    'Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic",',
    '          weight_block_size=[128,128], ignored_layers=[...648...], store_dtype=None,',
    '          use_deep_gemm=False, packed_modules_mapping={...})',
    '不含任何 kernel 信息：不知道在哪张卡上，也不知道用 CUTLASS 还是 DeepGEMM——那是每层建',
    'quant_method 时才决定的（下两张图）。',
], color="teal", ls=9.2, align="left", ts=12)
arrow(ax, (50, d[1]), (50, y + 0.3))

import sys
print("bottom =", e[1], file=sys.stderr)
save(fig, "quant_config_build.png")
