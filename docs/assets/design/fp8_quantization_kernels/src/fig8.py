from fig_common import *

fig, ax = new_fig(17, 21.1)
title(ax, "Qwen3.5-MoE-35B-A3B-FP8：checkpoint 事实与加载期",
      "/data/chengjie/models/Qwen3.6-35B-A3B-FP8   ·  config.json 与 safetensors 头实测")

a0 = box(ax, 4, 0, 92, None, top=91.0,
         title="① checkpoint 里写了什么", lines=[
    "config.json  architectures = [\"Qwen3_5MoeForConditionalGeneration\"]     model_type = \"qwen3_5_moe\"      -> vllm/model_executor/models/qwen3_5.py",
    "",
    "quantization_config = { quant_method: \"fp8\",  activation_scheme: \"dynamic\",  fmt: \"e4m3\",  weight_block_size: [128, 128],",
    "                        modules_to_not_convert: [ 648 项 ] }",
    "",
    "text_config   40 层，hidden 2048，head_dim 256，16 q-head / 2 kv-head，attn_output_gate=true",
    "              layer_types = [linear_attention x3, full_attention] x10      full_attention_interval = 4  ->  只有 10 层是 full attention",
    "              num_experts 256，num_experts_per_tok 8，moe_intermediate_size 512，另有 shared_expert",
    "              mtp_num_hidden_layers = 1（MTP 投机头）；vision_config 存在（多模态）",
], color="gray", ls=9.0, align="left", ts=12.5)

b0 = box(ax, 4, 0, 92, None, top=a0[1] - 3.0,
         title="② safetensors 头实测：权重确实是 128x128 block FP8，但 scale 是 BF16", lines=[
    "  model.language_model.layers.3.self_attn.q_proj.weight            F8_E4M3   [8192, 2048]      8192/128=64, 2048/128=16",
    "  model.language_model.layers.3.self_attn.q_proj.weight_scale_inv  BF16      [  64,   16]   <- 注意不是 fp32",
    "  model.language_model.layers.3.mlp.experts.0.gate_proj.weight     F8_E4M3   [ 512, 2048]",
    "  model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv     BF16   [4, 16]",
    "",
    "vLLM 侧 create_fp8_scale_parameter（fp8_utils.py:1266）默认建 torch.float32 的 BlockQuantScaleParameter，",
    "weight_loader 里 copy_ 时 bf16 自动 upcast 成 fp32——所以 checkpoint 的 bf16 scale 不需要额外处理，",
    "但它比 fp32 少 16 bit 尾数，等于这份权重的 block scale 本身就只有 bf16 的精度。",
], color="blue", ls=9.0, align="left", ts=12.5)

cy = b0[1] - 3.0
c0 = box(ax, 4, 0, 45, None, top=cy,
         title="③ 量化了的层（走 FP8 GEMM）", lines=[
    "full_attention 层（10 个）",
    "  q_proj  [8192, 2048]   k_proj [512, 2048]",
    "  v_proj  [ 512, 2048]   o_proj [2048, 4096]",
    "",
    "linear_attention 层（30 个，Gated DeltaNet）",
    "  in_proj_qkv [8192, 2048]",
    "  in_proj_z   [4096, 2048]",
    "  out_proj    [2048, 4096]",
    "",
    "MoE（每层）",
    "  experts.{0..255}.gate/up_proj [512, 2048]",
    "  experts.{0..255}.down_proj    [2048, 512]",
    "  shared_expert.gate/up/down    同上形状",
], color="green", ls=9.0, align="left", ts=12.5)

d0 = box(ax, 51, 0, 45, None, top=cy,
         title="③' modules_to_not_convert：648 项", lines=[
    "整个 visual 塔      27 层 x (qkv, proj, fc1, fc2)",
    "                   + deepstack_merger_list",
    "input/post_attention_layernorm      40 x 2",
    "mlp.gate（router）                  40",
    "mlp.shared_expert_gate              40",
    "self_attn.q_norm / k_norm           10 x 2",
    "linear_attn.A_log / conv1d / dt_bias 30 x 3",
    "linear_attn.in_proj_a / in_proj_b   30 x 2",
    "linear_attn.in_proj_ba / norm       30 x 2",
    "",
    "规律：router、门控、归一化、卷积、状态参数",
    "全部保持 BF16——它们要么太小，要么对量化",
    "误差敏感（router 一错，专家就选错）。",
], color="orange", ls=9.0, align="left", ts=12.5)

e0 = box(ax, 4, 0, 92, None, top=min(c0[1], d0[1]) - 3.0,
         title="④ 加载期：一条 config 变成两套 kernel 选择", lines=[
    "Fp8Config(weight_block_size=[128,128], activation_scheme=\"dynamic\")      fp8.py:95",
    "  -> block_quant = True，act_q_static = False",
    "  -> 激活 GroupShape(1, 128)   权重 GroupShape(128, 128)                  fp8.py:301",
    "",
    "     +-- 普通 Linear 层（q/k/v/o_proj、in_proj_qkv/z、out_proj、shared_expert.*）",
    "     |     Fp8LinearMethod -> _POSSIBLE_FP8_BLOCK_KERNELS[CUDA]           kernels/linear/__init__.py:355",
    "     |       [FlashInfer+DeepGEMM, DeepGemm, Cutlass, Marlin, Triton, Humming]  取第一个 can_implement 的",
    "     |     DeepGemm 的门禁 should_use_deepgemm_for_fp8_linear：out_dtype==bf16 且 N%64==0 且 K%128==0",
    "     |       这份权重全部满足（8192/512/2048/4096 都是 64 的倍数，2048/4096/512 都是 128 的倍数）",
    "     |",
    "     +-- FusedMoE 层（256 routed experts）",
    "           Fp8MoEMethod -> select_fp8_moe_backend                          fused_moe/oracle/fp8.py:271",
    "           默认优先级并不是 DeepGEMM 优先，见下一张图",
    "",
    "process_weights_after_loading（只在启动时跑一次）",
    "  DeepGemm 路径 -> deepgemm_post_process_fp8_weight_block                  fp8_utils.py:1089",
    "     use_e8m0 时 requant_weight_ue8m0_inplace：把权重反量化回 fp32，再用 2 的幂 scale 重新量化一遍",
    "     再 transform_sf_into_required_layout 把 scale 摆成目标架构的布局",
], color="purple", ls=9.0, align="left", ts=12.5)

import sys
print("bottom =", e0[1], file=sys.stderr)
save(fig, "qwen35_deepgemm_load.png")
