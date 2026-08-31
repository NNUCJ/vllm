from fig_common import *

fig, ax = new_fig(18, 19.4)
title(ax, "三张候选表的源码原文",
      "dense 两张在 kernels/linear/__init__.py；MoE 一张在 fused_moe/oracle/fp8.py")

y = 91.0

# ---- 左：非 block ----
lb = box(ax, 1.5, 0, 48, None, top=y,
         title="① dense · per-tensor / per-token", lines=[
    "vllm/model_executor/kernels/linear/__init__.py:322",
    "",
    "322  _POSSIBLE_FP8_KERNELS: dict[...] = {",
    "323      PlatformEnum.CUDA: [",
    "324          MarlinFP8ScaledMMLinearKernel,",
    "325          FlashInferFP8ScaledMMLinearKernel,",
    "326          CutlassFP8ScaledMMLinearKernel,",
    "327          PerTensorTorchFP8ScaledMMLinearKernel,",
    "328          ChannelWiseTorchFP8ScaledMMLinearKernel,",
    "329          HummingFP8ScaledMMLinearKernel,",
    "330      ],",
    "331      PlatformEnum.ROCM: [",
    "332          AiterHipbMMPerTokenFp8ScaledMMLinearKernel,",
    "333          AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,",
    "334          AiterPerTokenFp8ScaledMMLinearKernel,",
    "335          ROCmFP8ScaledMMLinearKernel,",
    "336          PerTensorTorchFP8ScaledMMLinearKernel,",
    "337          RowWiseTorchFP8ScaledMMLinearKernel,",
    "338          ChannelWiseTorchFP8ScaledMMLinearKernel,",
    "339      ],",
    "340      PlatformEnum.CPU: [",
    "341          PerTensorTorchFP8ScaledMMLinearKernel,",
    "342          ChannelWiseTorchFP8ScaledMMLinearKernel,",
    "343      ],",
    "344      PlatformEnum.XPU: [",
    "345          XPUW8A16FP8LinearKernel,",
    "346          XPUW8A8FP8LinearKernel,",
    "347      ],",
    "348  }",
], color="blue", ls=8.4, align="left", ts=12)

# ---- 右：block ----
rb = box(ax, 51, 0, 47.5, None, top=y,
         title="② dense · block（weight_block_size 存在）", lines=[
    "vllm/model_executor/kernels/linear/__init__.py:352",
    "",
    "351  # in priority/performance order (when available)",
    "352  _POSSIBLE_FP8_BLOCK_KERNELS: dict[",
    "353      PlatformEnum, list[type[Fp8BlockScaled...]]",
    "354  ] = {",
    "355      PlatformEnum.CUDA: [",
    "356          FlashInferFp8DeepGEMMDynamicBlockScaledKernel,",
    "357          DeepGemmFp8BlockScaledMMKernel,",
    "358          CutlassFp8BlockScaledMMKernel,",
    "359          MarlinFP8ScaledMMLinearKernel,",
    "360          TritonFp8BlockScaledMMKernel,",
    "361          HummingFP8ScaledMMLinearKernel,",
    "362      ],",
    "363      PlatformEnum.ROCM: [",
    "364          AiterFp8BlockScaledMMKernel,",
    "365          TritonFp8BlockScaledMMKernel,",
    "366      ],",
    "367      PlatformEnum.CPU: [",
    "368          CPUFp8BlockScaledMMKernel,",
    "369      ],",
    "370      PlatformEnum.XPU: [",
    "371          XPUFp8BlockScaledMMKernel,",
    "372          TritonFp8BlockScaledMMKernel,",
    "373      ],",
    "374  }",
], color="teal", ls=8.4, align="left", ts=12)

# ---- 下：MoE ----
y2 = min(lb[1], rb[1]) - 3.4
mb = box(ax, 1.5, 0, 97, None, top=y2,
         title="③ MoE · 唯一一张表，不分平台（平台差异靠后面的重排和门禁体现）", lines=[
    "vllm/model_executor/layers/fused_moe/oracle/fp8.py:80          —— 定义在 _get_priority_backends 函数体内，每次调用重建一份",
    "",
    "  80      _AVAILABLE_BACKENDS = [                      86          Fp8MoeBackend.TRITON,",
    "  81          Fp8MoeBackend.AITER,                     87          Fp8MoeBackend.MARLIN,",
    "  82          Fp8MoeBackend.FLASHINFER_TRTLLM,         88          Fp8MoeBackend.HUMMING,",
    "  83          Fp8MoeBackend.FLASHINFER_CUTLASS,        89          Fp8MoeBackend.BATCHED_DEEPGEMM,",
    "  84          Fp8MoeBackend.DEEPGEMM,                  90          Fp8MoeBackend.BATCHED_VLLM_CUTLASS,",
    "  85          Fp8MoeBackend.VLLM_CUTLASS,              91          Fp8MoeBackend.BATCHED_TRITON,",
    "                                                       92          Fp8MoeBackend.XPU,",
    "                                                       93          Fp8MoeBackend.CPU,",
    "                                                       94          Fp8MoeBackend.HPC,",
    "                                                       95      ]",
], color="orange", ls=8.6, align="left", ts=12)

# ---- 注 ----
y3 = mb[1] - 3.4
nb = box(ax, 1.5, 0, 97, None, top=y3,
         title="三张表怎么被消费", lines=[
    "  ①②  顺序即优先级，choose_scaled_mm_linear_kernel :504 从头扫，第一个同时过三道门（DISABLED_KERNELS / is_supported /",
    "      can_implement）的胜出，构造期定死。表本身是模块级常量，运行期不改。",
    "      走哪一张由粒度决定：activation_quant_key.scale.group_shape.is_per_group() 为真走 ②，否则走 ①（:594）。",
    "",
    "  ③   顺序只是起点。select_fp8_moe_backend :271 先用 _get_priority_backends 重排（_move_to_front，只挪不删），",
    "      再依次过 --moe-backend / VLLM_USE_DEEP_GEMM / VLLM_TEST_FORCE_FP8_MARLIN / AITER / allow_vllm_cutlass 五个开关",
    "      （这几个会 remove 元素或直接强制返回），最后才从头扫描试 is_supported_config()。",
    "",
    "  CUDA 上几个实际落不到的位置：①的 Marlin 需要 VLLM_TEST_FORCE_FP8_MARLIN；①的 FlashInfer 只在 cc>=100；",
    "  ②的 FlashInferFp8DeepGEMM... 只在 SM90（符号名硬编码 fp8_blockscale_gemm_sm90）；",
    "  ③的 VLLM_CUTLASS 和 BATCHED_VLLM_CUTLASS 在 Fp8MoEMethod 下恒被 remove（allow_vllm_cutlass=False）。",
], color="purple", ls=8.6, align="left", ts=12.5)

import sys
print("bottom =", nb[1], file=sys.stderr)
save(fig, "candidate_lists.png")
