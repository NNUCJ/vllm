# DeepSeek-V4 技术报告分析

> 来源：`https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf`
>
> 报告标题：*DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*
>
> 分析重点：模型架构、长上下文机制、训练/后训练流程、推理系统设计，以及对 vLLM 适配的影响。

---

## 1. 总体结论

DeepSeek-V4 的核心目标不是简单扩大参数规模，而是解决 **百万 token 上下文下的推理效率瓶颈**。报告发布了两个 MoE 模型：

| 模型 | 总参数 | 激活参数 | 上下文长度 | 定位 |
|---|---:|---:|---:|---|
| DeepSeek-V4-Flash | 284B | 13B | 1M tokens | 高性价比、低推理成本 |
| DeepSeek-V4-Pro | 1.6T | 49B | 1M tokens | 高能力、最大 reasoning effort |

和 DeepSeek-V3.2 相比，DeepSeek-V4-Pro 在 1M-token context 下虽然激活参数更多，但报告声称只需要：

- `27%` 的 single-token inference FLOPs；
- `10%` 的 KV cache；
- DeepSeek-V4-Flash 进一步降低到 `10%` FLOPs 和 `7%` KV cache。

这说明 V4 的主要创新点在 **attention 结构和 KV cache 组织**，而不是 MoE feed-forward 本身。

---

## 2. 相比 DeepSeek-V3/V3.2 的关键变化

DeepSeek-V4 继承了以下设计：

- DeepSeekMoE；
- Multi-Token Prediction，MTP；
- auxiliary-loss-free load balancing；
- DeepSeek 系列已有的 tokenizer、FIM、token splitting 等训练策略。

主要新增或修改点：

| 模块 | V4 变化 | 目的 |
|---|---|---|
| Attention | 引入 CSA + HCA 混合注意力 | 降低长上下文 FLOPs 和 KV cache |
| Residual | 引入 mHC | 改善深层信号传播和训练稳定性 |
| Optimizer | 大多数模块使用 Muon | 更快收敛、更稳定训练 |
| MoE routing | affinity 从 Sigmoid 改为 `Sqrt(Softplus)` | 调整路由打分行为 |
| 前几层 FFN | dense FFN 改为 Hash routing MoE | 让所有 Transformer blocks 使用 MoE |
| Quantization | FP4 QAT 用于 MoE expert weights 和 CSA indexer QK | 降低存储和计算成本 |
| 推理系统 | 异构 KV cache + on-disk KV cache | 支持百万上下文和 shared-prefix 复用 |

报告中有一个明显笔误：DeepSeek-V4-Pro 模型设置段落最后写成 “DeepSeek-V4-Flash comprises 1.6T total parameters”，根据上下文应为 DeepSeek-V4-Pro。

---

## 3. 模型规格

### 3.1 DeepSeek-V4-Flash

| 项 | 配置 |
|---|---:|
| Transformer layers | 43 |
| hidden size | 4096 |
| 总参数 | 284B |
| 激活参数 | 13B |
| 前两层 attention | pure sliding window attention |
| 后续 attention | CSA / HCA interleaved |
| MoE layers | 所有 Transformer blocks |
| 前 3 个 MoE layers | Hash routing |
| routed experts | 256 |
| shared experts | 1 |
| activated routed experts/token | 6 |
| expert intermediate size | 2048 |
| MTP depth | 1 |
| mHC expansion factor `n_hc` | 4 |
| Sinkhorn-Knopp iterations | 20 |

CSA/HCA 配置：

| 项 | 配置 |
|---|---:|
| CSA compression rate `m` | 4 |
| CSA indexer query heads | 64 |
| CSA indexer head dim | 128 |
| CSA attention top-k | 512 |
| HCA compression rate `m'` | 128 |
| query heads | 64 |
| head dim `c` | 512 |
| query compression dim `d_c` | 1024 |
| output projection groups `g` | 8 |
| intermediate output dim `d_g` | 1024 |
| sliding window size `n_win` | 128 |

### 3.2 DeepSeek-V4-Pro

| 项 | 配置 |
|---|---:|
| Transformer layers | 61 |
| hidden size | 7168 |
| 总参数 | 1.6T |
| 激活参数 | 49B |
| 前两层 attention | HCA |
| 后续 attention | CSA / HCA interleaved |
| MoE layers | 所有 Transformer blocks |
| 前 3 个 MoE layers | Hash routing |
| routed experts | 384 |
| shared experts | 1 |
| activated routed experts/token | 6 |
| expert intermediate size | 3072 |
| MTP depth | 1 |
| mHC expansion factor `n_hc` | 4 |
| Sinkhorn-Knopp iterations | 20 |

CSA/HCA 配置：

| 项 | 配置 |
|---|---:|
| CSA compression rate `m` | 4 |
| CSA indexer query heads | 64 |
| CSA indexer head dim | 128 |
| CSA attention top-k | 1024 |
| HCA compression rate `m'` | 128 |
| query heads | 128 |
| head dim `c` | 512 |
| query compression dim `d_c` | 1536 |
| output projection groups `g` | 16 |
| intermediate output dim `d_g` | 1024 |
| sliding window size `n_win` | 128 |

---

## 4. 混合注意力：CSA + HCA

DeepSeek-V4 的长上下文能力主要来自混合注意力机制。它没有继续沿用传统 dense attention 或普通 MLA 路线，而是组合两类 compressed attention：

- CSA：Compressed Sparse Attention；
- HCA：Heavily Compressed Attention。

### 4.1 CSA：Compressed Sparse Attention

CSA 的流程：

1. 每 `m` 个 token 的 KV 被压缩成一个 compressed KV entry；
2. 通过 lightning indexer 为 query 计算 compressed KV block 的相关性分数；
3. 只选择 top-k compressed KV entries 做 core attention；
4. 再额外拼接最近 `n_win` 个未压缩 sliding window KV entries，用于保留局部细粒度依赖。

V4-Pro 中 `m=4`，CSA top-k 为 `1024`。这意味着在 1M token 上下文下，原始 KV 数先压缩到约 `1M / 4 = 256K` 个 compressed entries，再从中选择 `1024` 个参与 core attention。

CSA 的关键性质：

- 长上下文复杂度从“对所有历史 token 做 attention”变成“对少量 selected compressed blocks 做 attention”；
- 需要额外维护 indexer KV；
- 需要 top-k selection，这对推理 kernel 和 KV cache layout 都是新挑战；
- 精度依赖压缩器和 sparse selector 是否能保留关键信息。

### 4.2 HCA：Heavily Compressed Attention

HCA 更激进：

1. 每 `m'=128` 个 token 的 KV 合成一个 compressed KV entry；
2. 不做 sparse selection；
3. query 对所有 heavily compressed KV entries 做 dense attention；
4. 同样拼接最近 `n_win=128` 的 sliding window branch。

HCA 的作用是用极高压缩率换稳定的全局覆盖。它不像 CSA 依赖 top-k selector，因此不会漏选 block，但压缩损失更大。

### 4.3 为什么 CSA 和 HCA 要混合

二者取舍不同：

| Attention | 优点 | 风险 |
|---|---|---|
| CSA | 保留更多局部/语义相关信息，top-k 选择更细 | selector 可能漏选关键上下文，kernel 更复杂 |
| HCA | 极高压缩，全局覆盖，计算稳定 | 单个 entry 压缩 128 token，信息损失更大 |

DeepSeek-V4 采用 interleaved layout，让不同层以不同方式处理长上下文。直观上，HCA 提供全局低成本感知，CSA 提供更细粒度的稀疏检索能力。

### 4.4 附加设计

报告还提到几个 attention 细节：

- Query 和 compressed KV entry 在 core attention 前做 RMSNorm，避免 attention logits 爆炸；
- RoPE 只应用到最后 64 维；
- 对 attention output 也做反向 RoPE，使输出携带相对位置信息；
- 使用 attention sink，使每个 head 的 attention mass 可以小于 1；
- KV 存储采用 BF16 + FP8 混合格式：RoPE 维度 BF16，其余维度 FP8；
- CSA indexer QK 路径使用 FP4。

---

## 5. mHC：Manifold-Constrained Hyper-Connections

mHC 是对 residual connection 的增强。普通 Transformer 的 residual stream 是一条 `d` 维向量，mHC 将 residual stream 扩展为 `n_hc × d`，其中 V4 使用 `n_hc=4`。

它引入三个映射：

- `A_l`：pre-block mixing，把扩展 residual stream 映射成当前层输入；
- `B_l`：residual mixing，在扩展 residual stream 内部混合；
- `C_l`：post-block mixing，把当前层输出写回扩展 residual stream。

核心创新是约束 `B_l` 为 doubly stochastic matrix，即每行每列和为 1 且元素非负。这个矩阵集合属于 Birkhoff polytope，有两个重要稳定性性质：

- spectral norm 不超过 1，避免 residual transformation 放大信号；
- doubly stochastic matrix 乘积仍在同一集合内，深层堆叠更稳定。

实现上，V4 使用 Sinkhorn-Knopp 迭代把原始参数投影到 doubly stochastic manifold，迭代次数为 20。

工程影响：

- mHC 会增加 activation memory 和额外矩阵运算；
- 报告通过 fused kernel 和 recomputation 将 mHC wall-time overhead 控制在约 `6.7%`；
- 对推理框架而言，mHC 不是普通 residual add，需要新增 fused residual mixing 路径。

---

## 6. Muon Optimizer

DeepSeek-V4 对大多数模块使用 Muon optimizer，只对以下模块继续使用 AdamW：

- embedding；
- prediction head；
- RMSNorm weights；
- mHC 的 static bias 和 gating factors。

Muon 的关键思想是对梯度/动量矩阵做近似正交化，再更新参数。V4 使用 hybrid Newton-Schulz iterations：

- 前 8 步使用快速收敛系数 `(3.4445, -4.7750, 2.0315)`；
- 后 2 步使用稳定系数 `(2, -1.5, 0.5)`；
- 共 10 步。

训练超参：

| 项 | 值 |
|---|---:|
| Muon momentum | 0.95 |
| Muon weight decay | 0.1 |
| update RMS rescale factor | 0.18 |
| AdamW beta1 | 0.9 |
| AdamW beta2 | 0.95 |
| AdamW eps | 1e-20 |
| AdamW weight decay | 0.1 |

报告认为 Muon 带来更快收敛和更好稳定性，但这也显著提高训练系统复杂度，因为 Muon update 需要完整 gradient matrix。

---

## 7. FP4 Quantization-Aware Training

DeepSeek-V4 使用 FP4，具体是 MXFP4，覆盖两类关键路径：

1. MoE routed expert weights；
2. CSA indexer QK path。

报告强调训练时并不是只做 fake quant，而是在 forward/backward 中直接使用真实 FP4 quantized weights。优化器状态会先量化到 FP4，再无损 dequant 到 FP8 做计算。

关键点：

- FP4 routed expert weights 可以显著降低 expert 参数存储；
- CSA indexer 使用 FP4 能降低百万上下文下 sparse selector 的 QK 计算成本；
- 当前硬件上 FP4 × FP8 peak FLOPs 和 FP8 × FP8 相同，但报告认为未来硬件可进一步释放 FP4 理论收益；
- 对推理框架而言，需要支持 FP4 权重、FP4/FP8 混合计算和对应 scale 管理。

---

## 8. 训练流程

### 8.1 数据

训练语料超过 32T tokens，包括：

- web pages；
- 数学内容；
- 代码；
- long documents；
- scientific papers；
- technical reports；
- multilingual data；
- agentic data。

相比 V3，V4 更强调：

- 去除 batched auto-generated 和 templated content，降低 model collapse 风险；
- 增强 long-document 数据；
- 加入 agentic data 提升代码和工具使用能力；
- 使用 sample-level attention masking；
- tokenizer 仍保持 128K vocab，但加入少量 context construction special tokens。

### 8.2 预训练长度扩展

训练从 `4K` sequence length 开始，逐步扩展到：

1. `16K`
2. `64K`
3. `1M`

稀疏注意力不是一开始就启用：

- 先用 dense attention warmup；
- Flash 前 `1T` tokens 使用 dense attention；
- 在 sequence length 到 `64K` 后引入 sparse attention；
- 引入时先 warmup CSA lightning indexer；
- 之后大部分训练使用 sparse attention。

### 8.3 DeepSeek-V4-Flash 训练配置

| 项 | 值 |
|---|---:|
| 训练 tokens | 32T |
| max batch size | 75.5M tokens |
| peak LR | 2.7e-4 |
| final LR | 2.7e-5 |
| warmup steps | 2000 |
| auxiliary-loss-free bias update speed | 0.001 |
| sequence-wise balance loss weight | 0.0001 |
| MTP loss weight | 0.3，大部分训练；LR decay 后 0.1 |

### 8.4 DeepSeek-V4-Pro 训练配置

| 项 | 值 |
|---|---:|
| 训练 tokens | 33T |
| max batch size | 94.4M tokens |
| peak LR | 2.0e-4 |
| final LR | 2.0e-5 |
| warmup steps | 2000 |
| auxiliary-loss-free bias update speed | 0.001 |
| sequence-wise balance loss weight | 0.0001 |
| MTP loss weight | 0.3，大部分训练；LR decay 后 0.1 |

---

## 9. 训练稳定性：Anticipatory Routing 与 SwiGLU Clamping

DeepSeek-V4 的训练稳定性问题主要来自 trillion-parameter MoE 中的 outliers 和 routing feedback loop。

### 9.1 Anticipatory Routing

普通 MoE 中，当前 step 的 hidden states 和 routing network 同步更新，routing 决策可能强化异常 token/expert 分布，导致 loss spike。

Anticipatory Routing 的做法：

- step `t` 使用当前参数 `theta_t` 计算 backbone features；
- routing indices 使用历史参数 `theta_{t-delta}` 预先计算；
- 实现上提前 fetch 数据并缓存 routing indices；
- 只在检测到 loss spike 后短暂启用，稳定后恢复标准训练。

报告称额外 wall-clock 开销约 `20%`，但由于只在 spike 时启用，整体训练开销可忽略。

### 9.2 SwiGLU Clamping

训练期间对 SwiGLU 做数值裁剪：

- linear component 限制在 `[-10, 10]`；
- gate component 上界限制为 `10`。

这属于直接压制 MoE outliers 的工程手段。报告承认其理论机制尚未完全理解。

---

## 10. 后训练：Specialist + OPD

DeepSeek-V4 的 post-training 采用两阶段范式：

1. 独立训练 domain-specific experts；
2. 使用 On-Policy Distillation，OPD，将多个专家能力合并到统一模型。

专家方向包括：

- mathematics；
- coding；
- agent；
- instruction following。

每个专家通常先做 SFT，再用 GRPO 做 RL。和传统 RLHF 不同，报告强调 hard-to-verify tasks 不再依赖 scalar reward model，而是使用 Generative Reward Model，GRM。模型本身也作为评估器，通过 rubric-guided RL data 优化 judge 能力。

### 10.1 Reasoning modes

DeepSeek-V4 支持三种 reasoning effort：

| 模式 | 特点 | 典型用途 | 输出格式 |
|---|---|---|---|
| Non-think | 快速、直觉响应 | 日常任务、低风险决策 | `</think> summary` |
| Think High | 显式逻辑分析 | 复杂问题、规划 | `<think>...</think> summary` |
| Think Max | 最大 reasoning effort | 探索模型能力边界 | 系统 prompt + `<think>...</think> summary` |

Think Max 会在 system prompt 注入 “Reasoning Effort: Absolute maximum...” 这类强指令。

### 10.2 Tool-call schema

V4 引入基于 XML 的 DSML tool-call schema，使用特殊 token：

- `<|DSML|tool_calls>`
- `<|DSML|invoke name="...">`
- `<|DSML|parameter ...>`

设计目的：

- 降低 JSON escaping failure；
- 减少 tool-call 格式错误；
- 更适合复杂 agent 场景。

### 10.3 Interleaved thinking

V3.2 在多轮工具调用中会保留部分 reasoning traces，但新用户消息可能清空思考上下文。V4 借助 1M context 做了区分：

- tool-calling 场景：完整保留跨轮 reasoning history；
- 普通对话场景：新用户消息后仍丢弃旧 reasoning，节省上下文。

这说明 V4 的 agent 能力不只是模型能力，也依赖 serving framework 正确识别 tool-calling path。

### 10.4 Quick Instruction

Quick Instruction 用特殊 token 执行辅助任务，例如：

- `<|action|>`：判断是否需要 web search；
- `<|query|>`：生成搜索 query；
- `<|authority|>`：判断权威性需求；
- `<|domain|>`：识别领域；
- `<|read_url|>`：判断 URL 是否需要读取。

它的工程意义是：这些辅助任务直接复用已有 KV cache，避免额外小模型重复 prefill，从而降低 TTFT。

---

## 11. 推理系统设计

### 11.1 异构 KV cache

DeepSeek-V4 的 KV cache 不再是传统每层同构结构。由于 CSA/HCA/SWA 并存，KV cache 至少包含：

- CSA compressed KV；
- CSA indexer KV；
- HCA compressed KV；
- sliding window uncompressed KV；
- unready-for-compression tail states。

报告将 KV cache 分成两类：

- classical KV cache：存 compressed KV entries；
- state cache：存 SWA 和尚未压缩的 tail tokens。

这对 vLLM 影响很大。传统 PagedAttention 假设每层 cache shape 比较规整，而 V4 要求不同 attention layer 和不同 cache component 有不同 block size、eviction policy 和 hit policy。

### 11.2 On-disk KV cache

V4 使用 on-disk KV cache 复用 shared-prefix request，避免重复 prefill。

策略：

- CSA/HCA compressed KV entries 直接存盘；
- sliding window KV cache 太大，不完整存盘；
- 恢复时可通过重算最近 `n_win * L` tokens 恢复 tail state。

这说明对于 1M context serving，内存 KV cache 和磁盘 cache 需要协同。对真实在线系统而言，这会引入：

- prefix cache 命中管理；
- SSD 带宽/延迟瓶颈；
- KV 序列状态恢复逻辑；
- request preemption 和 resume。

---

## 12. MoE 系统：通信计算融合

DeepSeek-V4 使用 Expert Parallelism，EP，但重点是把通信和计算融合成 wave-based pipeline。

一个 MoE layer 被拆成：

- Dispatch；
- Linear-1 GEMM；
- SwiGLU + FP8 cast；
- Linear-2 GEMM；
- Combine。

传统做法中 Dispatch/Combine 是通信瓶颈。V4 将 experts 拆成多个 wave：

- 当前 wave 做 GEMM；
- 下一 wave 同时做 token transfer；
- 已完成 wave 同时 send results；
- 形成 communication-computation overlap。

报告称相比强 non-fused baseline：

- 一般 inference workload 加速 `1.50x ~ 1.73x`；
- RL rollout/high-speed agent serving 等 latency-sensitive 场景最高 `1.96x`。

报告还开源了 CUDA MegaMoE kernel，作为 DeepGEMM 的组件。

---

## 13. 评测结果解读

### 13.1 Base model

Table 1 中，DeepSeek-V4-Pro-Base 相比 DeepSeek-V3.2-Base 在多数任务提升显著：

| Benchmark | V3.2-Base | V4-Flash-Base | V4-Pro-Base |
|---|---:|---:|---:|
| MMLU-Pro | 65.5 | 68.3 | 73.5 |
| Simple-QA verified | 28.3 | 30.1 | 55.2 |
| FACTS Parametric | 27.1 | 33.9 | 62.6 |
| HumanEval | 62.8 | 69.5 | 76.8 |
| MATH | 60.5 | 57.4 | 64.5 |
| LongBench-V2 | 40.2 | 44.7 | 51.5 |

关键观察：

- V4-Flash 只有 13B activated params，但多数指标已超过 V3.2-Base；
- V4-Pro 在知识类任务提升最明显，说明 1.6T 总参数对知识存储仍很关键；
- 代码/数学并非所有项都单调提升，例如 BigCodeBench 中 V4-Pro-Base 低于 V3.2-Base。

### 13.2 Post-training model

Table 6 中，DeepSeek-V4-Pro-Max 和闭源/开源模型对比：

| Benchmark | DS-V4-Pro-Max | 观察 |
|---|---:|---|
| SimpleQA-Verified | 57.9 | 强于开源，但低于 Gemini-3.1-Pro 的 75.6 |
| Chinese-SimpleQA | 84.4 | 接近 Gemini-3.1-Pro 的 85.9 |
| GPQA Diamond | 90.1 | 低于 Gemini-3.1-Pro/GPT-5.4/Opus |
| HLE | 37.7 | 低于 Gemini-3.1-Pro 的 44.4 |
| LiveCodeBench | 93.5 | 报告表中最高 |
| Codeforces rating | 3206 | 高于 GPT-5.4 xHigh 的 3168 |
| MRCR 1M | 83.5 | 低于 Opus-4.6 的 92.9，高于 Gemini-3.1-Pro 的 76.3 |
| CorpusQA 1M | 62.0 | 低于 Opus-4.6 的 71.7，高于 Gemini-3.1-Pro 的 53.8 |
| Terminal Bench 2.0 | 67.9 | 低于 GPT-5.4 的 75.1 |
| SWE Verified | 80.6 | 接近 Opus/Gemini/K2.6 |
| Toolathlon | 51.8 | 低于 GPT-5.4 的 54.6，高于其他列中多数模型 |

整体看，报告的定位比较克制：V4-Pro-Max 是最强开源模型之一，在代码竞赛和长上下文上很强，但在部分知识、agent、HLE w/ tools 等任务上仍落后最强闭源模型。

### 13.3 Flash vs Pro

Table 7 显示：

- Flash 在 knowledge tasks 上明显落后 Pro；
- Flash 在 reasoning tasks 上通过 Max mode 能接近 Pro；
- Pro 在 agentic/coding/knowledge 任务上更稳；
- Flash 更适合高性价比推理，Pro 更适合高难任务。

这符合参数分工：Flash activated params 仅 13B，知识容量较小；但推理任务可以通过 test-time compute 弥补部分差距。

---

## 14. 对 vLLM 适配的影响

DeepSeek-V4 对 vLLM 的挑战比 DeepSeek-V3/V3.2 更大，主要不在 MoE GEMM，而在 attention 和 KV cache。

### 14.1 需要新的 attention backend

CSA/HCA 不是标准 MLA、GQA 或 dense attention。至少需要支持：

- compressed KV entry 生成；
- CSA lightning indexer；
- compressed block top-k selection；
- CSA sparse core attention；
- HCA compressed dense attention；
- sliding window branch；
- attention sink；
- partial RoPE；
- query/KV per-head RMSNorm；
- grouped output projection。

这意味着不能简单复用现有 FlashAttention/PagedAttention backend。

### 14.2 KV cache 管理需要异构化

传统 KV cache 通常按 layer、block、head、head_dim 管理。V4 需要按 component 管理：

- CSA main compressed KV；
- CSA indexer KV；
- HCA KV；
- SWA KV；
- uncompressed tail state。

不同 component：

- 压缩率不同；
- block size 不同；
- 生命周期不同；
- eviction 策略不同；
- 是否可落盘不同；
- 是否需要恢复/重算不同。

vLLM 需要在 cache allocator、block table、prefix cache、scheduler 中新增抽象。

### 14.3 FP4/FP8 路径

当前很多推理系统主要支持 FP16/BF16/FP8/int8/int4。V4 要求：

- MoE routed expert FP4 weights；
- FP4-to-FP8 dequant；
- CSA indexer QK FP4 compute；
- 混合 BF16/FP8 KV storage。

如果没有原生 FP4 kernel，实际部署可能无法达到报告中的效率。

### 14.4 MoE kernel 需要更深通信计算融合

现有 vLLM fused_moe 路径通常围绕 token sorting、expert GEMM、combine。V4 报告中的 MegaMoE 是通信、GEMM、activation、combine 统一调度的 mega-kernel。

适配重点：

- EP dispatch/combine 与 GEMM overlap；
- expert wave scheduling；
- 小 batch/RL rollout 场景优化；
- deterministic/batch-invariant kernel；
- FP4 routed expert weights。

### 14.5 Serving 层要支持长上下文 agent

V4 的产品能力依赖 runtime 机制：

- tool-calling path 保留完整 thinking traces；
- 普通对话 path 丢弃旧 thinking；
- Quick Instruction 复用 KV cache 做辅助任务；
- on-disk KV cache 支持 shared-prefix；
- preemption/resume 需要保存 WAL 和 KV cache。

这已经超出普通 LLM forward 的范围，需要 serving protocol、scheduler、cache manager、tool runtime 一起适配。

---

## 15. 风险与局限

报告自身也承认几个问题：

- 架构较复杂，包含大量已验证但不够优雅的 trick；
- Anticipatory Routing 和 SwiGLU Clamping 的理论机制尚不充分；
- 百万上下文下 128K 之后 retrieval 性能开始下降；
- 部分知识和 agent benchmarks 仍落后最强闭源模型；
- 真实部署收益强依赖 kernel、FP4 硬件、KV cache 系统和 serving workload；
- 内部评测框架结果需要外部复现验证。

从工程角度看，DeepSeek-V4 是“能力强但系统复杂度高”的路线。Flash 版本可能更适合大规模部署，Pro 更适合高难度 reasoning、coding agent 和长上下文任务。

---

## 16. 总结

DeepSeek-V4 的技术路线可以概括为：

```text
MoE 负责参数容量
CSA/HCA 负责百万上下文效率
mHC 负责深层表达与稳定性
Muon + routing/clamping 负责大规模训练稳定
FP4 + fused kernels + 异构 KV cache 负责真实推理成本
OPD + reasoning modes 负责后训练能力整合
```

最值得关注的是 CSA/HCA 和异构 KV cache。它们改变了长上下文推理系统的基本假设：KV cache 不再是简单按 token 累积的同构结构，attention 也不再是对全部历史 token 或标准 paged blocks 的访问。对于 vLLM 来说，DeepSeek-V4 的适配更像是新增一个完整的 long-context sparse-compressed attention stack，而不是对现有 MLA/MoE 路径的小改动。

