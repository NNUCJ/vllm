# Dynamo调研报告

## 1. 调研说明

本文基于 2026-03-27 对公开资料的整理撰写，目标是在当前工程目录中形成一份可供方案评审和芯片适配讨论使用的技术路线报告。

需要先说明一点：用户给出的链接为 `https://github.com/NNUCJ/dynamo/tree/main`，但在本次检索中未能查到对应的公开仓库页面。公开可访问、且内容与 “Dynamo 推理编排框架” 明确对应的上游项目为 `ai-dynamo/dynamo`，并且其官方文档为 NVIDIA Dynamo 文档站。下文默认将调研对象解释为这一公开项目；如果后续确认 `NNUCJ/dynamo` 是内部镜像、私有 fork 或转存仓库，本报告的结论仍可作为主干参考，但应再做一次与私有分支的差异比对。

## 2. 项目背景与技术定位

### 2.1 技术背景

Dynamo 不是一个单独的 LLM 推理引擎，而是位于推理引擎之上的“数据中心级推理编排层”。它的核心目标不是替代 vLLM、SGLang、TensorRT-LLM，而是把这些后端组织成一个可跨多 GPU、多节点扩展的统一推理系统。

这一定位反映了当前大模型服务的几个现实问题：

1. 单机单卡优化已经不能满足大模型高并发与长上下文场景。
2. Prefill 和 Decode 的资源特征差异很大，混跑容易浪费 GPU。
3. KV Cache 成本越来越高，重复 Prefill 会显著拉高 TTFT 和成本。
4. 真正的生产环境不仅需要“能跑起来”，还需要自动扩缩、故障转移、可观测、K8s 集成和多租户治理。

因此，Dynamo 的技术路线不是“再造一个引擎”，而是：

1. 复用主流推理引擎的算子与执行能力。
2. 在引擎之上增加统一的服务入口、路由、KV 管理、调度与编排。
3. 让部署形态从单实例推理升级为集群级推理系统。

### 2.2 解决的问题

Dynamo 主要解决以下问题：

1. **分离式服务**：将 Prefill 与 Decode 分成不同工作池，独立扩缩容。
2. **KV 感知路由**：根据已有 KV Cache 命中情况选择 worker，减少重复计算。
3. **多层 KV Cache 管理**：支持把 KV 从 GPU 向 CPU、SSD、远端存储扩展。
4. **统一前端**：对外提供 OpenAI 兼容接口，隐藏后端差异。
5. **集群级调度与自动扩缩**：以 SLA、吞吐、时延为目标做部署优化。
6. **故障恢复与迁移**：worker 故障时尽量减少用户请求损失。

### 2.3 技术价值判断

从路线判断，Dynamo 更适合以下场景：

1. 多卡或多机的大模型在线服务。
2. 长上下文、高缓存复用、重推理链路场景。
3. 需要把不同推理引擎纳入统一平台治理的场景。
4. 希望把“芯片能力”封装在后端引擎层，把“服务治理能力”放在更上层的场景。

如果只是单卡、单模型、低并发部署，Dynamo 往往不是第一优先级，直接用 vLLM/SGLang 会更轻。

## 3. Dynamo 的总体技术路线

### 3.1 架构思路

Dynamo 可概括为“前端网关 + 编排层 + 后端推理引擎 + 基础设施”的四层结构。

#### 第一层：统一前端

前端负责暴露 OpenAI 兼容 HTTP API，也支持 KServe gRPC。它承担：

1. 请求接入。
2. 请求预处理。
3. 路由决策。
4. 响应格式化与流式输出。

这意味着对上层业务来说，后端是不是 vLLM、SGLang、TensorRT-LLM，可以被前端尽量屏蔽。

#### 第二层：编排与控制层

这是 Dynamo 的核心价值区，主要包括：

1. **KV-aware Routing**：根据 worker 负载和 KV cache 重叠度选路。
2. **Disaggregated Serving**：把 Prefill/Decode 解耦。
3. **Planner**：按 SLA、时延和成本做容量规划与扩缩。
4. **KVBM / KV 管理**：扩展上下文缓存层级。
5. **Fault Tolerance**：健康检查、请求迁移、异常恢复。

#### 第三层：后端引擎层

当前公开支持的后端主要是：

1. vLLM
2. SGLang
3. TensorRT-LLM

因此 Dynamo 的设计是“引擎无关但并非引擎无要求”。后端必须能够提供足够的运行时信息、缓存接口和扩展能力，才能充分接入 Dynamo 的高级能力。

#### 第四层：基础设施层

包括：

1. GPU / 多 GPU / 多节点网络
2. etcd 或文件式发现机制
3. NATS 消息系统
4. 容器运行时
5. Kubernetes
6. 对象存储/块存储/本地盘等 KV 扩展介质

### 3.2 与传统推理服务的区别

传统做法通常是“API Server + 单个引擎实例”。Dynamo 则进一步引入：

1. 集群级 worker 管理。
2. 请求级路由与缓存感知。
3. Prefill/Decode 角色拆分。
4. 自动扩缩与部署规划。
5. 多后端统一纳管。

因此，Dynamo 更像“LLM Serving Control Plane + Data Plane”的组合，而不仅是一个 SDK。

## 4. 技术栈分析

### 4.1 语言与实现技术

根据公开仓库 README，Dynamo 的总体实现路线是：

1. **Rust**：用于性能敏感、并发敏感的核心服务部分，尤其是高性能前端与部分系统组件。
2. **Python**：用于后端引擎适配、工作进程启动、生态扩展与开发者接口。

这条路线很典型：Rust 负责服务面高并发和低延迟，Python 负责拥抱 vLLM、PyTorch、SGLang 等 AI 生态。

### 4.2 推理后端栈

它本身不直接做所有模型计算，而是适配下列后端：

1. **vLLM**：偏通用，生态活跃，适合作为默认开源后端。
2. **SGLang**：对某些场景下的吞吐、推理工作流与 agent 模式更友好。
3. **TensorRT-LLM**：更偏 NVIDIA 深度优化路径。

### 4.3 服务与控制平面组件

从官方文档可归纳出 Dynamo 依赖的关键组件：

1. **Frontend**：统一 API 入口，OpenAI 兼容。
2. **Discovery Backend**：支持 `file`、`etcd`，K8s 下也有原生发现方式。
3. **NATS**：用于消息与事件通道，特别是 KV 事件相关能力。
4. **Planner**：做 SLA 驱动的资源规划。
5. **KVBM**：KV Block 管理。
6. **Grove / K8s Operator**：面向 Kubernetes 的调度、部署与资源拓扑管理。

### 4.4 部署与工程化栈

公开资料显示其工程化栈大致包括：

1. **Python 包管理**：`uv`、`pip`
2. **容器**：Docker / NGC 预构建镜像
3. **Kubernetes**：生产推荐路径
4. **Rust 构建**：Cargo
5. **Python 构建**：maturin、pyproject 体系

### 4.5 硬件与 CUDA 依赖现状

官方支持矩阵表明，Dynamo 的公开发布版本是建立在 NVIDIA GPU 和 CUDA 生态之上的。按公开文档，当前版本对 vLLM/SGLang/TensorRT-LLM 的支持依赖明确的 CUDA 版本组合，例如 vLLM 在近几个版本中主要围绕 CUDA 12.9/13.0 测试。

这意味着：

1. Dynamo 本身虽然是“后端无关”的编排层，
2. 但如果后端选用 vLLM，那么底层仍然强依赖 CUDA 兼容的软件栈，
3. 尤其是 PyTorch CUDA、vLLM 自定义 kernel、通信库和容器生态。

## 5. 后端使用 vLLM 时的技术路径

### 5.1 为什么优先考虑 vLLM

如果目标是让 Dynamo 在自研 GPGPU 芯片上尽快跑起来，vLLM 通常是一个现实的优先选项，原因有三点：

1. 社区成熟度高，接口与部署示例多。
2. Dynamo 对 vLLM 已有公开文档、容器与单机/多机样例。
3. 相比 TensorRT-LLM，vLLM 对“非 NVIDIA 专属优化链路”的依赖相对更容易抽象到 CUDA 兼容层。

### 5.2 vLLM 接入时 Dynamo 的运行模式

官方文档给出的 vLLM 接入模式包括：

1. Aggregated Serving
2. Aggregated Serving with KV Routing
3. Disaggregated Serving
4. Disaggregated Serving with KV Routing
5. Data Parallel / Expert Parallel 相关部署

建议技术路线如下：

1. **第一阶段**：先打通单机单 worker 的 aggregated 模式。
2. **第二阶段**：再打开多 worker 与 KV 路由。
3. **第三阶段**：再做 prefill/decode 解耦。
4. **第四阶段**：最后进入 K8s、自动扩缩和多节点优化。

这条路线最适合芯片 bring-up，因为每一层都能单独验证。

## 6. Dynamo + vLLM 调用示例

下面给出几种从简单到完整的调用方式。示例主要面向“先跑通，再增强”的顺序。

### 6.1 最小可运行示例：文件发现模式

这一模式不依赖 etcd，适合单机验证。

#### 6.1.1 启动 frontend

```bash
python3 -m dynamo.frontend --http-port 8000 --discovery-backend file
```

#### 6.1.2 启动 vLLM worker

```bash
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --discovery-backend file
```

如果显存不足，可进一步限制上下文长度或按 vLLM 参数做缩减，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --context-length 4096
```

#### 6.1.3 OpenAI 兼容接口调用示例

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "请用一句话介绍 Dynamo 的定位。"}
    ],
    "temperature": 0.2,
    "max_tokens": 128,
    "stream": false
  }'
```

### 6.2 流式调用示例

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "请分三点说明 Dynamo 和 vLLM 的关系。"}
    ],
    "temperature": 0.2,
    "max_tokens": 256,
    "stream": true
  }'
```

### 6.3 Python SDK 风格调用示例

如果业务侧已经使用 OpenAI SDK，可直接把 `base_url` 指向 Dynamo frontend。

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "说明 Dynamo 为什么适合做多机推理编排。"},
    ],
    temperature=0.2,
    max_tokens=128,
    stream=False,
)

print(resp.choices[0].message.content)
```

### 6.4 带 Prefill/Decode 拆分的示例

这类模式更接近 Dynamo 的核心价值。示意命令如下：

#### 6.4.1 启动 frontend

```bash
python3 -m dynamo.frontend --http-port 8000 --discovery-backend file
```

#### 6.4.2 启动 prefill worker

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --disaggregation-mode prefill
```

#### 6.4.3 启动 decode worker

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --disaggregation-mode decode
```

### 6.5 带 KV 传输配置的示例

官方 vLLM 文档中给出了 `--kv-transfer-config` 的能力入口。一个示意写法如下：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

说明：

1. `NixlConnector` 是 Dynamo/vLLM 协同时的重要数据传输能力之一。
2. 真正可用的参数组合需要和所选 Dynamo 版本、vLLM 版本、NIXL 版本严格对齐。
3. 在自研芯片场景下，这一层是后续高性能优化的重点，也是移植风险最大的部分之一。

### 6.6 容器方式示例

如果先不考虑自研芯片，仅用于理解标准部署方式，可使用官方容器：

```bash
docker run --gpus all --network host --rm -it \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.0
```

容器内再启动：

```bash
python3 -m dynamo.frontend --discovery-backend file > dynamo.frontend.log 2>&1 &
python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --discovery-backend file
```

## 7. 若在自研 GPGPU 芯片上做到 CUDA 兼容，需要具备哪些软件栈

这一部分是本报告最关键的结论。

首先要明确：**Dynamo 能否在自研芯片上执行，不取决于 Dynamo 本身，而主要取决于 “Dynamo 选定的后端引擎能否在该芯片上稳定运行”**。如果后端采用 vLLM，那么实际上要跑通的是：

1. Python
2. PyTorch CUDA 栈
3. vLLM
4. vLLM 所需的自定义 CUDA/Triton/通信组件
5. Dynamo 对这些组件的编排调用

因此，“CUDA 兼容”至少要分成四个层次来看。

### 7.1 第一层：基础系统软件栈

必须具备：

1. **Linux 内核驱动**
   负责设备枚举、内存管理、中断、DMA、上下文、进程隔离等。
2. **用户态驱动**
   至少要提供 `libcuda.so` 这一类 CUDA Driver API 入口。
3. **设备文件与运行时权限模型**
   支持容器、普通进程、监控进程访问设备。
4. **容器运行时对接**
   最好有类似 `nvidia-container-toolkit` 的能力，把设备、驱动库、环境变量注入容器。

如果这些能力不完整，Dynamo 即便代码能装上，也无法真正调用后端 GPU 计算。

### 7.2 第二层：CUDA 兼容执行栈

如果目标是“尽量不改上层代码直接跑 vLLM”，那么芯片软件栈至少要覆盖：

1. **CUDA Driver API 兼容**
   例如上下文、stream、event、memory、module、kernel launch 等基础接口。
2. **CUDA Runtime API 兼容**
   至少要支持 PyTorch 与 vLLM 高频调用的运行时语义。
3. **PTX / cubin / fatbin 装载与执行能力**
   这里要明确是源兼容、PTX 兼容还是二进制兼容。
4. **NVRTC 或等效 JIT 编译能力**
   很多上层组件会动态生成 kernel。
5. **CUDA Graph、Pinned Memory、Unified Addressing 等常见机制**
   不一定第一天就全量支持，但核心特性缺失会明显影响 vLLM 可用性。

这里要强调一个工程现实：

1. **只有 API 兼容，不足以支撑 vLLM 生产可用。**
2. **只有源码级兼容，也不足以保证第三方 wheel 直接可用。**
3. 真正想少改代码，需要尽量靠近“二进制兼容 + 行为兼容”。

### 7.3 第三层：AI 框架兼容栈

对 vLLM 来说，至少需要以下软件层可运行：

1. **PyTorch CUDA 版本可用**
   这是最核心前提。若 PyTorch 无法稳定识别并调度芯片，后面都无从谈起。
2. **Triton 编译与执行链路**
   vLLM 和相关依赖常用 Triton/kernel 生成技术。
3. **核心数学库**
   至少要有 CUDA 兼容的：
   - cuBLAS / cuBLASLt 等价物
   - cuDNN 等价物
   - 可能还需要 cuSPARSE / cuRAND 等部分能力
4. **通信库**
   多卡部署至少要有 NCCL 等价物，且需要对 PyTorch 分布式透明或近似透明。
5. **监控管理库**
   很多上层工具默认会依赖 NVML 或等效能力来获取显存、温度、设备状态。

如果这些库只有“API 壳子”，但性能、线程安全、并发语义或错误处理不兼容，上层框架仍然会频繁出问题。

### 7.4 第四层：vLLM 专项兼容栈

即使 PyTorch 已经跑通，vLLM 仍有专项要求。至少要评估：

1. **Paged Attention 相关 kernel**
2. **FlashAttention 或等效高性能注意力 kernel**
3. **量化推理相关 kernel**
   如 AWQ、GPTQ、Marlin 等常见路径
4. **KV Cache 管理与高速搬运能力**
5. **多流并发与异步执行能力**
6. **自定义 C++/CUDA 扩展的构建与装载**

这里的关键不是“能否编译”，而是：

1. 算子是否真的在芯片上执行；
2. 性能是否达到服务可用门槛；
3. 长时间运行是否稳定；
4. 内存碎片、OOM、stream 同步语义是否正确。

### 7.5 第五层：Dynamo 相关附加能力

当 vLLM 已跑通后，Dynamo 还会进一步要求：

1. **多进程/多 worker 稳定运行**
2. **前端与 worker 注册发现机制正常**
3. **KV 事件与指标上报可用**
4. **NIXL 或等效传输机制可用**
5. **多卡/多机通信路径可用**
6. **容器化、K8s、健康检查、日志与观测链路可用**

这意味着“芯片能跑单模型”与“芯片能承载 Dynamo 集群服务”之间还有很长一段工程距离。

## 8. 在自研 CUDA 兼容芯片上执行 Dynamo 的建议软件栈清单

下面给出一份更落地的分层清单。

### 8.1 必选基础层

1. Linux kernel driver
2. `libcuda.so` 兼容实现
3. `libcudart.so` 兼容实现
4. 基本内存/stream/event/kernel launch 机制
5. 设备管理工具与监控接口
6. 容器 runtime 注入工具

### 8.2 必选 AI 框架层

1. PyTorch CUDA 构建可用
2. Torch C++ Extension 构建链可用
3. Triton 可用，或提供能绕过 Triton 的替代路径
4. cuBLAS/cuBLASLt 等价库
5. cuDNN 等价库
6. NCCL 等价库

### 8.3 必选 vLLM 运行层

1. vLLM 对应版本源码可编译
2. 关键 CUDA/Triton kernel 可执行
3. KV cache 机制稳定
4. 分页注意力与采样相关 kernel 正常
5. 单卡、双卡、多卡模式都能跑通

### 8.4 必选 Dynamo 编排层

1. `ai-dynamo[vllm]` 能安装
2. `python -m dynamo.frontend` 能启动
3. `python -m dynamo.vllm` 能注册 worker
4. OpenAI 兼容接口可对外服务
5. file discovery 模式跑通
6. etcd + NATS 模式跑通

### 8.5 进阶优化层

1. KV-aware routing
2. Prefill/Decode disaggregation
3. NIXL 或自研高速 KV 传输
4. K8s operator 对接
5. 自动扩缩与 SLA planner
6. GPU 拓扑感知调度

## 9. 建议的芯片适配实施路线

### 阶段一：先验证 PyTorch 与 vLLM

目标：

1. 不上 Dynamo。
2. 先让单卡 PyTorch 推理稳定。
3. 再让单卡 vLLM 服务稳定。

验收标准：

1. 可加载常见 HuggingFace 模型。
2. 连续生成稳定无崩溃。
3. 至少支持基础 attention、采样与 KV cache。

### 阶段二：接入 Dynamo 最小闭环

目标：

1. 使用 `--discovery-backend file`。
2. 跑通 `frontend + 单个 dynamo.vllm worker`。
3. 通过 OpenAI 接口完成请求。

验收标准：

1. `curl /v1/chat/completions` 可用。
2. 流式输出可用。
3. worker 可自动发现与注册。

### 阶段三：多卡与通信栈验证

目标：

1. 双卡部署。
2. 验证数据并行或分角色 worker。
3. 引入 NCCL 等价通信库。

验收标准：

1. 多 worker 稳定。
2. 性能随 GPU 数量合理扩展。
3. 无明显死锁、hang、显存泄漏。

### 阶段四：KV 路由与分离式服务

目标：

1. 打开 KV 感知路由。
2. 打开 prefill/decode 解耦。
3. 验证 KV 迁移、事件和指标链路。

验收标准：

1. 命中缓存时 TTFT 明显改善。
2. Prefill/Decode worker 能独立扩缩。
3. 故障场景下服务能恢复。

### 阶段五：容器与集群化

目标：

1. 支持容器运行。
2. 支持 K8s 调度。
3. 形成标准化部署方式。

验收标准：

1. 容器内可以稳定识别芯片与驱动。
2. 可通过 K8s 完成部署与观测。
3. 运维链路可复用。

## 10. 风险判断

### 10.1 最大风险不在 Dynamo，而在 vLLM 栈

从实现路径看，真正的高风险点是：

1. PyTorch CUDA 兼容度不够。
2. Triton 与自定义 kernel 无法稳定运行。
3. NCCL 等价通信库不成熟。
4. FlashAttention/PagedAttention 性能或语义不兼容。

如果这些问题不解决，Dynamo 只能停留在“框架可安装但无法形成服务能力”的状态。

### 10.2 只做 CUDA API 兼容，仍可能不够

很多项目表面上调用的是 CUDA API，但实际依赖的是：

1. 特定编译器行为；
2. 特定 kernel 调度语义；
3. 特定数学库性能特征；
4. Triton/PTX/JIT 行为；
5. NCCL 拓扑与通信语义。

因此建议将目标定义为：

1. **优先实现 PyTorch + vLLM 的可用兼容，**
2. **再追求 Dynamo 的集群编排能力。**

### 10.3 推荐优先级

建议优先级如下：

1. PyTorch 单卡稳定
2. vLLM 单卡稳定
3. vLLM 多卡稳定
4. Dynamo 最小闭环
5. Dynamo 高级特性

## 11. 结论

Dynamo 的本质是一个面向数据中心级推理服务的编排层，而不是替代引擎的计算框架。它的技术路线非常清晰：以 Rust + Python 实现统一前端、路由、KV 管理、扩缩容和集群调度，并把 vLLM、SGLang、TensorRT-LLM 这些推理引擎组织成可治理的服务系统。

如果后端选用 vLLM，那么在自研 GPGPU 芯片上跑 Dynamo 的关键前提不是 Dynamo 本身，而是芯片是否具备完整、稳定、足够高兼容度的 CUDA 软件栈，尤其是 PyTorch、Triton、通信库、注意力 kernel 和 vLLM 自定义扩展这一整条链路。

从落地路径看，最现实的实施方案是：

1. 先打通 PyTorch；
2. 再打通 vLLM；
3. 再接入 Dynamo 的最小 file-discovery 模式；
4. 最后再上 KV 路由、P/D 分离、NATS、K8s 和多节点优化。

如果团队当前要制定芯片适配路线，建议把 “Dynamo 适配” 拆解成 “vLLM 兼容工程 + Dynamo 编排接入工程” 两条线并行推进，其中前者是决定成败的主路径。

## 12. 参考资料

1. `ai-dynamo/dynamo` GitHub 仓库：<https://github.com/ai-dynamo/dynamo>
2. Dynamo README：<https://github.com/ai-dynamo/dynamo/blob/main/README.md>
3. NVIDIA Dynamo Quickstart：<https://docs.nvidia.com/dynamo/getting-started>
4. NVIDIA Dynamo Frontend 文档：<https://docs.nvidia.com/dynamo/components/frontend>
5. NVIDIA Dynamo vLLM 后端文档：<https://docs.nvidia.com/dynamo/latest/components/backends/v-llm>
6. NVIDIA Dynamo Support Matrix：<https://docs.nvidia.com/dynamo/latest/getting-started/support-matrix>
