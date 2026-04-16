# vLLM 0.18.0 分布式并行机制深度分析报告

> **版本**: vLLM 0.18.0 (dev branch)  
> **模型**: deepseek-moe-16b-base (DeepSeekV2 架构)  
> **硬件**: 8× NVIDIA RTX 4090 (SM 8.9, Ada Lovelace)  
> **作者**: 内部技术分析  
> **日期**: 2026-04

---

## 目录

- [第一部分：总览与架构](#第一部分总览与架构)
  - [第1章 vLLM 分布式并行全景](#第1章-vllm-分布式并行全景)
  - [第2章 启动链路：从脚本到 Worker](#第2章-启动链路从脚本到-worker)
- [第二部分：通信组创建](#第二部分通信组创建)
  - [第3章 通信组创建机制](#第3章-通信组创建机制)
    - [3.1 all_ranks 五维张量的构造](#31-all_ranks-五维张量的构造)
    - [3.2 各组的切片方法](#32-各组的切片方法)
    - [3.3 init_model_parallel_group() 组工厂函数](#33-init_model_parallel_group-组工厂函数)
    - [3.4 GroupCoordinator 通信组核心对象](#34-groupcoordinator-通信组的核心对象)
    - [3.5 全局变量与访问器](#35-全局变量与访问器)
    - [3.6 _WORLD 组的创建与其同模型并行组的本质区别](#36-_world-组的创建与其同模型并行组的本质区别)
- [第三部分：通信栈](#第三部分通信栈)
  - [第4章 通信抽象层](#第4章-通信抽象层)
- [第四部分：张量并行权重切分（核心）](#第四部分张量并行权重切分核心)
  - [第5章 TP 切分原语层](#第5章-tp-切分原语层)
  - [第6章 DeepSeek-MoE-16B 端到端切分实例](#第6章-deepseek-moe-16b-端到端切分实例)
- [第五部分：其他并行维度](#第五部分其他并行维度)
  - [第7章 流水线并行 (PP)](#第7章-流水线并行-pp)
  - [第8章 专家并行 (EP)](#第8章-专家并行-ep)
  - [第9章 数据并行 (DP)](#第9章-数据并行-dp)
    - [9.6 DP Wave 调度机制（MoE 专用）](#96-dp-wave-调度机制moe-专用)
    - [9.8 组合场景实战分析 —— 以 DeepSeek-MoE-16B 为例](#98-组合场景实战分析--以-deepseek-moe-16b-为例)
- [第六部分：总结与实践](#第六部分总结与实践)
  - [第10章 设计总结与最佳实践](#第10章-设计总结与最佳实践)
- [第七部分：DCP / PCP 上下文并行深度分析](#第七部分dcp--pcp-上下文并行深度分析)
  - [第11章 DCP / PCP 概述与设计动机](#第11章-dcp--pcp-概述与设计动机)
  - [第12章 配置参数详解](#第12章-配置参数详解)
  - [第13章 通信组创建（DCP/PCP）](#第13章-通信组创建)
  - [第14章 KV Cache 切分策略](#第14章-kv-cache-切分策略)
  - [第15章 DCP Forward 计算流程](#第15章-dcp-forward-计算流程)
  - [第16章 两种 DCP 通信后端对比](#第16章-两种-dcp-通信后端对比)
  - [第17章 LSE 加权合并的数学原理](#第17章-lse-加权合并的数学原理)
  - [第18章 PCP 与 Expert Parallel 的交互](#第18章-pcp-与-expert-parallel-的交互)
  - [第19章 DCP + PCP 联合使用分析](#第19章-dcp--pcp-联合使用分析)
- [附录](#附录)
  - [附录 A：源码阅读路径](#附录-a源码阅读路径)
  - [附录 B：配置速查表](#附录-b配置速查表)
  - [附录 C：关键 API 索引](#附录-c关键-api-索引)
  - [附录 D：DCP/PCP 源码阅读路径](#附录-ddcppcp-源码阅读路径)
  - [附录 E：DCP/PCP 关键 API 索引](#附录-edcppcp-关键-api-索引)

---

# 第一部分：总览与架构

---

## 第1章 vLLM 分布式并行全景

### 1.1 六大并行维度一览

vLLM 0.18.0 支持 **六大并行维度**，每个维度解决不同的扩展瓶颈：

| 维度 | 缩写 | 切什么 | 通信原语 | 典型场景 |
|------|------|--------|----------|----------|
| **张量并行** | TP | 权重矩阵 | AllReduce / AllGather | 单层太大，单卡放不下 |
| **流水线并行** | PP | 层（stage） | Send / Recv (P2P) | 模型层数多，多卡分段 |
| **数据并行** | DP | 请求（batch） | AllReduce (梯度/同步) | 吞吐扩展，多副本推理 |
| **专家并行** | EP | MoE 专家 | All2All | MoE 模型专家数 > 单卡容量 |
| **预填充上下文并行** | PCP | 长序列 prefill | AllReduce / ReduceScatter | 长上下文 prefill 加速 |
| **解码上下文并行** | DCP | 解码阶段 KV | 复用 TP 内 GPU | 解码阶段 KV Cache 分摊 |

### 1.2 并行维度与 GPU 的映射关系

vLLM 使用一个五维张量来表达所有 GPU 的拓扑布局：

```
all_ranks = torch.arange(world_size).reshape(
    ExternalDP,          # 外部数据并行（verl 集成）
    DP,                  # 数据并行
    PP,                  # 流水线并行
    PCP,                 # 预填充上下文并行
    TP,                  # 张量并行
)
```

> **源码位置**: `vllm/distributed/parallel_state.py` → `initialize_model_parallel()` 约 L1553

**维度排列意义**：最内层是 TP（同一节点内高速互联），向外依次是 PCP、PP、DP、ExternalDP。这保证了通信量最大的 TP 组内 GPU 物理上最近。

**示例**：8 GPU，TP=4，DP=2，PP=1，PCP=1

```
all_ranks = [0,1,2,3,4,5,6,7].reshape(1, 2, 1, 1, 4)

TP groups:  [0,1,2,3], [4,5,6,7]
DP groups:  [0,4], [1,5], [2,6], [3,7]
```

### 1.3 整体架构层次

```
┌──────────────────────────────────────────────────────┐
│                    用户脚本层                          │
│         data_parallel.py / LLM() API                 │
├──────────────────────────────────────────────────────┤
│                    引擎层 (Engine)                     │
│      LLMEngine → Executor (mp/ray/external)          │
├──────────────────────────────────────────────────────┤
│                    Worker 层                           │
│   Worker.init_device() → init_distributed_env()      │
│   → initialize_model_parallel()                      │
├──────────────────────────────────────────────────────┤
│                  通信组管理层                          │
│   parallel_state.py: _TP, _PP, _DP, _EP, _PCP, _DCP │
│   GroupCoordinator + DeviceCommunicator               │
├──────────────────────────────────────────────────────┤
│                   模型层                               │
│   Linear layers: Column/Row/QKV/MergedColumn         │
│   Embedding: VocabParallelEmbedding                  │
│   MoE: FusedMoE + All2All                            │
├──────────────────────────────────────────────────────┤
│                 通信后端层                              │
│   CudaCommunicator: PyNCCL, CustomAllreduce,         │
│   FlashInfer, SymmMem, QuickAllReduce                │
└──────────────────────────────────────────────────────┘
```

---

## 第2章 启动链路：从脚本到 Worker

### 2.1 入口脚本 `data_parallel.py` 分析

`examples/offline_inference/data_parallel.py` 是 DP 推理的标准入口：

```python
# 关键环境变量设置
os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
os.environ["VLLM_DP_SIZE"] = str(dp_size)
os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
```

**执行流程**：

```
__main__
  ├── 解析参数: dp_size, tp, model, ...
  ├── 确定 dp_master_ip / dp_master_port
  ├── for local_dp_rank in range(dp_per_node):
  │     └── Process(target=main, args=(...))  # 每个 DP rank 一个进程
  └── 等待所有进程完成
```

每个 DP 进程内部：
1. 设置 `VLLM_DP_*` 环境变量
2. 创建 `LLM(**engine_args)` 实例
3. 调用 `llm.generate(prompts, sampling_params)`

### 2.2 从 LLM 到 Worker 的调用链

```
LLM.__init__()
  → LLMEngine.__init__()
    → ExecutorBase.create()           # 根据配置选择 Executor
      → MultiprocExecutor / RayExecutor / ...
        → Worker.__init__()
          → Worker.init_device()      # 关键初始化入口
```

### 2.3 Worker.init_device() 详解

> **源码位置**: `vllm/worker/gpu_worker.py` → `Worker.init_device()` 约 L230

```python
def init_device(self):
    # 1. 计算本地 rank（考虑 DP 偏移）
    local_rank = self.local_rank  
    # 对于 DP，local_rank 需要加上 DP rank 的偏移
    # local_rank = self.local_rank + dp_rank * tp_size
    
    # 2. 绑定 GPU 设备
    torch.cuda.set_device(local_rank)
    
    # 3. 调用分布式初始化
    init_worker_distributed_environment(...)
```

### 2.4 `init_worker_distributed_environment()` 三步走

> **源码位置**: `vllm/worker/gpu_worker.py` 约 L1030

```python
def init_worker_distributed_environment(...):
    # Step 1: 初始化自定义 AllReduce（如果可用）
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)
    
    # Step 2: 初始化分布式环境（torch.distributed + World group）
    init_distributed_environment(
        parallel_config.world_size,
        rank,
        distributed_init_method,
        local_rank,
        backend,
    )
    
    # Step 3: 创建所有并行通信组
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=parallel_config.tensor_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_parallel_size,
        ...
    )
```

**执行时序图**：

```
Worker.init_device()
    │
    ▼
set_custom_all_reduce()
    │
    ▼
init_distributed_environment()
    ├── torch.distributed.init_process_group()
    ├── 创建 _WORLD group (GroupCoordinator)
    └── 检测节点数 _NODE_COUNT
    │
    ▼
initialize_model_parallel()
    ├── 构建 all_ranks 五维张量
    ├── 创建 _TP  group
    ├── 创建 _DCP group
    ├── 创建 _PCP group
    ├── 创建 _PP  group
    ├── 创建 _DP  group
    └── 创建 _EP  group (仅 MoE 模型)
```

---

# 第二部分：通信组创建

---

## 第3章 通信组创建机制

### 3.1 all_ranks 五维张量的构造

> **源码位置**: `parallel_state.py` → `initialize_model_parallel()` 约 L1553

```python
all_ranks = torch.arange(world_size).reshape(
    -1,                                    # ExternalDP (自动推断)
    data_parallel_size,                    # DP
    pipeline_model_parallel_size,          # PP
    prefill_context_model_parallel_size,   # PCP
    tensor_model_parallel_size,            # TP
)
```

**这是 vLLM 分布式架构最核心的数据结构**。所有通信组都从这个五维张量通过 `transpose` + `reshape` + `unbind` 操作生成。

### 3.2 各组的切片方法

#### 3.2.1 TP 组 —— 最内层直接切

```python
# TP 是最内层维度，直接 view 后 unbind
group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
# 示例 (8GPU, TP=4, DP=2): [[0,1,2,3], [4,5,6,7]]
```

**原理**：五维张量的最后一维就是 TP，直接把前四维展平成一维，对最后一维 unbind 即可。

#### 3.2.2 PP 组 —— transpose 到最后再切

```python
# PP 在第3维(index=2)，transpose 到最后
group_ranks = (
    all_ranks.transpose(2, 4)          # PP ↔ TP 交换
    .reshape(-1, pipeline_model_parallel_size)
    .unbind(0)
)
# 示例 (8GPU, TP=2, PP=2, DP=2): [[0,2], [1,3], [4,6], [5,7]]
```

#### 3.2.3 DP 组 —— transpose DP 维到最后

```python
# DP 在第2维(index=1)，transpose 到最后
group_ranks = (
    all_ranks.transpose(1, 4)          # DP ↔ TP 交换
    .reshape(-1, data_parallel_size)
    .unbind(0)
)
# 示例 (8GPU, TP=4, DP=2): [[0,4], [1,5], [2,6], [3,7]]
```

#### 3.2.4 PCP 组 —— transpose PCP 维到最后

```python
group_ranks = (
    all_ranks.transpose(3, 4)          # PCP ↔ TP 交换
    .reshape(-1, prefill_context_model_parallel_size)
    .unbind(0)
)
```

#### 3.2.5 DCP 组 —— 复用 TP 内 GPU

```python
# DCP 在 TP 组内部进一步划分
group_ranks = all_ranks.reshape(
    -1, decode_context_model_parallel_size
).unbind(0)
```

**注意**：DCP 不增加总 GPU 数，而是将 TP 组内的 GPU 重新分组。要求 `dcp_size ≤ tp_size`。

#### 3.2.6 EP 组 —— 跨 DP 的大通信域

```python
# EP 将 DP × PCP × TP 展平为一个大组
group_ranks = (
    all_ranks.transpose(1, 2)          # DP ↔ PP 交换
    .reshape(
        -1,
        data_parallel_size * prefill_context_model_parallel_size 
        * tensor_model_parallel_size,
    )
    .unbind(0)
)
# EP_size = DP × PCP × TP
```

**关键设计**：EP 组包含了同一 PP stage 内的所有 DP × TP 卡。当 `enable_expert_parallel=True` 时，MoE 专家在 EP 组内按 rank 均分。

### 3.3 `init_model_parallel_group()` —— 组工厂函数

每个通信组都通过 `init_model_parallel_group()` 创建，它返回一个 `GroupCoordinator` 对象：

```python
def init_model_parallel_group(
    group_ranks: list[list[int]],
    local_rank: int,
    backend: str,
    use_message_queue_broadcaster: bool = False,
    group_name: str = "",
    use_device_communicator: bool = True,
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=use_device_communicator,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
    )
```

### 3.4 `GroupCoordinator` —— 通信组的核心对象

> **源码位置**: `parallel_state.py` → `class GroupCoordinator` 约 L290

每个 `GroupCoordinator` 维护以下关键属性：

| 属性 | 类型 | 含义 |
|------|------|------|
| `rank` | int | 全局 rank |
| `ranks` | list[int] | 组内所有全局 rank |
| `world_size` | int | 组大小 |
| `local_rank` | int | 本地 rank（设备绑定） |
| `rank_in_group` | int | 组内相对 rank |
| `cpu_group` | ProcessGroup | CPU 通信组 (Gloo) |
| `device_group` | ProcessGroup | 设备通信组 (NCCL) |
| `device_communicator` | DeviceCommunicatorBase | 高性能通信器 |
| `mq_broadcaster` | MessageQueue | 共享内存广播器 |

**双组设计** (cpu_group + device_group)：

```python
for ranks in group_ranks:
    # 设备通信组（NCCL）——用于 GPU 张量通信
    device_group = torch.distributed.new_group(
        ranks, backend=torch_distributed_backend
    )
    # CPU 通信组（Gloo）——用于协调、元数据传输
    cpu_group = torch.distributed.new_group(ranks, backend="gloo")
```

**为什么需要双组？**
- NCCL barrier 内部会偷偷创建 GPU 张量做 broadcast，容易搞乱当前设备（参见源码注释）
- Gloo 运行在 CPU 上，适合传输 Python 对象、控制信息和元数据
- GPU 张量通信走 NCCL（高带宽），CPU 协调走 Gloo（低延迟），两路互不阻塞

### 3.5 全局变量与访问器

`parallel_state.py` 中维护了一组全局单例：

```python
_WORLD: GroupCoordinator = None    # 全局 World 组
_TP:    GroupCoordinator = None    # 张量并行组
_PP:    GroupCoordinator = None    # 流水线并行组
_DP:    GroupCoordinator = None    # 数据并行组
_EP:    GroupCoordinator = None    # 专家并行组
_PCP:   GroupCoordinator = None    # 预填充上下文并行组
_DCP:   GroupCoordinator = None    # 解码上下文并行组
_EPLB:  GroupCoordinator = None    # EP 负载均衡组
```

每个组都有对应的 getter 函数：

```python
def get_world_group() -> GroupCoordinator: ...  # 全局 World 组
def get_tp_group() -> GroupCoordinator:    ...  # 张量并行组
def get_pp_group() -> GroupCoordinator:    ...  # 流水线并行组
def get_dp_group() -> GroupCoordinator:    ...  # 数据并行组
def get_ep_group() -> GroupCoordinator:    ...  # 专家并行组
def get_dcp_group() -> GroupCoordinator:   ...  # 解码上下文并行组
def get_pcp_group() -> GroupCoordinator:   ...  # 预填充上下文并行组
def get_eplb_group() -> GroupCoordinator:  ...  # EP 负载均衡组
```

### 3.6 `_WORLD` 组的创建与其同模型并行组的本质区别

#### 3.6.1 `_WORLD` 组的创建全链路

`_WORLD` 是 vLLM 分布式系统中**最先创建**的通信组，它的创建发生在 `init_distributed_environment()` 函数中，在任何模型并行组创建之前。完整链路如下：

```
用户启动入口（如 LLMEngine / Worker）
  │
  ▼
init_distributed_environment(world_size, rank, local_rank, backend="nccl", ...)
  │
  │  ① DP 地址/端口/rank 调整（如有 DP > 1）
  │  ② torch.distributed.init_process_group()    ← 全局 PyTorch 通信初始化
  │  ③ init_world_group(ranks, local_rank, backend)
  │  ④ _node_count(_WORLD.cpu_group)             ← 节点数检测
  │  ⑤ _INNER_DP_WORLD 创建（如有 DP > 1）
  │
  ▼
_WORLD 就绪 → 后续 initialize_model_parallel() 依赖 _WORLD 创建 TP/PP/DP/...
```

#### 3.6.2 Step ①：DP 场景下的 rank/world_size 调整

当 `data_parallel_size > 1` 或多节点时，`init_distributed_environment()` 首先将局部 rank 映射为跨 DP 副本的全局 rank：

```python
# parallel_state.py L1370-1393
if (config.parallel_config.data_parallel_size > 1
    or config.parallel_config.nnodes > 1):
    parallel_config = config.parallel_config
    # 将 rank 偏移到全局空间：global_rank = dp_rank * local_ws + rank
    rank = parallel_config.data_parallel_rank * world_size + rank
    # 扩展 world_size 为跨 DP 的总进程数
    world_size = parallel_config.world_size_across_dp
    # 选择合适的 IP/port 作为 rendezvous 地址
    if parallel_config.nnodes > 1:
        ip = parallel_config.master_addr
        port = parallel_config.master_port
    else:
        ip = parallel_config.data_parallel_master_ip
        port = parallel_config.get_next_dp_init_port()
    distributed_init_method = get_distributed_init_method(ip, port)
```

**示例**：假设 2 个 DP 副本，每个副本 4 个 GPU（TP=4）：
- 副本 0 的 rank 0~3 → 全局 rank 0~3
- 副本 1 的 rank 0~3 → 全局 rank 4~7
- `world_size` 从 4 调整为 8

#### 3.6.3 Step ②：`torch.distributed.init_process_group()`

这是 PyTorch 原生的全局进程组初始化：

```python
# parallel_state.py L1409-1416
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(
        backend=backend,               # 默认 "nccl"
        init_method=distributed_init_method,  # 如 "tcp://10.0.0.1:29500"
        world_size=world_size,          # 全局进程总数（含所有 DP 副本）
        rank=rank,                      # 全局 rank
        timeout=timeout,
    )
```

**进程模型与集合操作语义**

vLLM 为每个 GPU 启动一个独立的 Python Worker 进程，**一个进程对应一个 GPU，一个 rank**。进程总数 = `TP × PP × PCP`（不含 DP 维度，DP 由 Executor 层管理）：

```python
# multiproc_executor.py L112-115
tp_size, pp_size, pcp_size = self._get_parallel_sizes()
assert self.world_size == tp_size * pp_size * pcp_size

# L168-175 —— 按 local_world_size 启动进程，每个进程绑定一个 GPU
for local_rank in range(self.local_world_size):
    global_rank = global_start_rank + local_rank
    WorkerProc.make_worker_process(
        local_rank=local_rank,
        rank=global_rank,   # ← 每个进程一个唯一 rank
        ...
    )
```

需要注意的是，**每个进程不是"TP 进程"或"PP 进程"，而是同时属于多个并行维度的通信组**。以 TP=4, PP=2（共 8 GPU）为例：

```
进程 (rank=0):  GPU 0
  ├── TP 组 [0,1,2,3] 中的 rank_in_group=0   ← "TP 成员"
  ├── PP 组 [0,4]     中的 rank_in_group=0   ← "PP 成员"（stage 0）
  ├── DP 组 [0]       中的 rank_in_group=0   ← "DP 成员"
  └── WORLD [0..7]    中的 rank=0            ← "全局成员"

进程 (rank=5):  GPU 5
  ├── TP 组 [4,5,6,7] 中的 rank_in_group=1   ← "TP 成员"
  ├── PP 组 [1,5]     中的 rank_in_group=1   ← "PP 成员"（stage 1）
  ├── DP 组 [5]       中的 rank_in_group=0   ← "DP 成员"
  └── WORLD [0..7]    中的 rank=5            ← "全局成员"
```

> **每个 GPU 对应一个 Worker 进程，该进程同时属于 TP/PP/DP/EP/DCP/PCP 等多个通信组。**"TP 组"只是这个进程参与的众多通信组之一。

`init_process_group()` 是一个**集合操作（collective operation）**——**所有进程必须同时调用**，在 rendezvous 点汇合、交换信息、建立连接。任何一个进程缺席都会导致其余进程永久阻塞等待：

```
进程 0 (rank=0)                    进程 1 (rank=1)                    ...  进程 7 (rank=7)
     │                                  │                                       │
     ├─ init_process_group(rank=0) ─┐   ├─ init_process_group(rank=1) ─┐       ├─ init_process_group(rank=7) ─┐
     │                              │   │                              │       │                              │
     │                              ▼   │                              ▼       │                              ▼
     │                         ┌──────────────────────────────────────────┐    │
     │                         │     Rendezvous (tcp://ip:port)          │    │
     │                         │  所有 8 个进程在此汇合、交换信息、建立连接  │    │
     │                         └──────────────────────────────────────────┘    │
     │                              │                              │           │
     ◀──────────────────────────────┘              ┌───────────────┘           │
     │                                             │                           │
     ▼                                             ▼                           ▼
  is_initialized() = True                  is_initialized() = True     is_initialized() = True
```

因此，`init_process_group()` 的调用语义是：

| 维度 | 说明 |
|------|------|
| **每个进程各调用 1 次** | 8 个 Worker 进程 = 8 次调用（每个进程内只执行 1 次） |
| **`if not is_initialized()` 守卫** | 防止**同一进程内**重复初始化，而非限制跨进程调用 |
| **集合操作语义** | 所有进程必须同时调用，少一个就会 hang |

**底层发生了什么：**

```python
# 当 backend="nccl" 时，init_process_group() 内部做了两件事：

# 1. 建立 Store（通常是 TCPStore）—— 进程间的"通讯录"
#    所有进程通过 init_method (如 tcp://10.0.0.1:29500) 连接到同一个 TCPStore
#    在 Store 中注册自己的 rank、hostname、port 等元信息

# 2. 创建 ProcessGroupNCCL —— PyTorch 的默认全局通信组
#    此时还没有真正建立 NCCL 通信通道（NCCL 是 lazy init）
#    NCCL 通道在第一次实际通信操作时才建立
```

**与后续 `new_group()` 的区别：**

| | `init_process_group()` | `new_group()` |
|--|------------------------|---------------|
| **调用次数** | 每个进程调用 **1 次** | 每创建一个子组调用 1 次，可调用多次 |
| **作用** | 建立全局默认 ProcessGroup + TCPStore | 在已有全局组基础上创建子组 |
| **前置条件** | 无（这是第一步） | 必须先完成 `init_process_group()` |
| **是否集合操作** | 是（所有进程必须同时调用） | 是（所有进程必须同时调用，即使不在子组内） |

此步骤完成后：
- PyTorch 的**默认全局 ProcessGroup** 已经建立（每个进程内各持有一份引用）
- 所有进程可以通过 `torch.distributed` 互相通信
- 但 vLLM 还没有自己的 `GroupCoordinator`，需要下一步创建

#### 3.6.4 Step ③：`init_world_group()` — 创建 `_WORLD` GroupCoordinator

与 `init_process_group()` 同理，`init_world_group()` 遵循相同的集合操作语义——**每个 Worker 进程各调用一次**。

`_WORLD` 是 Python **模块级全局变量**（`parallel_state.py` 顶层的 `_WORLD: GroupCoordinator | None = None`），由于每个 Worker 是独立的 Python 进程，拥有各自独立的内存空间，因此每个进程持有**独立的一份 `_WORLD`**。`if _WORLD is None` 守卫的含义是防止**同一进程内**重复初始化，而非限制跨进程调用：

```python
# parallel_state.py L1443-1450
global _WORLD, _NODE_COUNT, _INNER_DP_WORLD

if _WORLD is None:                # ★ 守卫：同一进程内只进入一次
    ranks = list(range(torch.distributed.get_world_size()))  # [0, 1, ..., N-1]
    _WORLD = init_world_group(ranks, local_rank, backend)
else:
    # 同一进程内被再次调用时，仅做 world_size 一致性校验
    assert _WORLD.world_size == torch.distributed.get_world_size(), (
        "world group already initialized with a different world size"
    )
```

**各进程的 `_WORLD` 实例对比**（以 8 GPU 为例）：

```
进程 0 (rank=0)                          进程 1 (rank=1)                    ...
  │                                        │
  │  _WORLD = None  (进程 0 自己的全局变量)   │  _WORLD = None  (进程 1 自己的全局变量)
  │                                        │
  ├─ init_world_group(rank=0) ──┐          ├─ init_world_group(rank=1) ──┐
  │                             │          │                             │
  │  _WORLD = GroupCoordinator  │          │  _WORLD = GroupCoordinator  │
  │    .rank = 0                │          │    .rank = 1                │
  │    .local_rank = 0          │          │    .local_rank = 1          │
  │    .ranks = [0,1,...,7]     │          │    .ranks = [0,1,...,7]     │
  │                             │          │                             │
  ▼                             ▼          ▼                             ▼
  进程 0 的 _WORLD 就绪                     进程 1 的 _WORLD 就绪
```

各进程的 `_WORLD` 对象共享相同的 `ranks` 列表（都是 `[0,...,N-1]`），但 `.rank` 和 `.local_rank` 各不相同——它们记录的是**本进程**在组中的身份。`init_world_group()` 内部的 `GroupCoordinator.__init__()` 调用 `torch.distributed.new_group()` 创建 NCCL/Gloo ProcessGroup，这同样是集合操作，所有进程必须同时执行。

`init_world_group()` 是一个**极简的工厂函数**，只做一件事——用固定参数调用 `GroupCoordinator`：

```python
# parallel_state.py L1134-1142
def init_world_group(
    ranks: list[int], local_rank: int, backend: str
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],              # ← 只有一个组，包含全部 rank
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=False,     # ★ 关键：不创建 DeviceCommunicator
        group_name="world",               # ★ 固定名称 "world"
        # use_message_queue_broadcaster 默认 False → 不创建 MessageQueue
    )
```

**关键：ProcessGroup 创建次数分析**

在 `GroupCoordinator.__init__()` 内部，有一个遍历 `group_ranks` 的循环，为每个子组创建 NCCL + Gloo 两个 ProcessGroup：

```python
for ranks in group_ranks:
    device_group = torch.distributed.new_group(ranks, backend="nccl")
    cpu_group    = torch.distributed.new_group(ranks, backend="gloo")
    if self.rank in ranks:
        # 保留属于自己的那对 ProcessGroup
        self.device_group = device_group
        self.cpu_group = cpu_group
```

对于 `_WORLD` 组，`group_ranks = [[0,1,...,N-1]]`，**只有一个子列表**，所以：

| | `_WORLD` 组 | 模型并行组（如 TP=4, 8 GPU） |
|--|------------|---------------------------|
| `group_ranks` | `[[0,1,2,3,4,5,6,7]]` | `[[0,1,2,3], [4,5,6,7]]` |
| **子列表数量** | **1 个** | **2 个** |
| **循环次数** | **1 次** | **2 次** |
| 创建的 NCCL ProcessGroup 数 | **1 个**（全部 rank） | **2 个**（每个子组各 1 个） |
| 创建的 Gloo ProcessGroup 数 | **1 个** | **2 个** |
| 当前进程实际使用的 | 1 对（就是那唯一一对） | 1 对（只保留自己所在子组的） |

> **为什么模型并行组要循环多次？** 因为 `torch.distributed.new_group()` 是一个**集合操作**——所有进程必须同时调用，即使它们不属于该子组。例如 8 GPU 场景下 TP=4 有两个子组 `[0,1,2,3]` 和 `[4,5,6,7]`：**所有 8 个进程都必须执行两次 `new_group()`**，但 rank 0~3 只保留第一个子组的 ProcessGroup，rank 4~7 只保留第二个。
>
> 而 `_WORLD` 只有一个子组包含全部 rank，所以所有进程只需循环 1 次，创建的那一对 ProcessGroup 就是自己要用的。

**总结：整个 `_WORLD` 创建过程中，精确地创建了 1 个 NCCL ProcessGroup + 1 个 Gloo ProcessGroup，仅此而已。**

#### 3.6.5 Step ③ 内部：`GroupCoordinator.__init__()` 对 `_WORLD` 做了什么

当 `GroupCoordinator` 以 `_WORLD` 的参数被构造时，构造函数内部经历以下步骤：

```python
# parallel_state.py L319-393（摘要注释版）
class GroupCoordinator:
    def __init__(self, group_ranks, local_rank, torch_distributed_backend,
                 use_device_communicator, use_message_queue_broadcaster=False,
                 group_name=None):

        # ─── (a) 注册唯一名称 ───
        self.unique_name = _get_unique_name("world")  # → "world:0"
        _register_group(self)

        # ─── (b) 记录全局 rank 和 local_rank ───
        self.rank = torch.distributed.get_rank()       # 如 rank=3
        self.local_rank = local_rank                   # 如 local_rank=3

        # ─── (c) 为 group_ranks 中的每个子组创建 ProcessGroup ───
        # _WORLD 只有一个子组 [0,1,...,N-1]，所以只循环一次
        for ranks in group_ranks:  # group_ranks = [[0,1,2,...,7]]
            # 创建 NCCL 后端的 device ProcessGroup（包含全部 rank）
            device_group = torch.distributed.new_group(
                ranks, backend="nccl"
            )
            # 创建 Gloo 后端的 cpu ProcessGroup（包含全部 rank）
            cpu_group = torch.distributed.new_group(
                ranks, backend="gloo"
            )
            # 当前进程 rank 在 ranks 中，记录下来
            if self.rank in ranks:
                self.ranks = ranks           # [0,1,...,7]
                self.world_size = len(ranks) # 8
                self.rank_in_group = ranks.index(self.rank)  # 等于 self.rank
                self.device_group = device_group
                self.cpu_group = cpu_group

        # ─── (d) 设置 device ───
        self.device = torch.device(f"cuda:{local_rank}")

        # ─── (e) DeviceCommunicator：不创建！───
        self.use_device_communicator = False  # ★ _WORLD 传入 False
        self.device_communicator = None       # ★ 永远是 None
        # if use_device_communicator and self.world_size > 1:
        #     ... 这段代码不会执行

        # ─── (f) MessageQueue：不创建！───
        self.mq_broadcaster = None            # ★ 永远是 None
        # if use_message_queue_broadcaster and self.world_size > 1:
        #     ... 这段代码不会执行
```

**创建后 `_WORLD` 的状态快照**（以 8 GPU 为例）：

| 属性 | 值 | 说明 |
|------|----|----|
| `unique_name` | `"world:0"` | 全局唯一标识 |
| `rank` | 当前进程的全局 rank | 如 0~7 |
| `local_rank` | 节点内 GPU 编号 | 如 0~7（单节点） |
| `ranks` | `[0, 1, 2, 3, 4, 5, 6, 7]` | 包含**所有**进程 |
| `world_size` | `8` | = 全局进程总数 |
| `rank_in_group` | = `rank`（因为包含全部进程） | 与全局 rank 相同 |
| `device_group` | NCCL `ProcessGroup` | PyTorch 原生 NCCL 组 |
| `cpu_group` | Gloo `ProcessGroup` | PyTorch 原生 Gloo 组 |
| `device` | `cuda:local_rank` | 当前 GPU 设备 |
| `device_communicator` | **`None`** | ★ 无高性能通信器 |
| `mq_broadcaster` | **`None`** | ★ 无共享内存广播 |

#### 3.6.6 Step ④⑤：节点检测与 `_INNER_DP_WORLD`

`_WORLD` 创建后，立即利用其 `cpu_group` 完成后续初始化：

```python
# parallel_state.py L1448-1472
# ④ 节点数检测
if config is not None and config.parallel_config.nnodes > 1:
    _NODE_COUNT = config.parallel_config.nnodes      # 多节点：直接从配置读
else:
    _NODE_COUNT = _node_count(_WORLD.cpu_group)       # 单机：通过 Gloo 检测

# ⑤ _INNER_DP_WORLD（仅 DP > 1 且多节点 DP 时创建）
if parallel_config.data_parallel_size > 1:
    world_size_inner_dp = parallel_config.world_size  # 单个副本的 world_size
    group_ranks = [
        [dp_rank * world_size_inner_dp + i for i in range(world_size_inner_dp)]
        for dp_rank in range(parallel_config.data_parallel_size)
    ]
    # 注意：也用 init_model_parallel_group，但 use_device_communicator=False
    _INNER_DP_WORLD = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,       # ← 依赖 _WORLD 提供 local_rank
        backend,
        use_message_queue_broadcaster=True,  # ← 有 MessageQueue（用于副本内广播）
        group_name="inner_dp_world",
        use_device_communicator=False,       # ← 也是管理组，无 DeviceCommunicator
    )
else:
    _INNER_DP_WORLD = _WORLD  # ← 非 DP 场景下直接指向 _WORLD
```

| 对比 | `_WORLD` | `_INNER_DP_WORLD` |
|------|----------|-------------------|
| **范围** | 跨所有 DP 副本的全部 rank | 单个 DP 副本内的全部 rank |
| **DeviceCommunicator** | `None` | `None`（也是管理组） |
| **MessageQueue** | 无 | DP > 1 时有（用于副本内广播） |
| **DP=1 时** | 独立存在 | 直接等于 `_WORLD` |

至此，`init_distributed_environment()` 完成。`_WORLD`（及可能的 `_INNER_DP_WORLD`）已就绪，后续的 `initialize_model_parallel()` 可以基于 `_WORLD` 创建所有模型并行组。

---

#### 3.6.7 `_WORLD` 与模型并行组：两个不同的工厂函数

理解了 `_WORLD` 的创建流程后，我们来对比模型并行组的创建路径。两者使用**完全不同的工厂函数**：

```python
# ═══ _WORLD 组：通过 init_world_group() 创建 ═══
def init_world_group(
    ranks: list[int], local_rank: int, backend: str
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],           # ← 只有一个组，包含所有 rank
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=False,  # ← 不创建 DeviceCommunicator
        group_name="world",            # ← 固定名称
    )

# ═══ 模型并行组：通过 init_model_parallel_group() 创建 ═══
def init_model_parallel_group(
    group_ranks: list[list[int]],      # ← 多个子组
    local_rank: int,
    backend: str,
    use_message_queue_broadcaster: bool = False,
    group_name: str | None = None,
    use_device_communicator: bool = True,  # ← 默认创建 DeviceCommunicator
) -> GroupCoordinator:
    ...
```

#### 3.6.8 核心差异对比表

| 对比维度 | `_WORLD` 组 | 模型并行组 (TP/PP/DCP/PCP/EP) |
|---------|-------------|------------------------------|
| **工厂函数** | `init_world_group()` | `init_model_parallel_group()` |
| **创建时机** | `init_distributed_environment()` 阶段（最先创建） | `initialize_model_parallel()` 阶段（之后创建） |
| **group_ranks 结构** | `[ranks]` —— 单个列表包含全部 rank | `[子组1, 子组2, ...]` —— 多个子组列表 |
| **world_size** | = 全局进程数（跨所有 DP 副本） | = 组内成员数（如 TP=4 则 world_size=4） |
| **use_device_communicator** | `False` —— **不创建** | `True` —— **创建** DeviceCommunicator |
| **use_message_queue_broadcaster** | `False` —— 不创建 | TP 和 DCP 为 `True`，其余为 `False` |
| **DeviceCommunicator** | `None` | `CudaCommunicator`（含 CustomAllReduce/PyNCCL 等） |
| **mq_broadcaster** | `None` | TP/DCP 有 `MessageQueue` 共享内存广播器 |
| **主要用途** | 全局协调、元数据获取 | 模型计算中的高性能集合通信 |

#### 3.6.9 `_WORLD` 的角色：管理组而非计算组

**`_WORLD` 组的设计哲学是"管理组"，不是"计算组"**。它在后续流程中承担五个关键职责：

**（1）全局信息提供者** —— 所有模型并行组都依赖 `_WORLD` 获取 `local_rank`：

```python
# initialize_model_parallel() 中
_TP = init_model_parallel_group(
    group_ranks, get_world_group().local_rank, backend, ...  # ← 依赖 _WORLD
)
_DCP = init_model_parallel_group(
    group_ranks, get_world_group().local_rank, backend, ...  # ← 依赖 _WORLD
)
_PP = init_model_parallel_group(
    group_ranks, get_world_group().local_rank, backend, ...  # ← 依赖 _WORLD
)
```

**（2）通信后端探测**：

```python
# 从 _WORLD 的 device_group 获取默认后端，传递给所有子组
backend = backend or torch.distributed.get_backend(
    get_world_group().device_group
)
```

**（3）节点数检测**：

```python
_NODE_COUNT = _node_count(_WORLD.cpu_group)  # 利用 Gloo 组检测集群节点数
```

**（4）全局 barrier 同步**：

```python
get_world_group().barrier()  # 模型权重加载时做全局同步
```

**（5）全局身份标识**：

```python
def is_global_first_rank() -> bool:
    if _WORLD is not None:
        return _WORLD.is_first_rank  # 即 rank == 0
```

由于 `_WORLD` 组不参与模型前向/反向计算中的张量通信，因此：
- **不需要** NCCL 高性能集合通信（AllReduce/AllGather/ReduceScatter）
- **不需要** CustomAllReduce、PyNCCL 等自定义高性能后端
- **不需要** MessageQueue 共享内存广播器
- 只需要基础的 PyTorch `ProcessGroup`（`device_group` + `cpu_group`）即可

#### 3.6.10 各组 GroupCoordinator 构建参数全览

```
创建调用链：

init_distributed_environment()
  └─→ init_world_group([0..N-1])
        └─→ GroupCoordinator(
              group_ranks = [[0,1,...,N-1]],   # 全部 rank
              use_device_communicator = False,  # ★ 无设备通信器
              use_message_queue_broadcaster = False)

initialize_model_parallel()
  ├─→ init_model_parallel_group(tp_groups, ..., group_name="tp")
  │     └─→ GroupCoordinator(
  │           group_ranks = [[0,1],[2,3],...],
  │           use_device_communicator = True,   # ★ 有设备通信器
  │           use_message_queue_broadcaster = True)  # ★ 有共享内存广播
  │
  ├─→ init_model_parallel_group(dcp_groups, ..., group_name="dcp")
  │     └─→ GroupCoordinator(
  │           use_device_communicator = True,   # ★ 有设备通信器
  │           use_message_queue_broadcaster = True)  # ★ 有共享内存广播
  │
  ├─→ init_model_parallel_group(pcp_groups, ..., group_name="pcp")
  │     └─→ GroupCoordinator(
  │           use_device_communicator = True,   # ★ 有设备通信器
  │           use_message_queue_broadcaster = False)
  │
  ├─→ init_model_parallel_group(pp_groups, ..., group_name="pp")
  │     └─→ GroupCoordinator(
  │           use_device_communicator = True,   # ★ 有设备通信器
  │           use_message_queue_broadcaster = False)
  │
  ├─→ init_model_parallel_group(dp_groups, ..., group_name="dp")
  │     └─→ GroupCoordinator(
  │           use_device_communicator = True,   # ★ 有设备通信器
  │           use_message_queue_broadcaster = False)
  │
  └─→ init_model_parallel_group(ep_groups, ..., group_name="ep")
        └─→ GroupCoordinator(
              use_device_communicator = True,   # ★ 有设备通信器
              use_message_queue_broadcaster = False)
```

#### 3.6.11 `use_device_communicator=False` 的影响

当 `use_device_communicator=False` 时（仅 `_WORLD` 组），`GroupCoordinator.__init__()` 中以下代码**不执行**：

```python
# parallel_state.py L368-378
self.device_communicator = None  # ← 永远是 None
if use_device_communicator and self.world_size > 1:
    # 以下不会执行：
    device_comm_cls = resolve_obj_by_qualname(
        current_platform.get_device_communicator_cls()
    )
    self.device_communicator = device_comm_cls(  # 不创建 CudaCommunicator
        cpu_group=self.cpu_group,
        device=self.device,
        device_group=self.device_group,
        unique_name=self.unique_name,
    )
```

这意味着 `_WORLD` 组：
- **不初始化** `CudaCommunicator`（不加载 CustomAllReduce / DeepEP / QuickAllReduce / MoriAll2All 等）
- **不分配** 高性能集合通信所需的 GPU 缓冲区
- 调用 `_WORLD.all_reduce()` 等方法会触发 `ValueError("No device communicator found")`
- 只能使用 `barrier()`、`broadcast_object()`、`send_object()`、`recv_object()` 等基于 `cpu_group` (Gloo) 的方法

#### 3.6.12 设计总结

```
┌──────────────────────────────────────────────────────────────────────┐
│                          _WORLD 组                                   │
│  创建：init_distributed_environment() → init_world_group()           │
│  角色：全局管理者 / 元数据提供者                                       │
│  能力：barrier、broadcast_object、节点检测、rank/local_rank 查询       │
│  限制：不能做 GPU 张量集合通信                                        │
│  关键差异：use_device_communicator = False → 无 CudaCommunicator     │
├──────────────────────────────────────────────────────────────────────┤
│               模型并行组 (TP/PP/DP/PCP/DCP/EP)                       │
│  创建：initialize_model_parallel() → init_model_parallel_group()     │
│  角色：模型计算通信执行者                                              │
│  能力：AllReduce、AllGather、ReduceScatter、Send/Recv 等全部操作      │
│  额外：DeviceCommunicator（含 CustomAllReduce/PyNCCL 等）             │
│  关键差异：use_device_communicator = True → 完整的高性能通信栈         │
└──────────────────────────────────────────────────────────────────────┘

完整初始化时序：
  init_distributed_environment()
    ├── DP rank/world_size 调整
    ├── torch.distributed.init_process_group()   ← PyTorch 全局初始化
    ├── init_world_group([0..N-1])               ← 创建 _WORLD（管理基础设施）
    ├── _node_count() 节点检测
    └── _INNER_DP_WORLD 创建（如 DP > 1）
         │
         ▼
  initialize_model_parallel()                    ← 依赖 _WORLD 创建 TP/PP/DP/...
         │
         ▼
  模型推理 / 训练                                 ← 使用 TP/PP/... 做计算通信
                                                   使用 _WORLD 做全局协调
```

---

# 第三部分：通信栈

---

## 第4章 通信抽象层

### 4.1 三层通信架构

```
┌─────────────────────────────────────────┐
│   Layer 3: GroupCoordinator             │  ← 业务接口
│   all_reduce(), send(), recv(), ...     │
├─────────────────────────────────────────┤
│   Layer 2: DeviceCommunicatorBase       │  ← 设备抽象
│   CudaCommunicator / XpuCommunicator    │
├─────────────────────────────────────────┤
│   Layer 1: 具体通信后端                  │  ← 硬件实现
│   PyNCCL, CustomAllreduce, FlashInfer,  │
│   SymmMem, QuickAllReduce, ...          │
└─────────────────────────────────────────┘
```

### 4.2 后端选择机制：从 Platform 到具体通信器

vLLM 的通信后端选择是一个**两级派发**过程：第一级由 `Platform` 类决定使用哪个 `DeviceCommunicator`，第二级由该 `DeviceCommunicator` 内部决定使用哪些具体通信算法。

- **第一级派发**（Platform → DeviceCommunicator）：详见本节 4.2.1～4.2.4
- **第二级派发**（DeviceCommunicator 内部的具体通信算法选择）：以 `CudaCommunicator` 为例，其初始化过程详见 **4.3 节**，AllReduce 优先级瀑布（NCCL SymmMem → QuickAllReduce → FlashInfer → CustomAllreduce → Torch SymmMem → PyNCCL → torch.distributed 兜底）详见 **4.4 节**

#### 4.2.1 第一级派发：Platform → DeviceCommunicator

vLLM 通过 `Platform` 插件体系自动检测当前硬件平台，每个平台类通过 `get_device_communicator_cls()` 方法返回对应的 DeviceCommunicator **全限定类名（FQN）**：

```python
# ═══ 基类默认实现（vllm/platforms/interface.py）═══
class Platform:
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"

# ═══ CUDA 平台（vllm/platforms/cuda.py）═══
class CudaPlatform(Platform):
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"

# ═══ ROCm 平台（vllm/platforms/rocm.py）—— 也使用 CudaCommunicator ═══
class RocmPlatform(Platform):
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"

# ═══ XPU 平台（vllm/platforms/xpu.py）═══
class XpuPlatform(Platform):
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.xpu_communicator.XpuCommunicator"

# ═══ CPU 平台（vllm/platforms/cpu.py）═══
class CpuPlatform(Platform):
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.cpu_communicator.CpuCommunicator"
```

**平台与 DeviceCommunicator 的对应关系：**

| 硬件平台 | Platform 类 | DeviceCommunicator | 自动检测条件 |
|---------|------------|-------------------|------------|
| NVIDIA GPU | `CudaPlatform` | `CudaCommunicator` | `torch.cuda.is_available()` |
| AMD GPU (ROCm) | `RocmPlatform` | `CudaCommunicator` | ROCm 环境 |
| Intel XPU | `XpuPlatform` | `XpuCommunicator` | `torch.xpu.is_available()` |
| CPU | `CpuPlatform` | `CpuCommunicator` | 无 GPU 可用时 |
| TPU | `TpuPlatform` | 平台定制通信器 | TPU 环境 |
| Ascend NPU (OOT) | `NPUPlatform` | `NPUCommunicator` | `pip install vllm-ascend` |
| MUSA GPU (OOT) | `MUSAPlatform` | `CudaCommunicator` | `pip install vllm-musa` |
| 其他 OOT 平台 | 用户实现 | 用户实现 | 通过 `entry_points` 注册 |

#### 4.2.2 平台检测链路

平台自动检测发生在首次访问 `current_platform` 时，通过内置插件探测链完成：

```python
# vllm/platforms/__init__.py
builtin_platform_plugins = {
    "tpu":  tpu_platform_plugin,   # 检查 TPU 环境
    "cuda": cuda_platform_plugin,  # 检查 torch.cuda.is_available()
    "rocm": rocm_platform_plugin,  # 检查 ROCm 环境
    "xpu":  xpu_platform_plugin,   # 检查 torch.xpu.is_available()
    "cpu":  cpu_platform_plugin,   # 兜底：CPU
}

# resolve_current_platform_cls_qualname() 按顺序探测，取第一个激活的插件
# 外部（OOT）插件优先级 > 内置插件
```

#### 4.2.3 使用非默认 DeviceCommunicator 的三种方式

**方式一：通过 OOT Platform 插件（推荐）**

硬件厂商通过 Python 包的 `entry_points` 机制注册自定义平台，vLLM 启动时自动加载。以下以两个真实开源项目为例进行说明。

**示例 A：vllm-ascend（华为 Ascend NPU）**

> 项目地址：[https://github.com/vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)

**Step ① 注册入口**——在 `setup.py` 中通过 `entry_points` 声明插件：

```python
# vllm-ascend/setup.py（末尾）
setup(
    name="vllm_ascend",
    ...
    entry_points={
        "vllm.platform_plugins": [
            "ascend = vllm_ascend:register"          # ← 平台插件
        ],
        "vllm.general_plugins": [
            "ascend_kv_connector = vllm_ascend:register_connector",
            "ascend_model_loader = vllm_ascend:register_model_loader",
            "ascend_service_profiling = vllm_ascend:register_service_profiling",
        ],
    },
)
```

**Step ② register() 函数**——返回自定义 Platform 类的全限定类名（FQN）：

```python
# vllm_ascend/__init__.py
def register():
    """Register the NPU platform."""
    return "vllm_ascend.platform.NPUPlatform"
```

**Step ③ NPUPlatform 类**——继承 `Platform`，通过 `get_device_communicator_cls()` 指定自研通信器：

```python
# vllm_ascend/platform.py
class NPUPlatform(Platform):
    _enum = PlatformEnum.OOT           # 标记为 OOT 平台
    device_name: str = "npu"
    device_type: str = "npu"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator"
```

**Step ④ NPUCommunicator 类**——实现 Ascend NPU 特有的通信操作：

```python
# vllm_ascend/distributed/device_communicators/npu_communicator.py
class NPUCommunicator(DeviceCommunicatorBase):
    def __init__(self, cpu_group, device=None, device_group=None, unique_name=""):
        super().__init__(cpu_group, device, device_group, unique_name)
        self.device = torch.npu.current_device()
        self.ca_comm = None   # NPU 暂不支持 CustomAllReduce

    def all_to_all(self, input_, scatter_dim=0, gather_dim=-1,
                   scatter_sizes=None, gather_sizes=None):
        # 基于 torch.distributed.all_to_all 实现
        ...
```

> vllm-ascend 还额外封装了 `PyHcclCommunicator`（对 HCCL 库的 Python binding，类似 vLLM 对 NCCL 的 PyNcclCommunicator 封装），提供 `all_reduce()`、`broadcast()` 等底层 HCCL 通信原语。

**示例 B：vllm-musa（摩尔线程 MUSA GPU）**

> 项目地址：[https://github.com/MooreThreads/vllm-musa](https://github.com/MooreThreads/vllm-musa)

**Step ① 注册入口**——在 `pyproject.toml` 中通过 `entry-points` 声明插件：

```toml
# vllm-musa/pyproject.toml
[project.entry-points."vllm.platform_plugins"]
musa = "vllm_musa:musa_platform_plugin"

[project.entry-points."vllm.general_plugins"]
musa_custom_ops = "vllm_musa:register_custom_ops"
```

**Step ② musa_platform_plugin() 函数**——检测 MUSA 硬件可用性后返回 Platform FQN：

```python
# vllm_musa/__init__.py
def musa_platform_plugin() -> str | None:
    """vLLM platform plugin entry point."""
    if _torchada_available:
        import torchada
        if torchada.is_musa_platform():
            return "vllm_musa.musa.MUSAPlatform"
    try:
        import torch_musa              # Fallback 检测
        return "vllm_musa.musa.MUSAPlatform"
    except ImportError:
        pass
    return None                        # 非 MUSA 环境返回 None，vLLM 跳过该插件
```

**Step ③ MUSAPlatform 类**——继承 `Platform`，**复用 CudaCommunicator**：

```python
# vllm_musa/musa.py
class MUSAPlatformBase(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "musa"
    device_type: str = "musa"
    dispatch_key: str = "MUSA"
    dist_backend: str = "mccl"         # MUSA 的 NCCL 等价实现

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # 关键差异：复用 vLLM 内置的 CudaCommunicator（通过 torchada 兼容层）
        return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"

# 根据 MTML（摩尔线程管理库）可用性自动选择子类
MUSAPlatform = MtmlMUSAPlatform if mtml_available else NonMtmlMUSAPlatform
```

**两种 OOT 策略对比：**

| 维度 | vllm-ascend (Ascend NPU) | vllm-musa (MUSA GPU) |
|------|--------------------------|----------------------|
| 注册方式 | `setup.py` + `entry_points` | `pyproject.toml` + `entry-points` |
| register 函数 | 直接返回 FQN 字符串 | 先检测硬件再返回（含 fallback 逻辑） |
| DeviceCommunicator | **自研 NPUCommunicator**（继承 Base） | **复用 CudaCommunicator**（通过 torchada 兼容层） |
| 底层通信库 | HCCL（封装为 PyHcclCommunicator） | MCCL + torchada → torch.cuda API |
| 设计理念 | NPU 架构差异大，需自研通信路径 | MUSA 与 CUDA 高度兼容，复用已有实现 |

安装对应插件包后，vLLM 会自动探测并使用该平台（OOT 插件优先级高于内置插件）。可通过 `VLLM_PLUGINS` 环境变量控制加载哪些插件。

**方式二：通过环境变量控制 CudaCommunicator 内部的具体算法**

在使用默认 `CudaCommunicator` 的前提下，可以通过环境变量切换其内部的 AllReduce 实现：

```bash
# 启用/禁用 FlashInfer AllReduce（默认关闭）
export VLLM_ALLREDUCE_USE_FLASHINFER=1    # 开启
export VLLM_ALLREDUCE_USE_FLASHINFER=0    # 关闭（默认）

# 启用/禁用 Torch SymmMem AllReduce（默认开启）
export VLLM_ALLREDUCE_USE_SYMM_MEM=1      # 开启（默认）
export VLLM_ALLREDUCE_USE_SYMM_MEM=0      # 关闭

# 禁用 CustomAllReduce（通过 Python API）
from vllm.distributed.parallel_state import set_custom_all_reduce
set_custom_all_reduce(False)
```

**方式三：直接替换 `current_platform`（测试/开发用）**

```python
import vllm.platforms
vllm.platforms.current_platform = MyCustomPlatform()
# 之后创建的所有 GroupCoordinator 都会使用新平台的 DeviceCommunicator
```

#### 4.2.4 完整选择链路图

```
vLLM 启动
  │
  ├─ 探测硬件平台 → resolve_current_platform_cls_qualname()
  │    ├─ OOT 插件?  → 优先使用（如 vllm-ascend → NPUPlatform, vllm-musa → MUSAPlatform）
  │    ├─ CUDA?      → CudaPlatform
  │    ├─ ROCm?      → RocmPlatform
  │    ├─ XPU?       → XpuPlatform
  │    └─ 兜底       → CpuPlatform
  │
  ├─【第一级派发】获取 DeviceCommunicator 类名
  │    └─ current_platform.get_device_communicator_cls()
  │         → "vllm.distributed...CudaCommunicator"       (NVIDIA/AMD/MUSA)
  │         → "vllm.distributed...XpuCommunicator"        (Intel)
  │         → "vllm.distributed...CpuCommunicator"        (CPU)
  │         → "vllm_ascend...NPUCommunicator"             (Ascend NPU)
  │
  ├─ resolve_obj_by_qualname(FQN) → 动态 import 并实例化
  │
  ├─ GroupCoordinator.__init__()
  │    └─ self.device_communicator = DeviceCommunicatorCls(
  │           cpu_group=..., device=..., device_group=..., unique_name=...
  │       )
  │
  └─【第二级派发】DeviceCommunicator 内部初始化具体通信算法（详见 4.3 节、4.4 节）
       ├─ CudaCommunicator: PyNCCL / CustomAllReduce / SymmMem / FlashInfer / ...
       ├─ NPUCommunicator: PyHCCL / torch.distributed (HCCL 后端)
       └─ 其他 Communicator: 各自的底层通信库
```

> **设计哲学**：vLLM 通过 `Platform` 插件体系实现了硬件适配的**开闭原则（OCP）**——对扩展开放（新硬件只需实现新 Platform + DeviceCommunicator 插件包），对修改关闭（无需改动 vLLM 主仓库代码）。

### 4.3 `CudaCommunicator` 初始化

> **源码位置**: `vllm/distributed/device_communicators/cuda_communicator.py`

`CudaCommunicator` 在创建时会根据硬件条件初始化多种通信后端：

```python
class CudaCommunicator(DeviceCommunicatorBase):
    def __init__(self, ...):
        # 1. PyNCCL —— 始终初始化，作为兜底
        self.pynccl_comm = PyNcclCommunicator(...)
        
        # 2. CustomAllreduce —— CUDA 自定义 AllReduce
        if use_custom_allreduce:
            self.ca_comm = CustomAllreduce(...)
        
        # 3. QuickAllReduce —— 仅 ROCm
        if current_platform.is_rocm():
            self.qr_comm = QuickAllReduce(...)
        
        # 4. SymmMemCommunicator —— 对称内存通信
        self.symm_mem_comm = SymmMemCommunicator(...)
        
        # 5. FlashInferAllReduce —— FlashInfer 实现
        self.fi_ar_comm = FlashInferAllReduce(...)
        
        # 6. All2All Manager —— EP 专用
        self.all2all_manager = All2AllManager(...)
```

### QKVParallelLinear

> **源码位置**: `cuda_communicator.py` → `all_reduce()` 方法

`CudaCommunicator.all_reduce()` 实现了一个**优先级瀑布**（Priority Cascade）：

```python
def all_reduce(self, input_tensor, ...) -> torch.Tensor:
    # 优先级 1: NCCL SymmMem (最优，利用对称内存)
    if self.symm_mem_comm is not None and ...:
        return nccl_symm_mem_allreduce(...)
    
    # 优先级 2: QuickAllReduce (ROCm 专用)
    if self.qr_comm is not None and ...:
        return self.qr_comm.all_reduce(...)
    
    # 优先级 3: FlashInfer AllReduce
    if self.fi_ar_comm is not None and ...:
        return self.fi_ar_comm.all_reduce(...)
    
    # 优先级 4: CustomAllreduce (自定义 CUDA kernel)
    if self.ca_comm is not None and ...:
        return self.ca_comm.all_reduce(...)
    
    # 优先级 5: Torch SymmMem
    if self.symm_mem_comm is not None and ...:
        return torch_symm_mem_allreduce(...)
    
    # 优先级 6: PyNCCL (标准 NCCL)
    if self.pynccl_comm is not None and not self.pynccl_comm.disabled:
        self.pynccl_comm.all_reduce(input_tensor)
        return input_tensor
    
    # 优先级 7: torch.distributed 兜底
    torch.distributed.all_reduce(input_tensor, group=self.device_group)
    return input_tensor
```

**优先级选择总结**：

| 优先级 | 后端 | 条件 | 特点 |
|--------|------|------|------|
| 1 | NCCL SymmMem | 对称内存可用 + 张量对齐 | 零拷贝，最低延迟 |
| 2 | QuickAllReduce | ROCm 平台 | AMD GPU 优化 |
| 3 | FlashInfer | fi_ar_comm 可用 + 尺寸匹配 | 高效小张量 AllReduce |
| 4 | CustomAllreduce | ca_comm 可用 + 尺寸匹配 | 自定义 CUDA kernel |
| 5 | Torch SymmMem | symm_mem 可用（宽松条件） | PyTorch 原生对称内存 |
| 6 | PyNCCL | 始终可用 | 标准 NCCL，通用性最好 |
| 7 | torch.distributed | 终极兜底 | 最慢但最稳定 |

### 4.5 All2All 后端（EP 通信）

All2All 通信是 MoE 专家并行的核心，vLLM 支持 **8 种 All2All 后端**：

```python
class All2AllBackend(Enum):
    NAIVE = "naive"
    ALLGATHER_REDUCESCATTER = "allgather_reducescatter"
    DEEPEP_HIGH_THROUGHPUT = "deepep_high_throughput"
    DEEPEP_LOW_LATENCY = "deepep_low_latency"
    MORI = "mori"
    NIXL_EP = "nixl_ep"
    FLASHINFER_NVLINK_TWO_SIDED = "flashinfer_nvlink_two_sided"
    FLASHINFER_NVLINK_ONE_SIDED = "flashinfer_nvlink_one_sided"
```

| 后端 | 适用场景 | 特点 |
|------|----------|------|
| naive | 通用 | 基于 NCCL P2P send/recv |
| allgather_reducescatter | 小 EP 组 | AllGather + 本地 ReduceScatter |
| deepep_high_throughput | DeepSeek 专用 | 高吞吐，适合大 batch |
| deepep_low_latency | DeepSeek 专用 | 低延迟，适合小 batch |
| mori | 实验性 | 优化的通信调度 |
| nixl_ep | NIXL 加速 | 硬件加速通信 |
| flashinfer_nvlink_two_sided | NVLink 双边 | 利用 NVLink 高带宽 |
| flashinfer_nvlink_one_sided | NVLink 单边 | 单边 RDMA 风格 |

### 4.6 通信操作的 torch.compile 注册

vLLM 将核心通信操作注册为 custom op，以支持 `torch.compile`：

```python
direct_register_custom_op(op_name="all_reduce",    op_func=all_reduce,    ...)
direct_register_custom_op(op_name="reduce_scatter", op_func=reduce_scatter, ...)
direct_register_custom_op(op_name="all_gather",     op_func=all_gather,     ...)
```

这保证了通信操作在计算图编译时不会被优化掉或错误重排。

---

# 第四部分：张量并行权重切分（核心）

---

## 第5章 TP 切分原语层

### 5.1 TP 切分的数学基础

对于线性层 $Y = XA + b$，有两种切分方式：

**列切分 (Column Parallel)**：将 $A$ 按列切分
$$A = [A_1, A_2, ..., A_p]$$
$$Y_i = X \cdot A_i \quad \text{(各卡独立计算，无需通信)}$$

**行切分 (Row Parallel)**：将 $A$ 按行切分
$$A = \begin{bmatrix} A_1 \\ A_2 \\ \vdots \\ A_p \end{bmatrix}, \quad X = [X_1, X_2, ..., X_p]$$
$$Y = \sum_{i=1}^{p} X_i \cdot A_i \quad \text{(需要 AllReduce)}$$

### 5.2 `ColumnParallelLinear` 详解

> **源码位置**: `vllm/model_executor/layers/linear.py` 约 L406

```python
class ColumnParallelLinear(LinearBase):
    """Y = XA + b, A is parallelized along its second dimension (output dim)"""
```

**初始化时的关键切分逻辑**：

```python
def __init__(self, input_size, output_size, ...):
    self.tp_rank = get_tensor_model_parallel_rank()
    self.tp_size = get_tensor_model_parallel_world_size()
    
    # 输入维度不变，输出维度按 TP 均分
    self.input_size_per_partition = input_size          # 不切
    self.output_size_per_partition = divide(output_size, self.tp_size)  # 切
```

**权重形状**：

| | 原始权重 | 切分后（每卡） |
|---|---------|---------------|
| weight | `[output_size, input_size]` | `[output_size/tp, input_size]` |
| bias | `[output_size]` | `[output_size/tp]` |

**权重加载（切片逻辑）**：

```python
def weight_loader(self, param, loaded_weight):
    output_dim = getattr(param, "output_dim", None)
    if output_dim is not None:
        shard_size = param_data.shape[output_dim]
        start_idx = self.tp_rank * shard_size
        # 从完整权重中截取本 rank 的那一段
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
    param_data.copy_(loaded_weight)
```

**Forward 逻辑**：

```python
def forward(self, input_):
    # 1. 矩阵乘法（每卡各算自己的分片）
    output_parallel = self.quant_method.apply(self, input_, bias)
    
    # 2. 可选的 AllGather（通常不需要，因为后接 RowParallel）
    if self.gather_output and self.tp_size > 1:
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    return output
```

### 5.3 `RowParallelLinear` 详解

> **源码位置**: `linear.py` 约 L1383

```python
class RowParallelLinear(LinearBase):
    """Y = XA + b, A is parallelized along its first dimension (input dim)"""
```

**初始化切分**：

```python
def __init__(self, input_size, output_size, ...):
    # 输入维度按 TP 均分，输出维度不变
    self.input_size_per_partition = divide(input_size, self.tp_size)   # 切
    self.output_size_per_partition = output_size                       # 不切
```

**权重形状**：

| | 原始权重 | 切分后（每卡） |
|---|---------|---------------|
| weight | `[output_size, input_size]` | `[output_size, input_size/tp]` |

**Forward 逻辑**（重点：AllReduce）：

```python
def forward(self, input_):
    # 1. 输入已经是并行的（来自前面的 ColumnParallel 输出）
    if self.input_is_parallel:
        input_parallel = input_
    else:
        # 如果输入不是并行的，手动切分
        input_parallel = split_tensor_along_last_dim(input_, self.tp_size)[self.tp_rank]
    
    # 2. 矩阵乘法
    output_parallel = self.quant_method.apply(self, input_parallel, bias_)
    
    # 3. AllReduce 聚合所有 rank 的部分结果
    if self.reduce_results and self.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output_parallel)
    else:
        output = output_parallel
    return output
```

**Bias 处理的精妙设计**：

```python
# 只有 rank 0 才将 bias 融合到 GEMM 中
# 避免 AllReduce 后 bias 被加了 tp_size 次
bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
```

### 5.4 `MergedColumnParallelLinear` 详解

> **源码位置**: `linear.py` 约 L604

**用途**：将多个 ColumnParallel 层的权重合并为一个矩阵，减少内存碎片和 kernel launch 开销。

典型应用：MLP 中的 `gate_proj` + `up_proj` → `gate_up_proj`

```python
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size, output_sizes: list[int], ...):
        self.output_sizes = output_sizes  # e.g., [intermediate, intermediate]
        # 总 output_size = sum(output_sizes)
        super().__init__(input_size=input_size, output_size=sum(output_sizes), ...)
```

**权重加载的分片逻辑**：

对于 `gate_up_proj = [gate_proj | up_proj]`：
- `shard_id=0` (gate_proj): 偏移 = 0，大小 = intermediate_size / tp
- `shard_id=1` (up_proj): 偏移 = intermediate_size / tp，大小 = intermediate_size / tp

```python
def weight_loader(self, param, loaded_weight, loaded_shard_id):
    shard_offset = sum(self.output_sizes[:loaded_shard_id])
    shard_size = self.output_sizes[loaded_shard_id]
    shard_offset //= self.tp_size
    shard_size //= self.tp_size
    
    param_data = param_data.narrow(output_dim, shard_offset, shard_size)
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
    param_data.copy_(loaded_weight)
```

### 5.5 `QKVParallelLinear` 详解

> **源码位置**: `linear.py` 约 L952

**用途**：处理 Attention 层的 Q、K、V 投影，支持 GQA（Grouped-Query Attention）。

```python
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden_size, head_size, total_num_heads, 
                 total_num_kv_heads=None, ...):
        tp_size = get_tensor_model_parallel_world_size()
        
        # Q heads 按 TP 均分
        self.num_heads = divide(total_num_heads, tp_size)
        
        # KV heads 的特殊处理（GQA）
        if tp_size >= total_num_kv_heads:
            # TP 数 >= KV heads 数：每卡 1 个 KV head，部分复制
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
        else:
            # TP 数 < KV heads 数：正常均分
            self.num_kv_heads = divide(total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
```

**output_sizes 结构**：

```python
self.output_sizes = [
    self.num_heads * self.head_size * tp_size,      # Q 部分
    self.num_kv_heads * self.head_size * tp_size,    # K 部分  
    self.num_kv_heads * self.v_head_size * tp_size,  # V 部分
]
```

**shard 映射**：

```python
def _get_shard_offset_mapping(self, loaded_shard_id):
    return {
        "q": 0,
        "k": self.num_heads * self.head_size,
        "v": (self.num_heads + self.num_kv_heads) * self.head_size,
    }.get(loaded_shard_id)

def _get_shard_size_mapping(self, loaded_shard_id):
    return {
        "q": self.num_heads * self.head_size,
        "k": self.num_kv_heads * self.head_size,
        "v": self.num_kv_heads * self.v_head_size,
    }.get(loaded_shard_id)
```

### 5.6 `VocabParallelEmbedding`

**切分策略**：将词表按行均分到各 TP rank。

```
词表大小 V, TP = 4:
  Rank 0: tokens [0, V/4)
  Rank 1: tokens [V/4, V/2)
  Rank 2: tokens [V/2, 3V/4)
  Rank 3: tokens [3V/4, V)
```

**Forward 逻辑**：
1. 将不属于本 rank 范围的 token id 掩为 0
2. 在本地 embedding 表中查找
3. **AllReduce** 聚合所有 rank 的结果（不在本 rank 范围的位置贡献为 0）

### 5.7 TP 层配对模式总结

```
         Column Parallel          Row Parallel
输入 X ──→ [W₁|W₂|W₃|W₄] ──→ AllReduce ──→ 输出 Y
            ↑ 按列切          ↑ 按行切

标准配对：
  ColumnParallel (无通信) → RowParallel (AllReduce)
  
MLP 示例：
  gate_up_proj (MergedColumn, 无通信)
    → SiLU & Mul
      → down_proj (Row, AllReduce)

Attention 示例：
  qkv_proj (QKVParallel/Column, 无通信)
    → Attention 计算
      → o_proj (Row, AllReduce)
```

**关键原则**：**一对 Column+Row 只需一次 AllReduce**。这是 Megatron-LM 论文的核心思想，vLLM 完全沿用。

---

## 第6章 DeepSeek-MoE-16B 端到端切分实例

### 6.1 模型架构概览

DeepSeek-MoE-16B-base 基于 `DeepseekV2` 架构：

| 参数 | 值 |
|------|-----|
| hidden_size | 2048 |
| num_attention_heads | 16 |
| num_key_value_heads | 16 |
| intermediate_size (dense) | 10944 |
| moe_intermediate_size | 1408 |
| n_routed_experts | 64 |
| n_shared_experts | 2 |
| num_experts_per_tok | 6 |
| first_k_dense_replace | 1 |
| num_hidden_layers | 28 |
| vocab_size | 102400 |

**层结构**：
- **第 0 层**：Dense 层（普通 MLP）
- **第 1-27 层**：MoE 层（64 路由专家 + 2 共享专家）

### 6.2 DecoderLayer 的判断逻辑

> **源码位置**: `deepseek_v2.py` → `DeepseekV2DecoderLayer.__init__()` 约 L1042

```python
if (config.n_routed_experts is not None
    and layer_idx >= config.first_k_dense_replace   # >= 1
    and layer_idx % moe_layer_freq == 0):            # 每层都是 MoE
    self.mlp = DeepseekV2MoE(...)    # MoE 层
else:
    self.mlp = DeepseekV2MLP(...)    # Dense 层
```

### 6.3 Attention 层的 TP 切分

对于 DeepSeek-MoE-16B（非 MLA 架构，使用标准 MHA）：

```python
class DeepseekAttention(nn.Module):
    def __init__(self, ...):
        tp_size = get_tensor_model_parallel_world_size()
        
        # Q: 16 heads → 16/tp heads per rank
        self.num_heads = self.total_num_heads // tp_size
        
        # KV: 16 heads → 16/tp heads per rank
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        
        # QKV 投影 —— ColumnParallel (QKVParallelLinear)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,           # 2048，不切
            self.head_dim,         # 128
            self.total_num_heads,  # 16
            self.total_num_kv_heads,  # 16
        )
        
        # O 投影 —— RowParallel
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,  # 16*128=2048
            hidden_size,                            # 2048
        )
```

**TP=4 时各 rank 的 Attention 权重**：

| 子层 | 原始形状 | 每卡形状 | 通信 |
|------|---------|---------|------|
| qkv_proj (Q) | [2048, 2048] | [512, 2048] | 无 |
| qkv_proj (K) | [2048, 2048] | [512, 2048] | 无 |
| qkv_proj (V) | [2048, 2048] | [512, 2048] | 无 |
| o_proj | [2048, 2048] | [2048, 512] | AllReduce |

### 6.4 Dense MLP 的 TP 切分（第 0 层）

```python
class DeepseekV2MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, ...):
        # gate + up 合并为 MergedColumnParallel
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,                     # 2048
            [intermediate_size] * 2,         # [10944, 10944]
        )
        # down 为 RowParallel
        self.down_proj = RowParallelLinear(
            intermediate_size,   # 10944
            hidden_size,         # 2048
        )
```

**TP=4 时的 Dense MLP 权重**：

| 子层 | 原始形状 | 每卡形状 | 通信 |
|------|---------|---------|------|
| gate_up_proj | [21888, 2048] | [5472, 2048] | 无 |
| down_proj | [2048, 10944] | [2048, 2736] | AllReduce |

**Forward 数据流**：
```
x [B, 2048]
  → gate_up_proj → [B, 5472] (每卡)
    → SiluAndMul → [B, 2736] (每卡)
      → down_proj → [B, 2048] (AllReduce 后)
```

### 6.5 MoE 层的切分（第 1-27 层）

> **源码位置**: `deepseek_v2.py` → `DeepseekV2MoE.__init__()` 约 L256

MoE 层的并行化涉及 **TP + EP** 两个维度的交互：

```python
class DeepseekV2MoE(nn.Module):
    def __init__(self, config, parallel_config, ...):
        self.tp_size = get_tensor_model_parallel_world_size()
        
        # EP 组信息
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        
        # 64 路由专家
        self.n_routed_experts = config.n_routed_experts  # 64
        
        # Gate（路由器）—— 全复制
        self.gate = GateLinear(
            config.hidden_size,       # 2048
            config.n_routed_experts,  # 64
        )
        
        # 共享专家 —— 使用 DeepseekV2MLP
        intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        # = 1408 * 2 = 2816
        self.shared_experts = DeepseekV2MLP(
            hidden_size=config.hidden_size,       # 2048
            intermediate_size=intermediate_size,   # 2816
            reduce_results=False,                  # 不在内部 AllReduce
        )
        
        # 路由专家 —— SharedFusedMoE
        self.experts = SharedFusedMoE(
            num_experts=64,
            top_k=6,
            hidden_size=2048,
            intermediate_size=1408,
            reduce_results=False,
        )
```

#### 6.5.1 专家分配策略

当 `enable_expert_parallel=True` 时：

```
EP_size = DP × PCP × TP（例如 DP=2, PCP=1, TP=4 → EP_size=8）

64 experts / 8 EP ranks = 8 experts per rank

Rank 0: experts [0-7]
Rank 1: experts [8-15]
...
Rank 7: experts [56-63]
```

#### 6.5.2 MoE Forward 数据流

```python
def forward(self, hidden_states):
    # 1. Router 计算（全复制，每卡都算）
    router_logits, _ = self.gate(hidden_states)  # [B, 64]
    
    # 2. FusedMoE（包含 All2All 通信）
    #    - Dispatch: 根据 router 结果，通过 All2All 将 token 发送到拥有对应专家的 rank
    #    - Compute: 本地计算分配到的 token
    #    - Combine: 通过 All2All 将结果发送回原始 rank
    shared_output, final_hidden_states = self.experts(
        hidden_states=hidden_states, 
        router_logits=router_logits,
    )
    
    # 3. 应用路由缩放因子
    final_hidden_states *= self.routed_scaling_factor
    
    # 4. 加上共享专家的输出
    final_hidden_states += shared_output
    
    # 5. AllReduce（聚合 TP 组内的结果）
    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    
    return final_hidden_states
```

#### 6.5.3 共享专家的处理

共享专家 (`shared_experts`) 使用 `DeepseekV2MLP`，但有特殊设置：

- `reduce_results=False`：不在 MLP 内部做 AllReduce
- `is_sequence_parallel=False`（默认）：使用标准 TP 切分

共享专家与路由专家的输出在 MoE forward 末尾合并后，**统一做一次 AllReduce**，减少通信次数。

### 6.6 Embedding 与 LM Head 的 TP 切分

```python
class DeepseekV2Model(nn.Module):
    def __init__(self, ...):
        # 词嵌入 —— 词表并行
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,    # 102400
                config.hidden_size,   # 2048
            )

class DeepseekV2ForCausalLM(nn.Module):
    def __init__(self, ...):
        # LM Head —— 词表并行
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,    # 102400
                config.hidden_size,   # 2048
            )
```

**TP=4 时的 Embedding/LM Head**：

| 组件 | 原始形状 | 每卡形状 | 通信 |
|------|---------|---------|------|
| embed_tokens | [102400, 2048] | [25600, 2048] | AllReduce |
| lm_head | [102400, 2048] | [25600, 2048] | AllGather |

### 6.7 单层端到端通信开销总结

**Dense 层（第 0 层）**：每层 2 次 AllReduce

```
输入 [B, 2048]
  → Attention: qkv_proj(无通信) → attention → o_proj(AllReduce ①)
  → MLP: gate_up_proj(无通信) → SiLU → down_proj(AllReduce ②)
→ 输出 [B, 2048]
```

**MoE 层（第 1-27 层）**：每层 1 次 AllReduce + 2 次 All2All

```
输入 [B, 2048]
  → Attention: qkv_proj(无通信) → attention → o_proj(AllReduce ①)
  → MoE:
      Gate(无通信) → All2All Dispatch → Expert Compute → All2All Combine
      + SharedExpert(无通信)
      → AllReduce ②（合并 TP 结果）
→ 输出 [B, 2048]
```

---

# 第五部分：其他并行维度

---

## 第7章 流水线并行 (PP)

### 7.1 PP 的核心思想

PP 将模型的层按阶段（stage）分配到不同的 GPU：

```
Stage 0 (GPU 0): layers [0, ..., N/PP - 1]
Stage 1 (GPU 1): layers [N/PP, ..., 2*N/PP - 1]
...
```

### 7.2 vLLM 中的 PP 实现

#### 7.2.1 层分配

```python
# vllm/model_executor/models/utils.py → make_layers()
def make_layers(num_hidden_layers, layer_fn, prefix):
    start_layer, end_layer = get_pp_indices(num_hidden_layers, ...)
    layers = nn.ModuleList([
        layer_fn(f"{prefix}.{i}") if start_layer <= i < end_layer 
        else PPMissingLayer()
        for i in range(num_hidden_layers)
    ])
    return start_layer, end_layer, layers
```

`PPMissingLayer` 是一个空占位符，保证所有 rank 的层索引一致但不实际分配内存。

#### 7.2.2 阶段间通信

PP 使用 **P2P Send/Recv** 而非集合通信：

```python
# CudaCommunicator
def send(self, tensor, dst=None):
    if dst is None:
        dst = (self.rank_in_group + 1) % self.world_size
    pynccl_comm.send(tensor, dst)

def recv(self, size, dtype, src=None):
    if src is None:
        src = (self.rank_in_group - 1) % self.world_size
    tensor = torch.empty(size, dtype=dtype, device=self.device)
    pynccl_comm.recv(tensor, src)
    return tensor
```

#### 7.2.3 IntermediateTensors 传递

```python
# DeepseekV2Model.forward()
def forward(self, input_ids, positions, intermediate_tensors, ...):
    if get_pp_group().is_first_rank:
        hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        # 从上一个 stage 接收
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]
    
    for layer in self.layers[start:end]:
        hidden_states, residual = layer(positions, hidden_states, residual)
    
    if not get_pp_group().is_last_rank:
        # 发送给下一个 stage
        return IntermediateTensors({
            "hidden_states": hidden_states,
            "residual": residual,
        })
    
    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states
```

### 7.3 PP 与 TP 的交互

PP 和 TP 可以组合使用：

```
8 GPU, TP=2, PP=2:
  Stage 0: [GPU0, GPU1] (TP group)  → layers [0-13]
  Stage 1: [GPU2, GPU3] (TP group)  → layers [14-27]
  
  PP groups: [GPU0, GPU2], [GPU1, GPU3]
  TP groups: [GPU0, GPU1], [GPU2, GPU3]
```

同一 stage 内的 GPU 用 TP（AllReduce），stage 之间用 PP（Send/Recv）。

### 7.4 PP 的条件守卫

模型中的 Embedding 和 LM Head 仅在特定 stage 创建：

```python
if get_pp_group().is_first_rank:
    self.embed_tokens = VocabParallelEmbedding(...)
else:
    self.embed_tokens = PPMissingLayer()

if get_pp_group().is_last_rank:
    self.norm = RMSNorm(...)
    self.lm_head = ParallelLMHead(...)
else:
    self.norm = PPMissingLayer()
```

---

## 第8章 专家并行 (EP)

### 8.1 EP 的动机

DeepSeek-MoE-16B 有 64 个路由专家。当 TP=4 时，每个 TP rank 需要存储全部 64 个专家的权重。EP 的目标是**将专家分散到多个 rank**，减少每个 rank 的内存占用。

### 8.2 EP 组的构成

```python
# EP 组 = DP × PCP × TP 的所有 rank
EP_size = data_parallel_size * prefill_context_model_parallel_size * tensor_model_parallel_size
```

**示例**：DP=2, TP=4, PP=1
```
EP_size = 2 × 1 × 4 = 8
EP group: [0,1,2,3,4,5,6,7] (所有 GPU 在一个 EP 组中)
64 experts / 8 = 8 experts per rank
```

### 8.3 EP 通信的两层抽象架构

vLLM 的 EP 通信 **并非只使用单一 All2All 原语**，而是在统一的 `dispatch/combine` 语义下，实现了 **8 种完全不同的通信后端**。整体采用 **两层分离架构**：

| 层次 | 职责 | 核心代码 |
|------|------|---------|
| **通信层** — `All2AllManager` | 底层跨 rank 的数据搬运 | `vllm/distributed/device_communicators/all2all.py` |
| **MoE 集成层** — `PrepareAndFinalize` | 路由计算、量化、与 FusedMoE kernel 对接 | `vllm/model_executor/layers/fused_moe/` 下各文件 |

**通信层**定义在 `All2AllManagerBase` 基类中（`base_device_communicator.py`），提供统一的 `dispatch()` / `combine()` 接口：pall2all_utils.py

```python
# base_device_communicator.py — All2AllManagerBase
class All2AllManagerBase:
    def dispatch(self, hidden_states, topk_weights, topk_ids, ...):
        """Phase 1: 将 token 发送到拥有目标专家的 rank"""
        ...
    def combine(self, hidden_states, ...):
        """Phase 3: 将计算结果发送回 token 原始 rank"""
        ...
```

**MoE 集成层**通过 `PrepareAndFinalize` 类家族，将通信与 MoE 计算流水线对接。`all2all_utils.py` 中的 `maybe_make_prepare_finalize()` 函数根据后端配置选择对应实现：

```python
# all2all_utils.py — 选择 PrepareAndFinalize 实现
def maybe_make_prepare_finalize(moe, quant_config, ...):
    if moe.use_deepep_ht_kernels:
        return DeepEPHTPrepareAndFinalize(...)
    elif moe.use_deepep_ll_kernels:
        return DeepEPLLPrepareAndFinalize(...)
    elif moe.use_mori_kernels:
        return MoriPrepareAndFinalize(...)
    elif moe.use_fi_nvl_two_sided_kernels:
        return FlashInferNVLinkTwoSidedPrepareAndFinalize(...)
    elif moe.use_fi_nvl_one_sided_kernels:
        return FlashInferNVLinkOneSidedPrepareAndFinalize(...)
    elif moe.use_naive_all2all_kernels:
        return MoEPrepareAndFinalizeNaiveDPEPModular(...)
    elif moe.use_nixl_ep_kernels:
        return NixlEPPrepareAndFinalize(...)
```

### 8.4 EP 通信的三阶段执行流程

无论使用哪种后端，EP 通信都遵循统一的三阶段流程：

```
Phase 1 - Dispatch (分发):
  每个 rank 根据 router 结果，将 token 发送到拥有目标专家的 rank

Phase 2 - Compute (计算):
  每个 rank 在本地执行分配到的 token 与本地专家的计算

Phase 3 - Combine (聚合):
  每个 rank 将计算结果发送回 token 的原始 rank，加权求和得到最终输出
```

```python
# CudaCommunicator 中的 dispatch/combine 接口
def dispatch(self, hidden_states, topk_weights, topk_ids, ...):
    return self.all2all_manager.dispatch(hidden_states, topk_weights, topk_ids, ...)

def combine(self, hidden_states, ...):
    return self.all2all_manager.combine(hidden_states, ...)
```

### 8.5 后端选择机制

后端由用户参数 `--all2all-backend` 指定，在 `CudaCommunicator.__init__()` 中（`cuda_communicator.py`）通过 `if/elif` 链映射到具体的 Manager 类：

```python
# cuda_communicator.py — 后端选择（简化）
if self.all2all_backend == "naive":
    self.all2all_manager = NaiveAll2AllManager(...)
elif self.all2all_backend == "allgather_reducescatter":
    self.all2all_manager = AgRsAll2AllManager(...)
elif self.all2all_backend == "deepep_high_throughput":
    self.all2all_manager = DeepEPHTAll2AllManager(...)
elif self.all2all_backend == "deepep_low_latency":
    self.all2all_manager = DeepEPLLAll2AllManager(...)
elif self.all2all_backend == "mori":
    self.all2all_manager = MoriAll2AllManager(...)
elif self.all2all_backend == "nixl_ep":
    self.all2all_manager = NixlEPAll2AllManager(...)
elif self.all2all_backend in ("flashinfer_all2allv", "flashinfer_nvlink_two_sided"):
    self.all2all_manager = FlashInferNVLinkTwoSidedManager(...)
elif self.all2all_backend == "flashinfer_nvlink_one_sided":
    self.all2all_manager = FlashInferNVLinkOneSidedManager(...)
```

`MoEParallelConfig`（`fused_moe/config.py`）提供了便捷属性来判断当前使用的后端：

```python
# fused_moe/config.py — MoEParallelConfig 后端判断属性
@property
def use_all2all_kernels(self):
    return self.dp_size > 1 and self.use_ep

@property
def use_deepep_ht_kernels(self):
    return self.use_all2all_kernels and self.all2all_backend == "deepep_high_throughput"

@property
def use_deepep_ll_kernels(self):
    return self.use_all2all_kernels and self.all2all_backend == "deepep_low_latency"

@property
def use_naive_all2all_kernels(self):
    return self.use_all2all_kernels and self.all2all_backend in ["naive", "allgather_reducescatter"]

# ... 类似的 use_fi_nvl_two_sided_kernels, use_mori_kernels, use_nixl_ep_kernels
```

### 8.6 八种通信后端详解

#### 8.6.1 `naive` — 基于 AllReduce 的模拟

| 属性 | 值 |
|------|---|
| **Manager 类** | `NaiveAll2AllManager`（`all2all.py`） |
| **底层通信原语** | AllReduce（广播 + 切片） |
| **SM 开销** | 高 |
| **适用场景** | 仅用于测试/调试 |

**实现原理**：dispatch 时每个 rank 把自己的数据通过 AllReduce 广播给所有 rank，接收方再切片取出属于自己的部分；combine 同理。通信量为 $O(N^2)$，极不高效，仅作为正确性验证的参考实现。

```python
# NaiveAll2AllManager.dispatch（简化）
class NaiveAll2AllManager(All2AllManagerBase):
    def dispatch(self, hidden_states, topk_weights, topk_ids, ...):
        # 1. AllReduce: 每个 rank 的数据广播给所有 rank
        # 2. 根据 topk_ids 切片提取属于本 rank 专家的 token
        ...
```

#### 8.6.2 `allgather_reducescatter` — AllGatherV + ReduceScatterV

| 属性 | 值 |
|------|---|
| **Manager 类** | `AgRsAll2AllManager`（`all2all.py`） |
| **dispatch 原语** | AllGatherV（变长 all-gather） |
| **combine 原语** | ReduceScatterV（变长 reduce-scatter） |
| **SM 开销** | 中 |
| **适用场景** | 通用兼容方案，不依赖特殊硬件 |

**实现原理**：这 **不是** 真正的 All2All 原语。dispatch 使用 `AllGatherV`，每个 rank 收集所有 rank 发给自己的 token（变长，因为每个 rank 发给不同专家的 token 数不同）；combine 使用 `ReduceScatterV`，将计算结果按 token 归属做 reduce-scatter。通过 AllGather + ReduceScatter 的组合实现了等价于 All2All 的语义。

```python
# AgRsAll2AllManager — 使用 AllGatherV + ReduceScatterV
class AgRsAll2AllManager(All2AllManagerBase):
    def dispatch(self, ...):
        # AllGatherV: 变长 gather，收集各 rank 发来的 token
        ...
    def combine(self, ...):
        # ReduceScatterV: 变长 reduce-scatter，将结果归还原始 rank
        ...
```

#### 8.6.3 `deepep_high_throughput` — DeepEP 高吞吐模式

| 属性 | 值 |
|------|---|
| **Manager 类** | `DeepEPHTAll2AllManager`（`all2all.py`） |
| **底层通信原语** | GPU-initiated NVLink + RDMA（DeepEP 库） |
| **SM 开销** | ~20 SMs |
| **适用场景** | 高吞吐 prefill 场景 |
| **PrepareAndFinalize** | `DeepEPHTPrepareAndFinalize`（`deepep_ht_prepare_finalize.py`） |

**实现原理**：使用 DeepSeek 团队开源的 [DeepEP](https://github.com/deepseek-ai/DeepEP) 库。通信通过 `deep_ep.Buffer` 对象完成，支持 **节点内 NVLink** 和 **节点间 RDMA** 的融合通信。需要占用约 20 个 SM 做通信（`num_sms=20`）。

```python
# DeepEPHTAll2AllManager（简化）
class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
    def num_sms(self):
        return 20  # 通信占用 ~20 个 SM

    # dispatch/combine 均通过 deep_ep.Buffer 完成
```

**PrepareAndFinalize 层**：`DeepEPHTPrepareAndFinalize` 直接调用 `deep_ep.Buffer.dispatch()` 和 `deep_ep.Buffer.combine()`，并支持 **DBO（Dual-Batch Overlap）** 双批次重叠优化——可以在一个 micro-batch 做通信的同时，另一个 micro-batch 做计算：

```python
# deepep_ht_prepare_finalize.py — DeepEP 高吞吐 PrepareAndFinalize
class DeepEPHTPrepareAndFinalize(FusedMoEPrepareAndFinalizeModular):
    def prepare(self, a1, topk_weights, topk_ids, ...):
        # 直接调用 DeepEP buffer 的 dispatch
        expert_x, expert_x_scale, handle, event = self.buffer.dispatch(
            a1, topk_ids, num_experts,
            use_fp8=self.use_fp8_dispatch, async_finish=False
        )
        self.handles[ubatch_id] = (handle, event)
        return expert_x, expert_x_scale, expert_tokens_meta, ...

    def finalize(self, output, fused_expert_output, topk_weights, ...):
        # 直接调用 DeepEP buffer 的 combine
        combined = self.buffer.combine(
            fused_expert_output, handle, topk_weights,
            async_finish=False
        )
        output.copy_(combined)
```

#### 8.6.4 `deepep_low_latency` — DeepEP 低延迟模式

| 属性 | 值 |
|------|---|
| **Manager 类** | `DeepEPLLAll2AllManager`（`all2all.py`） |
| **底层通信原语** | Pure RDMA（DeepEP 库） |
| **SM 开销** | **0 SMs**（完全不占用计算 SM） |
| **适用场景** | 低延迟 decode 场景 |
| **PrepareAndFinalize** | `DeepEPLLPrepareAndFinalize`（`deepep_ll_prepare_finalize.py`） |

**实现原理**：同样基于 DeepEP 库，但优化目标不同——追求 **最低通信延迟**。通信完全通过 RDMA 完成，`max_sms_used() = 0`，不消耗任何 GPU SM 资源。支持 FP8 量化 dispatch（`DEEPEP_QUANT_BLOCK_SHAPE = [128, 128]`）。

```python
# DeepEPLLAll2AllManager（简化）
class DeepEPLLAll2AllManager(DeepEPAll2AllManagerBase):
    def max_sms_used(self):
        return 0  # 纯 RDMA 通信，0 SM 开销
```

**PrepareAndFinalize 层**：`DeepEPLLPrepareAndFinalize` 仅支持特定 hidden_size（`SUPPORTED_HIDDEN_SIZES = [2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]`），调用 DeepEP 低延迟 buffer 进行通信。

#### 8.6.5 `nixl_ep` — NIXL EP（NVIDIA 网络库）

| 属性 | 值 |
|------|---|
| **Manager 类** | `NixlEPAll2AllManager`（`all2all.py`） |
| **底层通信原语** | RDMA-based（NIXL EP 库） |
| **SM 开销** | **0 SMs** |
| **适用场景** | 弹性 EP、低延迟 |
| **PrepareAndFinalize** | `NixlEPPrepareAndFinalize`（`nixl_ep_prepare_finalize.py`） |

**实现原理**：使用 NVIDIA 的 NIXL EP 库，同样基于 RDMA。支持 **Elastic EP**（动态调整 EP 大小）、FP8 dispatch 以及 ue8m0 scale 格式。`max_sms_used() = 0`。

```python
# NixlEPAll2AllManager（简化）
class NixlEPAll2AllManager(All2AllManagerBase):
    def max_sms_used(self):
        return 0  # RDMA 通信，0 SM 开销
```

**PrepareAndFinalize 层**：`NixlEPPrepareAndFinalize` 支持异步通信（`supports_async = True`），通过 `prepare_async()` / `finalize_async()` 实现通信与计算的重叠。dispatch 前可选做 FP8 量化以减少传输数据量：

```python
# nixl_ep_prepare_finalize.py（简化）
class NixlEPPrepareAndFinalize(FusedMoEPrepareAndFinalizeModular):
    SUPPORTED_HIDDEN_SIZES = [2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]

    def supports_async(self) -> bool:
        return True  # 支持异步 dispatch/combine

    def prepare_async(self, a1, topk_weights, topk_ids, ...):
        # 通过 nixl_ep.Buffer.dispatch 发起异步通信
        expert_x, expert_num_tokens, handle, _, hook = self.buffer.dispatch(
            a1, topk_ids, max_tokens, num_experts,
            use_fp8=self.use_fp8_dispatch, async_finish=False,
            return_recv_hook=True
        )
        return hook, receiver  # 返回 hook 供后续同步
```

#### 8.6.6 `flashinfer_nvlink_two_sided` — FlashInfer NVLink 双边

| 属性 | 值 |
|------|---|
| **Manager 类** | `FlashInferNVLinkTwoSidedManager`（`all2all.py`） |
| **底层通信原语** | NVLink 双边 alltoall（FlashInfer 库） |
| **SM 开销** | 中 |
| **适用场景** | 节点内 NVLink 互联 |
| **PrepareAndFinalize** | `FlashInferNVLinkTwoSidedPrepareAndFinalize`（`flashinfer_nvlink_two_sided_prepare_finalize.py`） |

**实现原理**：使用 FlashInfer 库的 `alltoall` 原语，基于 NVLink 双边通信（发送方和接收方都参与通信操作）。配置名也支持旧名 `flashinfer_all2allv`。

```python
# FlashInferNVLinkTwoSidedManager（简化）
class FlashInferNVLinkTwoSidedManager(All2AllManagerBase):
    # 使用 FlashInfer 的 alltoall API
    # dispatch/combine 均通过 FlashInfer 的 NVLink 双边通信完成
```

**PrepareAndFinalize 层**：`FlashInferNVLinkTwoSidedPrepareAndFinalize` 通过 `self.all2all_manager`（从 EP 组获取）调用 dispatch/combine，在层级做路由计算和输入准备。

#### 8.6.7 `flashinfer_nvlink_one_sided` — FlashInfer NVLink 单边

| 属性 | 值 |
|------|---|
| **Manager 类** | `FlashInferNVLinkOneSidedManager`（`all2all.py`） |
| **底层通信原语** | NVLink 单边（`MoeAlltoAll`，源自 TRT-LLM/FlashInfer） |
| **SM 开销** | 低 |
| **适用场景** | 节点内低延迟 |
| **PrepareAndFinalize** | `FlashInferNVLinkOneSidedPrepareAndFinalize`（`flashinfer_nvlink_one_sided_prepare_finalize.py`） |

**实现原理**：使用 `MoeAlltoAll`（源自 TensorRT-LLM 的实现），基于 NVLink **单边通信**——只需接收方主动拉取数据，无需发送方配合，延迟更低。

```python
# FlashInferNVLinkOneSidedManager（简化）
class FlashInferNVLinkOneSidedManager(All2AllManagerBase):
    # 使用 MoeAlltoAll (TRT-LLM) 的 NVLink 单边通信
    # 接收方主动拉取数据，延迟更低
```

**PrepareAndFinalize 层**：`FlashInferNVLinkOneSidedPrepareAndFinalize` 在 dispatch 前会先做输入量化（FP8），减少 NVLink 传输带宽占用。

#### 8.6.8 `mori` — MoRI for AMD ROCm

| 属性 | 值 |
|------|---|
| **Manager 类** | `MoriAll2AllManager`（`all2all.py`） |
| **底层通信原语** | MoRI kernels（AMD ROCm 专用） |
| **支持硬件** | AMD gfx942 (MI300X)、gfx950 (MI350) |
| **适用场景** | AMD GPU 集群 |
| **PrepareAndFinalize** | `MoriPrepareAndFinalize`（`mori_prepare_finalize.py`） |

**实现原理**：AMD ROCm 平台专用，使用 MoRI（Mixture of RDMA Interconnects）库。通过 `mori.ops.EpDispatchCombineOp` 执行通信，支持节点内/节点间分离通信。

```python
# mori_prepare_finalize.py（简化）
class MoriPrepareAndFinalize(FusedMoEPrepareAndFinalizeModular):
    def __init__(self, mori_op: mori.ops.EpDispatchCombineOp, ...):
        self.mori_op = mori_op

    def prepare(self, a1, topk_weights, topk_ids, ...):
        # 可选 FP8 量化（使用 aiter 库的 hip_quant）
        dispatch_a1, dispatch_weights, dispatch_scale, dispatch_ids, recv_num = \
            self.mori_op.dispatch(a1, topk_weights, scale, topk_ids)
        return dispatch_a1, dispatch_scale, expert_tokens_meta, ...

    def finalize(self, output, fused_expert_output, topk_weights, topk_ids, ...):
        result = self.mori_op.combine(fused_expert_output, None, topk_ids)[0]
        output.copy_(result[:num_token])
```

### 8.7 八种后端对比总结

| 后端名称 | 底层通信原语 | SM 开销 | 典型场景 | 硬件平台 |
|---------|-------------|---------|---------|---------|
| `naive` | AllReduce（广播+切片） | 高 | 测试/调试 | CUDA |
| `allgather_reducescatter` | AllGatherV + ReduceScatterV | 中 | 通用兼容 | CUDA |
| `deepep_high_throughput` | NVLink + RDMA（GPU-initiated） | ~20 SMs | 高吞吐 prefill | CUDA |
| `deepep_low_latency` | Pure RDMA | 0 SMs | 低延迟 decode | CUDA |
| `nixl_ep` | RDMA（NIXL） | 0 SMs | 弹性 EP、低延迟 | CUDA |
| `flashinfer_nvlink_two_sided` | NVLink 双边 alltoall | 中 | 节点内 NVLink | CUDA |
| `flashinfer_nvlink_one_sided` | NVLink 单边（TRT-LLM） | 低 | 节点内低延迟 | CUDA |
| `mori` | MoRI kernels | — | AMD MI300X/MI350 | ROCm |

> **关键结论**："All2All" 在 vLLM 中是一个 **统一的接口抽象名称**（dispatch + combine 语义），而非单一的通信原语。实际底层涵盖了 AllReduce 模拟、AllGather+ReduceScatter 组合、RDMA 单边通信、NVLink 双边/单边通信等截然不同的通信技术。不同后端在 SM 占用、通信延迟、吞吐量和硬件适配方面各有侧重，用户可根据部署场景通过 `--all2all-backend` 参数灵活选择。

### 8.8 EPLB（专家并行负载均衡）

vLLM 支持 **Expert Parallel Load Balancing**，通过冗余专家缓解负载不均：

```python
# DeepseekV2MoE 中的 EPLB 配置
self.n_redundant_experts = eplb_config.num_redundant_experts
self.n_logical_experts = self.n_routed_experts          # 64
self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
self.n_local_physical_experts = self.n_physical_experts // self.ep_size
```

EPLB 创建了独立的 `_EPLB` 通信组，与 EP 组有相同的 rank 组成但独立的 ProcessGroup，避免 EPLB 通信与 MoE forward 的 All2All 死锁。

### 8.9 EP 通信的替代实现方案：AllGather 与 AllReduce

除了 All2All 语义外，EP 通信还可以通过 **AllGather** 和 **AllReduce** 等集合通信原语来实现。这两种方案在语义和性能上有本质区别。

#### 8.9.1 三种 EP 通信范式对比

```
方案 A — All2All（点对点交换）:
  每个 rank 只发送目标 rank 需要的 token，只接收自己需要的 token
  通信量: O(tokens × hidden_dim) — 与实际路由结果成正比

方案 B — AllGather + ReduceScatter（全量收集）:
  Dispatch: 每个 rank 收集所有 rank 的全部 token（AllGather）
  Combine:  每个 rank 将结果按 token 归属做 reduce（ReduceScatter）
  通信量: O(N × tokens × hidden_dim) — 与 EP 组大小 N 成正比

方案 C — AllReduce（广播+切片）:
  Dispatch: 通过 AllReduce 使每个 rank 获得所有 token
  Combine:  通过 AllReduce 使结果回到原始 rank
  通信量: O(N × tokens × hidden_dim) — 与 AllGather 方案相当
```

**关键权衡**：

| 维度 | All2All | AllGather/AllReduce |
|------|---------|-------------------|
| **通信量** | 只传必要数据，通信量最优 | 传输全量数据，通信量 $\times N$ |
| **实现复杂度** | 需要变长通信、路由索引管理 | 简单直接，标准集合通信 |
| **硬件依赖** | 需要专用 kernel（DeepEP/NIXL 等）| NCCL 原生支持 |
| **适用场景** | EP_size 较大时通信优势明显 | EP_size 较小或混合 EP+TP 场景 |

#### 8.9.2 vLLM 中的 AllGather 方案实现

vLLM 已经实现了 AllGather-based EP 方案，即 `allgather_reducescatter` 后端（`AgRsAll2AllManager`，参见 8.6.2 节）：

```python
# all2all.py — AgRsAll2AllManager
class AgRsAll2AllManager(All2AllManagerBase):
    def dispatch(self, hidden_states, topk_weights, topk_ids, ...):
        # AllGatherV: 变长 gather，收集各 rank 的 token
        gathered_tensors = dist_group.all_gatherv(
            [hidden_states, topk_weights, topk_ids],
            dim=0, sizes=sizes,
        )
        return gathered_tensors[0], gathered_tensors[1], gathered_tensors[2]

    def combine(self, hidden_states, ...):
        # ReduceScatterV: 变长 reduce-scatter，归还原始 rank
        hidden_states = dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
        return hidden_states
```

**工作流程**：

```
Rank 0 有 token [A, B]，Rank 1 有 token [C, D, E]

Dispatch (AllGatherV):
  Rank 0: [A, B] → AllGatherV → [A, B, C, D, E]  (收集所有 token)
  Rank 1: [C, D, E] → AllGatherV → [A, B, C, D, E]

Compute:
  每个 rank 对全量 token 执行本地专家计算
  但只有路由到本 rank 专家的 token 产生有效输出

Combine (ReduceScatterV):
  全量结果 → ReduceScatterV → 每个 rank 只保留自己原始 token 的结果
  Rank 0: 拿回 [A, B] 的最终结果
  Rank 1: 拿回 [C, D, E] 的最终结果
```

此外，`naive` 后端（`NaiveAll2AllManager`）使用 **AllReduce** 实现类似效果——广播全量数据后每个 rank 切片取出自己的部分。AllReduce 方案更简单但通信量更大，仅用于测试/调试。

#### 8.9.3 SGLang 中的 EP 实现对比

SGLang 的 EP 实现采用了与 vLLM 不同的架构设计，但后端种类高度重合。

**架构差异**：

| 维度 | vLLM | SGLang |
|------|------|--------|
| **抽象层次** | 两层分离：`All2AllManager`（通信层）+ `PrepareAndFinalize`（MoE 集成层） | 单层统一：`BaseDispatcher` 子类同时封装通信和 MoE 集成 |
| **配置参数** | `--all2all-backend` | `--moe-a2a-backend` |
| **异步接口** | 部分后端支持 `prepare_async` / `finalize_async` | 全部后端通过 `dispatch_a/b` + `combine_a/b` 两阶段异步 |
| **TBO 支持** | DBO（Dual-Batch Overlap） | TBO（Two-Batch Overlap）+ SBO（Single-Batch Overlap） |

**SGLang 的后端枚举**（`MoeA2ABackend` 枚举类，`sglang/srt/layers/moe/utils.py`）：

```python
class MoeA2ABackend(Enum):
    NONE = "none"            # AllReduce/AllGather — 默认后端
    DEEPEP = "deepep"        # DeepEP (normal + low_latency 模式)
    MOONCAKE = "mooncake"    # Mooncake EP (RDMA, 弹性推理)
    NIXL = "nixl"            # NIXL-EP (RDMA, 弹性 EP)
    MORI = "mori"            # MoRI (AMD ROCm)
    ASCEND_FUSEEP = "ascend_fuseep"  # 昇腾 NPU 融合 EP
    FLASHINFER = "flashinfer"        # FlashInfer A2A
    CUSTOMIZED = "customized"        # 自定义后端
```

**SGLang 的 Dispatcher 体系**（`sglang/srt/layers/moe/token_dispatcher/` 目录）：

| Dispatcher 类 | 对应后端 | vLLM 对应 |
|---------------|---------|----------|
| `StandardDispatcher` | `none`（默认） | `NaiveAll2AllManager` / `AgRsAll2AllManager` |
| `DeepEPDispatcher` | `deepep` | `DeepEPHTAll2AllManager` / `DeepEPLLAll2AllManager` |
| `MooncakeEPDispatcher` | `mooncake` | —（vLLM 无此后端） |
| `NixlEPDispatcher` | `nixl` | `NixlEPAll2AllManager` |
| `MoriEPDispatcher` | `mori` | `MoriAll2AllManager` |
| `FlashinferDispatcher` | `flashinfer` | `FlashInferNVLinkOneSidedManager` |
| `NpuFuseEPDispatcher` | `ascend_fuseep` | —（vLLM 无此后端） |

**SGLang 的 `none` 后端（StandardDispatcher）详解**：

当 `--moe-a2a-backend none`（默认）时，SGLang 使用 `StandardDispatcher`。这是一种 **不做显式 All2All 通信** 的 EP 方案——每个 rank 持有部分专家，直接用本地专家处理所有 token，然后通过 AllReduce 汇总结果：

```python
# SGLang — StandardDispatcher (standard.py)
class StandardDispatcher(BaseDispatcher):
    def dispatch(self, hidden_states, topk_output):
        if should_use_flashinfer_cutlass_moe_fp4_allgather():
            # FP4 AllGather 路径：先量化，AllGatherV 收集，再计算
            x, x_sf = fp4_quantize(hidden_states, global_scale)
            topk_weights, topk_ids, x, x_sf = get_tp_group().all_gatherv(
                [topk_weights, topk_ids, x, x_sf], sizes=get_dp_global_num_tokens()
            )
            return StandardDispatchOutput(hidden_states=x, ...)
        else:
            # 默认路径：直接传递，不做通信
            # topk_ids 通过 local_expert_mapping 映射到本地专家
            topk_ids = self.local_expert_mapping[topk_ids]
            return StandardDispatchOutput(hidden_states=hidden_states, ...)

    def combine(self, combine_input):
        if should_use_flashinfer_cutlass_moe_fp4_allgather():
            # ReduceScatterV 归还结果
            get_tp_group().reduce_scatterv(global_hidden_states, output=local, sizes=...)
        return hidden_states
```

对于 `none` 后端的非 FP4 路径，EP 的 AllReduce 通信发生在 MoE 层外部的模型代码中：

```python
# SGLang — deepseek_v2.py (Qwen3MoE, DeepSeek 等模型)
class Qwen3MoeSparseMoeBlock(nn.Module):
    def forward_normal(self, hidden_states, ...):
        final_hidden_states = self.experts(hidden_states, topk_output)
        if self.ep_size > 1:
            # EP AllReduce: 每个 rank 只计算了部分专家，需要 AllReduce 汇总
            final_hidden_states = moe_expert_parallel_all_reduce(final_hidden_states)
        return final_hidden_states
```

> **关键差异**：SGLang 的 `none` 后端是"**计算冗余 + AllReduce 聚合**"模式——每个 rank 对全量 token 运行本地专家（非目标专家的输出为 0），然后 AllReduce 求和。这与 vLLM 的 AllGather 方案不同：vLLM 的 `allgather_reducescatter` 后端先 AllGather 收集 token，再每个 rank 只计算本地专家，最后 ReduceScatter 归还结果。

**SGLang 独有的 Mooncake 后端**：

SGLang 还支持 `mooncake` 后端（`MooncakeEPDispatcher`），这是月之暗面开源的 Mooncake 项目，是 DeepEP 的扩展版本，专为弹性推理（Elastic Inference）设计，支持通过 RDMA 进行高性能数据传输。vLLM 目前不支持此后端。

**SGLang 独有的 Ascend FuseEP 后端**：

`ascend_fuseep` 后端（`NpuFuseEPDispatcher`）是华为昇腾 NPU 平台的融合 EP 实现，将 dispatch + expert compute + combine 融合为单个 NPU 算子调用。

#### 8.9.4 两阶段异步通信（TBO/DBO）

SGLang 和 vLLM 都支持将 EP 通信拆分为两个阶段以实现通信与计算的重叠，但命名和实现方式不同：

**vLLM — DBO（Dual-Batch Overlap）**：
```
dispatch_async → [通信进行中] → 本 batch 计算 → 等待 dispatch 完成
compute_experts
combine_async → [通信进行中] → 下一步计算 → 等待 combine 完成
```

**SGLang — TBO（Two-Batch Overlap）**：
```
dispatch_a (发起异步 dispatch) → hook/其他操作
dispatch_b (等待 dispatch 完成) → 获取 dispatch 结果
experts (本地专家计算)
combine_a (发起异步 combine) → hook/其他操作  
combine_b (等待 combine 完成) → 获取最终结果
```

SGLang 的 `dispatch_a/b` + `combine_a/b` 四步接口是所有 EP Dispatcher 的标准接口，使得 TBO 调度器可以交错两个 micro-batch 的通信和计算。

#### 8.9.5 EP 方案选择建议

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| EP_size 较小（≤8），节点内 NVLink | AllGather/AllReduce 或 FlashInfer NVLink | 通信量可控，实现简单 |
| EP_size 较大（>8），跨节点 | DeepEP / NIXL EP | All2All 通信量优势显著 |
| AMD ROCm 平台 | MoRI | AMD 原生优化 |
| 弹性 EP（动态扩缩容） | NIXL EP / Mooncake | 支持 Elastic EP |
| 混合 EP+TP（ep_size < tp_size） | AllGather/AllReduce（`none`/`allgather_reducescatter`） | 其他后端通常要求 ep_size = tp_size |
| 测试/调试 | `naive` | 最简实现，易于验证正确性 |

---

## 第9章 数据并行 (DP)

### 9.1 DP 的实现方式

vLLM 的 DP 实现与传统训练框架不同——不需要梯度同步，而是：

1. **每个 DP rank 独立加载完整模型**
2. **每个 DP rank 处理不同的 prompt 子集**
3. **KV Cache 等状态独立维护**

```python
# data_parallel.py 中的 prompt 分配
floor = len(prompts) // dp_size
remainder = len(prompts) % dp_size

def start(rank):
    return rank * floor + min(rank, remainder)

prompts = prompts[start(global_dp_rank) : start(global_dp_rank + 1)]
```

### 9.2 DP 环境变量协议

| 环境变量 | 含义 |
|----------|------|
| `VLLM_DP_RANK` | 全局 DP rank |
| `VLLM_DP_RANK_LOCAL` | 本地 DP rank |
| `VLLM_DP_SIZE` | DP 组大小 |
| `VLLM_DP_MASTER_IP` | DP 主节点 IP |
| `VLLM_DP_MASTER_PORT` | DP 主节点端口 |

### 9.3 DP 组的通信场景

DP 组内的通信主要发生在：
- **EP 路由**：当 EP 跨 DP 时，All2All 在整个 EP 组（含多个 DP rank）内进行
- **EPLB 负载统计**：DP 组内同步专家负载信息
- **元数据同步**：调度器决策的广播

### 9.4 DP + TP 的 GPU 绑定

```python
# Worker.init_device() 中的 local_rank 计算
# 对于 DP=2, TP=4:
# DP rank 0 → local_rank [0,1,2,3]
# DP rank 1 → local_rank [4,5,6,7]
```

### 9.5 多节点 DP

`data_parallel.py` 支持多节点 DP：

```bash
# Node 0
python data_parallel.py -dp=2 -tp=4 \
    --dp-num-nodes=2 --dp-node-rank=0 \
    --dp-master-addr=10.99.48.128 --dp-master-port=13345

# Node 1  
python data_parallel.py -dp=2 -tp=4 \
    --dp-num-nodes=2 --dp-node-rank=1 \
    --dp-master-addr=10.99.48.128 --dp-master-port=13345
```

### 9.6 DP Wave 调度机制（MoE 专用）

> 本节整合了原 9.6（机制设计）与 9.7（代码链路）两章，按"**为什么 → 架构 → 初始化 → 主循环 → 终止 → 时序 → 边界 → 索引**"单一叙事推进,所有结论均带 v0.18.0 源码行号。
>
> **入口脚本**：`examples/offline_inference/data_parallel.py`
> **示例模型**：`/data/models/deepseek/deepseek-moe-16b-base`（64 routed + 2 shared experts, top-k=6, 28 层）

---

#### 9.6.1 为什么需要 Wave：死锁问题的根源

Wave 机制不是可选的性能优化,而是 **MoE × DP 场景下 All2All 通信模式的必然产物**。下面从死锁场景出发推导其必要性。

**(1) MoE 层的集体通信约束**

DeepSeek-MoE-16B 的层 1–27 为 MoE。当 `enable_expert_parallel=True` 时,每层 forward 会触发一次 Expert Parallel 的 **All2All**（或同语义的 combine/dispatch）集体通信。All2All 是 **全员同步**原语 —— 所有 DP ranks 必须在同一"逻辑时刻"集合,才能完成数据交换。

**(2) 天然不对齐的请求时间线**

`data_parallel.py` 的入口下,每个 DP rank 是独立 `Process`（L287–310 `Process(target=main)`),各 rank 的请求队列独立:
- Rank 0: 当前 batch 有 20 个 decode 步
- Rank 1: 当前 batch 只有 3 个 decode 步

若 Rank 1 提前 return 退出 busy loop,Rank 0 再进入下一层 MoE 的 All2All —— **Rank 0 在等待已退出的 Rank 1**,NCCL 集合卡死,整个任务 hang。

**(3) Wave 的解决方案**

Wave 将"请求生命周期"抽象为一次**同步波次**：

```
所有 DP rank 共同进入 wave → 全部跑完自己的活 → 用 AllReduce 对齐状态
   ↓                                                   ↓
若某 rank 无请求 → 跑 dummy forward 占位                 ↓
                                                    全部完成才退出 wave
```

**两条核心保证**：
- 只要**任一** rank 还有未完成请求,**所有** rank 继续 forward（有请求跑真实 batch,无请求跑 dummy batch)
- 退出时机由**全局 AllReduce** 决定,而非本地状态 —— 确保所有 rank 同步退出

**(4) Wave 的死对头：非 MoE 模型**

非 MoE 模型（如 Qwen-7B dense）的 forward 不含 All2All,各 DP rank 完全独立,**不需要 Wave**。`core.py` L1062–1073 的分支逻辑：

```python
# vllm/v1/engine/core.py: run_engine_core() L1062-1073
if not vllm_config.model_config.is_moe:
    # 非 MoE → 强制 dp_size=1,走普通 EngineCoreProc,不启 wave
    vllm_config.parallel_config.data_parallel_size = 1
    ...
    engine_core = EngineCoreProc(...)
else:
    engine_core = DPEngineCoreProc(...)
```

`DPEngineCoreProc.__init__` 更进一步用硬断言把守：

```python
# vllm/v1/engine/core.py: L1584
assert vllm_config.model_config.is_moe, \
    "DPEngineCoreProc is only for MoE models"
```

**结论**：Wave = MoE 专用。非 MoE 走不同分支,不存在 wave 概念。

---

#### 9.6.2 Wave 架构总览

**(1) 两种执行模式**

| 模式 | 触发条件 | 入口 | Wave 终止判定 |
|------|---------|------|-------------|
| **离线 SPMD** | `data_parallel.py` 直接 `Process.spawn`,无外部协调器 | `DPEngineCoreProc` | 32 步一次 **AllReduce(MAX)** 对齐 |
| **在线 Coordinator** | API Server 下发,存在 `DPCoordinator` | `DPEngineCoreProc` + `Coordinator` | Coordinator 分发 `start_wave` 消息 |

本节主线以**离线 SPMD** 为主（对应 `data_parallel.py`),Coordinator 模式在 9.6.9 补充。

**(2) 三层状态机**

```
┌──────────────────────── 前端（LLM/LLMEngine）────────────────────────┐
│  SyncMPClient.engines_running        ← get_output() 收 wave_complete │
│  has_unfinished_requests()           ← _run_engine while 的退出条件   │
└───────────────────────────────↕ ZMQ output queue ─────────────────────┘
┌──────────────────────── 后端（DPEngineCoreProc）────────────────────────┐
│  engines_running: bool               ← 本 rank 是否在 wave 内         │
│  current_wave: int                   ← 当前波次编号（从 0 递增）        │
│  step_counter: int                   ← 32 步 AllReduce 计数器         │
└──────────────────────────────↕ dp_group AllReduce ─────────────────────┘
┌──────────────────────── 集体状态（stateless dp_group gloo）────────────┐
│  torch.distributed.all_reduce(has_unfinished, ReduceOp.MAX)            │
└───────────────────────────────────────────────────────────────────────┘
```

**(3) 四类关键状态变量**

| 变量 | 位置 | 含义 |
|------|------|------|
| `engines_running` | `EngineCoreProc.__init__` L802 / `SyncMPClient` | 当前是否处于 wave 中（True=不得退出） |
| `current_wave` | `DPEngineCoreProc.__init__` L1591 | 波次编号,随每次 wave 结束递增 |
| `step_counter` | 同上 L1590 | 调度步计数,每 32 步触发一次全局 AllReduce |
| `last_counts` | 同上 L1592 | 上一轮 `(num_waiting, num_running)` 用于变化检测 |

---

#### 9.6.3 初始化路径

`data_parallel.py` 启动后,每个 DP rank 在本进程内按以下顺序建立 wave 基础设施。

**(1) 进程启动与环境变量**

```python
# examples/offline_inference/data_parallel.py L160-164
os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
os.environ["VLLM_DP_SIZE"] = str(dp_size)
os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
```

`Process(target=main)` spawn 后,每个子进程各自 `LLM(...)`,触发 `DPEngineCoreProc` 创建。

**(2) DPEngineCoreProc.__init__（`core.py` L1571-1608）**

关键步骤：

```python
# 1. 断言必须是 MoE 模型
assert vllm_config.model_config.is_moe          # L1584

# 2. 初始化 wave 状态
self.step_counter = 0                            # L1590
self.current_wave = 0                            # L1591
self.last_counts = (0, 0)                        # L1592

# 3. 通过父类 EngineCoreProc 继承
self.engines_running = False                     # L802 (父类)
self.has_coordinator = ...                       # L813 (父类)
```

**(3) DP 组建立（`_init_data_parallel`,`core.py` L1610-1623)**

```python
self.dp_group, self.dp_rank = stateless_init_dp_group(return_store=True)
```

`stateless_init_dp_group` 位于 `vllm/config/parallel.py` L536-575,使用 **gloo 后端**(CPU) 建立 DP 专用通信组,和 NCCL 的 TP/EP 组完全解耦。这样做的原因：
- wave AllReduce 是**控制平面**信号(bool),数据量极小,不需要 NCCL
- 与数据平面的 TP/EP NCCL 通信不冲突,避免流/组互锁

---

#### 9.6.4 运行时核心循环

`run_busy_loop`（`core.py` L1682-1734）是 wave 机制的运转中心。精简后的骨架:

```python
def run_busy_loop(self):                                     # L1682
    while True:
        # 1. 正常执行调度 + forward
        executed = self._process_engine_step()

        # 2. 分支判断
        if not executed and self.engines_running:           # L1700-1708
            # 本 rank 无请求但仍在 wave 中 → 跑 dummy forward
            self.execute_dummy_batch()

        # 3. 每 32 步触发一次全局对齐
        if self._has_global_unfinished_reqs():              # → L1736
            continue

        # 4. AllReduce 显示全员完成 → 发 wave_complete
        if self.engines_running:                            # L1714-1734
            self._send_wave_complete()
            self.engines_running = False
            self.current_wave += 1
```

**(1) Dummy forward：不是空 batch,是最小真实 forward**

当本 rank 无请求但其他 rank 还在跑时,不能直接 idle —— 必须参与 MoE 层的 All2All,否则其他 rank 会卡死。

```python
# vllm/v1/worker/gpu_worker.py L908-909
def execute_dummy_batch(self) -> None:
    self.model_runner._dummy_run(1, uniform_decode=True)
```

`_dummy_run(1, uniform_decode=True)` 的语义：
- 构造 **1 token 的 uniform decode 输入**
- 执行**完整的 forward pass**（包括 attention、所有 MoE 层、每层的 All2All）
- 不产生输出 tokens,纯粹为了参与集体通信

**这是常见误解点**：dummy batch ≠ 空 batch ≠ no-op。它是"最小 forward"—— 数据量最小,但 kernel/通信路径完整。

**(2) 32 步 AllReduce 优化**

`_has_global_unfinished_reqs`（`core.py` L1736-1742）：

```python
def _has_global_unfinished_reqs(self) -> bool:        # L1736
    self.step_counter += 1
    if self.step_counter % 32 != 0:                   # L1738-1740
        return True                                    # 暂不判定,直接继续
    return has_unfinished_dp(self.dp_group, self.has_unfinished_local())
```

**优化动机**：每步都 AllReduce 成本过高（gloo CPU,不过仍有 RTT 延迟）。观察到 decode 阶段典型请求 >>32 步,因此每 32 步查一次全局状态,误差 ≤32 步 × decode 延迟,性价比极高。

`has_unfinished_dp`（`vllm/config/parallel.py` L619-628)：

```python
tensor = torch.tensor([has_unfinished], dtype=torch.int32, device="cpu")
torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
return bool(tensor.item())
```

`ReduceOp.MAX` 的语义：**任意 rank 有请求 → 全员继续**。这就是 Wave 的数学保证。

**(3) Wave 激活路径**

`SyncMPClient.add_request`（`core_client.py` L847-850）：

```python
def add_request(self, request):
    ...
    if self.is_dp:
        self.engines_running = True     # 前端置位,下一次 get_output 不会误判
```

后端侧 `engines_running` 的置位由 `DPEngineCoreProc` 在收到请求并开始 forward 时处理 —— 一旦 forward 被调度,本 rank 就进入 wave。

---

#### 9.6.5 Wave 终止与前端感知

**(1) 后端：发出 wave_complete**

当 `_has_global_unfinished_reqs` 返回 False（AllReduce 所有 rank 本地均无请求)：

```python
# vllm/v1/engine/core.py L1714-1734
self._send_wave_complete()              # 把 wave_complete=True 塞进 output_queue
self.engines_running = False             # 本 rank 退出 wave
self.current_wave += 1                   # 波次编号 +1
```

`output_queue` 经 ZMQ socket 推送到前端。

**(2) 前端：收到 wave_complete 并清状态**

```python
# vllm/v1/engine/core_client.py SyncMPClient.get_output() L810-820
outputs = self.output_queue.get(...)
if outputs.wave_complete:
    self.engines_running = False        # 前端状态翻转
    return outputs
```

**(3) LLM 层退出条件**

```python
# vllm/v1/engine/llm_engine.py has_unfinished_requests() L189-193
def has_unfinished_requests(self) -> bool:
    has_unfinished = self.scheduler.has_unfinished_seqs()
    if self.is_dp:
        return has_unfinished or self.engine_core.dp_engines_running()
    return has_unfinished
```

```python
# vllm/entrypoints/llm.py _run_engine() L1977-1979
while self.llm_engine.has_unfinished_requests():
    step_outputs = self.llm_engine.step()
    ...
```

**退出链**：
```
所有 rank 本地无请求
  → AllReduce(MAX)=0
  → 后端 send wave_complete
  → 前端 engines_running=False
  → LLM has_unfinished_requests()=False
  → while 退出 → generate 返回
```

---

#### 9.6.6 完整状态时序图

以 DP=2（Rank 0、Rank 1）为例,演示从添加请求到 wave 结束的完整流转：

```
Time  Rank 0 (busy loop)         Rank 1 (busy loop)         dp_group (gloo)
 ─┬── ──────────────────────     ──────────────────────     ──────────────
  │   idle (无请求)               idle (无请求)               -
  │
  ▼   add_request(A,B,C,D)
t1    engines_running=True       engines_running=True
       (前端 set + 本地 forward)    (收到广播/同 batch 分发)
  │
  ▼   decode step 1..31          decode step 1..31           -
       step_counter=1..31         step_counter=1..31        (无 AllReduce)
  │
  ▼   decode step 32             decode step 32
t2     step_counter=32%32==0     step_counter=32%32==0
       AllReduce(local=1) ────────────────────────────────► MAX → 1
       → continue                 → continue
  │
  ▼   Rank 0 剩 A,B,C,D 继续      Rank 1 的 A,B,C,D 在不同时
                                  刻先完成（比如 step 50 后清空）
  │
  ▼   decode step 64             local=0 → execute_dummy_batch()
t3     AllReduce(local=1) ────────────────────────────────► MAX → 1
                                 dummy forward 参与 All2All
  │
  ▼   ...继续...                  ...dummy 参与...            -
  │
  ▼   A,B,C,D 也完成,local=0
t4     AllReduce(local=0) ────────────────────────────────► MAX → 0
       send wave_complete         send wave_complete
       engines_running=False      engines_running=False
       current_wave++             current_wave++
  │
  ▼   idle                        idle
```

**关键观察点**：
- `t2→t3`：Rank 1 先清空本地队列,但因 AllReduce=1 仍需参与 wave,跑 dummy forward 保证 All2All 能完成
- `t4`：只有**两侧本地都清空**才能让 AllReduce=0,触发 wave_complete
- 32 步是 AllReduce 粒度,`t2→t4` 之间最多相差 31 步（弱同步,可接受）

---

#### 9.6.7 边界条件

**(1) 请求高频插入：next-wave 复用**

前端在上一波未完成时新 `add_request`：`engines_running=True` 保持,新请求直接被编入本轮 wave 的 scheduler。不等 wave_complete,提升吞吐。

**(2) 请求 wave 之间的空窗**

若前端 Pythonic 代码在两次 `generate` 之间有延迟,后端 `engines_running=False` 进入 idle:

```python
# 后端 run_busy_loop 中无请求且非 wave 状态:
if not self.engines_running:
    time.sleep(small_interval)       # 轻睡眠,等 ZMQ 消息
```

新请求到来 → ZMQ 唤醒 → 进入新 wave。

**(3) rank 数量失衡（例如 DP=4,只有 1 个有请求）**

其他 3 个 rank 全走 dummy forward 路径。性能代价：3× dummy 的 forward 通信带宽被占用。优化策略：
- 前端负载均衡 (round-robin/least-load dispatcher)
- 粒度：`ReduceOp.MAX` 使得"1 忙 3 闲"与"4 均忙"走同一逻辑,不存在功能缺陷

**(4) 异常中止**

- rank 进程 crash：dp_group gloo AllReduce 抛异常,其他 rank 捕获后统一 abort
- KeyboardInterrupt：`run_busy_loop` 外层 `try/except` 捕获,所有 rank 走清理流程

---

#### 9.6.8 Wave 使用前提与例外路径

**(1) Wave 使用前提矩阵**

| 前提 | 要求 | 校验位置 |
|------|------|---------|
| 模型是 MoE | `model_config.is_moe == True` | `core.py` L1584 `assert` |
| `dp_size > 1` | `parallel_config.data_parallel_size > 1` | `core.py` L1062-1073 |
| 通过 `DPEngineCoreProc` 路径启动 | 非 `EngineCoreProc` | `core.py` L1062-1073 |
| 未使用 external launcher DP | `parallel_config.distributed_executor_backend != "external_launcher"` | `llm_engine.py` L82-89 |

任何一项不满足,都走非 wave 路径。

**(2) 例外：External Launcher DP**

verl、TRL 等 RLHF 框架使用 external launcher DP（跳过 vllm 的 Process spawn,由外部 torchrun 拉起）。此时走 `LLMEngine` 直接路径,**不走 `DPEngineCoreProc`**：

```python
# vllm/v1/engine/llm_engine.py L82-89
if parallel_config.distributed_executor_backend == "external_launcher":
    if parallel_config.data_parallel_size > 1:
        self.dp_group = init_external_dp_group(...)   # 自建 dp_group
```

```python
# vllm/v1/engine/llm_engine.py L189-193
def has_unfinished_requests(self) -> bool:
    has_unfinished = self.scheduler.has_unfinished_seqs()
    if self.dp_group is not None:       # external launcher 路径
        # 每步都 AllReduce,不做 32 步聚合
        tensor = torch.tensor([has_unfinished], ...)
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=self.dp_group)
        return bool(tensor.item())
    return has_unfinished
```

**与 Wave 路径的差异**：
- 无 `engines_running` / `current_wave` 状态
- 无 32 步聚合,每步 AllReduce（成本更高,但语义更简单）
- Dummy forward 由 `gpu_model_runner.py` L3601-3612 的 `num_scheduled_tokens == 0` 分支直接触发

External launcher 的核心诉求是 RLHF 训推一体,控制粒度优先于吞吐,因此接受每步 AllReduce 的代价。

---

#### 9.6.9 在线 Coordinator 模式补充

API Server 场景下,前端是 `AsyncMPClient`,由 `DPCoordinator` 统一调度 wave 起止：

| 环节 | 离线 SPMD | 在线 Coordinator |
|------|----------|-----------------|
| wave 激活 | 前端 `add_request` 置 `engines_running=True` | Coordinator 下发 `start_wave` 消息 |
| wave 终止 | 后端 AllReduce=0 → `wave_complete` | 后端上报 `stats`,Coordinator 聚合判断 |
| 跨副本同步 | dp_group AllReduce(MAX) | Coordinator ZMQ pub/sub |
| 主要代码 | `DPEngineCoreProc` | `DPCoordinator` + `AsyncMPClient` |

Coordinator 模式的优势：
- 多 API Server 副本间 wave 可对齐
- 可观测性更好（Coordinator 聚合所有 stats）
- 支持动态伸缩（新 rank 加入 wave）

核心循环机制仍是"有人干活就全员保持 forward,全员无事才结束 wave",只是信号通道从 dp_group 换成了 Coordinator。

---

#### 9.6.10 代码锚点总索引

**运行时核心（v0.18.0)**

| 功能 | 文件 | 符号 / 行号 | 说明 |
|------|------|------------|------|
| 非 MoE 降级分支 | `vllm/v1/engine/core.py` | `run_engine_core()` L1062-1073 | dp_size→1,不走 DPEngineCoreProc |
| MoE 专用进程类 | `vllm/v1/engine/core.py` | `class DPEngineCoreProc` L1571-1608 | wave 核心容器 |
| **MoE 断言** | `vllm/v1/engine/core.py` | `__init__` L1584 | `assert is_moe` |
| wave 状态初始化 | `vllm/v1/engine/core.py` | `__init__` L1590-1592 | step_counter/current_wave/last_counts |
| DP 组建立 | `vllm/v1/engine/core.py` | `_init_data_parallel` L1610-1623 | gloo 后端 |
| **主循环** | `vllm/v1/engine/core.py` | `run_busy_loop` L1682-1734 | wave 机制主体 |
| dummy batch 触发 | `vllm/v1/engine/core.py` | L1700-1708 | `not executed and engines_running` |
| wave 完成通知 | `vllm/v1/engine/core.py` | L1714-1734 | `wave_complete` → `output_queue` |
| **32 步 AllReduce** | `vllm/v1/engine/core.py` | `_has_global_unfinished_reqs` L1736-1742 | 控制平面对齐 |

**通信与同步**

| 功能 | 文件 | 符号 / 行号 | 说明 |
|------|------|------------|------|
| dp_group AllReduce | `vllm/config/parallel.py` | `has_unfinished_dp` L619-628 | `ReduceOp.MAX` over gloo CPU |
| stateless dp_group | `vllm/config/parallel.py` | `stateless_init_dp_group` L536-575 | 不占用全局通信组 |

**Worker 侧**

| 功能 | 文件 | 符号 / 行号 | 说明 |
|------|------|------------|------|
| **Dummy forward（非空!）** | `vllm/v1/worker/gpu_worker.py` | `execute_dummy_batch` L908-909 | `_dummy_run(1, uniform_decode=True)` |
| External launcher dummy | `vllm/v1/worker/gpu_model_runner.py` | L3601-3612 | `num_scheduled_tokens==0` 时降级 |

**前端 / 控制链**

| 功能 | 文件 | 符号 / 行号 | 说明 |
|------|------|------------|------|
| engines_running 初值 | `vllm/v1/engine/core.py` | `EngineCoreProc.__init__` L802 | 父类基础状态 |
| 前端置位 | `vllm/v1/engine/core_client.py` | `SyncMPClient.add_request` L847-850 | wave 激活 |
| 前端接收 wave_complete | `vllm/v1/engine/core_client.py` | `SyncMPClient.get_output` L810-820 | `engines_running=False` |
| 退出条件 | `vllm/v1/engine/llm_engine.py` | `has_unfinished_requests` L189-193 | wave/external launcher 两路 |
| while 退出 | `vllm/entrypoints/llm.py` | `_run_engine` L1977-1979 | 最外层循环 |
| **External launcher DP** | `vllm/v1/engine/llm_engine.py` | L82-89, L189-198 | 自建 dp_group,非 wave |

**入口脚本**

| 功能 | 文件 | 行号 | 说明 |
|------|------|-----|------|
| DP 环境变量 | `examples/offline_inference/data_parallel.py` | L160-164 | VLLM_DP_* |
| 进程 spawn | `examples/offline_inference/data_parallel.py` | L287-310 | `Process(target=main)` |

---

> **一句话总结**：Wave = **MoE 专用的全员同步调度原语**,用 32 步粒度的 AllReduce(MAX) 协调"有事就全员跑（真实 batch 或 dummy forward）、全员没事才退出"的生命周期,避免 MoE All2All 在 DP 下因请求时间线不对齐而死锁。非 MoE、external launcher DP、dp_size=1 这三类场景都不走 wave 路径。

---

### 9.8 组合场景实战分析 —— 以 DeepSeek-MoE-16B 为例

> **入口脚本**: `examples/offline_inference/data_parallel.py`
> **模型路径**: `/data/models/deepseek/deepseek-moe-16b-base`
> **关键架构参数**（来自 `config.json`）:
> - `hidden_size = 2048`, `moe_intermediate_size = 1408`, `num_attention_heads = 16`
> - `n_routed_experts = 64`, `n_shared_experts = 2`, `num_experts_per_tok = 6`
> - `first_k_dense_replace = 1` → 层 0 为 dense MLP，层 1–27 为 MoE
> - `num_hidden_layers = 28`, `vocab_size = 102400`

本节系统比较 `(TP, DP) × (EP on/off)` 六种组合下的 **专家部署、权重分布、通信模式**，并指出 vLLM 如何选择 All2All 后端。硬件设定：8× RTX 4090（单节点，NVLink 不全互联）。

#### 9.8.1 关键决策点：`use_ep` 如何被推导

`FusedMoE` 构造时根据 `ParallelConfig.enable_expert_parallel` 决定 MoE 内部并行模式（源码 `vllm/model_executor/layers/fused_moe/config.py::FusedMoEParallelConfig.make` L1083-1130）：

```python
use_ep = (dp_size * pcp_size * tp_size > 1) and enable_expert_parallel

# flatten：MoE 层面把 DP×PCP×TP 视为一个"大 TP"
flatten_tp_size = dp_size * pcp_size * tp_size
flatten_tp_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank

if not use_ep:
    # TP/DP 保留原样；ep_size=1，专家权重沿 flatten_tp 切分
    return FusedMoEParallelConfig(tp_size=flatten_tp_size, ep_size=1, use_ep=False, ...)
else:
    # MoE 内部 tp_size 置 1，ep_size=flatten_tp_size，专家权重整块放置
    return FusedMoEParallelConfig(tp_size=1, ep_size=flatten_tp_size, use_ep=True, ...)
```

**核心洞察**：
- `enable_expert_parallel=False`：MoE 层把 `DP×PCP×TP` 所有 rank 拉平当成一个大 TP 组，每个 expert 权重被切成 `flatten_tp_size` 份；每卡持有全部 64 个 expert 的 1/N 片段。
- `enable_expert_parallel=True`：MoE 层 `tp_size=1`，`ep_size=flatten_tp_size`；每个 expert 权重整块驻留在某一张卡上，`local_num_experts = 64 / ep_size`（不均时多出的 remainder 散到前几个 rank，见 `layer.py::determine_expert_map` L108-122）。

> **注意**：非 MoE 层（Attention、Dense MLP、Embedding、LM Head）**始终按 TP 切分**，与 `enable_expert_parallel` 无关；这意味着 TP=1 时 Attention 也不切分。

#### 9.8.2 场景矩阵总览（DeepSeek-MoE-16B × 8×4090）

| 场景 | 启动参数 | 引擎数 | TP | DP | EP | flatten MoE tp/ep | 每卡 expert 数 | 每专家权重形态 |
|-----|---------|-------|----|----|----|-----------------|--------------|------------|
| **A** | `-tp 8` | 1 | 8 | 1 | OFF | tp=8, ep=1 | 64（全部） | `1408/8 = 176` 切片 |
| **B** | `-tp 8 --enable-expert-parallel` | 1 | 8 | 1 | ON | tp=1, ep=8 | 8（整块） | 完整 `2048×1408` |
| **C** | `--dp-size 8 -tp 1` | 8 | 1 | 8 | OFF | tp=1, ep=1 | 64（全部） | 完整 `2048×1408` |
| **D** | `--dp-size 8 -tp 1 --enable-expert-parallel` | 8 | 1 | 8 | ON | tp=1, ep=8 | 8（整块） | 完整 `2048×1408` |
| **E** | `--dp-size 2 -tp 4` | 2 | 4 | 2 | OFF | tp=4, ep=1 | 64（全部） | `1408/4 = 352` 切片 |
| **F** | `--dp-size 2 -tp 4 --enable-expert-parallel` | 2 | 4 | 2 | ON | tp=1, ep=8 | 8（整块） | 完整 `2048×1408` |

> **引擎数**对应 `data_parallel.py` 中 `dp_per_node` 个独立子进程，每个子进程内部自己建 TP 组；DP 组由 `VLLM_DP_*` 环境变量跨进程握手后补建。

#### 9.8.3 MoE 层权重存储计算（每层，64 routed + 2 shared）

每个 routed expert 持有 `gate_proj + up_proj + down_proj` = `3 × 2048 × 1408 × 2 bytes ≈ 16.5 MB`（bf16）。64 experts 每层 `≈ 1056 MB`。shared_experts 另计 2 个。以下给出每层 routed 部分每卡显存：

| 场景 | 每卡 routed 专家权重 | 说明 |
|-----|-------------------|------|
| A (TP=8, EP=OFF) | `1056 / 8 = 132 MB` | 64 experts 各切 1/8（intermediate 维度） |
| B (TP=8, EP=ON)  | `1056 / 8 = 132 MB` | 每卡 8 个整块 experts |
| C (DP=8, EP=OFF) | `1056 MB` | 每引擎独立保存全部 64 experts ⚠️ 最差 |
| D (DP=8, EP=ON)  | `1056 / 8 = 132 MB` | 每引擎 8 个整块 experts |
| E (TP=4, DP=2, EP=OFF) | `1056 / 4 = 264 MB` | 每引擎内 TP=4 切，引擎间冗余 |
| F (TP=4, DP=2, EP=ON)  | `1056 / 8 = 132 MB` | flatten ep=8，跨引擎统一切 |

**结论**：只要 **EP=ON**，无论走 TP 还是 DP，每层显存开销相同（`sum(weight) / 8`）。场景 C（纯 DP 无 EP）是**显存最差的部署**，每引擎完整保存全部专家权重。

#### 9.8.4 Router / Dispatch / Combine 三阶段的通信对照

DeepSeek-MoE-16B 每 token 选 6 个专家（top-k=6），伪代码流程固定为：
```
gate(x)  →  top-k 路由  →  dispatch 到 expert-owner rank  →  expert compute  →  combine 回原 token rank
```

| 场景 | gate 计算 | dispatch 通信 | expert compute | combine 通信 | 层末 AllReduce |
|-----|---------|--------------|---------------|-------------|---------------|
| **A** | 每卡本地 | **无**（全部 expert 在本卡） | 每卡跑全部 top-k expert 的 1/8 切片 | **无** | ✅ `tensor_model_parallel_all_reduce`（TP=8） |
| **B** | 每卡本地 | **All2All dispatch** | 只跑本地 8 个 expert 对应的 token | **All2All combine** | ❌（ep_size=tp_size，已在 A2A 内部完成聚合） |
| **C** | 每卡本地 | **无**（全部 expert 在本卡，且 batch 不跨引擎） | 每引擎跑所有 64 expert | **无** | ❌（TP=1） |
| **D** | 每卡本地 | **All2All dispatch（跨引擎）** | 每卡跑 8 个 expert | **All2All combine** | ❌ |
| **E** | 每卡本地 | **无** | 每卡跑全部 top-k expert 的 1/4 切片 | **无** | ✅ AllReduce（TP=4） |
| **F** | 每卡本地 | **All2All dispatch（跨引擎 EP=8）** | 每卡跑 8 个 expert | **All2All combine** | ❌ |

**关键理解**：
- **EP=OFF 且 TP>1（A/E）**：每卡保存"每个专家的权重切片"，所以不需要 token 在 rank 间移动，只需在 expert 输出处做一次 `AllReduce` 聚合 hidden 维度。通信量 = `tokens × hidden × sizeof(bf16)`。
- **EP=ON（B/D/F）**：每卡只有部分 expert 的完整权重，必须把 token dispatch 到对应 rank 并 combine 回来。通信量 = `2 × tokens × topk × hidden`（两次 A2A）。
- **场景 A vs B** 通信量对比（`tokens × hidden = T`）：A = `1 × T`（一次 AR），B = `2 × 6 × T = 12T`（两次 A2A，topk=6）。**A 的 MoE 层通信量显著更小**。但 B 没有专家权重切片带来的 GEMM 粒度损失，kernel 效率更高，且支持更大 batch。

#### 9.8.5 All2All 后端选择（`use_all2all_kernels`）

触发条件（`config.py::FusedMoEParallelConfig.use_all2all_kernels` L944-946）：
```python
use_all2all_kernels = (dp_size > 1) and use_ep
```

**注意**：**仅当 `DP>1` 且 `EP=ON` 才走 A2A 内核路径**。也就是说：
- 场景 B（TP=8, EP=ON, DP=1）：`use_all2all_kernels=False`，`tp_size` 被置 1，`ep_size=8`，但**不走外部 A2A 后端**，而是复用 TP 通信组内的 naive `all_to_all_single`（通过 `GroupCoordinator`）。
- 场景 D/F（DP>1, EP=ON）：走完整的 A2A 后端选择逻辑，可用 `naive`/`allgather_reducescatter`/`deepep_high_throughput`/`deepep_low_latency`/`flashinfer_nvlink_*`/`mori`/`nixl_ep`。

对 4090（无 NVLink 全互联）的实用建议：
- **默认 `naive`**：基于 NCCL P2P send/recv，prefill/decode 通吃。
- **`allgather_reducescatter`**：用集合通信替代 P2P，小 batch 下延迟更稳定。
- **`deepep_low_latency`**：decode 阶段专用，要求 NVLink 或 IB RDMA，4090 收益有限。

#### 9.8.6 DP Wave 同步与 dummy batch 的必要性

**EP=OFF（场景 C/E）**：每引擎独立运行，`has_unfinished_dp` 依然会每 32 步做一次 `ReduceOp.MAX` AllReduce 以协调整体退出，但**没有 dummy batch 需求**——即便某些引擎空转，也不会阻塞其他引擎。

**EP=ON（场景 D/F）**：MoE 层的 A2A 集合通信要求**所有 EP rank 同时到达**。若某引擎请求已处理完，它必须用 `execute_dummy_batch` 参与 A2A（`vllm/v1/engine/core.py::run_busy_loop` L1700-1708），否则其他引擎会在 A2A 处死锁。Wave 机制 + dummy batch **只有在 EP=ON 时才真正关键**。

> **验证路径**：用 `-tp 8 --enable-expert-parallel` 启动时观察 `grep -n "execute_dummy_batch\|START_DP_WAVE" vllm/v1/engine/core.py`，但因为单引擎（DP=1）不会触发 wave；真正能触发是 `--dp-size 8 -tp 1 --enable-expert-parallel`。

#### 9.8.7 Shared Experts（2 个共享专家）的部署

DeepSeek-MoE-16B 的 2 个 shared expert 走**独立路径**，不参与路由，每个 token 都经过它们。vLLM 中通过 `SharedFusedMoE` 融合调用（`deepseek_v2.py::DeepseekV2MoE.forward` L348-398）：

- **EP=OFF**：shared expert 权重按 TP 切分（与 dense MLP 相同逻辑），forward 后参与 TP AllReduce。
- **EP=ON**：`SharedFusedMoE` 把 shared + routed 融合到同一 A2A 调用中；shared 输出与 routed 输出相加后一起 combine。
  - 但 shared expert 因为"每个 token 都需要"，在 A2A 语义下等价于**每个 rank 都持有完整 shared 权重副本**（不切），以避免引入额外的 AllGather。

#### 9.8.8 选型建议矩阵

| 目标 | 推荐组合 | 理由 |
|------|---------|------|
| **单请求最低延迟** | A（TP=8, EP=OFF） | MoE 层通信 = 1×AR，显存均衡，无 A2A 抖动 |
| **小 batch 高吞吐** | B（TP=8, EP=ON） | 专家权重整块，kernel 效率高；单引擎无 DP 开销 |
| **大 batch 高吞吐** | F（TP=4, DP=2, EP=ON） | 引擎级并行提供请求独立性，EP 均摊显存 |
| **极致吞吐（多副本）** | D（DP=8, EP=ON） | 8 个独立引擎，MoE 显存靠 EP 解决 |
| **⚠️ 不推荐** | C（DP=8, EP=OFF） | 每引擎冗余保存 64 experts，显存 8× 浪费 |

#### 9.8.9 `data_parallel.py` 对应启动命令

以上场景均可通过修改 `examples/offline_inference/data_parallel.py` 对应参数启动（节选关键 CLI 参数）：

```bash
# 场景 A：纯 TP
python -c "from vllm import LLM; LLM(model='/data/models/deepseek/deepseek-moe-16b-base', tensor_parallel_size=8)"

# 场景 B：TP + EP
python -c "from vllm import LLM; LLM(model='...', tensor_parallel_size=8, enable_expert_parallel=True)"

# 场景 D：DP + EP（走 data_parallel.py）
python examples/offline_inference/data_parallel.py \
    --model /data/models/deepseek/deepseek-moe-16b-base \
    --dp-size 8 --tp-size 1 --enable-expert-parallel

# 场景 F：DP + TP + EP
python examples/offline_inference/data_parallel.py \
    --model /data/models/deepseek/deepseek-moe-16b-base \
    --dp-size 2 --tp-size 4 --enable-expert-parallel
```

> `data_parallel.py` 内部会根据 `dp_size` 和 `dp_per_node` 启动多个子进程，每个子进程通过 `VLLM_DP_*` 环境变量加入同一个 DP 组；TP 组由子进程内部 `LLM()` 构造时自建。

---

# 第六部分：总结与实践

---

## 第10章 设计总结与最佳实践

### 10.1 vLLM 分布式设计的五大原则

#### 原则一：五维张量统一建模

所有通信组从一个 `all_ranks` 五维张量派生。通过 `transpose → reshape → unbind` 的统一操作，任何并行维度的组创建都是同一模式的不同实例。

**好处**：新增并行维度只需新增一个 transpose 方向，代码改动极小。

#### 原则二：优先级瀑布通信

AllReduce 不绑定单一后端，而是按优先级逐一尝试：SymmMem → FlashInfer → CustomAllreduce → PyNCCL → torch.distributed。

**好处**：自动利用最优硬件特性，同时保证在任何环境下都能正确运行。

#### 原则三：Column-Row 配对最小化通信

严格遵循 Megatron-LM 的 Column → Row 配对模式：Column 层不通信，Row 层做 AllReduce。一对 Column+Row 只需一次 AllReduce。

**好处**：对于标准 Transformer，每层只需 2 次 AllReduce（Attention + MLP 各一次）。

#### 原则四：权重加载时切片

TP 切分在**权重加载时**完成（`weight_loader`），而非在 forward 时动态切。每个 rank 只持有自己那份权重。

**好处**：推理时零额外开销，内存占用最小化。

#### 原则五：双组隔离（Device + CPU）

每个 `GroupCoordinator` 同时维护 NCCL device group 和 Gloo CPU group：
- NCCL：高性能 GPU 张量通信
- Gloo：CPU 端协调和元数据传输

**好处**：两类通信不互相阻塞，控制面与数据面解耦。

### 10.2 通信开销分析

**以 DeepSeek-MoE-16B, TP=4 为例**：

| 层类型 | 每层通信次数 | 通信类型 | 数据量 |
|--------|-------------|---------|--------|
| Attention (全部 28 层) | 1 | AllReduce (o_proj) | B × 2048 × sizeof(dtype) |
| Dense MLP (第 0 层) | 1 | AllReduce (down_proj) | B × 2048 × sizeof(dtype) |
| MoE (第 1-27 层) | 1 + 2 | AllReduce + All2All | AllReduce: B × 2048; All2All: 取决于路由 |

**整个模型的 AllReduce 次数**：
- 28 层 × 1 (Attention) + 1 (第0层 MLP) + 27 (MoE) = 28 + 1 + 27 = **56 次 AllReduce**
- 27 层 × 2 = **54 次 All2All**（仅当 EP > 1 时）

### 10.3 配置建议

| 场景 | 推荐配置 | 原因 |
|------|---------|------|
| 单机 8×4090, 16B MoE | TP=4, EP=4 | 充分利用卡间互联 |
| 单机 8×4090, 7B Dense | TP=2 或 TP=4 | Dense 模型不需要 EP |
| 2 机 16 卡, 70B Dense | TP=8, PP=2 | PP 跨机，TP 机内 |
| 2 机 16 卡, 大 MoE | TP=4, DP=2, EP=8 | EP 跨机，最大化专家分散 |

### 10.4 常见问题与排查

| 问题 | 可能原因 | 排查方法 |
|------|---------|---------|
| 启动挂死 | 通信组创建死锁 | 检查所有 rank 是否同步调用 `init_process_group` |
| AllReduce 结果错误 | TP 组内 rank 不一致 | 打印 `get_tp_group().ranks` 验证 |
| MoE 输出全零 | EP 路由错误 | 检查 `get_ep_group().rank_in_group` 和专家分配 |
| OOM | TP 不够大 | 增大 TP 或开启 EP |
| FP4 量化失败 | SM 版本不支持 | SM < 9.0 时需要 try/except guard |

---

# 第七部分：DCP / PCP 上下文并行深度分析

> 本部分详细剖析 vLLM 的 **Decode Context Parallel (DCP)** 与 **Prefill Context Parallel (PCP)** 机制，涵盖设计动机、配置参数、通信组创建、KV Cache 切分策略、Forward 计算流程、通信后端对比、LSE 加权合并的数学原理，以及 DCP+PCP 联合使用的交互关系。

---

## 第11章 DCP / PCP 概述与设计动机

### 11.1 问题背景

长上下文推理（如 128K+ token 序列）面临的核心瓶颈：

| 阶段 | 瓶颈 | 原因 |
|------|------|------|
| Prefill | 计算量 | $O(n^2 \cdot d)$ 的注意力计算随序列长度二次增长 |
| Decode | 显存 | 单个序列的 KV Cache 可达数十 GB，超出单卡容量 |
| Decode | 带宽 | 每步需访问完整 KV Cache，显存带宽成为瓶颈 |

传统 TP（张量并行）按注意力头切分权重，但 **不切分 KV Cache**——每张卡仍持有完整的 KV Cache 副本。当序列极长时，单卡显存无法容纳。

### 11.2 DCP 设计思路

**DCP（Decode Context Parallel）** 将 KV Cache **在 TP 组内部**进一步按 token 维度切分到多张 GPU 上：

```
传统 TP (tp=4):
  GPU0: head[0]  + 完整 KV Cache
  GPU1: head[1]  + 完整 KV Cache  ← 显存瓶颈
  GPU2: head[2]  + 完整 KV Cache
  GPU3: head[3]  + 完整 KV Cache

DCP (tp=4, dcp=4):
  GPU0: all_heads + KV Cache[tokens 0,4,8,...]   ← 显存降为 1/4
  GPU1: all_heads + KV Cache[tokens 1,5,9,...]
  GPU2: all_heads + KV Cache[tokens 2,6,10,...]
  GPU3: all_heads + KV Cache[tokens 3,7,11,...]
```

**关键约束**: `tp_size % dcp_size == 0`，DCP 复用 TP 组的 GPU，不增加总 GPU 数。

### 11.3 PCP 设计思路

**PCP（Prefill Context Parallel）** 将长序列 Prefill 的计算按序列维度切分到多张 GPU：

```
PCP (pcp=2, tp=4):
  PCP_group_0 (4 GPUs): 处理序列前半部分
  PCP_group_1 (4 GPUs): 处理序列后半部分
```

PCP **增加了总 GPU 数**: `world_size = TP × PP × PCP`。

### 11.4 DCP vs PCP 对比

| 维度 | DCP | PCP |
|------|-----|-----|
| 阶段 | Decode | Prefill |
| 切分对象 | KV Cache (token 维度) | 序列计算 (sequence 维度) |
| GPU 开销 | 不增加 GPU（复用 TP 组） | 增加 GPU（PCP 是独立维度） |
| world_size 影响 | 不影响 | world_size *= pcp_size |
| TP 约束 | dcp_size ≤ tp_size 且整除 | 无约束，独立维度 |
| 通信组 | TP 组内切分 | 与 TP 正交的独立组 |

---

## 第12章 配置参数详解

> 源文件: `vllm/config/parallel.py`

### 12.1 核心参数

```python
@dataclass
class ParallelConfig:
    # Prefill Context Parallel: 增加 GPU
    prefill_context_parallel_size: int = 1          # line ~105

    # Decode Context Parallel: 复用 TP 组 GPU
    decode_context_parallel_size: int = 1           # line ~297

    # DCP 通信后端: "ag_rs" (AllGather+ReduceScatter) 或 "a2a" (All-to-All)
    dcp_comm_backend: DCPCommBackend = "ag_rs"      # line ~309

    # KV Cache 交织粒度 (round-robin 分配的块大小)
    cp_kv_cache_interleave_size: int = 1            # line ~317
```

### 12.2 参数验证逻辑

```python
# ParallelConfig.__post_init__() 中的验证
assert self.tensor_parallel_size % self.decode_context_parallel_size == 0, (
    "Tensor parallel size must be divisible by decode context parallel size"
)

# world_size 计算：PCP 影响 world_size，DCP 不影响
self.world_size = (
    self.pipeline_parallel_size
    * self.tensor_parallel_size
    * self.prefill_context_parallel_size  # PCP 乘入 world_size
)
# 注意: decode_context_parallel_size 不参与 world_size 计算
```

### 12.3 GQA/MQA 模型的 DCP 约束

> 源文件: `vllm/config/model.py`, lines ~1090-1114

对于非 MLA 的 GQA/MQA 模型，DCP 有更严格的约束：

```python
if decode_context_parallel_size > 1 and not self.use_mla:
    total_num_kv_heads = self.get_total_num_kv_heads()
    # TP 必须大于 KV head 数（确保每个 GPU 持有所有 KV head 的子集）
    assert tensor_parallel_size > total_num_kv_heads

    # DCP 不能超过 TP / KV_head 的比值
    max_dcp_size = tensor_parallel_size // total_num_kv_heads
    assert decode_context_parallel_size <= max_dcp_size

    # Q per KV 必须被 DCP 整除
    num_q_per_kv = total_num_attention_heads // total_num_kv_heads
    assert num_q_per_kv % decode_context_parallel_size == 0
```

### 12.4 CLI 使用示例

```bash
# DCP 使用 AG+RS 后端 (默认)
vllm serve model --tp 8 --dcp 8

# DCP 使用 A2A 后端 (更低通信延迟)
vllm serve model --tp 16 --dcp 16 --dcp-comm-backend a2a

# DCP + KV Cache 交织
vllm serve model --tp 8 --dcp 4 --cp-kv-cache-interleave-size 256

# PCP + TP (增加 GPU)
vllm serve model --tp 4 --pcp 2  # 总共 8 GPU

# DCP + PCP 联合使用
vllm serve model --tp 8 --dcp 4 --pcp 2  # 总共 16 GPU
```

---

## 第13章 通信组创建（DCP/PCP）

> 源文件: `vllm/distributed/parallel_state.py`, `initialize_model_parallel()` 函数

### 13.1 五维 all_ranks 张量回顾

```python
all_ranks = torch.arange(world_size).reshape(
    -1,                                    # ExternalDP
    data_parallel_size,                    # DP
    pipeline_model_parallel_size,          # PP
    prefill_context_model_parallel_size,   # PCP
    tensor_model_parallel_size,            # TP
)
# 布局顺序: ExternalDP × DP × PP × PCP × TP
```

### 13.2 DCP 组创建 —— TP 内部切分

```python
# DCP 组直接从 all_ranks 最内层 reshape 切分
# 因为 DCP 复用 TP 组的 GPU, dcp_size 必须整除 tp_size
group_ranks = all_ranks.reshape(
    -1, decode_context_model_parallel_size
).unbind(0)
```

**示例**: `tp=8, dcp=4`

```
all_ranks (TP维): [0, 1, 2, 3, 4, 5, 6, 7]

reshape(-1, 4):
  [[0, 1, 2, 3],    ← DCP 组 0
   [4, 5, 6, 7]]    ← DCP 组 1

GPU 0-3 构成 DCP 组 0: 共享同一组序列的 KV Cache
GPU 4-7 构成 DCP 组 1: 共享另一组序列的 KV Cache
```

**注**: 当 `dcp_size == tp_size` 时，DCP 组与 TP 组完全相同。当 `dcp_size < tp_size` 时，一个 TP 组被分割为多个 DCP 子组。

### 13.3 PCP 组创建 —— 与 TP 正交

```python
# PCP 在第 4 维 (index=3)，需要 transpose 到最后再 reshape
group_ranks = (
    all_ranks.transpose(3, 4)        # 交换 PCP 和 TP 维度
    .reshape(-1, prefill_context_model_parallel_size)
    .unbind(0)
)
```

**示例**: `tp=4, pcp=2`, 8 GPUs

```
all_ranks 原始布局 (PP=1, DP=1):
  shape: [1, 1, 1, 2, 4] → PCP × TP
  [[0,1,2,3],     ← PCP rank 0
   [4,5,6,7]]     ← PCP rank 1

transpose(3,4) → shape: [1, 1, 1, 4, 2]
reshape(-1, 2):
  [[0,4], [1,5], [2,6], [3,7]]  ← 4 个 PCP 组

含义: GPU 0 和 GPU 4 在同一个 PCP 组
      GPU 1 和 GPU 5 在同一个 PCP 组 ...
```

PCP 组中的 GPU 来自**不同的 TP 组**，它们负责同一个 TP 位置但不同的序列片段。

### 13.4 DCP + PCP 联合时的 Rank 关系

> 源文件: `vllm/v1/attention/backend.py`, `AttentionImplBase.__new__()`

```python
# 联合 rank 计算
self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
```

**示例**: `dcp=4, pcp=2`

```
total_cp_world_size = 2 × 4 = 8

PCP rank 0, DCP rank 0 → total_cp_rank = 0
PCP rank 0, DCP rank 1 → total_cp_rank = 1
PCP rank 0, DCP rank 2 → total_cp_rank = 2
PCP rank 0, DCP rank 3 → total_cp_rank = 3
PCP rank 1, DCP rank 0 → total_cp_rank = 4
PCP rank 1, DCP rank 1 → total_cp_rank = 5
...
```

---

## 第14章 KV Cache 切分策略

> 源文件: `vllm/v1/worker/gpu/cp_utils.py`, `vllm/v1/attention/backends/utils.py`

### 14.1 Round-Robin 分配原理

DCP 将 KV Cache 按 **round-robin（轮询）** 方式分配到各 rank，分配粒度由 `cp_kv_cache_interleave_size`（简称 `I`）控制：

```
序列 tokens: [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, ...]

dcp_size=4, interleave=1:
  Rank 0: [t0, t4, t8, ...]
  Rank 1: [t1, t5, t9, ...]
  Rank 2: [t2, t6, t10, ...]
  Rank 3: [t3, t7, t11, ...]

dcp_size=4, interleave=3:
  Rank 0: [t0,t1,t2,   t12,t13,t14,  ...]
  Rank 1: [t3,t4,t5,   t15,t16,t17,  ...]
  Rank 2: [t6,t7,t8,   t18,t19,t20,  ...]
  Rank 3: [t9,t10,t11, t21,t22,t23,  ...]
```

### 14.2 Local Seq Lens 计算

每个 DCP rank 持有的 token 数量通过以下公式计算：

$$
\text{rounds} = \left\lfloor \frac{\text{seq\_lens}}{\text{dcp\_size} \times I} \right\rfloor
$$

$$
\text{total\_remainder} = \text{seq\_lens} \mod (\text{dcp\_size} \times I)
$$

$$
\text{rank\_remainder} = \text{clamp}(\text{total\_remainder} - \text{dcp\_rank} \times I,\ 0,\ I)
$$

$$
\text{local\_seq\_lens} = \text{rounds} \times I + \text{rank\_remainder}
$$

### 14.3 Triton Kernel 实现

> 源文件: `vllm/v1/worker/gpu/cp_utils.py`

```python
@triton.jit
def _dcp_local_seq_lens_kernel(
    out_ptr, seq_lens_ptr,
    dcp_size, dcp_rank, cp_interleave,
    num_reqs, max_num_reqs, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    seq_lens = tl.load(seq_lens_ptr + block, mask=block < num_reqs)

    # Round-robin 分配
    rounds = seq_lens // (dcp_size * cp_interleave)
    remainder = seq_lens % (dcp_size * cp_interleave)
    remainder = tl.maximum(remainder - dcp_rank * cp_interleave, 0)
    remainder = tl.minimum(remainder, cp_interleave)
    local_seq_lens = rounds * cp_interleave + remainder

    local_seq_lens = tl.where(block < num_reqs, local_seq_lens, 0)
    tl.store(out_ptr + block, local_seq_lens, mask=block < max_num_reqs)
```

此 kernel 是 **CUDA Graph 安全的**（在 persistent buffer 上原地计算），在每次 attention forward 前执行。

### 14.4 PyTorch 参考实现

> 源文件: `vllm/v1/attention/backends/utils.py`, `get_dcp_local_seq_lens()`

```python
def get_dcp_local_seq_lens(seq_lens, dcp_size, dcp_rank, cp_kv_cache_interleave_size):
    base = seq_lens // cp_kv_cache_interleave_size // dcp_size * cp_kv_cache_interleave_size
    remainder = seq_lens - base * dcp_size
    remainder = torch.clip(
        remainder - rank_offsets * cp_kv_cache_interleave_size,
        0, cp_kv_cache_interleave_size
    )
    return base + remainder
```

### 14.5 数值示例

```
seq_lens = 100, dcp_size = 4, interleave = 1:

Rank 0: rounds = 100 // 4 = 25, remainder = clamp(0 - 0, 0, 1) = 0  → 25
Rank 1: rounds = 25, remainder = clamp(0 - 1, 0, 1) = 0              → 25
Rank 2: rounds = 25, remainder = clamp(0 - 2, 0, 1) = 0              → 25
Rank 3: rounds = 25, remainder = clamp(0 - 3, 0, 1) = 0              → 25
总计: 25 × 4 = 100 ✓

seq_lens = 101, dcp_size = 4, interleave = 1:

Rank 0: rounds = 25, remainder = clamp(1 - 0, 0, 1) = 1  → 26
Rank 1: rounds = 25, remainder = clamp(1 - 1, 0, 1) = 0  → 25
Rank 2: rounds = 25, remainder = clamp(1 - 2, 0, 1) = 0  → 25
Rank 3: rounds = 25, remainder = clamp(1 - 3, 0, 1) = 0  → 25
总计: 26 + 25 + 25 + 25 = 101 ✓
```

---

## 第15章 DCP Forward 计算流程

> 源文件: `vllm/v1/attention/backends/flash_attn.py`, `FlashAttentionImpl._forward_with_dcp()`

### 15.1 整体流程图

```
┌─────────────────────────────────────────────────────────┐
│                   _forward_with_dcp()                    │
│                                                          │
│  ① AllGather Query (across DCP group)                    │
│     query [B, H_local, D] → query_all [B, H_all, D]    │
│                                                          │
│  ② Context Attention (local KV shard)                    │
│     FA(query_all, KV_local) → ctx_out [B,H_all,D]      │
│     causal=False, return_lse=True                        │
│                                                          │
│  ③ DCP Combine (cross-rank LSE-weighted merge)           │
│     dcp_combine(ctx_out, ctx_lse) → ctx_cor [B,H_l,D]  │
│                                                          │
│  ④ Query Attention (current step's new KV)               │
│     FA(query, new_K, new_V) → q_out [B, H_l, D]        │
│     causal=True, return_lse=True                         │
│                                                          │
│  ⑤ Merge (LSE-weighted combination)                      │
│     merge_attn_states(output, ctx_cor, ctx_lse,          │
│                       q_out, q_lse)                      │
└─────────────────────────────────────────────────────────┘
```

### 15.2 步骤详解

#### Step 1: AllGather Query

```python
query = query.contiguous()
query_across_dcp = get_dcp_group().all_gather(query, dim=1)
# query:           [num_tokens, H_local, head_dim]
# query_across_dcp: [num_tokens, H_all,   head_dim]
# H_all = H_local × dcp_world_size
```

**为什么需要 AllGather Q?** DCP 模式下，每个 rank 只持有部分注意力头的 Q。但为了对本地 KV shard 做完整的 attention，需要所有 head 的 Q。

#### Step 2: Context Attention（对历史 KV Cache 做 Attention）

```python
context_attn_out, context_lse = flash_attn_varlen_func(
    q=query_across_dcp,            # 所有 head 的 Q
    k=key_cache, v=value_cache,    # 本地 KV shard
    seqused_k=dcp_context_kv_lens, # 每个 rank 的本地 KV 长度
    causal=False,                  # KV 是 round-robin 分配的，不连续，不能用 causal
    return_softmax_lse=True,       # 返回 log-sum-exp 用于后续合并
)
# context_attn_out: [num_tokens, H_all, head_dim]
# context_lse:      [H_all, num_tokens]  (FA 返回格式)
```

**关键**: `causal=False`。因为 KV Cache 是 round-robin 分配的，token 顺序不连续，因此不能使用 causal mask。

#### Step 3: DCP Combine（跨 rank 合并部分注意力输出）

```python
context_attn_out_cor, context_lse_cor = self.dcp_combine(
    context_attn_out,
    context_lse.transpose(0, 1),  # [H,B] → [B,H]
    get_dcp_group(),
    return_lse=True,
)
```

`dcp_combine` 根据配置选择 AG+RS 或 A2A 后端（详见第16章）。

#### Step 4: Query Attention（对当前步新 KV 做 Attention）

```python
query_attn_out, query_lse = flash_attn_varlen_func(
    q=query,                  # 本地 head 的 Q
    k=key, v=value,           # 当前步新写入的 K, V
    cu_seqlens_k=cu_seqlens_q,
    causal=attn_metadata.causal,  # 当前步的 KV 是连续的，可以用 causal
    return_softmax_lse=True,
)
```

#### Step 5: Merge

```python
merge_attn_states(
    output,
    context_attn_out_cor, context_lse_cor,
    query_attn_out, query_lse,
)
```

使用 LSE 加权合并两部分注意力输出（context 部分 + query 部分）。

### 15.3 Attention Backend 选择

> 源文件: `vllm/v1/attention/backends/flash_attn.py`, `FlashAttentionImpl.__init__()`

```python
vllm_config = get_current_vllm_config_or_none()
dcp_a2a = (
    vllm_config is not None
    and vllm_config.parallel_config.decode_context_parallel_size > 1
    and vllm_config.parallel_config.dcp_comm_backend == "a2a"
)
self.dcp_combine = dcp_a2a_lse_reduce if dcp_a2a else cp_lse_ag_out_rs
```

---

## 第16章 两种 DCP 通信后端对比

### 16.1 AG+RS 后端（AllGather + ReduceScatter）

> 源文件: `vllm/v1/attention/ops/common.py`, `cp_lse_ag_out_rs()`

#### 通信流程

```
Step 1: AllGather LSE
  各 rank 的 lse [B, H] → 合并为 lses [N, B, H]

Step 2: Correct Output (本地 Triton Kernel)
  利用全局 LSE 修正本地 attention 输出
  factor = exp(local_lse - global_lse)
  corrected_out = local_out × factor

Step 3: ReduceScatter Output
  各 rank 的 corrected_out [B, H, D] → 按 head 维度 scatter
  每个 rank 获得 [B, H/N, D]
```

#### 代码实现

```python
def cp_lse_ag_out_rs(cp_attn_out, cp_attn_lse, cp_group, ...):
    # Step 1+2: AllGather LSE + 修正
    out, lse = _cp_lse_common(cp_attn_out, cp_attn_lse, cp_group, ...)

    # Step 3: ReduceScatter（按 head 维度 scatter）
    out = cp_group.reduce_scatter(out, dim=1)

    if return_lse:
        cp_num_heads = lse.shape[1] // cp_group.world_size
        cp_rank = cp_group.rank_in_group
        lse = lse[:, cp_num_heads * cp_rank : cp_num_heads * (cp_rank + 1)]
        return out, lse
    return out
```

#### NCCL 调用次数: **3 次**
1. AllGather Q (在 `_forward_with_dcp` 开头)
2. AllGather LSE (在 `_cp_lse_common` 中)
3. ReduceScatter output (在 `cp_lse_ag_out_rs` 中)

### 16.2 A2A 后端（All-to-All）

> 源文件: `vllm/v1/attention/ops/dcp_alltoall.py`, `dcp_a2a_lse_reduce()`
> 参考论文: https://arxiv.org/abs/2507.07120

#### 通信流程

```
Step 1: Reshape for A2A
  output [B, H, D] → [N, B, H/N, D]  (按 head 切分)
  lse    [B, H]    → [N, B, H/N]

Step 2: All-to-All (两次并行)
  send_output [N, B, H/N, D] ←→ recv_output [N, B, H/N, D]
  send_lse    [N, B, H/N]    ←→ recv_lse    [N, B, H/N]
  (异步执行，overlap 通信)

Step 3: Local Triton Combine (LSE-weighted merge)
  recv [N, B, H/N, D] + lse [N, B, H/N] → [B, H/N, D]
```

#### 代码实现

```python
def dcp_a2a_lse_reduce(cp_attn_out, cp_attn_lse, cp_group, ...):
    B, H, D = cp_attn_out.shape
    H_per_rank = H // world_size

    # Reshape: [B,H,D] → [N,B,H/N,D], [B,H] → [N,B,H/N]
    send_output = cp_attn_out.view(B, world_size, H_per_rank, D) \
                             .permute(1, 0, 2, 3).contiguous()
    send_lse = cp_attn_lse.view(B, world_size, H_per_rank) \
                           .permute(1, 0, 2).contiguous()

    # All-to-All (async overlap)
    work_output = dist.all_to_all_single(recv_output, send_output, ..., async_op=True)
    work_lse = dist.all_to_all_single(recv_lse, send_lse, ..., async_op=True)
    work_output.wait()
    work_lse.wait()

    # Local Triton combine
    return dcp_lse_combine_triton(recv_output, recv_lse, ...)
```

#### NCCL 调用次数: **3 次**（但通信量不同）
1. AllGather Q (在 `_forward_with_dcp` 开头)
2. All-to-All output (在 `dcp_a2a_lse_reduce` 中)
3. All-to-All LSE (在 `dcp_a2a_lse_reduce` 中，与上一步 async overlap)

### 16.3 两种后端通信量对比

设 $B$ = batch tokens, $H$ = total heads, $D$ = head dim, $N$ = DCP world size:

| 操作 | AG+RS | A2A |
|------|-------|-----|
| AllGather Q | $B \times H \times D$ | $B \times H \times D$ |
| LSE 通信 | AllGather: $N \times B \times H$ (每 rank 收到全量) | A2A: $B \times H$ (每 rank 收发 $B \times H/N$，共 $N$ 份) |
| Output 通信 | ReduceScatter: $B \times H \times D$ | A2A: $B \times H \times D$ |
| **总 NCCL 延迟** | 3 次独立调用 | 2+1 次（A2A output 和 LSE 可 overlap） |

**结论**: A2A 后端在长上下文 decode 场景下，由于 NCCL 调用次数少、可 overlap，**每步通信延迟更低**。但 AG+RS 后端实现更简单，对短上下文的 overhead 更可控。

---

## 第17章 LSE 加权合并的数学原理

### 17.1 问题定义

DCP 将 KV Cache 分为 $N$ 个 shard，每个 rank $i$ 独立计算：

$$
\text{out}_i = \text{softmax}\left(\frac{QK_i^T}{\sqrt{d}}\right) V_i
$$

$$
\text{lse}_i = \log \sum_j \exp\left(\frac{q \cdot k_{i,j}}{\sqrt{d}}\right)
$$

需要合并为全局结果：

$$
\text{out} = \text{softmax}\left(\frac{Q[K_0; K_1; \ldots; K_{N-1}]^T}{\sqrt{d}}\right) [V_0; V_1; \ldots; V_{N-1}]
$$

### 17.2 LSE 合并公式

**核心思想**: 利用 LSE (Log-Sum-Exp) 值计算每个 shard 的权重，无需重新计算全局 softmax。

$$
\text{LSE}_{\text{global}} = \log \left( \sum_{i=0}^{N-1} \exp(\text{lse}_i) \right)
$$

使用数值稳定的 log-sum-exp trick:

$$
\text{lse\_max} = \max_i(\text{lse}_i)
$$

$$
\text{LSE}_{\text{global}} = \text{lse\_max} + \log \left( \sum_{i=0}^{N-1} \exp(\text{lse}_i - \text{lse\_max}) \right)
$$

每个 shard 的权重:

$$
w_i = \frac{\exp(\text{lse}_i)}{\sum_j \exp(\text{lse}_j)} = \exp(\text{lse}_i - \text{LSE}_{\text{global}})
$$

全局输出:

$$
\text{out} = \sum_{i=0}^{N-1} w_i \cdot \text{out}_i
$$

### 17.3 正确性证明

对于全序列的 attention:

$$
\text{out} = \frac{\sum_j \exp(s_j) v_j}{\sum_j \exp(s_j)}
$$

其中 $s_j = q \cdot k_j / \sqrt{d}$。将 $j$ 按 shard 分组:

$$
\text{out} = \frac{\sum_i \sum_{j \in S_i} \exp(s_j) v_j}{\sum_i \sum_{j \in S_i} \exp(s_j)}
$$

对于第 $i$ 个 shard:

$$
\text{out}_i = \frac{\sum_{j \in S_i} \exp(s_j) v_j}{\sum_{j \in S_i} \exp(s_j)}, \quad \exp(\text{lse}_i) = \sum_{j \in S_i} \exp(s_j)
$$

因此:

$$
\text{out} = \frac{\sum_i \exp(\text{lse}_i) \cdot \text{out}_i}{\sum_i \exp(\text{lse}_i)} = \sum_i w_i \cdot \text{out}_i \quad \square
$$

### 17.4 Triton Kernel 实现（A2A 后端）

> 源文件: `vllm/v1/attention/ops/dcp_alltoall.py`, `_dcp_lse_combine_kernel`

```python
@triton.jit
def _dcp_lse_combine_kernel(recv_output_ptr, recv_lse_ptr, out_ptr, out_lse_ptr, ...):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Pass 1: 找最大 LSE (数值稳定)
    lse_max = -float("inf")
    for n in tl.static_range(N):
        lse_val = tl.load(recv_lse_ptr + n * rl_stride_N + base_offset)
        lse_max = tl.maximum(lse_max, lse_val)

    # Pass 2: 计算 exp 求和 → global_lse
    lse_sum = 0.0
    for n in tl.static_range(N):
        lse_val = tl.load(recv_lse_ptr + n * rl_stride_N + base_offset)
        lse_sum += tl.exp(lse_val - lse_max)
    global_lse = tl.log(lse_sum) + lse_max

    # Pass 3: 加权合并 HEAD_DIM 维度
    d_offsets = tl.arange(0, HEAD_DIM)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for n in tl.static_range(N):
        lse_val = tl.load(recv_lse_ptr + n * rl_stride_N + base_offset)
        weight = tl.exp(lse_val - global_lse)
        out_vals = tl.load(recv_output_ptr + n * ro_stride_N + batch_idx * ro_stride_B
                           + head_idx * ro_stride_H + d_offsets)
        acc += out_vals * weight

    tl.store(out_ptr + batch_idx * o_stride_B + head_idx * o_stride_H + d_offsets, acc)
    tl.store(out_lse_ptr + batch_idx * ol_stride_B + head_idx, global_lse)
```

**Grid**: `(B, H_local, 1)`，每个 program 处理一个 `(batch, head)` 对的所有 $D$ 维元素。三趟遍历确保数值稳定。

### 17.5 Triton Kernel 实现（AG+RS 后端）

> 源文件: `vllm/v1/attention/ops/common.py`, `_correct_attn_cp_out_kernel`

AG+RS 后端使用**修正**（correct）而非合并：先用全局 LSE 修正本地输出，再通过 ReduceScatter 隐式完成加权求和。

```python
@triton.jit
def _correct_attn_cp_out_kernel(outputs_ptr, new_output_ptr, lses_ptr, ...):
    # 计算全局 LSE (与 A2A 版本相同)
    lse_max = tl.max(lse, axis=0)
    lse_exp = tl.exp(lse - lse_max)
    lse_acc = tl.sum(lse_exp, axis=0)
    global_lse = tl.log(lse_acc) + lse_max

    # 修正本地输出: factor = exp(local_lse - global_lse)
    factor = tl.exp(local_lse - global_lse)
    output = tl.load(outputs_ptr + ...) * factor
    tl.store(new_output_ptr + ..., output)
```

修正后，ReduceScatter 将各 rank 的 `output × factor` 相加，等价于全局 LSE 加权合并。

---

## 第18章 PCP 与 Expert Parallel 的交互

### 18.1 PCP 对 EP Rank 的影响

> 源文件: `vllm/model_executor/layers/fused_moe/config.py`

PCP 引入额外的并行维度，影响 Expert Parallel 中的 rank 计算:

```python
# EP 组 = DP × PCP × TP 的所有 rank
flatten_tp_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
```

这意味着 **PCP 的不同 rank 在 EP 组中被视为不同的 rank**，它们可能负责不同的专家子集。

### 18.2 模型权重加载中的 PCP

> 源文件: `vllm/model_executor/model_loader/default_loader.py`

```python
# 模型加载时的 ep_rank 计算
ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
```

专家权重根据 `ep_rank` 进行切片加载，确保 PCP 各 rank 加载正确的专家子集。

### 18.3 EP 组构成（含 PCP）

```python
# parallel_state.py 中的 EP 组创建
group_ranks = (
    all_ranks.transpose(1, 2)  # 交换 DP 和 PP
    .reshape(-1, data_parallel_size * pcp_size * tp_size)
    .unbind(0)
)
# EP 组 = 同一 PP stage 内的所有 DP × PCP × TP rank
```

**示例**: `tp=4, pcp=2, dp=2`, 16 GPUs

```
EP 组 (PP=1):
  [GPU_0 ... GPU_15]  ← 所有 16 个 GPU 在同一个 EP 组中
  EP_size = DP × PCP × TP = 2 × 2 × 4 = 16

ep_rank 映射:
  DP=0, PCP=0, TP=0 → ep_rank = 0
  DP=0, PCP=0, TP=1 → ep_rank = 1
  ...
  DP=0, PCP=1, TP=0 → ep_rank = 4
  ...
  DP=1, PCP=0, TP=0 → ep_rank = 8
  ...
```

---

## 第19章 DCP + PCP 联合使用分析

### 19.1 联合使用的场景

长上下文推理的典型部署:

```bash
# 超长上下文 (256K+) 推理
vllm serve deepseek-v3 --tp 16 --dcp 16 --pcp 2

# GPU 总量: 16 (TP) × 2 (PCP) = 32 GPUs
# Prefill: PCP 将序列切为 2 份，各 16 GPU 处理一份
# Decode:  DCP 将 KV Cache 切为 16 份，每 GPU 持有 1/16
```

### 19.2 AttentionImplBase 的统一抽象

> 源文件: `vllm/v1/attention/backend.py`

```python
class AttentionImplBase(ABC, Generic[T]):
    # DCP 能力声明
    can_return_lse_for_decode: bool = False    # 必须为 True 才能启用 DCP
    supports_pcp: bool = False                  # 必须为 True 才能启用 PCP

    dcp_world_size: int
    dcp_rank: int
    pcp_world_size: int
    pcp_rank: int
    total_cp_world_size: int
    total_cp_rank: int

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        # 自动从全局通信组获取 rank 信息
        self.dcp_world_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group
        self.pcp_world_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group
        self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        self.need_to_return_lse_for_decode = (
            self.dcp_world_size > 1 and self.can_return_lse_for_decode
        )
        return self
```

### 19.3 兼容性检查

> 源文件: `vllm/v1/worker/cp_utils.py`

```python
def check_attention_cp_compatibility(vllm_config):
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size

    if pcp_size * dcp_size > 1:
        layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
        for layer in layers.values():
            impl = getattr(layer, "impl", None)
            if impl is None:
                continue
            if dcp_size > 1:
                assert impl.need_to_return_lse_for_decode  # DCP 要求
            if pcp_size > 1:
                assert impl.supports_pcp                    # PCP 要求
```

### 19.4 当前 Backend 支持状态

| Attention Backend | can_return_lse_for_decode | supports_pcp | DCP | PCP |
|-------------------|:------------------------:|:------------:|:---:|:---:|
| FlashAttention    | ✅ True | 待确认 | ✅ | 待确认 |
| 其他 Backends | ❌ False | ❌ False | ❌ | ❌ |

> 注: DCP/PCP 目前主要由 FlashAttention 后端支持，其他后端需要实现 `can_return_lse_for_decode` 和 `supports_pcp` 才能启用。

### 19.5 设计总结

```
┌─────────────────────────────────────────────────────┐
│              vLLM Context Parallel 设计              │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌─── PCP (Prefill) ────┐  ┌─── DCP (Decode) ────┐ │
│  │ • 按序列切分          │  │ • 按 token 切分 KV  │ │
│  │ • 增加 GPU            │  │ • 复用 TP 组 GPU    │ │
│  │ • 独立通信组          │  │ • 2 种通信后端      │ │
│  │ • 影响 EP rank        │  │ • Round-Robin 分配  │ │
│  └──────────────────────┘  └──────────────────────┘ │
│                    │                │                 │
│                    └───── 联合 ─────┘                 │
│                           │                          │
│              total_cp_rank = pcp_rank × dcp_size     │
│                          + dcp_rank                  │
│                                                      │
│  ┌─── AttentionImplBase ─────────────────────────┐  │
│  │ can_return_lse_for_decode → 启用 DCP           │  │
│  │ supports_pcp              → 启用 PCP           │  │
│  │ dcp_combine = ag_rs | a2a → 选择通信后端       │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

# 附录

---

## 附录 A：源码阅读路径

### 推荐阅读顺序

```
1. 入口理解
   examples/offline_inference/data_parallel.py
   └── 理解 DP 环境变量和进程创建

2. Worker 初始化
   vllm/worker/gpu_worker.py
   └── Worker.init_device() → init_worker_distributed_environment()

3. 通信组创建（核心）
   vllm/distributed/parallel_state.py
   ├── init_distributed_environment()     # torch.distributed 初始化
   ├── initialize_model_parallel()        # all_ranks 张量 + 组创建
   └── class GroupCoordinator             # 通信组抽象

4. 通信后端
   vllm/distributed/device_communicators/cuda_communicator.py
   └── CudaCommunicator: AllReduce 优先级链, All2All 后端

5. TP 切分原语
   vllm/model_executor/layers/linear.py
   ├── ColumnParallelLinear               # 列切分
   ├── MergedColumnParallelLinear         # 合并列切分
   ├── QKVParallelLinear                  # QKV 专用
   └── RowParallelLinear                  # 行切分 + AllReduce

6. 模型实例
   vllm/model_executor/models/deepseek_v2.py
   ├── DeepseekAttention / DeepseekV2Attention  # Attention 层
   ├── DeepseekV2MLP                            # Dense MLP
   ├── DeepseekV2MoE                            # MoE 层
   └── DeepseekV2ForCausalLM                    # 顶层模型

7. Embedding
   vllm/model_executor/layers/vocab_parallel_embedding.py
   └── VocabParallelEmbedding             # 词表并行
```

---

## 附录 B：配置速查表

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `VLLM_DP_SIZE` | 1 | 数据并行大小 |
| `VLLM_DP_RANK` | 0 | 当前 DP rank |
| `VLLM_DP_RANK_LOCAL` | 0 | 本地 DP rank |
| `VLLM_DP_MASTER_IP` | 127.0.0.1 | DP 主节点 IP |
| `VLLM_DP_MASTER_PORT` | auto | DP 主节点端口 |
| `CUDA_VISIBLE_DEVICES` | all | 可见 GPU 列表 |

### LLM/EngineArgs 关键参数

| 参数 | 说明 |
|------|------|
| `tensor_parallel_size` | TP 大小 |
| `pipeline_parallel_size` | PP 大小 |
| `data_parallel_size` | DP 大小 |
| `enable_expert_parallel` | 启用 EP |
| `enforce_eager` | 禁用 CUDA Graph（调试用） |
| `disable_custom_all_reduce` | 禁用自定义 AllReduce |

---

## 附录 C：关键 API 索引

### parallel_state.py

| API | 功能 |
|-----|------|
| `init_distributed_environment()` | 初始化 torch.distributed + World group |
| `initialize_model_parallel()` | 创建所有并行通信组 |
| `get_world_group()` | 获取全局 World 通信组 |
| `get_tp_group()` | 获取 TP 通信组 |
| `get_pp_group()` | 获取 PP 通信组 |
| `get_dp_group()` | 获取 DP 通信组 |
| `get_ep_group()` | 获取 EP 通信组 |
| `get_dcp_group()` | 获取 DCP 通信组 |
| `get_pcp_group()` | 获取 PCP 通信组 |
| `get_tensor_model_parallel_world_size()` | 获取 TP 大小 |
| `get_tensor_model_parallel_rank()` | 获取当前 TP rank |
| `get_pipeline_model_parallel_world_size()` | 获取 PP 大小 |

### linear.py

| API | 功能 |
|-----|------|
| `ColumnParallelLinear` | 列并行线性层 |
| `MergedColumnParallelLinear` | 合并列并行层 |
| `QKVParallelLinear` | QKV 投影并行层 |
| `RowParallelLinear` | 行并行线性层 |

### cuda_communicator.py

| API | 功能 |
|-----|------|
| `CudaCommunicator.all_reduce()` | AllReduce（优先级链） |
| `CudaCommunicator.reduce_scatter()` | ReduceScatter |
| `CudaCommunicator.all_gatherv()` | 变长 AllGather |
| `CudaCommunicator.send() / recv()` | P2P 通信 |
| `CudaCommunicator.dispatch() / combine()` | All2All 通信 |
| `CudaCommunicator.broadcast()` | 广播 |

---

## 附录 D：DCP/PCP 源码阅读路径

### 推荐阅读顺序

| 序号 | 文件 | 内容 |
|------|------|------|
| 1 | `vllm/config/parallel.py` | DCP/PCP 配置定义与参数验证 |
| 2 | `vllm/distributed/parallel_state.py` | DCP/PCP 通信组创建 (`_DCP`, `_PCP`) |
| 3 | `vllm/v1/worker/cp_utils.py` | 兼容性检查 `check_attention_cp_compatibility` |
| 4 | `vllm/v1/worker/gpu/cp_utils.py` | DCP local seq lens Triton kernel |
| 5 | `vllm/v1/attention/backends/utils.py` | `get_dcp_local_seq_lens()` PyTorch 实现 |
| 6 | `vllm/v1/attention/backend.py` | `AttentionImplBase` DCP/PCP 属性初始化 |
| 7 | `vllm/v1/attention/backends/flash_attn.py` | `_forward_with_dcp()` 核心 forward 逻辑 |
| 8 | `vllm/v1/attention/ops/common.py` | AG+RS 后端: `cp_lse_ag_out_rs`, `correct_attn_out` |
| 9 | `vllm/v1/attention/ops/dcp_alltoall.py` | A2A 后端: `dcp_a2a_lse_reduce`, Triton combine |
| 10 | `vllm/config/model.py` | GQA/MQA 模型的 DCP 约束验证 |
| 11 | `vllm/model_executor/layers/fused_moe/config.py` | PCP 对 EP rank 映射的影响 |

## 附录 E：DCP/PCP 关键 API 索引

### parallel_state.py

| API | 功能 |
|-----|------|
| `get_dcp_group()` | 获取 DCP 通信组 |
| `get_pcp_group()` | 获取 PCP 通信组 |
| `get_context_model_parallel_group()` | `get_dcp_group()` 的别名（向后兼容） |

### cp_utils.py

| API | 功能 |
|-----|------|
| `check_attention_cp_compatibility()` | 检查 attention backend 是否支持 DCP/PCP |
| `get_total_cp_world_size()` | 返回 `dcp_size × pcp_size` |
| `prepare_dcp_local_seq_lens()` | Triton kernel 计算各 rank 的本地 KV 长度 |

### dcp_alltoall.py (A2A 后端)

| API | 功能 |
|-----|------|
| `dcp_a2a_lse_reduce()` | A2A 通信 + LSE 加权合并 |
| `dcp_lse_combine_triton()` | Triton LSE 加权合并 kernel 的 Python wrapper |
| `_dcp_lse_combine_kernel` | Triton kernel: 3-pass LSE 加权合并 |
| `_lse_weighted_combine()` | CPU 参考实现（用于测试验证） |

### common.py (AG+RS 后端)

| API | 功能 |
|-----|------|
| `cp_lse_ag_out_rs()` | AllGather LSE + Correct + ReduceScatter |
| `cp_lse_ag_out_ar()` | AllGather LSE + Correct + AllReduce |
| `correct_attn_out()` | 利用全局 LSE 修正本地 attention 输出 |
| `_correct_attn_cp_out_kernel` | Triton kernel: LSE 修正 |

---

> **报告结束**  
> 本报告基于 vLLM 0.18.0 源码分析，所有代码引用均标注了源文件和近似行号。  
> 如有疑问或需要更深入某一部分，请随时提出。
