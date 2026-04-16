# vLLM KV Cache 管理机制详解

> 本文档以 `examples/basic/offline_inference/basic.py` 推理脚本为入口，深入分析 vLLM V1 引擎中 KV Cache 的管理机制，重点阐述 Block Table、Slot Mapping 的创建与更新逻辑，并结合 Attention Metadata 说明 Attention Backend 中如何使用 Slot Mapping。

---

## 目录

1. [推理脚本入口](#1-推理脚本入口)
2. [KV Cache 整体架构](#2-kv-cache-整体架构)
3. [核心数据结构](#3-核心数据结构)
4. [Block Pool：物理块管理](#4-block-pool物理块管理)
5. [KV Cache Manager：块分配与调度](#5-kv-cache-manager块分配与调度)
6. [Block Table：块表管理与 Slot Mapping 计算](#6-block-table块表管理与-slot-mapping-计算)
7. [Attention Metadata 的构建](#7-attention-metadata-的构建)
8. [Attention Backend 中 Slot Mapping 的使用](#8-attention-backend-中-slot-mapping-的使用)
9. [CUDA Kernel：reshape_and_cache_flash](#9-cuda-kernelreshape_and_cache_flash)
10. [完整数据流总结](#10-完整数据流总结)

---

## 1. 推理脚本入口

```python
# examples/basic/offline_inference/basic.py
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * 2

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

llm = LLM(model="/data/models/deepseek/deepseek-moe-16b-base",
           enforce_eager=True,
           gpu_memory_utilization=0.8,
           tensor_parallel_size=2)

outputs = llm.generate(prompts, sampling_params)
```

当调用 `llm.generate()` 时，vLLM 内部经历以下关键阶段：

```
LLM.generate()
  → EngineCore (V1)
    → Scheduler：决定哪些请求参与本次推理，分配 KV Cache 块
    → GPUModelRunner：构建模型输入，计算 slot_mapping
    → Model Forward：使用 Attention Backend 进行注意力计算，写入 KV Cache
```

以 8 个 prompt、`max_tokens=50` 为例，每个请求最终需要约 55-60 个 token 的 KV Cache 存储（prompt + generated tokens）。

---

## 2. KV Cache 整体架构

vLLM 采用 **PagedAttention** 技术管理 KV Cache，核心思想借鉴了操作系统的虚拟内存分页机制：

```
┌──────────────────────────────────────────────────────────┐
│                    KV Cache 分页架构                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  逻辑视图 (每个请求)           物理视图 (GPU 显存)         │
│  ┌─────────────────┐          ┌─────────────────────┐    │
│  │ Token 0..15     │ ───────→ │ Block 42            │    │
│  │ Token 16..31    │ ───────→ │ Block 7             │    │
│  │ Token 32..47    │ ───────→ │ Block 103           │    │
│  │ Token 48..55    │ ───────→ │ Block 89 (部分填充)  │    │
│  └─────────────────┘          └─────────────────────┘    │
│                                                          │
│  Block Table (映射表)                                     │
│  req_0: [42, 7, 103, 89]                                 │
│  req_1: [15, 23, 67, ...]                                │
│  ...                                                     │
└──────────────────────────────────────────────────────────┘
```

### 关键概念

| 概念 | 说明 |
|------|------|
| **Block** | KV Cache 的最小分配单位，包含固定数量的 token 的 KV 向量（通常 block_size=16） |
| **Block Table** | 每个请求维护的映射表，记录逻辑块到物理块的映射 |
| **Slot** | Block 内某个 token 位置的全局唯一标识，`slot = block_id × block_size + block_offset` |
| **Slot Mapping** | 将当前 batch 中每个 token 映射到其在 KV Cache 中的具体 slot |
| **Block Pool** | 所有物理块的资源池，管理分配、释放和缓存 |

---

## 3. 核心数据结构

### 3.1 KVCacheBlock —— 物理块的元数据

> 源码：`vllm/v1/core/kv_cache_utils.py`

```python
class KVCacheBlock:
    """KV-cache block metadata."""
    block_id: int                                      # 物理块 ID（全局唯一）
    ref_cnt: int = 0                                   # 引用计数（有多少请求使用该块）
    _block_hash: BlockHashWithGroupId | None = None    # 块哈希（用于 prefix caching）
    prev_free_block: "KVCacheBlock | None" = None      # 双向链表：前驱空闲块
    next_free_block: "KVCacheBlock | None" = None      # 双向链表：后继空闲块
    is_null: bool = False                              # 是否为空块（占位用）
```

每个 `KVCacheBlock` 仅存储**元数据**，不持有实际的 KV 向量数据。真正的 KV 数据存储在 GPU 显存的一个大的连续 tensor 中，通过 `block_id` 索引。

### 3.2 FreeKVCacheBlockQueue —— 空闲块双向链表

> 源码：`vllm/v1/core/kv_cache_utils.py`

```python
class FreeKVCacheBlockQueue:
    """
    空闲块的双向链表，支持 O(1) 时间的：
    - popleft(): 从头部弹出（分配最旧的空闲块）
    - append():  添加到尾部（释放的块追加到末尾）
    - remove():  从中间移除（命中 prefix cache 时移除）
    
    淘汰策略：LRU（Least Recently Used）
    - 链表头部 = 最久未使用的块（优先被淘汰）
    - 链表尾部 = 最近释放的块
    """
```

### 3.3 BlockTable —— GPU 端块表

> 源码：`vllm/v1/worker/block_table.py`

```python
class BlockTable:
    def __init__(self, block_size, max_num_reqs, max_num_blocks_per_req,
                 max_num_batched_tokens, pin_memory, device, 
                 kernel_block_size, cp_kv_cache_interleave_size):
        # 核心存储：二维数组 [max_num_reqs, max_num_blocks_per_req]
        self.block_table = CpuGpuBuffer(
            max_num_reqs, max_num_blocks_per_req, dtype=torch.int32
        )
        # Slot Mapping：一维数组 [max_num_batched_tokens]
        self.slot_mapping = CpuGpuBuffer(
            max_num_batched_tokens, dtype=torch.int64
        )
```

`BlockTable` 使用 `CpuGpuBuffer` 实现 CPU-GPU 双缓冲：
- **CPU 端（numpy）**：用于 Scheduler 的快速计算
- **GPU 端（torch.Tensor）**：用于 Attention Kernel 的读取

---

## 4. Block Pool：物理块管理

> 源码：`vllm/v1/core/block_pool.py`

### 4.1 初始化

```python
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, hash_block_size):
        # 创建所有物理块（例如 num_gpu_blocks=2000）
        self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
        
        # 构建空闲块链表（初始时所有块都在链表中）
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        
        # Prefix Cache 哈希表：hash → block
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        
        # 空块（block_id=0），用于 sliding window 等场景的占位
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
```

初始化时根据 `gpu_memory_utilization=0.8` 等参数计算出可用的 GPU block 数量。例如，对于一个 80GB 的 GPU，使用 80% 显存，扣除模型权重后，可能分配约 2000 个 block。

### 4.2 块分配流程

```python
def get_new_blocks(self, num_blocks):
    """从空闲队列分配 num_blocks 个块"""
    ret = self.free_block_queue.popleft_n(num_blocks)
    for block in ret:
        # 如果该块之前被缓存（prefix caching），需要先驱逐
        self._maybe_evict_cached_block(block)
        block.ref_cnt += 1    # 引用计数 +1
    return ret
```

### 4.3 Prefix Caching 支持

当一个块被填满后，会计算其哈希值并缓存：

```python
def cache_full_blocks(self, request, blocks, num_cached_blocks, 
                      num_full_blocks, block_size, kv_cache_group_id):
    """将已填满的块添加到 prefix cache"""
    new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
    for i, blk in enumerate(new_full_blocks):
        block_hash = block_hashes[num_cached_blocks + i]
        block_hash_with_group_id = make_block_hash_with_group_id(
            block_hash, kv_cache_group_id
        )
        blk.block_hash = block_hash_with_group_id
        self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
```

查找缓存命中：
```python
def get_cached_block(self, block_hash, kv_cache_group_ids):
    """通过哈希值查找缓存的块"""
    cached_blocks = []
    for group_id in kv_cache_group_ids:
        block = self.cached_block_hash_to_block.get_one_block(
            make_block_hash_with_group_id(block_hash, group_id)
        )
        if not block:
            return None
        cached_blocks.append(block)
    return cached_blocks
```

---

## 5. KV Cache Manager：块分配与调度

> 源码：`vllm/v1/core/kv_cache_manager.py`

### 5.1 调度阶段 —— 分配块

在 Scheduler 调度每个 step 时，对每个请求调用 `allocate_slots()`：

```python
def allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0,
                   new_computed_blocks=None, ...):
    """
    为请求分配新的 KV Cache 块。
    
    块布局:
    |----- cached -----|----- new_computed -----|----- new ------|
    |                  |                        |               |
    | 之前已计算的块     | prefix cache 命中的块    | 本次需新分配的块|
    """
    # 1. 计算需要多少个新块
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=num_tokens_need_slot, ...
    )
    
    # 2. 检查是否有足够的空闲块
    if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
        return None  # 资源不足，无法调度
    
    # 3. 分配新块
    new_blocks = self.coordinator.allocate_new_blocks(
        request.request_id, num_tokens_need_slot, ...
    )
    
    # 4. 缓存已填满的块（prefix caching）
    self.coordinator.cache_blocks(request, num_tokens_to_cache)
    
    return self.create_kv_cache_blocks(new_blocks)
```

### 5.2 块分配结果 —— KVCacheBlocks

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    blocks[i][j] = 第 i 个 kv_cache_group 的第 j 个块
    """
    
    def get_block_ids(self) -> tuple[list[int], ...]:
        """转换为 block_id 列表，传给 Block Table"""
        return tuple([blk.block_id for blk in group] for group in self.blocks)
```

### 5.3 从 Scheduler 到 Worker 的数据流

```
Scheduler:
  KVCacheManager.allocate_slots(request)
    → KVCacheBlocks (包含 block_id 列表)
    → SchedulerOutput.new_block_ids = {req_id: [42, 7, 103, 89]}

Worker (GPUModelRunner):
  收到 SchedulerOutput
    → BlockTable.append_row(block_ids=[42, 7, 103, 89], row_idx=req_idx)
    → BlockTable.compute_slot_mapping(req_indices, positions)
    → BlockTable.commit_slot_mapping(num_tokens)  # CPU → GPU 拷贝
```

---

## 6. Block Table：块表管理与 Slot Mapping 计算

> 源码：`vllm/v1/worker/block_table.py`

这是最关键的部分，理解 Block Table 如何管理映射关系以及 Slot Mapping 的计算逻辑。

### 6.1 Block Table 的更新操作

#### append_row —— 追加新分配的块

当 Scheduler 为某个请求分配了新的块后：

```python
def append_row(self, block_ids: list[int], row_idx: int) -> None:
    """将新分配的 block_ids 追加到 block_table 的第 row_idx 行"""
    num_blocks = len(block_ids)
    start = self.num_blocks_per_row[row_idx]       # 当前已有的块数
    self.num_blocks_per_row[row_idx] += num_blocks  # 更新块数
    # 写入 block_table 的 numpy 数组
    self.block_table.np[row_idx, start:start + num_blocks] = block_ids
```

**示例**：假设 `block_size=16`，请求 0 有 50 个 token，需要 4 个块：

```
初始 prefill:
  block_table.np[0] = [42, 7, 103, 89, 0, 0, ...]
  num_blocks_per_row[0] = 4

生成第 51 个 token（需要新块）:
  append_row([156], row_idx=0)
  block_table.np[0] = [42, 7, 103, 89, 156, 0, ...]
  num_blocks_per_row[0] = 5
```

#### add_row —— 完整设置一行

```python
def add_row(self, block_ids: list[int], row_idx: int) -> None:
    """重置并设置 block_table 的第 row_idx 行"""
    self.num_blocks_per_row[row_idx] = 0
    self.append_row(block_ids, row_idx)
```

#### commit_block_table —— CPU 到 GPU 同步

```python
def commit_block_table(self, num_reqs: int) -> None:
    """将 block_table 从 CPU 拷贝到 GPU"""
    self.block_table.copy_to_gpu(num_reqs)
```

### 6.2 Slot Mapping 的计算 ⭐

这是本文档的核心部分。`compute_slot_mapping` 将 token 的逻辑位置映射到 KV Cache 中的物理 slot。

```python
def compute_slot_mapping(self, req_indices: np.ndarray, positions: np.ndarray) -> None:
    """
    参数:
        req_indices: 每个 token 所属的请求索引，形如 [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        positions:   每个 token 在其请求中的位置，形如 [0, 1, 2, 0, 1, 2, 3, 0, 1, 2]
    
    计算公式:
        slot = block_table[req_idx][position // block_size] * block_size 
               + position % block_size
    """
    # 步骤 1: 计算每个 token 对应的 block_table 索引
    block_table_indices = (
        req_indices * self.max_num_blocks_per_req    # 行偏移
        + positions // self.block_size                # 列偏移（第几个块）
    )
    
    # 步骤 2: 查找物理块号
    block_numbers = self.block_table.np.ravel()[block_table_indices]
    
    # 步骤 3: 计算块内偏移
    block_offsets = positions % self.block_size
    
    # 步骤 4: 合成最终的 slot
    # slot = block_number * block_size + block_offset
    np.add(
        block_numbers * self.block_size,
        block_offsets,
        out=self.slot_mapping.np[:req_indices.shape[0]]
    )
```

#### 详细计算示例

假设 `block_size = 16`，当前 batch 有 3 个请求：

```
请求 0: prompt "Hello, my name is" → 5 个 token，分配了块 [42]
请求 1: prompt "The president of the United States is" → 8 个 token，分配了块 [7]
请求 2: prompt "The capital of France is" → 6 个 token，分配了块 [103]

block_table (CPU numpy):
  row 0: [42, 0, 0, ...]    (1 个块)
  row 1: [ 7, 0, 0, ...]    (1 个块)
  row 2: [103, 0, 0, ...]   (1 个块)

输入参数:
  req_indices = [0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1,  2, 2, 2, 2, 2, 2]
  positions   = [0, 1, 2, 3, 4,  0, 1, 2, 3, 4, 5, 6, 7,  0, 1, 2, 3, 4, 5]
```

**计算过程**:

```
步骤 1: block_table_indices
  对于请求 0, token 位置 0: 0 * K + 0//16 = 0   → block_table[0][0] = 42
  对于请求 0, token 位置 4: 0 * K + 4//16 = 0   → block_table[0][0] = 42
  对于请求 1, token 位置 0: 1 * K + 0//16 = K   → block_table[1][0] = 7
  对于请求 2, token 位置 0: 2 * K + 0//16 = 2K  → block_table[2][0] = 103

步骤 2: block_numbers = [42,42,42,42,42, 7,7,7,7,7,7,7,7, 103,103,103,103,103,103]

步骤 3: block_offsets  = [0,1,2,3,4, 0,1,2,3,4,5,6,7, 0,1,2,3,4,5]

步骤 4: slot_mapping   = [42*16+0, 42*16+1, ..., 42*16+4,
                           7*16+0,  7*16+1,  ...,  7*16+7,
                          103*16+0,103*16+1, ..., 103*16+5]
                        = [672, 673, 674, 675, 676,
                           112, 113, 114, 115, 116, 117, 118, 119,
                          1648,1649,1650,1651,1652,1653]
```

### 6.3 Decode 阶段的 Slot Mapping 更新

在 decode 阶段，每个请求每步仅生成 1 个新 token。假设请求 0 已经生成了 18 个 token（跨越了第 2 个块）：

```
block_table.np[0] = [42, 156, 0, ...]   (第 2 个块是新分配的 156)

新 token 的 position = 18
  block_table_index = 0 * K + 18 // 16 = 0 * K + 1 → block_table[0][1] = 156
  block_offset = 18 % 16 = 2
  slot = 156 * 16 + 2 = 2498
```

### 6.4 commit_slot_mapping —— 传输到 GPU

```python
def commit_slot_mapping(self, num_tokens: int) -> None:
    """将计算好的 slot_mapping 从 CPU 内存拷贝到 GPU 显存"""
    self.slot_mapping.copy_to_gpu(num_tokens)
```

---

## 7. Attention Metadata 的构建

> 源码：`vllm/v1/worker/gpu_model_runner.py`

### 7.1 GPUModelRunner 中的完整流程

在 `GPUModelRunner._execute_model_on_device()` 执行过程中：

```python
# 步骤 1: 计算 slot_mapping（CPU numpy 计算）
self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)

# 步骤 2: 将 slot_mapping 拷贝到 GPU
self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

# 步骤 3: 获取各 KV Cache Group 的 slot_mapping
slot_mappings_by_gid, slot_mappings_by_layer = self._get_slot_mappings(
    num_tokens_padded, num_reqs_padded, num_tokens_unpadded
)

# 步骤 4: 构建 CommonAttentionMetadata
cm_base = CommonAttentionMetadata(
    query_start_loc=self.query_start_loc.gpu[:num_reqs_padded + 1],
    seq_lens=self.seq_lens.gpu[:num_reqs_padded],
    num_reqs=num_reqs_padded,
    num_actual_tokens=num_tokens_padded,
    max_query_len=max_query_len,
    max_seq_len=max_seq_len,
    block_table_tensor=block_table_gid_0,   # GPU tensor
    slot_mapping=slot_mapping_gid_0,         # GPU tensor
)
```

### 7.2 _get_slot_mappings —— 构建多层级映射

```python
def _get_slot_mappings(self, num_tokens_padded, num_reqs_padded, 
                       num_tokens_unpadded):
    """
    构建两个维度的 slot_mapping 字典:
    1. slot_mappings_by_gid:   KV Cache Group → slot_mapping
    2. slot_mappings_by_layer: Layer Name → slot_mapping
    """
    def _get_slot_mapping(kv_cache_gid):
        blk_table = self.input_batch.block_table[kv_cache_gid]
        slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]
        # 未使用的 slot 填充 -1（用于 CUDA Graph padding）
        slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)
        return slot_mapping
    
    # 按 KV Cache Group ID 构建
    slot_mappings_by_gid = {
        gid: _get_slot_mapping(gid)
        for gid in range(len(self.kv_cache_config.kv_cache_groups))
    }
    
    # 按 Layer Name 构建（同一个 group 的 layer 共享 slot_mapping）
    slot_mappings_by_layer = {}
    for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
        for layer_name in group.layer_names:
            slot_mappings_by_layer[layer_name] = slot_mappings_by_gid[gid]
    
    return slot_mappings_by_gid, slot_mappings_by_layer
```

**关键设计**：padded token 的 slot 设为 `-1`，在 CUDA kernel 中会跳过这些位置，确保不会写入无效数据。

### 7.3 CommonAttentionMetadata 结构

```python
@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor      # 每个请求的 query 起始位置
    seq_lens: torch.Tensor             # 每个请求的序列长度
    num_reqs: int                      # 请求数量
    num_actual_tokens: int             # 实际 token 数（含 padding）
    max_query_len: int                 # 最大 query 长度
    max_seq_len: int                   # 最大序列长度
    block_table_tensor: torch.Tensor   # GPU 上的 block table（用于 attention 读取）
    slot_mapping: torch.Tensor         # GPU 上的 slot mapping（用于 KV cache 写入）
    causal: bool = True
```

### 7.4 FlashAttentionMetadata 的构建

在 `FlashAttentionMetadataBuilder.build()` 中，从 `CommonAttentionMetadata` 提取数据：

```python
def build(self, common_prefix_len, common_attn_metadata):
    slot_mapping = common_attn_metadata.slot_mapping
    block_table_tensor = common_attn_metadata.block_table_tensor
    
    attn_metadata = FlashAttentionMetadata(
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        query_start_loc=query_start_loc,
        max_seq_len=max_seq_len,
        seq_lens=seq_lens,
        block_table=block_table_tensor,     # 用于 attention 计算时读取 KV
        slot_mapping=slot_mapping,           # 用于写入 KV cache
        use_cascade=use_cascade,
        ...
    )
    return attn_metadata
```

注意 `FlashAttentionMetadata` 中同时包含：
- **`block_table`**：用于 attention 计算时 **读取** KV Cache
- **`slot_mapping`**：用于 **写入** KV Cache

---

## 8. Attention Backend 中 Slot Mapping 的使用

> 源码：`vllm/v1/attention/backends/flash_attn.py`

### 8.1 关键设计：KV Cache 写入与读取分离

在 vLLM V1 中，FlashAttention 的一个关键设计是：

```python
class FlashAttentionBackend(AttentionBackend):
    # KV Cache 的更新（写入）不在 forward() 中执行
    forward_includes_kv_cache_update: bool = False
```

这意味着 KV Cache 的写入是通过 **独立的 `do_kv_cache_update()` 方法** 完成的，而不是在 `forward()` 中。这种分离设计使得：
1. 可以使用 `torch.compile` 优化，避免编译图中的副作用
2. 可以独立控制 KV Cache 写入的时机

### 8.2 KV Cache 写入 —— do_kv_cache_update()

```python
class FlashAttentionImpl(AttentionImpl):
    def do_kv_cache_update(
        self, layer, key, value, kv_cache, slot_mapping
    ) -> None:
        """
        将 key 和 value 写入到 KV Cache 的指定 slot 位置。
        
        参数:
            layer: 注意力层（包含 k_scale, v_scale）
            key:   当前 token 的 key，shape = [num_tokens, num_heads, head_size]
            value: 当前 token 的 value，shape = [num_tokens, num_heads, head_size]
            kv_cache: KV Cache tensor，shape = [2, num_blocks, block_size, num_heads, head_size]
            slot_mapping: slot 映射，shape = [num_actual_tokens]
        """
        key_cache, value_cache = kv_cache.unbind(0)
        
        # 注意：key/value 可能有 padding，但 slot_mapping 没有 padding
        # reshape_and_cache_flash 使用 slot_mapping 的长度来确定实际 token 数
        reshape_and_cache_flash(
            key, value,
            key_cache, value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale, layer._v_scale,
        )
```

### 8.3 调用链 —— 从模型层到 KV Cache 写入

```python
# vllm/model_executor/models/extract_hidden_states.py
def unified_kv_cache_update(to_cache, layer_name):
    """
    在模型的每一层 attention 之前，通过 custom op 调用 KV Cache 更新。
    """
    forward_context = get_forward_context()
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
    
    # 从 ForwardContext 获取当前层的 slot_mapping
    slot_mapping = forward_context.slot_mapping  # dict[str, Tensor]
    layer_slot_mapping = slot_mapping.get(layer_name)
    
    if layer_slot_mapping is not None:
        # 调用 attention 实现的 KV Cache 更新方法
        attn_layer.impl.do_kv_cache_update(
            attn_layer, to_cache, kv_cache, layer_slot_mapping
        )
```

完整调用链：

```
Model.forward()
  → 每层 Attention:
    1. unified_kv_cache_update(key_value, "model.layers.0.self_attn")
       → ForwardContext.slot_mapping["model.layers.0.self_attn"]  # 获取该层的 slot_mapping
       → FlashAttentionImpl.do_kv_cache_update()
         → reshape_and_cache_flash(key, value, k_cache, v_cache, slot_mapping, ...)
           → CUDA Kernel: reshape_and_cache_flash_kernel
    
    2. FlashAttentionImpl.forward(query, key, value, kv_cache, attn_metadata)
       → flash_attn_varlen_func(q, k_cache, v_cache, block_table=attn_metadata.block_table, ...)
       # 使用 block_table 从 KV Cache 读取，进行注意力计算
```

### 8.4 Attention 计算 —— forward()

在 `forward()` 中，**不使用 slot_mapping**，而是使用 **block_table**：

```python
def forward(self, layer, query, key, value, kv_cache, attn_metadata):
    key_cache, value_cache = kv_cache.unbind(0)
    
    # 使用 FlashAttention 的 varlen 接口
    # block_table 告诉 FlashAttention 每个请求的 KV 存储在哪些物理块中
    flash_attn_varlen_func(
        q=query[:num_actual_tokens],
        k=key_cache,                          # 整个 KV cache（paged）
        v=value_cache,                        # 整个 KV cache（paged）
        cu_seqlens_q=attn_metadata.query_start_loc,
        seqused_k=attn_metadata.seq_lens,     # 每个请求实际使用的 KV 长度
        block_table=attn_metadata.block_table, # 页表！指导如何在 paged cache 中读取
        ...
    )
```

**两种索引机制的区别**：

| 机制 | 用途 | 粒度 | 使用者 |
|------|------|------|--------|
| **slot_mapping** | KV Cache **写入** | token 级别（精确到块内偏移） | `reshape_and_cache` kernel |
| **block_table** | KV Cache **读取** | block 级别（整块读取） | FlashAttention kernel |

---

## 9. CUDA Kernel：reshape_and_cache_flash

> 源码：`csrc/cache_kernels.cu`

这是 slot_mapping 最终被消费的地方——CUDA kernel。

### 9.1 Kernel 实现

```cuda
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,            // [num_blocks, block_size, num_heads, head_size]
    cache_t* __restrict__ value_cache,          // [num_blocks, block_size, num_heads, head_size]
    const int64_t* __restrict__ slot_mapping,   // [num_actual_tokens]
    ...) 
{
    // 每个 CUDA block 处理一个 token
    const int64_t token_idx = blockIdx.x;
    
    // 从 slot_mapping 获取该 token 的目标 slot
    const int64_t slot_idx = slot_mapping[token_idx];
    
    // slot_idx == -1 表示 padding token，跳过
    if (slot_idx < 0) {
        return;
    }
    
    // 从 slot 反算出物理块索引和块内偏移
    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;
    
    // 计算源地址（输入 key/value）和目标地址（cache）
    const scalar_t* key_src = key + token_idx * key_stride;
    const scalar_t* value_src = value + token_idx * value_stride;
    
    cache_t* key_dst = key_cache + block_idx * block_stride + block_offset * page_stride;
    cache_t* value_dst = value_cache + block_idx * block_stride + block_offset * page_stride;
    
    // 向量化拷贝 key 和 value 到 cache
    // 支持 FP8 量化 (通过 CopyWithScaleOp)
    vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, n_elems, ...);
    vectorize_with_alignment<VEC_SIZE>(value_src, value_dst, n_elems, ...);
}
```

### 9.2 Host 端调用

```cuda
void reshape_and_cache_flash(...) {
    // 关键：使用 slot_mapping 的大小（而非 key 的大小）作为 grid 维度
    // 因为 key 可能有 padding，但 slot_mapping 只包含实际的 token
    int num_tokens = slot_mapping.size(0);
    
    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * head_size, 512));
    
    reshape_and_cache_flash_kernel<<<grid, block, 0, stream>>>(
        key.data_ptr(), value.data_ptr(),
        key_cache.data_ptr(), value_cache.data_ptr(),
        slot_mapping.data_ptr(),
        ...
    );
}
```

### 9.3 图解 Kernel 执行

```
slot_mapping = [672, 673, 674, 675, 676, 112, 113, ..., -1, -1]
                 ↓    ↓    ↓    ↓    ↓    ↓    ↓         ↓   ↓
CUDA block 0:  token 0 → slot 672
                block_idx = 672/16 = 42
                block_off = 672%16 = 0
                → key_cache[42][0] = key[0]

CUDA block 1:  token 1 → slot 673
                block_idx = 673/16 = 42
                block_off = 673%16 = 1
                → key_cache[42][1] = key[1]

...

CUDA block N:  token N → slot -1
                slot_idx < 0 → return (跳过 padding)
```

---

## 10. 完整数据流总结

以 `basic.py` 的 8 个 prompt 为例，以下是一次完整推理 step 中 KV Cache 相关的数据流：

```
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 1: Scheduler 分配块                                             │
│                                                                     │
│ KVCacheManager.allocate_slots(request, num_new_tokens)              │
│   ↓                                                                 │
│ BlockPool.get_new_blocks(num_blocks)                                │
│   → 从 FreeKVCacheBlockQueue 弹出空闲块                              │
│   → 设置 ref_cnt = 1                                                │
│   → 返回 [KVCacheBlock(42), KVCacheBlock(7), ...]                   │
│   ↓                                                                 │
│ SchedulerOutput.new_block_ids = {                                   │
│   "req_0": ([42], ),     # 第 0 个 kv_cache_group 的块               │
│   "req_1": ([7], ),                                                 │
│   ...                                                               │
│ }                                                                   │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 2: Worker 更新 Block Table（CPU）                                │
│                                                                     │
│ GPUModelRunner 收到 SchedulerOutput:                                 │
│   BlockTable.append_row([42], row_idx=0)    # 请求 0 的 block table  │
│   BlockTable.append_row([7],  row_idx=1)    # 请求 1 的 block table  │
│   ...                                                               │
│                                                                     │
│ block_table.np (CPU numpy):                                         │
│   [[ 42,  0,  0, ...],     # 请求 0                                 │
│    [  7,  0,  0, ...],     # 请求 1                                 │
│    [103,  0,  0, ...],     # 请求 2                                 │
│    [ 89,  0,  0, ...],     # 请求 3                                 │
│    ...]                                                             │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 3: 计算 Slot Mapping（CPU numpy 向量化计算）                      │
│                                                                     │
│ BlockTable.compute_slot_mapping(req_indices, positions)              │
│                                                                     │
│ 公式: slot = block_table[req][pos // block_size] * block_size       │
│              + pos % block_size                                     │
│                                                                     │
│ 输入:                                                               │
│   req_indices = [0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2, ...]   │
│   positions   = [0,1,2,3,4, 0,1,2,3,4,5,6,7, 0,1,2,3,4,5, ...]   │
│                                                                     │
│ 输出:                                                               │
│   slot_mapping.np = [672,673,674,675,676,                           │
│                      112,113,114,115,116,117,118,119,               │
│                     1648,1649,1650,1651,1652,1653, ...]             │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 4: CPU → GPU 传输                                               │
│                                                                     │
│ BlockTable.commit_block_table(num_reqs=8)   # block_table 拷贝到 GPU│
│ BlockTable.commit_slot_mapping(num_tokens)   # slot_mapping 拷贝到GPU│
│                                                                     │
│ 填充 padding: slot_mapping[num_actual:num_padded] = -1              │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 5: 构建 Attention Metadata                                      │
│                                                                     │
│ _get_slot_mappings()                                                │
│   → slot_mappings_by_gid:   {0: GPU Tensor}                        │
│   → slot_mappings_by_layer: {"model.layers.0.self_attn": GPU Tensor,│
│                               "model.layers.1.self_attn": GPU Tensor,│
│                               ...}                                  │
│                                                                     │
│ CommonAttentionMetadata(                                            │
│   block_table_tensor = GPU Tensor,  # 用于 attention 读取            │
│   slot_mapping = GPU Tensor,        # 用于 KV cache 写入             │
│ )                                                                   │
│   ↓                                                                 │
│ FlashAttentionMetadataBuilder.build()                               │
│   → FlashAttentionMetadata(                                         │
│       block_table = block_table_tensor,                             │
│       slot_mapping = slot_mapping,                                  │
│     )                                                               │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 6: 模型前向传播 —— KV Cache 写入（每层执行）                       │
│                                                                     │
│ unified_kv_cache_update(key_value, "model.layers.0.self_attn")      │
│   → ForwardContext.slot_mapping["model.layers.0.self_attn"]         │
│   → FlashAttentionImpl.do_kv_cache_update(key, value, cache, slot)  │
│     → reshape_and_cache_flash(key, value, k_cache, v_cache, slot)   │
│       → CUDA Kernel:                                                │
│           对每个 token:                                               │
│             slot_idx = slot_mapping[token_idx]                      │
│             if slot_idx < 0: return  // 跳过 padding                 │
│             block_idx = slot_idx / block_size                       │
│             block_off = slot_idx % block_size                       │
│             k_cache[block_idx][block_off] = key[token_idx]          │
│             v_cache[block_idx][block_off] = value[token_idx]        │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 7: 模型前向传播 —— Attention 计算（每层执行）                      │
│                                                                     │
│ FlashAttentionImpl.forward(query, key, value, kv_cache, metadata)   │
│   → flash_attn_varlen_func(                                        │
│       q = query,                                                    │
│       k = key_cache,         # 整个 paged KV cache                   │
│       v = value_cache,       # 整个 paged KV cache                   │
│       block_table = metadata.block_table,  # 页表                    │
│       seqused_k = metadata.seq_lens,       # 每个请求的 KV 长度      │
│     )                                                               │
│   # FlashAttention kernel 根据 block_table 从 paged cache 中        │
│   # 读取正确的 KV block，完成注意力计算                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 核心要点回顾

1. **分页管理**：KV Cache 被分割为固定大小的 block，通过 BlockPool 统一管理，避免了内存碎片。

2. **两级索引**：
   - **Block Table**（粗粒度）：请求 → 物理块列表，用于 FlashAttention 读取
   - **Slot Mapping**（细粒度）：token → 物理 slot，用于 KV Cache 写入

3. **CPU-GPU 协同**：Block Table 和 Slot Mapping 的计算在 CPU 上用 numpy 高效完成，然后通过 `CpuGpuBuffer` 同步到 GPU。

4. **读写分离**：在 V1 引擎中，KV Cache 的写入（`do_kv_cache_update`）与 Attention 计算（`forward`）分离，便于使用 `torch.compile` 优化。

5. **Padding 处理**：`slot_mapping` 中的 `-1` 值在 CUDA kernel 中被跳过，支持 CUDA Graph 的固定大小输入。

6. **Prefix Caching**：填满的 block 会被缓存（哈希→块映射），后续相同前缀的请求可复用，避免重复计算。
