# vLLM 自研 CUDA 兼容硬件适配 Checklist

## 1. 文档目的

这份 checklist 面向硬件、驱动、运行时、PyTorch 后端、通信栈团队，目标不是解释 vLLM 原理，而是回答一个更实际的问题：

> 如果要让自研 CUDA 兼容硬件稳定支撑 vLLM 的异步调度和 PD 分离，哪些能力必须具备，应该怎么验收，推进顺序是什么。

建议把它作为联调门禁文档使用。

---

## 2. 总体结论

### 必须先统一的认知

- “CUDA 兼容”不等于“vLLM 高性能特性可用”。
- `async scheduling` 依赖的不只是 kernel 可执行，还依赖 stream、event、pinned memory、async copy、无隐式同步。
- PD 分离不只是两卡通信，而是“两套 vLLM 实例 + KV connector + lookup buffer + 异步 load/save”。
- 建议按阶段推进，不要一开始同时打开 graph、async scheduling、TP/PP、PD、spec decode。

### 推荐推进顺序

1. 单卡 eager 跑通基础 generate。
2. 单卡验证 async output / async scheduling。
3. 多卡验证 TP/PP 与 `torch.distributed`。
4. 跑通一个最简单的 KV connector。
5. 跑通 PD 分离。
6. 最后补 graph capture、spec decode、复杂 multimodal 路径。

---

## 3. 阶段门禁 Checklist

## Phase 0 - 最低可运行门槛

### 目标

让 vLLM 在你们设备上以 eager 模式完成最基本的离线生成。

### 必备能力

- PyTorch 设备张量创建正常。
- 基础算子可运行。
- 显存分配、释放、OOM 报错可用。
- `to(device)`、`to(cpu)` 行为正确。
- 单卡 `LLM.generate()` 能完整结束。

### 验收项

- 能运行 `examples/offline_inference/basic/basic.py` 的等价最小样例。
- `enforce_eager=True` 下无崩溃、无 silent hang。
- 输出文本正确返回到 CPU。
- 失败时能看到明确异常，而不是卡死。

### 风险信号

- 首次 forward 可以跑，第二次挂死。
- 输出偶发为空或 token 数错误。
- CPU/GPU 拷贝偶发阻塞，进程不退出。

---

## Phase 1 - 异步执行基础能力

### 目标

确认设备和 runtime 具备支持 async scheduling 的最小异步语义。

### 必备能力

- 独立 stream 能创建和执行。
- event 能正确记录和等待。
- `non_blocking=True` 的 H2D / D2H copy 真实异步。
- pinned memory 有效，不退化成普通 host memory。
- 没有明显隐式同步把异步路径变成串行。

### 验收项

- 能创建并使用独立 copy stream。
- CPU 在发起 GPU 执行后可以继续准备下一批数据。
- 输出回传线程或 future 不会因为 copy 同步而把主路径卡住。
- 开启 `async_scheduling=True` 后程序稳定结束。

### 推荐验证方式

- 对照 `vllm/v1/worker/gpu_model_runner.py` 中 async output copy stream 相关路径。
- 对照 `vllm/v1/executor/uniproc_executor.py` / `multiproc_executor.py` 中 `max_concurrent_batches > 1` 的路径。
- 观察 `async_scheduling=False` 与 `True` 时行为差异，至少确认功能不回退。

### 风险信号

- `non_blocking=True` 但 CPU 时间线没有任何重叠。
- 打开 async scheduling 后吞吐不升反降很多。
- 打开后偶发重复 token、少 token、请求乱序。

---

## Phase 2 - PyTorch 后端与 runtime 完整性

### 目标

让 vLLM 常见执行路径能稳定依赖你们的 PyTorch 设备后端。

### 必备能力

- 张量生命周期稳定。
- stream / event / memory copy 接口语义与 CUDA 路径一致。
- Host/Device 内存模型行为可预期。
- profiler、错误码、异常传播可用。
- 必要时支持 page/block 粒度内存拷贝。

### 验收项

- 高频小 batch、多轮 step 下不泄漏、不 hang。
- OOM、非法访问、copy 失败能暴露到 Python 层。
- 多线程输出回传不会触发竞态崩溃。

### 风险信号

- 只有单线程/单 stream 场景稳定。
- 报错信息缺失，只有进程退出或 device reset。
- 长时间运行后出现随机错误。

---

## Phase 3 - Kernel 与编译栈

### 目标

补齐 vLLM 高性能路径所需的 kernel 基础。

### 必备能力

- attention backend 可用。
- sampler 相关 kernel 可用。
- 至少一种高性能 kernel 生态可维护：Triton 兼容或自研替代。
- 对需要的 custom op / extension 有稳定承载方式。

### 验收项

- 模型 forward 与 sampling 都可稳定执行。
- 长上下文场景不过早退化。
- kernel 失败时能被 runtime 捕获并定位。

### 风险信号

- 基础 matmul 能跑，但 attention/sampling 路径大量 fallback 到 CPU。
- kernel 行为正确但性能极差，导致 async scheduling 无收益。

---

## Phase 4 - 分布式通信栈

### 目标

让 vLLM 的 TP/PP/DP 与 KV 传输具备基础通信能力。

### 必备能力

- `torch.distributed` 或等价兼容层。
- point-to-point 与 collective 通信能力。
- 多进程 / 多卡初始化稳定。
- executor / worker 进程间通信不会异常死锁。

### 验收项

- `tensor_parallel_size > 1` 能稳定初始化。
- worker 进程 READY / shutdown 流程稳定。
- 多卡场景下请求可正确结束，输出不丢失。

### 风险信号

- 单卡稳定，多卡启动就 hang。
- 某一 worker 异常退出后其他进程无法感知。
- shutdown 阶段大量僵尸进程或 socket 未释放。

---

## Phase 5 - KV Connector 与 PD 分离基础能力

### 目标

支撑 Prefill / Decode 分离所需的 KV 传输路径。

### 必备能力

- 至少一个可工作的 connector。
- 支持 scheduler-side metadata 与 worker-side load/save 配合。
- 有可用的数据面传输 KV blocks。
- 允许请求乱序到达时仍能正确匹配 KV。

### 优先建议

- 第一阶段先做最简单 connector，优先验证功能正确性。
- 不要一开始就做最复杂、高性能、跨机版本。

### 验收项

- prefill 实例生成的 KV 能被 decode 实例加载。
- decode 侧能识别命中的远端 KV。
- KV 缺失时要么正确 fallback/recompute，要么明确 fail。
- Connector 统计、错误信息、完成通知可见。

### 风险信号

- Prefill 完成但 decode 永远等不到 KV。
- 请求顺序轻微变化就 KV 命中错乱。
- 远端 KV 已完成发送，但 decode 侧仍重复本地 prefill。

---

## Phase 6 - Graph 与进阶能力

### 目标

在功能稳定后再追求主线路径性能。

### 必备能力

- graph capture / replay 稳定。
- graph 与 async scheduling 不互相破坏。
- graph 模式下错误可定位、可回退。

### 验收项

- 关闭 graph 稳定，开启 graph 也稳定。
- graph 模式下吞吐提升符合预期。
- graph 失败时能自动回落 eager 或明确报错。

### 风险信号

- eager 正常，graph 模式偶发错误或结果不一致。
- graph 打开后 async overlap 消失。

---

## 4. 专门针对 async scheduling 的验收 Checklist

### 功能正确性

- `async_scheduling=True` 后所有请求都能正常结束。
- 无重复 token、漏 token、输出顺序错乱。
- preemption / abort / reset cache 后状态仍一致。

### 异步真实性

- scheduler 与 worker 准备工作能和 GPU 执行重叠。
- copy stream 与 compute stream 能并存。
- future/队列机制不会退化成全同步等待。

### 稳定性

- 连续多轮请求无 hang。
- 压测下无随机失败。
- 异常路径可以恢复或至少可观测。

### 建议观察指标

- step 级延迟是否出现明显空泡。
- H2D / D2H copy 时间是否与 compute 重叠。
- GPU utilization 是否较同步路径更稳定。

---

## 5. 专门针对 PD 分离的验收 Checklist

### 功能正确性

- 两个 vLLM 实例可独立启动。
- prefill 侧能保存 KV，decode 侧能加载 KV。
- 对相同 prompt，PD 路径结果与非 PD 路径基本一致。

### 匹配正确性

- 请求乱序时 lookup buffer 仍能正确匹配 KV。
- 部分命中时行为可解释。
- 请求结束后 connector 能正确回收状态。

### 故障处理

- KV load failure 能按策略处理：`recompute` 或 `fail`。
- 对端退出、链路中断、超时等情况能被上报。
- decode 侧不会无限等待不可达 KV。

### 性能观察

- prefill 和 decode 是否真正解耦。
- connector 是否成为新瓶颈。
- 异步 load/save 是否和前向有重叠。

---

## 6. 建议内部团队分工

### 硬件 / 驱动团队

- stream / event / async memcpy / graph 基础能力。
- 内存一致性与错误上报。

### PyTorch 后端团队

- 设备 backend、张量与 copy 语义。
- `torch.distributed` 与多进程兼容。

### Kernel 团队

- attention、sampling、关键 custom op。
- Triton 兼容或替代实现。

### 通信团队

- 多卡通信。
- KV transfer 数据面。
- connector 所需底层传输接口。

### 推理框架团队

- vLLM 平台适配。
- feature gate。
- 回退策略与验证脚本。

---

## 7. 最后建议

- 先做“可关闭的兼容”，不要一开始追求“全部特性默认开启”。
- 对每一项高阶特性建立单独门禁：eager、async scheduling、TP/PP、PD、graph。
- 任何时候只要发现隐式同步、随机 hang、跨实例 KV 错配，都应先回退到上一个稳定阶段，而不是继续叠加特性。

一句话总结：

> 让 vLLM 在自研 CUDA 兼容硬件上稳定支持 async scheduling 和 PD 分离，本质上不是“补几个 kernel”，而是补齐一整套异步执行、内存语义、通信、KV 传输和回退机制。
