<!--
  本文档为 TensorRT-LLM 官方 PyTorch Backend Scheduler 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/torch/scheduler.md
-->

# 调度器（Scheduler）

TensorRT LLM PyTorch 后端采用飞行中批处理（inflight batching）：在每个 LLM 步骤动态进行批处理和调度。
调度器被调用以确定当前步骤调度哪些请求。

## 调度器介绍

有两种调度器：

- `CapacityScheduler`（容量调度器）：此调度器决定是否为每个活动请求分配资源。
它考虑 KV cache 容量和其他资源（如果适用）。
`CapacityScheduler` 的输入包括所有需要处理的活动请求。
主要输出是 `fitting_requests`，表示当前步骤为其预留资源的请求。
另一个输出是 `paused_requests`，支持 C++ 运行时中的请求暂停。
- `MicroBatchScheduler`（微批调度器）：此调度器从 `CapacityScheduler` 选出的 `fitting_requests` 中选择一些请求。
另一个输入是 `inflight_request_ids`，支持 C++ 运行时中的流水线并行或重叠执行。
由于 PyTorch 流程不支持流水线并行，`inflight_request_ids` 是空集。
输出是 `context_requests` 和 `generation_requests`，即被调度的上下文和生成请求。
不在这些列表中的请求不会被选中进行模型前向。

> 💡 **AI Infra 视角**：对照 arch_overview.md 复习：容量调度（能不能进）→ 微批调度（谁先跑）。注意 `paused_requests`（暂停）和 `inflight_request_ids`（流水线并行用）是为 C++ 运行时预留的接口——**PyTorch 流程用不到，但接口保持兼容**。设计启示：**为未来能力预留接口是框架设计的一部分**（即使当前实现为空）。

`SimpleScheduler` 组合这两个调度器，先使用 `CapacityScheduler` 再使用 `MicroBatchScheduler`，得到最终调度结果。
`SimpleScheduler` 的输入包括 `active_requests` 和 `inflight_request_ids`，输出是 `context_requests`、`generation_requests` 和 `paused_requests`。

## 自定义你自己的调度器

要定制调度器或批处理机制，通过继承各自的类来实现你自己的 `CapacityScheduler` 和 `MicroBatchScheduler`。
如果不需要两步调度，直接继承 `RequestScheduler` 并实现 `schedule_request`。

`CapacityScheduler` 实现的一个例子是 `GuaranteedNoEvictScheduler` 类，见 [scheduler.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/scheduler.py)。
这个类在 C++ 绑定的 `CapacityScheduler` 之前使用，最初基于 Python 调度器。
它继承 `CapacityScheduler` 并实现自己的 `schedule_request` 方法。
该方法处理所有 `active_requests`，尝试调度更多能装进 KV cache 的请求。
资源估计应与 `kv_cache_manager` 中的资源分配和释放对齐。

代码片段：

```python
class GuaranteedNoEvictScheduler(CapacityScheduler):
    # 只调度 no_schedule_until_state <= state < no_schedule_after_state 的请求
    no_schedule_until_state = LlmRequestState.CONTEXT_INIT
    no_schedule_after_state = LlmRequestState.GENERATION_COMPLETE

    def __init__(self, max_num_requests: int, kv_cache_manager):
        super(GuaranteedNoEvictScheduler, self).__init__()
        self.max_num_requests = max_num_requests
        self.kv_cache_manager = kv_cache_manager

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest]]:
        scheduled_requests = []
        pending_requests = []
        reserved_blocks = 0
        max_blocks = self.kv_cache_manager.get_max_resource_count()
        for request in active_requests:
            req_state = request.state
            # 如果请求还不能调度或不应再被调度，跳过
            if req_state.value < self.no_schedule_until_state.value or req_state.value >= self.no_schedule_after_state.value:
                continue

            if len(scheduled_requests
                   ) >= self.max_num_requests or reserved_blocks >= max_blocks:
                break
            elif req_state == LlmRequestState.GENERATION_IN_PROGRESS or req_state == LlmRequestState.GENERATION_TO_COMPLETE:
                scheduled_requests.append(request)
                reserved_blocks += self.kv_cache_manager.get_needed_resource_to_completion(
                    request)
            else:
                pending_requests.append(request)

        available_blocks = max_blocks - reserved_blocks
        for request in pending_requests:
            req_state = request.state
            if len(scheduled_requests) >= self.max_num_requests:
                break
            elif req_state == LlmRequestState.CONTEXT_INIT:
                needed_blocks = self.kv_cache_manager.get_needed_resource_to_completion(
                    request)
                if needed_blocks <= available_blocks:
                    scheduled_requests.append(request)
                    available_blocks -= needed_blocks
                elif needed_blocks > available_blocks:
                    # 如果一个请求调度失败，跳出
                    break

        assert len(scheduled_requests) > 0, (
            "no pending request can get enough resource to complete, "
            "please increase KV cache pool size.")
        return scheduled_requests, []
```

> 💡 **AI Infra 视角**：读懂这个类 = 读懂调度器的心脏。逻辑拆解：
> 1. **先收编生成中的请求**（GENERATION_IN_PROGRESS）：它们在跑，不能踢（保证不驱逐的名字由来），按 `get_needed_resource_to_completion` 预留到完成所需的全部块；
> 2. **剩下的块给新请求**（CONTEXT_INIT）：块够就进，不够就 break——注意这个 break 的含义：**如果有个大请求装不下，后面更小的也不试了**（这个实现按序扫描，没做最优排序——简单但够用）；
> 3. **断言"总得有请求能跑"**：如果所有新请求都装不下，直接报错提示"增大 KV cache 池"——**宁可报错也不空转**。
> 这个"保证不驱逐"策略就是 perf-benchmarking.md 输出里 `Scheduling Policy: Guaranteed No Evict` 的实现。

实现自己的调度器后，将其集成到 `PyExecutor` 中。
对于 PyTorch 后端，代码在 [py_executor_creator.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/pyexecutor/py_executor_creator.py)。
在 `create_py_executor` 函数中，有两行创建 `CapacityScheduler`：

```python
    capacity_scheduler = BindCapacityScheduler(max_num_requests,
                                               kv_cache_manager.impl)
```

`MicroBatchScheduler` 可以做类似的调整。这允许 `PyExecutor` 使用你的自定义调度逻辑执行。

> 💡 **AI Infra 视角**：整篇的工程启示：**调度器是可插拔的**——C++ 绑定（性能）和 Python 实现（可定制）走同一接口。你在生产里想实验新调度策略（如 SLO 感知调度、公平队列），改这一个文件就行。**"用接口隔离性能实现和策略实现"**是推理引擎留给社区扩展空间的经典手法。
