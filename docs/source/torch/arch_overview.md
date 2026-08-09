<!--
  本文档为 TensorRT-LLM 官方 PyTorch Backend Architecture 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/torch/arch_overview.md
-->

# 架构总览

TensorRT LLM 是一个为大型语言模型（LLM）推理创建优化解决方案的工具包。
除了 TensorRT，PyTorch 也可以作为 TensorRT-LLM 的后端。本文档提供 PyTorch 后端架构的总览。

> 💡 **AI Infra 视角**：这篇是"PyTorch 后端地图"，与 developer-guide/overview.md（全系统架构）配套。**学习顺序建议**：已经读过架构总览的读者，这篇会带你把组件从"概念"落到"代码文件"——每个组件都标注了源码位置，读代码时对照本文是最好的打开方式。

## 顶层 API

PyTorch 后端的接口是 `tensorrt_llm.LLM`。

```python
from tensorrt_llm import LLM
llm = LLM(model=<path_to_llama_from_hf>)
```

`LLM` 还管理输入的 tokenization 和 detokenization 过程。

## PyExecutor

`PyExecutor` 的关键组件包括：

- Model Engine（模型引擎）：持有语言模型，高效支持单步模型前向。
- Decoder（解码器）：基于 Model Engine 输出生成输出 token。
- Scheduler（调度器）：决定是否为请求分配资源（如 KV Cache），以及当前步骤是否为每个请求运行前向。

PyExecutor 的单步流程包括：

- 从请求队列获取新请求（如果有）。
- 调度一些请求。
- 为已调度的请求运行模型前向。
- 使用已调度请求的模型前向输出来运行解码器。
- 为每个请求添加输出 token，并处理已完成的请求。

> 💡 **AI Infra 视角**：这就是架构总览那 5 步循环的"PyTorch 版"——对照记忆：取请求 → 调度 → forward → 采样（decoder）→ 输出处理。**本学习路线到这里，你已经在三个文档里见到同一个流程了**（developer-guide/overview.md、本文），说明这就是 PyExecutor 的全部——理解它，PyTorch 后端对你就不再是黑盒。

## Model Engine

`PyExecutor` 的核心组件是 `ModelEngine`，负责在 GPU 上高效执行模型的前向传播。
`ModelEngine` 的关键方法是 `forward`，处理前向计算。
对于 PyTorch 后端，派生类是 `PyTorchModelEngine`，声明在 [model_engine.py](../../../tensorrt_llm/_torch/pyexecutor/model_engine.py)。

## Decoder

Decoder 基于 Model Engine 输出生成输出 token，支持贪心搜索解码。

> 💡 **AI Infra 视角**：注意 Decoder 在这里只是"贪心解码"的抽象——完整的采样（greedy/top-k/top-p/beam）在 sampling.md 讲过的 Torch Sampler 里（`tensorrt_llm/_torch/sampler`）。**架构分层：ModelEngine 只负责"算出 logits"，怎么从 logits 选 token 是 Decoder/Sampler 的事**。

## Scheduler

调度器分两步运行：

1. CapacityScheduler（容量调度器）：判断是否有足够的资源容纳一个请求。
2. MicroBatchScheduler（微批调度器）：为模型前向选择一些请求。

CapacityScheduler 和 MicroBatchScheduler 目前都使用 C++ 绑定。
不过，由于接口在 Python 中实现，因此可以进行定制。
文档 [scheduler.md](./scheduler.md) 解释了如何实现自定义调度逻辑。

> 💡 **AI Infra 视角**：两步调度的分工（重要设计）：
> - **容量检查**：这个请求要的 KV cache/显存够不够？不够就别进来（否则跑到一半 OOM）——"能不能进"；
> - **微批选择**：在能进的请求里，这一步选谁上 GPU（token 预算、优先级）——"谁先跑"。
> 两层分离的好处：容量逻辑（C++ 快，遍历资源表）和策略逻辑（可 Python 定制）互不干扰。**"策略与机制分离"是调度器设计的经典模式**（类比 Linux 的调度器：机制在内核、策略可调）。

## ResourceManager

`ResourceManager` 帮助分配和管理单个请求推理可能需要的资源。
它是继承自 `BaseResourceManager` 的对象的容器，每个对象管理一种特定类型的资源。
`BaseResourceManager` 有三个重要接口：

- `prepare_resources`：PyExecutor 中每一步模型前向之前为当前 batch 调用。
- `update_resources`：每一步结束时为当前 batch 调用。
- `free_resources`：每个请求完成时调用。

Transformer 模型的一个关键资源是 KV Cache。KV Cache 的 `BaseResourceManager` 是 `KVCacheManager`。

> 💡 **AI Infra 视角**：三个接口对应 KV cache 的生命周期：
> - **prepare**（前向前）：给这批请求分配/确认 KV 块——"准备好弹药"；
> - **update**（前向后）：把新算的 K/V 写进缓存、更新块状态——"记账"；
> - **free**（请求结束）：释放块回池子——"还弹药"。
> **资源管理器是"显存财务系统"**：prepare/update/free 的时机错乱会导致显存泄漏或重复分配——这也是并发 bug 的高发区。

### KVCacheManager

目前，KVCacheManager 使用 C++ 绑定。但 Python 定制是可能的，因为其接口在 Python 中实现。
文档 [kv_cache_manager.md](./kv_cache_manager.md) 详细介绍了如何实现自定义 KVCacheManager。
