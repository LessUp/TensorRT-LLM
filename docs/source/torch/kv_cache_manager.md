<!--
  本文档为 TensorRT-LLM 官方 PyTorch Backend KV Cache Manager 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/torch/kv_cache_manager.md
-->

# KV Cache Manager

在基于 Transformer 的模型中，KV（Key-Value）Cache 是用于优化解码效率的机制，特别是在自回归生成任务中。
由于 KV Cache 需要显存来存储，它也是一种重要资源。
在 TensorRT LLM 中，KV Cache 由 `KVCacheManager` 管理。

KVCacheManager 实现的细节参见 [KV Cache 管理](../legacy/advanced/kv-cache-management.md)。

> 💡 **AI Infra 视角**：回顾 kvcache.md（特性层）和 arch_overview.md（资源管理层），这篇是**接口层**——回答"KVCacheManager 暴露哪些方法给调度器和模型引擎用"。这些接口的名字会反复出现在源码里，值得逐个记住。

## KV Cache Manager 介绍

`KVCacheManager` 是一种资源管理器，继承自 `BaseResourceManager`。
因此，它实现了 `BaseResourceManager` 声明的接口。

注意：随着项目演进，这些接口可能会变化。

## 接口

`BaseResourceManager` 的接口包括：

- **prepare_resources**：在 `PyExecutor` 中每一步模型前向之前为当前 batch 调用。
  在 `KVCacheManager` 中，这涉及分配 KV Cache 显存。分配因请求类型而异。
  对于首次进入上下文阶段的请求，需要为整个上下文分配 KV Cache。
  对于已处于生成阶段的请求，为即将到来的步骤分配 KV Cache。
  如果 KV Cache 以块组织且块内有空闲空间，则可能不发生实际分配。
- **update_resources**：在每一步结束时为当前 batch 调用，用于更新已分配资源。
  对于 KV Cache，更新可能不是必需的，因此该函数当前不执行任何操作。
  如果在 Python 中支持 KV Cache 复用，诸如 KV Cache Radix Tree 管理的更新会在这里进行。
- **free_resources**：请求完成时调用，释放为该请求分配的资源。
  对于 KV Cache，如果未启用复用，应回收该请求使用的 KV Cache 显存。
  在 C++ 绑定实现中，这可能涉及调用绑定的 `remove_sequence` 方法来释放与该请求相关的 KV Cache 显存。

> 💡 **AI Infra 视角**：三个接口与"请求生命周期"对齐：请求进（prepare）→ 请求跑（update）→ 请求走（free）。注意 prepare 的细节：**"块内有空闲就不实际分配"**——paged 结构的按需分配；还有 context 请求一次分配整段（它要一次算完 prompt），decode 请求只分配下一步的。**分配粒度与请求阶段相关**，这是 KV cache 管理和显存规划的核心逻辑。

还有两个为 `CapacityScheduler` 设计的接口：

- **get_max_resource_count**：查询可用资源的最大数量。对于 `KVCacheManager`，通常是最大 KV Cache 块数。
- **get_needed_resource_to_completion**：计算单个请求完成所需的资源。
  `CapacityScheduler` 用它求和总资源需求，判断是否能容纳新请求。

> 💡 **AI Infra 视角**：这两个接口就是 scheduler.md 里 `GuaranteedNoEvictScheduler` 用到的 `max_blocks` 和 `get_needed_resource_to_completion`——**调度器判断"能不能进"全靠这两个接口**。注意"to completion"：为生成中的请求预留**到生成结束**的块（保证不驱逐的前提），而不是只预留下一步。**资源预留的"保守度"是调度策略的核心参数**：预留太狠 → 并发低；预留太松 → 中途 OOM。

除了 `BaseResourceManager` 接口，`KVCacheManager` 还有与所用 `ModelEngine` 相关的接口。
对于 `PyTorchModelEngine`，常见接口包括：

- **get_batch_cache_indices**：接受 `LlmRequest` 列表，返回 `Dict[List[int]]`，表示每个请求的块 ID。
- **get_buffers**：返回给定层的 KV Cache 池缓冲区，供注意力后端使用。形状可能是 [`num_blocks`, 2, `num_tokens_per_block`, `num_kv_heads`, `head_dim`]。
- **get_num_free_blocks**：返回可用于分配的空闲块数。

> 💡 **AI Infra 视角**：这些是**注意力 kernel 的"取数接口"**：`get_buffers` 给出块池的显存布局（形状 = [块数, K/V 两半, 每块 token 数, KV 头数, 头维度]——对照 attention.md 讲的 paged KV cache），`get_batch_cache_indices` 告诉 kernel "这批请求各用哪些块"。**kernel 拿到的就是这两个信息 + 块索引，就能完成注意力的读取**——这是 paged attention 的接口层全景。

还有用于预热 `PyTorchModelEngine` 的接口，特别是使用 CUDA graphs 时：

- **add_padding_request**：向 KV Cache 添加一个上下文长度为 1 的序列作为预热请求。
  如果你的概念验证（proof of concept）不使用 CUDA Graph，这是可选的。

## 自定义 KV Cache Manager

要自定义 `KVCacheManager`，实现所有必要接口。
然后将其集成到 `PyExecutor` 中。对于 PyTorch 后端，相关代码在 [pytorch_model_registry.py](../../../tensorrt_llm/_torch/pyexecutor/backend_registries/pytorch_model_registry.py)。
在 `create_pytorch_model_based_executor` 函数中，`KVCacheManager` 的实例化如下：

```python
    kv_cache_manager = KVCacheManager(
        executor_config.kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=model_engine.model.config.num_hidden_layers,
        num_kv_heads=model_engine.model.config.num_key_value_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_num_requests,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )
```

> 💡 **AI Infra 视角**：注意构造参数与前面所有文档的对应：`num_layers`（每层一个 KV cache）、`num_kv_heads`（GQA 的 KV 头数）、`tokens_per_block`（块大小，2 的幂）、`max_seq_len`/`max_batch_size`（容量预算）、`dtype`（KV cache 精度，可 fp8）。**你在 kvcache.md 里学到的每个概念，都落在这里的构造参数上**——知识闭环了。

对于本地测试或概念验证，更新这些行以使用你的实现。
然后测试，确保 `PyExecutor` 使用你的自定义 `KVCacheManager` 运行。
