<!--
  本文档为 TensorRT-LLM 官方 Long Sequences 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/features/long-sequence.md
-->

# 长序列（Long Sequences）

在许多真实场景中，如长文档摘要或多轮对话，LLM 需要跨长序列执行认知任务以获得更好的结果。这给 LLM 推理带来了挑战。TensorRT LLM 支持不同的方法来高效处理长序列。本文档将介绍这些优化技术。

> 💡 **AI Infra 视角**：长序列为什么难？三个成本随序列长度增长：
> 1. **KV cache 显存**：每生成一个 token 都要存 KV——长度 L 的序列要存 O(L) 的 KV（每个请求）；
> 2. **attention 计算**：prefill 阶段 O(L²) 的注意力计算（每个 token 要看前面所有 token）；
> 3. **调度阻塞**：一个 128K 的长 prompt 如果一次性 prefill，会长时间霸占 GPU（max_num_tokens 预算被吃光），把其他请求堵死。
> 本章的三种技术（分块上下文、分块注意力、滑动窗口）分别从"调度层、算子层、缓存层"解决长序列问题——这也是"面对同一个问题，不同层级各有解法"的经典案例。

## 分块上下文（Chunked Context）

分块上下文允许 TensorRT LLM 将输入 token 分成更小的块，并将这些块与 decode 请求一起批处理。

使用分块上下文有两个好处：
- 可以防止上下文阶段成为瓶颈，与 decode 阶段的 token 实现更多并行化，提高 GPU 利用率。
- 分块上下文允许 TensorRT LLM 在处理更长上下文的同时实现更高的并发度。由于显存占用取决于每次迭代处理的 token 数，分块上下文将显存消耗与输入请求的上下文长度解耦，改为取决于更小的块大小。这使得 TensorRT LLM 无需增加显存需求即可处理更长的上下文，也有助于在相同显存消耗下提高并发度。

> 💡 **AI Infra 视角**：第二点的"解耦"是精髓：没有分块时，显存峰值 ≈ 最长 prompt 的长度（prefill 必须一次性算完）；分块后，显存峰值 ≈ 块大小，与 prompt 多长无关。所以**长上下文（128K 甚至 1M）能跑起来，分块 prefill 是前提之一**。代价是单个请求的 TTFT 略增（前文讲过）。

要启用分块上下文，请在 `LLM` API 中设置 `enable_chunked_prefill` 为 `True`。
```python
    llm = LLM(
        ...
        enable_chunked_prefill=True,
        ...
    )
```

注意，如果启用了分块上下文，请将 `max_num_tokens` 设置为 kv-cache 块大小 `tokens_per_block`（默认 64）的整数倍。

> 💡 **AI Infra 视角**：为什么要整数倍？分块上下文切出的块要能正好放进 KV cache 块里——如果 chunk 不是 block 的整数倍，就会出现"块装不满、跨块存储"的低效情况（前文 paged-attention 文档也提到过"除最后一块外，上下文块大小必须是 kv-cache 块大小的整数倍"）。**这类"对齐约束"是底层 kernel/显存设计与上层参数之间的隐形契约**，调参时容易踩坑。

## 分块注意力（Chunked attention）

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/media/feat_long_seq_chunked_attention.png" alt="feat_long_seq_chunked_attention" width="240" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 1. 分块注意力示意图 </em></sub></p>

与把输入 token 为整个模型拆成小块不同，分块注意力是另一种方法，只应用于模型中的注意力层。

使用分块注意力时，上下文请求中的 token 被拆成指定大小的块。然后 token 只能关注同一块中的其他 token。例如，如果块大小为 3，我们可能得到如图 1 所示的掩码。每个 token 只需要关注过去最多块大小个 token。因此，KV cache 大小和注意力计算都可以显著减少。

> 💡 **AI Infra 视角**：注意区分两个概念（名字像，实质不同）：
> - **分块上下文（chunked context / chunked prefill）**：把 prefill 在**时间**上拆开——分批算，**计算内容不变**（该看的 token 还是全看），只是调度层面的分时处理；
> - **分块注意力（chunked attention）**：把注意力的**范围**剪掉——每个 token 只看同块内的 token，**这是模型算法的近似**（部分 token 之间的注意力被砍掉）。
> 前者无损、后者有损（精度受影响）。分块注意力本质上是滑动窗口的一种变体（块状窗口）。

目前 TensorRT LLM 只能在 llama4 模型（使用 TRTLLM 注意力后端）中支持分块注意力。TensorRT LLM 会从模型配置中读取 `attention_chunk_size`。如果它不是 None，分块注意力将以 `attention_chunk_size` 作为块大小启用。如果你想在其他模型上启用分块注意力，可以在 attention API 中将 `attention_chunk_size` 设置为有效值。

注意，分块注意力只能应用于上下文请求。

## 滑动窗口注意力（Sliding Window Attention）

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/media/feat_long_seq_chunked_attention.png" alt="feat_long_seq_sliding_win_attn" width="240" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 2. 滑动窗口注意力示意图 </em></sub></p>


由于在处理长序列请求时，注意力层通常是性能瓶颈，滑动窗口注意力是一种有效的方法：将每个 token 的注意力范围限制在它周围固定大小的窗口内，大幅减少所需的计算量和显存。

图 2 展示了滑动窗口注意力的掩码。每个 token 只关注过去的 `N` 个 token。如果过去的 token 数超过最大 attention 窗口大小，`Sliding Window Attention` 将被激活。

TensorRT LLM 将 kv cache 视为环形缓冲区来支持此特性，也称为 `Cyclic KV Cache`。它只存储最近 `N` 个 token 的 kv cache，其中 `N` 由 `LLM` API 中的 `KvCacheConfig.max_attention_window` 参数决定。TensorRT LLM 允许每层有不同的 `N` 值，用户只需向 `KvCacheConfig.max_attention_window` 提供一个 `list[int]`。要启用此特性，用户可以设置
```python
    kv_cache_config = KvCacheConfig(
        ...
        max_attention_window = [...],
        ...
    )
    llm = LLM(
        ...
        kv_cache_config=kv_cache_config,
        ...
    )
```
如果 `KvCacheConfig.max_attention_window` 中提供的元素数量少于层数，提供的列表会重复多次以覆盖层数，为每层设置唯一的值。但需要注意的是，kv cache 的显存分配仍然取决于缓冲区的最大值。

> 💡 **AI Infra 视角**：最后一句是关键陷阱：**列表重复只决定每层"逻辑窗口大小"，但显存按列表最大值分配**。如果你给 60 层的模型提供 `[4096, 256]`，每层要么全窗口要么 256 窗口，但显存预留按 4096 算——省显存的效果要窗口大小真实生效才行（模型本身是滑窗架构，如 Mistral）。"配置项看似生效、实际显存没省"是排查性能问题时常见的坑。

注意：`Sliding Window Attention` 特性目前不能与 beam search 一起使用，因为上下文 kv cache 在 beams 之间共享。
