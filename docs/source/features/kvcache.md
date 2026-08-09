<!--
  本文档为 TensorRT-LLM 官方 KV Cache System 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/features/kvcache.md
-->

# KV Cache 系统

KV cache 存储先前计算过的键值对，在生成过程中复用，以避免重复计算。TensorRT LLM 的 KV cache 系统还支持跨请求复用，并使用主机内存卸载（offloading）和优先级驱逐（eviction）等工具提高复用率。它支持可变 attention 窗口大小，以及 MHA 优化技术（如 MQA 和 GQA）。

> 💡 **AI Infra 视角**：KV cache 是推理引擎中"显存管理"的核心战场。先建立三个直觉：
> 1. **为什么要缓存**：自回归生成第 n 个 token 时，attention 需要所有已生成 token 的 K/V。不缓存就得每步重算，O(n²) 计算 → 缓存后每步只算新 token，显存换速度；
> 2. **为什么难管理**：每个请求的 KV cache 大小随生成动态增长，请求来了又走，显存会碎（fragmentation）——这就是 PagedAttention（分页）要解决的问题；
> 3. **KV cache 是显存大户**：并发 1000 个请求 × 8192 上下文 × 80 层，KV cache 动辄几十 GB。所以 TRT-LLM 花了大量功夫做块复用、驱逐、卸载——**显存管理能力直接决定服务能撑多少并发**，这是 AI Infra 的核心优化点。

## 基础

KV cache 是一个块（block）池，每个块可以容纳固定数量 token 的 KV 状态。多个层被打包在同一个块中，这要求所有层具有相同的 head 数和相同的 attention 窗口大小。对于每种 attention 窗口大小与 head 数的组合，会分别创建一个独立的池，以支持可变 attention 窗口大小和 GQA 等优化技术。

> 💡 **AI Infra 视角**：理解"块"和"池"的设计：
> - **块（block）**：显存分配的最小单位。一个块里打包了模型所有层的 K/V（一个 token 在所有层的 KV 放一起）。类比操作系统的页（page）——PagedAttention 就是借鉴了虚拟内存分页思想；
> - **为什么每步只能算"2 的幂"个 token 每块**：块大小必须是大于 1 的 2 的幂（如 128），方便地址计算和对齐；
> - **为什么按 (窗口大小, head 数) 分组建池**：MQA/GQA 的 head 数不同，块内要装的数据量就不同，混在一起无法统一管理。
> 这些设计你会在阶段 4 的 `kv_cache_manager.md` 里看到具体代码实现。

单个块能存储的 token 数可以由用户在创建模型引擎时设置。它必须是大于 1 的 2 的幂。块按需分配给请求。请求填满块后，块会被存储到一个搜索结构中，这样后续请求如果有匹配的前缀，就可以复用这些 KV 状态。

如果创建了多个池，可用显存会在池之间分配。每个池的分配比例在初始化时确定并且是静态的。这并非最优方案，我们正在努力提供更好的解决方案。

## 跨请求复用

包含先前请求计算出的 KV 状态的块，一旦填满就会被存储到基数搜索树（radix search tree）中。新请求加入时执行一次搜索，匹配到的块直接复用而不是重新计算。被复用的块可以在多个请求之间共享，因此复用既省显存又省计算。

> 💡 **AI Infra 视角**：这是 KV cache 复用最巧妙的一环——**前缀共享**。两个请求如果 prompt 开头相同（比如系统提示词相同、或多轮对话的前几轮相同），前面部分的 KV 计算完全一样。radix tree（前缀树的一种）让相同前缀只存一份，多个请求共享。典型收益场景：
> - **多轮对话**：第 2 轮请求的 prompt = 第 1 轮完整对话 + 新问题，前段全部命中缓存；
> - **RAG/Agent 应用**：大量请求共享同样的 system prompt 和知识库前缀；
> - 业界同样技术：vLLM 叫 prefix caching / automatic prefix caching，SGLang 的 RadixAttention 也是同样的思想。
> 面试常问："多轮对话怎么优化？"——答"KV cache 前缀复用"是标准答案。

块在被从搜索树驱逐（evict）之前始终保持可复用。驱逐发生在新（空白）块被需要时。核心驱逐方案是**优先级 LRU**。所有块被赋予 0 到 100 之间的优先级（100 为最重要）。所有最低优先级的块必须先于次低优先级的块被驱逐。如果所有块优先级相同，则驱逐最久未使用的块（LRU）。

> 💡 **AI Infra 视角**：驱逐（eviction）就是"显存不够了，踢谁出去"的策略。优先级 LRU 的思路：重要的块（高优先级）留着，不重要的（低优先级）先踢；同优先级内踢最久没用过的。优先级由 retention policy（保留策略）决定——比如系统 prompt 的块优先级高、用户输入中段优先级低。**这是显存管理策略设计中的经典权衡：驱逐错了块 = 下次要重算（浪费 GPU 算力）**。

当块从主显存被驱逐时，其 KV 状态会被复制到二级显存（secondary memory）中的块。二级显存中的块仍在搜索树中，所以该块在被从二级显存驱逐之前仍然可复用。二级显存的驱逐发生在新块被需要来卸载主存块时。主存和二级块的驱逐方案相同。

> 💡 **AI Infra 视角**：这里"二级显存"通常指**主机（CPU）内存**。GPU 显存满了，先把不常用的 KV 挪到 CPU 内存（叫 offloading），CPU 内存满了才真正丢弃。用 PCIe 带宽换显存容量——本质是分层存储（hierarchy）：GPU 显存（快而贵）→ CPU 内存（慢而便宜）。后面"Enable Offloading to Host Memory"小节会讲怎么配置。

当前代码有一个限制：只有叶子块才能被驱逐（叶子是指基数树中没有后代的块）。这个设计对全注意力（full attention）层工作良好，但对有限注意力（limited attention）层不适用。这将在未来版本中修复。

### 保留策略（Retention Policy）

块的优先级与请求的[保留策略](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.KvCacheRetentionConfig)一致。优先级得分低的块会比得分高的块更先被释放。保留策略是一个 [TokenRangeRetentionConfig](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.KvCacheRetentionConfig.TokenRangeRetentionConfig) 对象列表，每个对象为给定 token 范围指定优先级，例如"给第 10 到 61 个 token 分配优先级 X"。你还可以设置一个毫秒时长（duration）来限定此策略的生效时间。优先级在块可复用后经过 `duration_ms` 时长后恢复为默认值 35。TokenRangeRetentionConfig 只适用于输入（prompt）token。`decode_retention_policy` 属性指定给生成（解码）token 的块分配什么优先级，`decode_duration_ms` 指定此优先级持续多久。到期后优先级恢复默认。任何需要时长的属性都可以设为 None，表示该部分保留策略永不过期。

> 💡 **AI Infra 视角**：为什么要把优先级策略做得这么细？因为不同 token 的"复用价值"不同：
> - prompt 开头的系统提示词：几乎每个请求都有 → 高价值；
> - 多轮对话的历史：同会话的后续请求会复用 → 中等价值；
> - 已生成的回答：同会话内后续轮次也会用到 → 有 decode_retention_policy 单独管。
> 精细化的驱逐优先级 = 让最该留的留下。默认优先级 35 是"中立"档。

未使用：`transfer_mode` 是调试选项，不应使用。

参见[此示例](../examples/kvcacheretentionconfig.md)了解如何通过修改请求的保留策略来改变特定请求的块优先级。

### 投机解码（Speculative Decoding）

所有投机解码模型都支持跨请求复用。详见[投机解码](speculative-decoding.md)。

## 有限 Attention 窗口大小

TensorRT LLM 利用有限 attention 窗口大小的层来减少计算和显存占用。离开 attention 窗口的块会被释放并放入基数搜索树以便复用。

> 💡 **AI Infra 视角**：有些模型（如 Longformer、某些长上下文模型）不是所有层都做全注意力——部分层只关注最近的 N 个 token（滑动窗口）。这些层不需要保留窗口之外的 KV：出了窗口就释放。这既省显存，也省计算（注意力只算窗口内）。代价是这些 token 的信息"丢失"（模型设计时已考虑），且与前缀复用有冲突（见上文"叶子块驱逐"的限制）。

## MQA / GQA

TensorRT LLM 利用分组查询注意力（grouped query attention）来节省显存。KV cache 只为离散的查询头组（query head groups）创建足够空间的块。对于 MHA，每个 head 一组；对于 MQA，所有 head 共享一组。GQA 是两者之间的平衡。

> 💡 **AI Infra 视角**：三个概念是 attention 的"KV 压缩"演进：
> - **MHA（多头注意力）**：每个查询头都有自己的 K/V → KV cache 最大；
> - **MQA（多查询注意力）**：所有查询头共享一组 K/V → KV cache 缩小约 H 倍，但精度下降；
> - **GQA（分组查询注意力）**：把查询头分成几组，每组共享一组 K/V → 折中（Llama 2/3、Mistral 等主流模型都用 GQA）。
> 为什么省的是 KV 而不是 Q？因为 Q 用完即弃（每步只算一个 token 的 Q），而 K/V 要缓存。KV 头数越少，KV cache 越小，能装的并发请求越多。**GQA 是当前开源模型的事实标准**。

## 控制 KV Cache 行为

KV cache 系统的许多功能是可选的，或有用户定义的属性来改变其工作方式。用户可以通过 [KVCacheConfig](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.KvCacheConfig) 类控制 KV cache 功能。本节其余部分描述如何更改 KV cache 系统最重要的行为。

参见[此示例](../examples/kvcacheconfig.md)了解如何使用 KvCacheConfig 控制 KV cache 行为。

### 数据类型

也许最重要的属性是 `dtype`，它指定 KV cache 中保存的数据类型。默认值 `'auto'` 表示数据类型从模型配置推断。

> 💡 **AI Infra 视角**：KV cache 也可以用低精度！默认情况下 KV cache 与模型权重同精度（FP16/BF16）。显存紧张时可以把 KV cache 降到 FP8/INT8，甚至 FP4——精度损失通常可接受（K/V 对噪声的容忍度高）。这被称为 KV cache 量化，是显存优化的重要手段。

### KV Cache 分配多少显存

属性 `free_gpu_memory_fraction` 是 > 0 且 < 1 的比例，指定将空闲 GPU 显存中的多少分配给 KV cache。默认是 90%（0.9 的比例）。如果同时设置了 `max_tokens`，KV cache 将计算容纳 `max_tokens` 所需的内存量，并分配 `max_tokens` 和 `free_gpu_memory_fraction` 两者中较小的值。

> 💡 **AI Infra 视角**：默认把空闲显存的 90% 给 KV cache！这反映了 KV cache 的重要性——剩下的 10% 留给计算图、激活值和其他临时缓冲。`max_tokens` 本质是"显存预算上限"：设成你预期要服务的最大并发 token 数，防止 KV cache 无上限吞噬显存。生产调优时这两个参数配合：显存不够就降 `free_gpu_memory_fraction` 或 `max_tokens`（减少并发容量），想多撑并发就提高。

### 启用/禁用跨请求复用

默认启用跨请求的块复用，可以通过将 `enable_block_reuse` 设为 False 来禁用。

`scheduler_config.enable_prefix_aware_scheduling` 只控制调度器侧对前缀复用估计的使用。当它为
`True`（默认）时，调度器可以使用估计的可复用 KV token 数，推迟重复的首块上下文（first-chunk context）请求，
并减少对预期复用缓存前缀块的请求的 token 预算核算。当它为 `False` 时，
这些调度器估计被禁用，可复用 token 估计保持为零，但实际的 KV 块复用仍由
`kv_cache_config.enable_block_reuse` 控制。

例如，以下配置保持运行时 KV 块复用启用，同时禁用前缀感知调度（prefix-aware scheduling）的准入和 token 预算核算：

```yaml
kv_cache_config:
  enable_block_reuse: true
scheduler_config:
  enable_prefix_aware_scheduling: false
```

> 💡 **AI Infra 视角**：这里区分了两层机制——**调度层**（让新请求"心里有数"地等一等复用机会、少算 token 预算）和**缓存层**（实际共享 KV 块）。生产排查时注意：命中率不高，先确认 `enable_block_reuse`；调度行为异常，再看 `enable_prefix_aware_scheduling`。

### Mamba 快照边界（Snapshot Boundaries）

混合 Mamba 模型必须将循环 Mamba 状态与 attention KV 前缀一起保留。快照策略归在
`kv_cache_config.mamba_state_config` 下。`periodic_snapshot_interval` 控制
周期性边界。默认禁用；将该间隔设为正值即可启用。已弃用的
`kv_cache_config.mamba_state_cache_interval` 别名仍被接受以保持
兼容性，会在验证期间复制到嵌套字段。新代码和配置文件应使用嵌套字段。
原型选项 `additional_snapshot_offsets_from_start` 和
`additional_snapshot_offsets_from_end` 用于添加固定边界。起始偏移从 prompt 开头计数 token。
结束偏移从 prompt 末尾向前计数，结束偏移 `0` 选择最后一个
prompt 边界。`per_conversation` 块复用策略会禁用周期性
Mamba 快照，因此将其与混合 Mamba 模型一起使用时，请配置一个或多个显式的稳定边界（通常
是结束偏移 `0`）。例如：

```yaml
kv_cache_config:
  enable_block_reuse: true
  use_kv_cache_manager_v2: true
  avg_seq_len: 2048
  block_reuse_config:
    policy: per_conversation
    max_num_turns: 2
  mamba_state_config:
    periodic_snapshot_interval: 0
    additional_snapshot_offsets_from_start: [128]
    additional_snapshot_offsets_from_end: [0, 32]
```

这会在前 128 个 token 之后、prompt 结尾处以及 prompt 末尾 32 个 token 之前保留快照。
特定 prompt 之外的位置会被忽略。设置 `avg_seq_len` 为工作负载的平均总序列长度，
以便 V2 按正确比例划分 attention KV 和 Mamba 状态池。
如果既没有配置 `avg_seq_len` 也没有显式的 `pool_ratio`，混合
Mamba 模型会警告并回退到 `max_seq_len` 的一半，这可能导致
次优的池划分。精确的显式边界目前要求
`MambaHybridCacheManagerV2`、`max_beam_width=1`，并且不使用 KV connector。混合
Mamba 模型在 `use_kv_cache_manager_v2: auto` 时默认选择 V2；设为 `false` 可选择 V1 C++
兼容性管理器。在分离式服务中，V2 Mamba 需要 Python
NIXL transceiver（`transceiver_runtime: PYTHON`）；V1 路由仅支持周期性
快照。

> 💡 **AI Infra 视角**：Mamba 是 SSM（状态空间模型）架构，推理时需要维护"循环状态"（类似 RNN 的隐状态）。混合模型（attention + Mamba 层混合，如 Nemotron H）要同时缓存两类状态。这里的"快照"概念类似 checkpoint：在特定 token 位置存一份状态副本，供前缀复用。这段对新手较偏，了解"混合架构的缓存更复杂"即可，不必深究。

### KV Cache 加盐（Salting）实现安全复用

KV cache 加盐提供一种安全机制，控制哪些请求可以复用缓存的 KV 状态。当请求带 `cache_salt` 参数时，KV cache 系统只允许具有相同 cache salt 值的请求复用缓存块。这可以防止提示词窃取攻击（prompt theft attack）等潜在安全问题——恶意用户可能试图从其他用户请求的缓存状态中推断信息。

要使用加盐，请在创建请求时指定字符串参数 `cache_salt`。只有 cache salt 值匹配的请求才能共享缓存 KV 块。salt 值可以是任何非空字符串，如用户 ID、租户 ID 或哈希字符串。

> 💡 **AI Infra 视角**：多租户场景的安全问题：A 用户和 B 用户如果 prompt 前缀相同（可能只是巧合或恶意构造），B 的请求可能复用 A 缓存的计算结果，甚至通过时序侧信道推断 A 的 prompt 内容。加盐 = 给缓存键混入租户身份，不同租户永远不共享。**做多租户推理服务时必须考虑缓存隔离**——这也是面试可以展示深度的点。

这种隔离完全由块键哈希强制执行：salt 被混入哈希输入，前缀匹配仅由摘要相等性决定（块不会逐 token 重新比较）。因此块键哈希必须是具有强抗碰撞性的密码学哈希，且摘要为 256 位（SHA-256 提供约 128 位抗碰撞性，在这里已经足够）；替换为非密码学哈希将允许构造碰撞绕过 salt 隔离，绝不能这样做。

### 多模态 UUID 支持缓存标识

使用多模态模型（如视觉语言模型）时，KV cache 系统需要识别哪些缓存块对应哪些多模态输入（图像、视频等）。默认情况下，系统使用基于内容的哈希为每个多模态输入生成唯一标识符。然而，这种方法在跨会话缓存管理上有局限，因为相同内容必须重新处理才能生成相同哈希。

要启用确定性缓存管理，你可以在创建请求时使用 `multi_modal_uuids` 参数为多模态数据提供自定义 UUID 字符串。提供后，这些 UUID 会出现在 KV cache 事件中（代替计算出的内容哈希），而缓存键本身则由 **UUID 和内容两者**共同计算以保证正确性。

**用法示例：**

```python
from tensorrt_llm.inputs import TextPrompt

# 为你的图像提供自定义 UUID
prompt = TextPrompt(
    prompt="Describe these images.",
    multi_modal_data={"image": [image1, image2]},
    multi_modal_uuids={"image": ["image-uuid-001", "image-uuid-002"]}
)
```

**关键特性：**

- **缓存正确性**：提供 UUID 时，缓存键由 UUID 和内容共同通过 `BLAKE3(UUID || Content)` 计算。这确保即使 UUID 相同，不同内容也总是产生不同的缓存条目。
- **用户隔离**：相同内容搭配不同 UUID 会产生不同缓存条目，实现按用户或按会话的缓存隔离。
- **稳定事件标识符**：原始 UUID 字符串被保留，并通过 `get_kv_cache_events()` 在 KV cache 事件中返回，实现确定性的外部缓存管理。
- **部分 UUID 支持**：你可以为部分条目提供 UUID，其他条目用 `None` 回退到仅内容哈希。
- **跨模态支持**：不同模态（图像、视频）可以各自拥有自己的 UUID。

**UUID 格式：**

- 可以是任何字符串（例如 "image-123"、"user-session-img-a"、数据库键）
- 原始 UUID 字符串会被保留并在 KV cache 事件中返回

### 启用主机内存卸载（Offloading）

在块被从 GPU 显存驱逐之前，可以选择将其卸载到主机（CPU）内存。该块在被从主机内存驱逐之前始终保持可复用。当卸载的块被复用时，它会先被复制回 GPU 显存。卸载由 `host_cache_size` 属性控制，指定分配多少主机内存（字节）用于卸载。默认值为 0。

启用卸载时，客户端可以通过切换块优先级阻止特定块被卸载。优先级低于某个阈值的块不会被卸载；它们会直接从 GPU 显存被驱逐，以减少 GPU 与主机之间的流量。该优先级由 `secondary_offload_min_priority` 设置。默认值为 35，即任何优先级低于 35 的块不会被卸载。

> 💡 **AI Infra 视角**：卸载（offload）的直觉：GPU 显存贵且满，CPU 内存便宜且大。把"可能还要用但暂时不常用"的 KV 挪到 CPU，PCIe 传一次花点时间，但换来 GPU 显存腾出来接更多请求。代价是复用时要先拷回来（有延迟）。`secondary_offload_min_priority` 防止"低价值块"来回折腾——价值太低就直接丢，省得浪费 PCIe 带宽。显存/带宽/算力三者的权衡，是 AI Infra 调优的日常。

这里是[示例](../../../examples/llm-api/llm_kv_cache_offloading.py)，展示如何启用主机卸载。

### 部分复用（Partial Reuse）

当部分（但不是全部）token 匹配时，就会发生块的部分复用。默认启用，可以通过将 `enable_partial_reuse` 设为 False 来禁用。

属性 `copy_on_partial_reuse` 指定是否应复制块以允许部分复用。如果禁用复制，部分匹配的块只有在没有其他请求使用它时才能被复用。如果启用复制，部分匹配的块不会直接复用，而是创建一个新块，将匹配的 token 复制到新块中。这允许多个请求部分复用同一个块。

> 💡 **AI Infra 视角**：部分复用是"前缀共享"的进阶：两个请求前缀只重合一半怎么办？方案 A：不共享（浪费）；方案 B：复制一部分（花点拷贝时间，但省了重复计算）。`copy_on_partial_reuse` 的取舍：块被多个请求占用时，直接共享会破坏其他请求的数据，所以复制出新块。**以拷贝带宽换计算量**，在重复计算比拷贝贵时划算。

### Attention 窗口大小

属性 `max_attention_window` 以整数列表形式指定模型中每层的最大 attention 窗口大小。如果列表长度小于层数，列表会按需重复。例如，如果模型只有全注意力层，最大序列长度为 4096，你可以设置 `max_attention_window = [4096]`。如果第一层是全注意力、第二层是窗口大小为 256 的有限注意力，后续层重复此模式，则设置 `max_attention_window = [4096,256]`。这意味着第一层全注意力、第二层有限注意力、第三层全注意力、第四层有限注意力，依此类推。

### 已弃用属性

属性 `use_uvm` 已弃用，将在未来版本中移除。

属性 `sink_token_length` 已弃用，在 PyTorch 后端会被静默忽略。
PyTorch attention kernel 不支持 StreamingLLM，因此任何非 `None` 的值都会在
到达执行器之前被丢弃。
