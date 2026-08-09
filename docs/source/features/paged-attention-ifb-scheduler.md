<!--
  本文档为 TensorRT-LLM 官方 Paged Attention, IFB, and Request Scheduling 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/features/paged-attention-ifb-scheduler.md
-->

# Paged Attention、IFB 与请求调度

## 飞行中批处理（In-flight Batching）

TensorRT LLM 支持请求的飞行中批处理（又称连续批处理 continuous batching 或迭代级批处理 iteration-level batching），以提升服务吞吐。有了这个特性，处于上下文（context）阶段的序列可以与处于生成（generation）阶段的序列一起处理。该技术的目的是更好地交错请求，既降低延迟，又更好地利用 GPU。
出于效率考虑 (1)，对飞行中批处理的支持 ***要求输入张量被打包（packed，无 padding）***。

> 💡 **AI Infra 视角**：这是全文最重要的一段，先理解"没有 IFB 的世界"：
> - **静态批处理（static batching）**：一批请求一起进来，一起算完一起返回。问题：有的请求 prompt 短生成快，算完也只能干等——GPU 大量空闲。就像自助餐一桌人必须等最慢的吃完才能翻台；
> - **飞行中批处理（IFB）**：每个请求独立推进，"算一步 token 就重新排一次队"。新请求随时插进来，完成的请求随时撤走。GPU 每步都满负荷——像流水线餐厅，每个人吃完就走、新客随时入座。
> IFB 是 vLLM 首创并发扬光大的（vLLM 论文的核心贡献之一），现在所有主流引擎都有。**"连续批处理/迭代级调度"是 AI Infra 面试必考概念**。

**在当前实现中，处于上下文阶段的序列必须排在生成阶段的序列之前**出现在输入张量中。例如，对于序列 `S0`、`S1` 和 `S2`，如果 `S0` 和 `S2` 处于上下文阶段（而 `S1` 处于生成阶段），则 `S0` 和 `S2` 的 token 必须出现在 `S1` 的 token 之前。这个约束可能在未来版本中放宽。

> 💡 **AI Infra 视角**：为什么有这个排序约束？因为 GPU kernel 是按"输入张量布局"写的：prefill（上下文）和 decode（生成）的计算模式不同（prefill 是大矩阵乘，decode 是每序列单 token 的小操作），kernel 需要知道"张量里哪一段是 prefill、哪一段是 decode"才能正确高效地算。这就是工程实现里的"约定"——理解了它，看代码时就不会疑惑为什么数据要这样排。

_(1) 将生成阶段只含单个 token 的序列 padding 到最大输入序列长度，是对资源的低效使用。

> 💡 **AI Infra 视角**：这句话解释"为什么要无 padding 打包"：decode 阶段的序列每步只有 1 个新 token，如果按传统方式 padding 到定长，假设 max_seq_len=4096，一个 1 token 的序列要 pad 4095 个无效 token——99.97% 是浪费。**无 padding 打包 = 只存真实 token**，是 IFB 可行的前提，也是显存和算力优化的根基。

### `max_batch_size`、`max_seq_len` 和 `max_num_tokens`

<p align="center">
    <img src="https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/media/max_bs_toks_len.svg?raw=true" alt="解释 `max_batch_size`、`max_seq_len` 和 `max_num_tokens`" width="30%" height="auto">
</p>

> 💡 **AI Infra 视角**：这三个参数是推理引擎的"容量预算"，必须先分清（面试必考）：
> - `max_batch_size`：**同时服务的请求数上限**（并行多少"桌客人"）；
> - `max_seq_len`：**单个请求的序列长度上限**（一个客人最多吃多长）；
> - `max_num_tokens`：**每步实际送进 GPU 的 token 总数上限**（每步最多"上几道菜"）。
> 三者乘积关系大致是显存预算：batch × seq_len 决定 KV cache 需求上限，max_num_tokens 决定单步计算规模。调参的本质就是在显存和延迟约束下找最大吞吐。

#### `max_batch_size`

`max_batch_size` 定义引擎可以处理的最大请求数。​

它控制运行时可以调度的最大请求数。

构建引擎时设置足够高的 `max_batch_size`，使其不会成为吞吐的瓶颈，并使用运行时（runtime）的 `max_batch_size` 来调吞吐或延迟，而无需重新构建引擎。

> 💡 **AI Infra 视角**：注意"构建时 vs 运行时"的区分：早期 TRT-LLM 的很多参数在**构建引擎时就要固化**（改参数 = 重新编译，很慢），现在改成了运行时可调。`max_batch_size` 尤其如此——它是 kernel 编译时的形状参数（CUDA Graph 捕获的尺寸也与之相关）。生产实践中：构建时给大一点，运行时用小值调吞吐/延迟，不用重编译。

#### `max_seq_len`

`max_seq_len` 定义单个请求的最大序列长度。

从 TensorRT LLM v0.11 开始，当启用 `--remove_input_padding` 和 `--context_fmha` 时，`max_seq_len` 可以替代 `max_input_len` 和 `max_output_len`，默认值为 `max_position_embeddings`。

使用默认的 `max_seq_len`（即 `max_position_embeddings`）即可，除非你非常确定工作负载的最大序列长度，否则无需调整。如果 GPU 显存紧张到连一个达到 `max_seq_len` 的请求都撑不住，则需要调小它。

> 💡 **AI Infra 视角**：`max_position_embeddings` 是模型训练时的最大位置编码长度（如 Llama 3 是 8192）。推理引擎默认对齐它。注意：**这个值直接决定 KV cache 的预留上限**——max_seq_len 越大，为每个请求预留的显存预算越大，能装的并发就越少。所以长上下文模型（128K 等）的 KV cache 优化格外重要。

#### `max_num_tokens`

`max_num_tokens` 定义移除 padding 后每个 batch 中批处理的输入 token 总数上限。​

从 v0.11 开始 `max_num_tokens` 默认值为 8192。建议调优 `--max_num_tokens` 以获得最佳性能。

当输入 padding 未被移除时，最大 token 数不生效。当输入 padding 被移除时，不同序列的 token 被打包在一起，最大 token 数可以设置为不同（更低）的值，默认是 8192。

必须考虑两个方面。首先，一些输入序列会比最大输入长度短。其次，当启用飞行中序列批处理时，上下文阶段的请求会与生成阶段的请求一起执行。后者产生的 token 远比 `max_input_len` 少（最多 `beam_width` 个 token）。

`max_num_tokens` 影响要分配的 workspace 缓冲区大小，以及其中一个矩阵乘法的维度。因此，为 `max_num_tokens` 使用更实际的值，可以让 TensorRT LLM 分配更多显存来存储 KV cache，并同时执行更多请求。这会提升效率。

> 💡 **AI Infra 视角**：关键洞察：`max_num_tokens` 设大了，KV cache 的显存就少了（workspace 占掉），设小了 GPU 每步算得少、吞吐低。它是"单步计算规模"和"总显存"之间的杠杆。业界常见的调法：设成 `max_batch_size × 平均序列长度` 的量级，而不是 max_batch × max_seq_len（那个太大）。

GPU 在更大的矩阵乘法中利用率更高。因此，适当增大 `max_num_tokens` 有利于性能。到某个点后 GPU 利用率会趋于饱和，超过饱和点可能会同时伤害首 token 延迟和端到端总延迟。总之，应该选择"合理偏高"的 `max_num_tokens` 以获得高 token 吞吐和良好的 GPU 算力利用率，但不能过高，以满足 SLO 中的 TTFT（首 token 时间）和 TPOT（每输出 token 时间）。

> 💡 **AI Infra 视角**：两个必须认识的行业指标（面试必考）：
> - **TTFT（Time To First Token）**：从发出请求到收到第一个 token 的时间——用户"感觉响应快不快"；
> - **TPOT（Time Per Output Token）**：生成每个输出 token 的平均时间——用户"打字速度";
> - 还有 ITL（Inter-Token Latency，token 间延迟）等变体。SLO（服务等级目标）就是公司对延迟的承诺，比如"P95 TTFT < 2s"。**所有引擎调参最终都是为了在成本约束下满足 SLO**。吞吐（throuphput）与延迟（latency）的矛盾贯穿始终：batch 越大吞吐越高，但每步排队的人越多、单个请求的延迟越差。

## 分块上下文（Chunked Context，又称 Chunked Prefill）

最初的行为是一次性处理所有上下文 token。但这个特性会把上下文拆成多个块（chunk）。这样，上下文块可以在生成阶段与更多 token 一起批量执行，从而提高整体吞吐。分块上下文还消除了对输入长度的限制。要启用此特性，还需要启用 FMHA 分页 kv-cache。除最后一个块外，每个上下文块的大小都必须是 kv-cache 块大小的整数倍。

> 💡 **AI Infra 视角**：分块 prefill 解决什么问题？长 prompt（比如 32K 的 RAG 文档）如果一次性 prefill，会占掉单步的全部预算，把其他请求都堵在后面——而且大矩阵乘需要大显存。拆成小块后：
> 1. 长 prompt 请求可以"分期付款"，穿插在 decode 之间执行；
> 2. 新请求无需等"巨无霸" prefill 完成才能上车；
> 3. 代价：单个请求的 TTFT 可能变高（被拆散了），但整体公平性和吞吐更好。
> 这也是 vLLM 的 chunked prefill 特性，业界标配。**"为什么大 prompt 会阻塞服务"+"怎么解决"——面试高频题**。

## KV Cache

在生成阶段，一个常见优化是给 MHA kernel 提供一个缓存，保存已经计算过的历史 K 和 V 元素的值。这个缓存就是 KV cache。TensorRT LLM 用这项技术加速生成阶段。在 TensorRT LLM 中，每个 Transformer 层有一个 KV cache，也就是说 KV cache 的数量等于模型的层数。当前版本的 TensorRT LLM 支持两种 KV cache：**连续（contiguous）** 和 **分页（paged）**。

> 💡 **AI Infra 视角**：注意"每层一个 KV cache"——KV 是按层组织的，每层都要存一份自己的 K/V（不同层的 K/V 不同）。这就是为什么 KV cache 显存 ≈ 层数 × 每层 K/V 大小。模型的层数越多（DeepSeek R1 有 61 层）、上下文越长，KV cache 越大。

### 连续 KV Cache（Contiguous）

连续 KV cache 是一个单一的大张量。它的形状是：
```
[max_batch_size * max_beam_width, 2, num_heads, max_seqlen, hidden_dim_per_head].
```

当序列比最大序列长度短时，这种实现使用的显存比实际需要的多得多。即使序列在生成许多输出 token 后接近上限，也需要很多步骤才能到达那个点。

> 💡 **AI Infra 视角**：连续 KV cache 的问题：给每个请求**预分配** max_seq_len 的整块空间（按最大长度预留，不管实际用多少）。100 个请求平均只用 1/4 长度，却有 3/4 显存空置浪费。这就是"显存碎片化+预分配浪费"——paged 方案就是为了解决它。

### 分页 KV Cache（Paged）

分页 KV cache 将 KV cache 分解成块，由缓存管理器在处理过程中分配给不同的请求。缓存管理器跟踪序列，从池中分配新块，并在需要时回收这些块。参见简化实现
[`tensorrt_llm.runtime.KVCacheManager`](source:tensorrt_llm/runtime/kv_cache_manager.py)。
Batch Manager 中包含了更高效的 C++ 实现
[Batch Manager](source:cpp/include/tensorrt_llm/batch_manager)。

> 💡 **AI Infra 视角**：PagedAttention 的核心思想（借鉴操作系统虚拟内存分页）：**按需分配、用多少分多少**。不再按"最大长度"预留，而是按固定大小的块（block）动态分配——就像 OS 给进程按页分配物理内存。好处：
> 1. 显存几乎零浪费（只有最后一块可能不满）；
> 2. 没有外部碎片（块可任意组合）；
> 3. 跨请求共享前缀（上一篇文章的块复用）成为可能。
> PagedAttention 是 vLLM 的成名论文（2023），现在 TRT-LLM、SGLang 全部采用。**"解释 PagedAttention 与显存分页的类比"是 AI Infra 面试经典题**。

## 调度器（Schedulers）

本节用可视化展示 TensorRT LLM 如何基于 max-batch size 和 max-num tokens 调度请求。示例从一个新初始化的引擎和几个到达的未调度请求开始。为了示例，设置 `max batch size = 4` 和 `max num tokens = 12`。每个方块代表一个 token，颜色代表它属于哪个请求。

![TRT-LLM 调度器可视化 1](../media/TRTLLM_Scheduler_Vis_1.svg)

> 💡 **AI Infra 视角**：解读上图：5 个请求排着队（R1~R5），引擎能力 = 同时最多 4 个请求（max_batch_size=4），每步最多 12 个 token（max_num_tokens=12）。接下来看调度器怎么一步步"放行"。

现在调度器取出前两个请求（请求 1 和 请求 2），调度它们执行上下文阶段。然而，它不能再调度更多请求，因为前两个请求的 prompt 各有 5 个 token，受 max num tokens 限制，预算只剩 2 个 token。由于其余请求的 prompt token 都超过 2 个，没有一个能被调度（上下文分块可以解决这种情况，见下面的 paged context attention 部分）。token 上标有 "C"，表示它们是在上下文阶段处理的 prompt token。

> 注：不同请求的 token 显示在不同行仅为可视化目的，不代表实际内存布局

![TRT-LLM 调度器可视化 2](../media/TRTLLM_Scheduler_Vis_2.svg)

> 💡 **AI Infra 视角**：注意这里的预算逻辑：max_num_tokens=12 是"每步能处理的 token 总数"。两个 5-token 的 prompt 共 10 个，剩 2——剩下的请求 prompt 都 ≥5 个，装不下。**这就是"token 预算"概念**：调度器按 token 数而不是请求数做预算，这也是无 padding 打包带来的精确性。

现在引擎执行一次迭代，两个已调度请求的上下文阶段都完成了。完成后，两个请求的 prompt 的 kv-cache 已创建，第一个 token 也已生成。生成的 token 标记为 "G(n)"——例如标记 "G1" 的 token 表示它是该请求生成的第一个 token。

TRT-LLM 优先调度生成阶段的请求，所以两个生成的 token 被排队在下次迭代处理。现在，由于之前调度的两个请求已进入生成阶段，只占 12 个 token 预算中的 2 个，调度器能够再调度两个请求（请求 3 和 请求 4）。它无法调度最后一个请求（请求 5），即使 max num tokens 预算中有空间——因为 max batch size 上限是 4。

![TRT-LLM 调度器可视化 3](../media/TRTLLM_Scheduler_Vis_3.svg)

> 💡 **AI Infra 视角**：两个关键设计点：
> 1. **生成阶段优先**：decode 的请求先排队——因为 decode 每步只需 1 个 token 的算力，先满足它们能保证生成不断流（卡顿 = 用户体验差），剩下的预算再塞 prefill；
> 2. **预算按 token 算**：decode 阶段每请求只占 1 个 token，所以 4 个请求只占 4 个 token 预算——空出的 8 个可以塞新请求的 prefill。**IFB 的精髓：prefill 和 decode 共享每步预算，谁都不闲着**。

执行完下一次迭代后，请求 1 和 2 的第二个 token 已生成，请求 3 和 4 的第一个 token 已生成。假设 G2（为请求 1 生成）是停止 token（stop token），表示请求 1 已完成。在这种情况下，调度器会在执行下一次迭代前驱逐请求 1，并准备将其结果返回给用户。这次驱逐使引擎状态低于 max batch size 上限，从而允许调度请求 5。

还要注意，G1（为请求 2 生成）已添加到请求 2 的 kv-cache 中，这展示了请求的 kv-cache 如何随着生成更多 token 而增长。

![TRT-LLM 调度器可视化 4](../media/TRTLLM_Scheduler_Vis_4.svg)

> 💡 **AI Infra 视角**：两个细节值得注意：
> 1. **驱逐（evict）= 完成**：请求生成完，立即从 batch 里移除并释放 KV cache——每步都在做"增删请求"的动态操作，这是 IFB 调度器与静态批处理最本质的区别；
> 2. **KV cache 动态增长**：每生成一个 token 就往 KV cache 里追加一块——paged 结构天然支持这种增长，连续结构就只能按最大长度预留。

总的来说，max batch size 和 max num tokens 上限在决定请求何时执行方面起着关键作用。调整这些参数可以显著影响吞吐，以及引擎如何平衡生成阶段的已有请求与上下文阶段的新请求。

> 注：这只是调度器的简化可视化，用于突出 max batch size 和 max num tokens 的影响。调度器还会考虑可用于 kv-cache 的空闲显存量等因素，并具有其他可配置选项。更多信息请参见 Additional Options 页面的 Runtime Flags。

## 再看 Paged Context Attention 与上下文分块

之前我们建议启用 paged context attention，尽管在我们的案例研究中它并没有显著影响性能。现在了解了 TensorRT LLM 调度器后，我们可以解释为什么它有益。简言之，我们建议启用它，因为它支持上下文分块（context chunking），允许请求的上下文阶段被拆成多块，分多次执行迭代处理，让引擎在上下文和生成阶段的执行之间提供更稳定的平衡。启用分块会降低所有请求的平均 TTFT，但可能会增加少数"幸运"请求的 TTFT——这些请求在没有分块时本来可以一次迭代处理完。

上文[调度器可视化](#the-schedulers)显示，最初请求 3 无法被调度，因为会让调度器超出 max-num tokens 限制。但有了上下文分块，情况就不同了：请求 3 的第一块可以被调度。

![TRT-LLM 调度器可视化 分块上下文 1](../media/TRTLLM_Scheduler_Vis_Chunked_Context_1.svg)

这有几个极其重要的原因。首先，它消除了大 prompt 请求（相对于 max num tokens）因其他请求已在飞行中而无法被调度的可能性。在生产负载中，这可以缓解排队效应和最坏情况下的 TTFT。其次，它允许设置更小的 max num tokens 值，因为不再需要 max num tokens 至少与你想要支持的最长 prompt 一样大。对长上下文场景这一点极其重要，因为设置极大的 max-num tokens 会占用本可用于 kv-cache 的显存。鉴于在最坏情况下，分块上下文对性能影响很小，而在许多情况下能显著受益，NVIDIA 建议始终启用它。
