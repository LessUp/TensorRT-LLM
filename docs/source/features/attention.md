<!--
  本文档为 TensorRT-LLM 官方 Multi-Head, Multi-Query, and Group-Query Attention 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/features/attention.md
-->

(attention)=


# 多头、多查询与分组查询注意力

本文档详细介绍 TensorRT LLM PyTorch 后端中自回归模型的多头注意力（MHA）、多查询注意力（MQA）和分组查询注意力（GQA）的实现。

多头注意力包含一系列批处理矩阵乘法、一次 softmax 运算和又一次批处理矩阵乘法，
如 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 论文所述。
[多查询注意力（MQA）](https://arxiv.org/abs/1911.02150) 和 [分组查询注意力（GQA）](https://arxiv.org/abs/2307.09288) 是
MHA 的变体，使用的 KV 头数少于查询头数。
TensorRT LLM 在 `tensorrt_llm/_torch/attention_backend/` 中提供了使用不同后端的多种实现。
以下章节说明如何使用这些实现，并提供实现新后端的简要指南。

> 💡 **AI Infra 视角**：为什么 MHA/MQA/GQA 的知识这么重要？attention 是 transformer 推理计算量最大、也是优化空间最大的部分（占了生成阶段绝大部分 FLOPs）。理解三个变体（MHA 每头一组 KV / MQA 全头一组 / GQA 折中），你就能理解主流模型的配置（Llama 3 用 GQA，8 个 KV 头 / 32 个查询头）。**"这个模型为什么省 KV 显存"、"KV cache 大小怎么估算"都从这里来**（详见 kvcache.md 的讲解）。

## 注意力后端（Attention Backends）

目前有三个可用的注意力后端：vanilla 后端、TRT-LLM 后端和 Flashinfer 后端。
你可以使用 `PyTorchConfig.attn_backend` 指定想要的注意力后端。例如，要使用 Flashinfer 后端，可以像这样给 `LLM` 构造函数传 `attn_backend="flashinfer"`：`LLM(attn_backend="flashinfer")`。这将为你的模型启用 Flashinfer 后端。

vanilla 后端 `VanillaAttention` 是一个参考实现，主要设计用于飞行中批处理和线性 KV cache 支持。虽然它可以作为有用的基线，但由于优化有限，不建议在生产中使用。

相比之下，Flashinfer 后端 `FlashInferAttention` 经过性能优化，支持飞行中批处理和分页 KV cache。它还包括以下高级特性：

1. **FP8 量化**：此特性支持将输入和 KV cache 量化为 FP8 格式，显著降低显存占用并提高计算吞吐。
2. **RoPE 融合**：通过将旋转位置编码（RoPE）直接集成到注意力计算中，提高效率并降低开销。

TRT-LLM 后端 `TrtllmAttention` 是默认后端，支持 Flashinfer 后端的所有特性，并进一步优化以获得更好的性能。它是生产环境的推荐选择。此外，它还提供以下高级特性：

1. **融合 QKV 输入**：可以接受单个 QKV 张量作为输入，比使用单独的 Q、K、V 张量更高效。
2. **FP8 输出**：支持以 FP8 格式输出注意力结果，将量化融合到注意力计算过程中。

> 💡 **AI Infra 视角**：三个后端的定位（生产上很重要）：
> - **Vanilla**：教学/参考实现，纯 PyTorch 算子拼出来的 attention——能跑但慢，别用于生产；
> - **FlashInfer**：社区库（flashinfer.ai）的 CUDA kernel，第三方开源项目（vLLM 等也用）；
> - **TRTLLM**：NVIDIA 自研 kernel（底层是 fmha 系列 CUDA kernel），默认且推荐。
> 为什么"融合"这么重要？RoPE 融合、QKV 融合、FP8 输出融合——**每次算子融合都省一次 kernel 启动和一次显存读写**。GPU 上"读数据"比"算数据"贵，融合的本质是把多次中间结果读写压缩掉。这是所有 GPU 算子优化的通用思路（flash attention 的核心也是减少 HBM 读写）。

## 实现一个新的注意力后端

你可以实现一个新的注意力后端来集成其他注意力库。
一个注意力后端由一个 `AttentionBackend` 类和 `AttentionMetadata` 类组成。
PyTorch 后端有三个涉及注意力后端的阶段：

1. 模型构建：在模型的 `__init__` 中，调用 `AttentionBackend.__init__` 为每一层创建一个注意力后端。
2. 元数据准备：在模型每次前向步骤之前：
   1. 如果元数据未初始化，调用 `AttentionMetadata.__init__` 创建注意力元数据。
   2. 如果使用 CUDA graphs，调用 `AttentionMetadata.create_cuda_graph_metadata` 将元数据转换为 CUDA graph 元数据，它会预分配所有张量，可用于捕获 CUDA graphs。使用 CUDA graphs 时，在初始预热运行后不要重新分配 `AttentionMetadata` 中存储的任何张量。
   3. 为准备输入和 KV cache 的参数，调用 `AttentionMetadata.prepare`，从现有元数据和 KV cache 管理器转换。
3. 单步前向：在每层注意力的前向过程中，调用 `AttentionBackend.forward` 执行注意力操作。`AttentionMetadata` 将作为前向参数提供。

### 实现 `AttentionMetadata`

`AttentionMetadata` 类存储来自批处理输入和 KV cache 的元数据，供注意力后端使用。
它包含以下预定义字段：

| 字段 | 类型 | 描述 |
| ----- | ---- | ----------- |
| max_num_requests | int | 单个 batch 中的最大请求数。 |
| num_contexts | int | batch 中上下文阶段序列的数量。 |
| num_generations | int | batch 中生成阶段序列的数量。 |
| max_num_tokens | int | 单个 batch 中所有请求的最大 token 数。 |
| num_tokens | int | batch 中的 token 数。 |
| num_ctx_tokens | int | 上下文阶段序列中的 token 数。 |
| kv_cache_manager | KVCacheManager | KV cache 管理器。 |
| is_cuda_graph | bool | 是否启用 CUDA graph。 |
| seq_lens | Tensor | batch 中每个序列的长度。形状为 (batch_size)，位于 CPU 内存。 |
| seq_lens_cuda | Tensor | 存储在 GPU 上的 `seq_lens` 副本。 |
| context_lens | Tensor | batch 中每个上下文阶段序列的长度。形状为 (`num_contexts`)。 |
| position_ids | Optional[Tensor] | 每个序列中每个 token 的位置。如果在后端外部应用位置嵌入，可能为 None。 |
| request_ids | List[int] | batch 中每个序列的请求 ID。 |
| prompt_lens | List[int] | batch 中每个序列的 prompt 长度。 |
| kv_cache_params | KVCacheParams | KV cache 的参数。 |

> 💡 **AI Infra 视角**：这张表是理解 attention 算子输入的关键——**GPU 算子除了数据还要"元数据"**：每个序列多长、哪些在 prefill 哪些在 decode、KV cache 在哪。因为无 padding 打包后，张量是"一长条 token"，没有元数据就无法告诉 kernel"从哪到哪属于哪个序列"。这也是 flashinfer/TRTLLM kernel 的 API 设计核心（indptr 等指针数组）。

在 `AttentionMetadata.__init__` 期间，你可以为新注意力元数据初始化额外字段。
例如，Flashinfer 元数据在这里初始化 `decode_wrapper`。
在 `AttentionMetadata.prepare` 期间，运行时将填充所有预定义字段，你可以根据这些预定义字段填充自定义字段。
例如，Flashinfer 元数据在这里通过组合 `context_lens` 和 `num_generations` 填充 `qo_indptr`。

### 实现 `AttentionBackend`

`AttentionBackend` 将注意力操作委托给后端实现。

其 `__init__` 接受以下参数：

| 字段 | 类型 | 描述 |
| ----- | ---- | ----------- |
| layer_idx | int | 模型中的注意力层索引。 |
| num_heads | int | 查询头数量。 |
| head_dim | int | 每个注意力头的大小 `(hidden_size // num_heads)`。 |
| num_kv_heads | Optional[int] | KV 头数量。为 None 时默认为 num_heads。 |
| quant_config | QuantConfig | 可选的量化配置。为 None 时不应用量化。 |
| pos_embd_params | PositionalEmbeddingParams | 可选参数，定义如何应用位置嵌入。为 None 时，模型应在调用后端前应用位置嵌入。否则，后端负责应用位置嵌入，并且可以先缓存 K 而不加嵌入。 |

其 `forward` 接受以下参数：

| 字段 | 类型 | 描述 |
| ----- | ---- | ----------- |
| q | Tensor | 查询张量，形状 `(num_tokens, num_heads * head_dim)`。 |
| k | Tensor | 键张量，形状 `(num_tokens, num_kv_heads * head_dim)`。 |
| v | Tensor | 值张量，形状 `(num_tokens, num_kv_heads * head_dim)`。 |
| metadata | AttentionMetadata | 注意力操作的元数据。 |
| forward_args | AttentionForwardArgs | 可选的每前向参数，如注意力掩码、输出缓冲区和缩放因子、RoPE 和 MRoPE 输入、MLA 缓冲区以及稀疏注意力输入。 |
| **kwargs | Any | 为 `AttentionForwardArgs` 声明的字段提供的临时兼容路径；未知字段会报错。 |

例如，FlashInfer 后端在它拥有 KV cache 更新权时调用 `append_paged_kv_cache`，然后使用 `FlashInferAttentionMetadata` 中缓存的 plan 调用 prefill、decode 或 ragged-prefill wrapper 的 `run` 方法。

## `TrtllmAttention` 后端的特性

以下章节介绍默认 `TrtllmAttention` 后端的一些特性。

### 打包张量（Packed Tensors）

在 `TrtllmAttention` 后端中，注意力算子支持打包（即无 padding）的 QKV 输入。
QKV 输入的一个朴素布局是将短于 `max_sequence_length` 的序列 padding 到最大
长度。这可能导致过多的显存消耗以及对 padding token 的无谓计算（在 MHA 块周围的
各种矩阵乘法中）。
为了解决这个问题，TensorRT LLM 支持无 padding 模式：不同 token 被打包在一起，
用户向算子提供一个包含不同序列长度的一维张量。

### 上下文与生成阶段

`TrtllmAttention` 后端将上下文和生成阶段的不同实现封装到一个自定义 torch 算子中。

#### 上下文阶段（Context Phase）

未优化的上下文阶段实现映射为一串 GPU kernel：先把中间的 `Q*K^T` 张量存到显存，
再调用 softmax 算子。这是最慢的方法，显存占用也很大（随序列长度二次增长）。

`TrtllmAttention` 后端会改为触发一个单 kernel 来执行 MHA/MQA 块。
对于短序列，该 kernel 使用 MHA/MQA 的 vanilla 实现。对于较长的序列，该 kernel 使用
Flash Attention 算法，如
[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
和
[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
所述。

> 💡 **AI Infra 视角**：FlashAttention 是过去几年 GPU 计算领域最重要的论文之一（面试必考）。核心洞察：**attention 的瓶颈不是计算，是显存带宽**。朴素实现中 `Q*K^T` 的中间结果（O(n²)）要写回显存再读出来做 softmax——两次 HBM 大流量往返。FlashAttention 用分块（tiling）+ 在线 softmax 技巧，把中间结果留在片上（SRAM），HBM 读写减少数倍。**"IO-aware"（感知 IO 的）就是它的精髓：优化显存流量而非 FLOPs**。所有现代推理引擎的 attention kernel 都是它的后裔。

目前，该实现会触发额外的 kernel 对元素做预处理（如 RoPE）并填充 KV cache（见下文）。在未来的
版本中，可能会减少此类 kernel 的数量以提升整体性能。

#### FP8 上下文 FMHA

当激活 FP8 量化时，可以通过启用 FP8 Context FMHA 进一步加速注意力。

FP8 Paged Context FMHA 也支持 fp8 量化工作流。
你需要为注意力算子指定 `use_paged_context_fmha = True`。

请注意，此特性仅在 Ada、Hopper 及更新架构上受支持。

#### 生成阶段（Generation Phase）

生成阶段使用单个 kernel 实现，称为 masked multi-head attention（掩码多头注意力）。该 kernel 能
在飞行中对 Q、K、V 元素进行预处理：添加 QKV 偏置、应用
RoPE、执行反量化和量化。TensorRT LLM 将在未来版本中继续添加（或启用）
更多特性，如支持 IA3。

masked MHA kernel 有一个特殊版本，在 GPU 占用率低时会将工作分发到
GPU 上的多个 CUDA thread-block。这种称为 multi-block 的模式始终启用。
NVIDIA 建议用户在 batch 大小和模型 head 数都相对较小的场景中测试该模式。
这里"小"的定义难以量化，因为它取决于 GPU 型号。
不过 NVIDIA 目前建议在 `batch_size * num_heads` 小于 GPU 上多处理器（SM）数量时测试该模式。
此建议将来可能会更改。

> 💡 **AI Infra 视角**：为什么 batch 小的时候要多线程块？decode 阶段每个序列每步只有 1 个 token：如果每 head 只用一个 thread-block，batch=1、8 个 head 的模型在 132 个 SM 的 H100 上只用了 8 个 SM——GPU 大部分闲着。multi-block 模式把每个 head 的工作拆到多个 block，让更多 SM 参与。**"占用率（occupancy）"思维**：kernel 设计要考虑 GPU 有多少并行单位可用，利用率低时想办法把工作拆细。

注意，即使启用了 multi-block 模式，注意力算子也不会立即触发 GPU kernel 的 multi-block 版本。multi-block 版本要变得比"每 head 单 CUDA thread-block"的 vanilla 实现更高效，需要一定的最小 token（输入 + 生成）数量。这由内部启发式算法控制。

另请注意，masked MHA kernel 使用的共享内存大小与序列长度成正比，因此在未启用 multi-block 模式时，某些情况下 GPU 的共享内存可能不足。为了在这些情况下让 masked MHA kernel 正常工作，multi-block 模式会被强制开启，并在日志中打印警告。

#### XQA 优化

XQA 优化是生成阶段 MQA/GQA 的另一个优化。
它目前只支持有限的模型配置，例如 LLAMA2 70B 模型。

XQA 优化的支持矩阵：
 - FP16 / BF16 计算数据类型。
 - FP16 / BF16 / FP8 / INT8 KV cache 数据类型。
 - 分页 KV cache（每块 8 / 16 / 32 / 64 / 128 tokens）。

默认启用。注意，还有一个启发式算法
决定使用 XQA kernel 还是 masked MHA kernel 以获得
更好的性能。
如果你想尽可能使用该 kernel，设置 `TRTLLM_FORCE_XQA=1` 在模型配置受支持时强制使用 XQA kernel。
支持的配置可以通过 `cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h` 中
`DecoderXQARunner` 类的 `shouldUse` 函数查看。

> 💡 **AI Infra 视角**：XQA 是"专用 kernel"思路的体现：decode 阶段每个查询只关注少量 KV（当前窗口），而 masked MHA kernel 是通用实现——XQA 针对 MQA/GQA 的"多查询头共享少量 KV 头"结构做专门优化，用广播（broadcast）技巧避免重复读 KV。**生产经验：内核选择经常要"启发式 + 强制开关"两手准备**——启发式选默认，出问题时用环境变量强制指定来对比定位。

(inflight-batching)=

### 飞行中批处理（In-flight Batching）

TensorRT LLM 支持请求的飞行中批处理（又称连续
批处理或迭代级批处理）以提升服务吞吐。有了这个特性，
上下文阶段的序列可以与生成阶段的序列一起处理。该技术的目的是更好地交错
请求，降低延迟并更好地利用 GPU。
出于效率原因 (1)，对飞行中批处理的支持 ***要求输入张量被打包（无 padding）***。

***在当前实现中，处于上下文阶段的序列必须位于生成阶段的序列之前**出现在输入张量中。例如，对于序列 `S0`、`S1` 和 `S2`，如果 `S0` 和 `S2` 处于上下文阶段（而 `S1` 处于生成阶段），则 `S0` 和 `S2` 的 token 必须出现在 `S1` 的 token 之前***。

_(1) 将生成阶段只含单个 token 的序列 padding 到最大输入序列长度，是对资源的低效使用_。

> 💡 **AI Infra 视角**：此段与 paged-attention-ifb-scheduler.md 内容相同——提醒你：attention 算子层面必须遵循"prefill 在前、decode 在后"的布局约定（见前文讲解）。

### 分块上下文（Chunked Context）

最初的行为是一次性处理所有上下文 token。此特性将上下文拆分成多个块。这样，
上下文块可以在生成阶段与更多 token 一起批处理，预计会提高总吞吐。分块上下文还消除了
对输入长度的限制。除最后一个块外，上下文块的大小需要是 kv-cache 块大小的整数倍。

> 要启用此特性，还需要启用 FMHA 分页 kv-cache。

### KV Cache

在生成阶段，一个常见优化是给 MHA kernel 提供
包含已计算的历史 K 和 V 元素的缓存。这个缓存就是 KV cache。TensorRT LLM 用
这项技术加速生成阶段。在 TensorRT LLM 中，每个 Transformer 层有一个 KV cache，
这意味着 KV cache 的数量等于模型的层数。当前版本支持两种
不同的 KV cache：**连续（contiguous）** 和 **分页（paged）**。

#### 连续 KV Cache

连续 KV cache 是一个单一的大张量。其形状为：
```
[max_batch_size * max_beam_width, 2, num_heads, max_seqlen, hidden_dim_per_head].
```

当序列短于最大序列长度时，该实现使用的显存比实际需要的多得多（即使序列在生成大量输出 token 后接近上限，也需要很多步骤才能到达那个点）。

#### 分页 KV Cache

分页 KV cache 将 KV cache 分解为块，由缓存管理器在处理过程中分配给
不同的请求。缓存管理器跟踪序列，从池中分配新块并在需要时回收
这些块。参见
[`KVCacheManager`](source:tensorrt_llm/_torch/pyexecutor/resource_manager.py) 的实现。

#### INT8/FP8 KV Caches

在当前实现中，即使网络其余部分以 INT8 或 FP8 运行，注意力算子仍然使用 FP32、FP16 和 BFloat16 输入输出。然而，TensorRT LLM 支持 INT8 和 FP8
（`QuantMode.INT8_KV_CACHE` 和
`QuantMode.FP8_KV_CACHE`）KV cache。

注意力算子填充 KV cache。启用 INT8 或 FP8 KV cache 时，输入值必须使用缩放
因子量化到 8 位。量化时，缩放因子存储在
`kv_cache_scaling_factor` 张量中。其形状为 `[1]`，当前版本仅支持逐张量（per-tensor）
量化。量化使用倒数缩放因子，
因为在插件中以 `fp_value * (1.0 / kv_cache_scaling_factor)` 的方式相乘。

生成期间，从缓存读出的值在 MHA/MQA kernel 中即时反量化。反量化定义为
`quantized_value * kv_cache_scaling_factor`。

> 💡 **AI Infra 视角**：KV cache 量化的关键细节——**只有 K/V 被量化，计算本身还是高精度**（Q 和注意力输出保持 FP16/BF16）。量化/反量化发生在"写缓存时"和"读缓存时"，scale 是 per-tensor 的（当前实现）。KV cache 量化是显存优化的大杀器（INT8 减半、FP8 减半），精度损失通常可控——K/V 分布比较稳定。

### 滑动窗口注意力（Sliding Window Attention）与循环（滚动缓冲）KV Cache

TensorRT LLM 有一个叫 `Cyclic KV Cache` 的特性，把 kv cache 当作环形缓冲区。这意味着它只存储最近 N 个 token 的 kv cache，其中 N 由 `TrtllmAttention.forward` 中的 `attention_window_size` 参数决定。缓存满时，新 token 的 kv cache 会覆盖"最久未使用"的缓存。

在上下文阶段，如果输入长度超过 `attention_window_size`，
`Sliding Window Attention` 会被激活。它的作用与滑动窗口大小相同。

该特性有助于在处理非常长的序列时减少 kv cache 的显存占用。

_注意：cyclic kv cache 特性目前不能与 beam search 一起使用，因为上下文 kv cache 在 beams 之间共享。_

> 💡 **AI Infra 视角**：滑动窗口 = 只保留最近 N 个 token 的 KV（Mistral 等模型用这个设计）。前面 kvcache.md 讲过：出窗口的 KV 被释放，循环缓冲让"释放+覆盖"变得天然高效（不用移动数据，覆盖即可）。**注意它和前缀复用有冲突**（出窗口的 KV 没了），用长上下文 + 滑动窗口组合时要小心。

### Beam-Search

注意力算子支持 beam-search。在上下文阶段，每个输入序列计算单个 beam。在生成阶段，MHA/MQA/GQA kernel 使用一个额外的张量来为每个 beam 重建正确的路径。
该张量称为 `cache_indirection`。其形状为 `[batch_size,
beam_width, max_seqlen]`。

对于序列 `si`、beam `bi` 和 token `ti`，元素
`cache_indirection[si][bi][ti]` 是 0 到 `beam_width-1` 之间的整数，
指示从 KV cache 的哪条路径读取 K 和 V 元素。该张量在采样阶段填充。

> 💡 **AI Infra 视角**：beam search 在生成阶段每个请求同时推进 beam_width 条序列——但 KV cache 不用存 beam_width 份！因为多个 beam 共享同一个前缀（直到分支点）。`cache_indirection` 就是"路径指针表"：告诉 kernel 每个 beam 的每个位置该从哪个 beam 的 KV 读。这是 beam search 节省显存的关键实现细节。

### 输入 QKV 张量

输入 QKV 张量在隐藏状态投影后把 Q、K、V 张量打包（沿最后一维拼接）。它是一个 3D 张量。RoPE
和到 INT8 或 FP8 的量化（需要时）由 GPT attention 算子执行。

在打包模式下，其形状为 `[num_tokens, 3 * hidden_dim]`，其中
`num_tokens` 是 batch 中的 token 总数。对于上下文阶段的序列，序列的 token 数等于其输入
长度（即使 beam search 的 beam width 大于 `1`）。对于生成阶段的序列，每个序列有 `beam_width` 个 token。
每个序列的 beam width 可以不同。

以下伪代码解释了 token 数如何计算：

```python
num_tokens = 0

# 加上上下文阶段每个序列的长度。
for seq in context_phase:
    num_tokens += seq.length

# 加上生成阶段每个序列的 beam 宽度。
for seq in generation_phase:
    num_tokens += seq.beam_width
```

### 旋转位置嵌入（RoPE）

注意力算子可以执行旋转位置嵌入（RoPE）的计算。启用该操作时
（`rotary_embedding_dim` 设置为大于 0 的值），它会与其他操作融合。GPT 算子通过将
`position_embedding_type` 设置为 `PositionEmbeddingType.rope_gpt_neox`
或 `PositionEmbeddingType.rope_gptj` 支持 GPT-NeoX 和 GPT-J 两种形式的 RoPE。

> 💡 **AI Infra 视角**：RoPE 是现在几乎所有主流 LLM 使用的位置编码（Llama、Qwen、DeepSeek、Gemma...）。它的数学巧妙之处：把"位置信息"编码成旋转矩阵的形式，让 attention 天然感知"相对距离"。对推理引擎来说，RoPE 要每步对 K（有时 Q）做旋转——**融合进 attention kernel 就省一次单独的 kernel 启动和显存读写**（前面说过融合的价值）。

### ALiBi

注意力算子可以对 `Q*K^T` 的结果应用 ALiBi。偏置在优化 kernel 中根据 ALiBi 斜率即时计算。

### 缩放因子（Scaling factor）

在 MHA 中，`Q*K^T` 的输出乘以一个常量，计算方式为：

```
norm_factor = 1.f / (q_scaling * sqrt(head_size)).
```

### 交叉注意力（Cross Attention）

在 GPT 风格 decoder-only 模型所需的自注意力 MHA 之上，注意力算子还支持交叉注意力。

这使注意力算子可以更广泛地用作通用 decoder 组件。例如，Encoder-Decoder 模型在 Decoder 中使用它来同时发起自注意力和交叉注意力模块。
