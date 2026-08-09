<!--
  本文档为 TensorRT-LLM 官方 PyTorch Backend Attention 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/torch/attention.md
-->

(attention)=

# 注意力（Attention）

本文档详细说明 TensorRT-LLM PyTorch 后端中自回归模型的多头注意力（MHA）、
多查询注意力（MQA）和分组查询注意力（GQA）的实现。快速回顾：多头注意力
包含一系列批处理矩阵乘法、一次 softmax 运算和又一次批处理矩阵乘法，
如 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 论文所述。
[多查询注意力（MQA）](https://arxiv.org/abs/1911.02150) 和 [分组查询注意力（GQA）](https://arxiv.org/abs/2307.09288) 是
MHA 的变体，使用的 KV 头数少于查询头数。
TensorRT LLM 在 `tensorrt_llm/_torch/attention_backend/` 中提供了使用不同后端的多种实现。
以下章节说明如何使用这些实现，并提供实现新后端的简要指南。

> 💡 **AI Infra 视角**：本篇与 [features/attention.md](../features/attention.md) 内容基本一致（后者多了 TrtllmAttention 特性的详细介绍），此处只保留后端对比和实现指南，详细讲解请看那篇。**为什么有两份？** features/ 面向使用者（选后端、配参数），torch/ 面向开发者（写新后端）——**同一主题按读者分层**是优秀文档体系的做法。

## 注意力后端

目前有三个可用的注意力后端：vanilla 后端、TRT-LLM 后端和 Flashinfer 后端。
你可以使用 `PyTorchConfig.attn_backend` 指定想要的注意力后端。例如，要使用 Flashinfer 后端，可以给 `LLM` 构造函数传 `attn_backend="flashinfer"`：`LLM(attn_backend="flashinfer")`。这将为你的模型启用 Flashinfer 后端。

vanilla 后端 `VanillaAttention` 是一个参考实现，主要设计用于飞行中批处理和线性 KV cache 支持。虽然它可以作为有用的基线，但由于优化有限，不建议在生产中使用。

相比之下，Flashinfer 后端 `FlashInferAttention` 经过性能优化，支持飞行中批处理和分页 KV cache。它还包括以下高级特性：

1. **FP8 量化**：此特性支持将输入和 KV cache 量化为 FP8 格式，显著降低显存占用并提高计算吞吐。
2. **RoPE 融合**：通过将旋转位置编码（RoPE）直接集成到注意力计算中，提高效率并降低开销。

TRT-LLM 后端 `TrtllmAttention` 是默认后端，支持 Flashinfer 后端的所有特性，并进一步优化以获得更好的性能。它是生产环境的推荐选择。此外，它还提供以下高级特性：

1. **融合 QKV 输入**：可以接受单个 QKV 张量作为输入，比使用单独的 Q、K、V 张量更高效。
2. **FP8 输出**：支持以 FP8 格式输出注意力结果，将量化融合到注意力计算过程中。

## 实现一个新的注意力后端

你可以实现一个新的注意力后端来集成其他注意力库。
一个注意力后端由 `AttentionBackend` 类和 `AttentionMetadata` 类组成。
PyTorch 有三个涉及注意力后端的阶段：

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
