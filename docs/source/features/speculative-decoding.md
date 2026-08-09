<!--
  本文档为 TensorRT-LLM 官方 Speculative Decoding 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/features/speculative-decoding.md
-->

# 投机解码（Speculative Decoding）

投机解码是一种在低 batch 大小下加速 LLM 推理的技术。一个轻量级的草稿（drafting）机制提出候选 token，目标模型在**一次前向传播**中验证它们。匹配的 token 被接受，从而减少所需的串行前向传播次数。

> 💡 **AI Infra 视角**：投机解码为什么能加速？核心洞察：**自回归生成是串行的**——每步只能等上一个 token 算完才能算下一个，而 GPU 在单 token 的 decode 上算力严重闲置（一个 token 的计算只占 GPU 很小一部分）。投机解码的思路：
> 1. **草稿模型（draft model）**：一个很小的模型（或轻量模块）快速猜出接下来 K 个 token（猜得快，因为小）；
> 2. **目标模型（target model）**：一次前向验证这 K 个 token 的概率；
> 3. **接受/拒绝**：概率匹配的 token 直接采用（最多一次前向拿 K 个 token！），不匹配的从那里重新开始。
> 效果：**把 K 次串行前向压缩成 1 次**——大模型算力不再闲置。代价：草稿模型的计算开销 + 接受率（acceptance rate）不高时会**更慢**（白白多算）。
> **适用前提**：低 batch（高 batch 时 GPU 本来就不闲，投机没收益）。**"投机解码的原理和适用场景"是面试必考题**。

## 快速开始

对于所有投机算法，启用投机时，每个请求都会创建一条长度为 `max_draft_len` 的草稿 token 序列。目前没有办法动态禁用投机，因此只有在低 batch 大小下才能观察到加速效果。

> 💡 **AI Infra 视角**：`max_draft_len` 是核心参数（默认 3~7）：一次草稿几个 token。设大了：每轮验证的 token 多，但草稿模型猜错的概率也高、浪费增多；设小了收益有限。生产实践通常从 3~5 开始调。**注意"没有动态禁用"**——所以服务配置了投机，高并发时反而可能变慢，要按峰值负载做决策。

### 草稿/目标（Draft/Target）

草稿/目标是投机解码最简单的形式。这种方法用一个任意的草稿模型来产生草稿 token。务必确保草稿模型和目标模型使用**相同的 tokenizer** 训练，否则接受率极低，性能会倒退。

```python
from tensorrt_llm.llmapi import DraftTargetDecodingConfig

# 选项 1：使用 HuggingFace Hub 模型 ID（自动下载）
speculative_config = DraftTargetDecodingConfig(
    max_draft_len=3, speculative_model="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")

# 选项 2：使用本地路径
# speculative_config = DraftTargetDecodingConfig(
#     max_draft_len=3, speculative_model="/path/to/draft_model")

llm = LLM("/path/to/target_model", speculative_config=speculative_config, disable_overlap_scheduler=True)
```

> 💡 **AI Infra 视角**：经典的自监督小模型方案：拿一个 1B 小模型给 70B 大模型打草稿。"tokenizer 必须一致"是硬前提——词表不同，token id 对不上，验证无从谈起。注意代码里 `disable_overlap_scheduler=True`：投机解码和重叠调度有交互问题（草稿/验证流程破坏了重叠流水线的时序），**配置组合的兼容性是 AI Infra 排查的老大难，文档里的每个标志都有存在的原因**。

### EAGLE 3

EAGLE 3 算法描述在论文 [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/pdf/2503.01840) 中。
默认情况下，每个请求使用一条长度为 `max_draft_len` 的草稿 token 序列（线性链）。可选地，可以启用动态树草稿生成来提高接受率——见下面的 [动态树模式](#dynamic-tree-mode)。

以下草稿模型 checkpoint 可用于 EAGLE 3：
* Llama 3 变体：[使用原始 EAGLE 3 论文作者的 checkpoint](https://huggingface.co/yuhuili)。
* Llama 4 Maverick：[使用 NVIDIA HuggingFace 仓库的 checkpoint](https://huggingface.co/nvidia/Llama-4-Maverick-17B-128E-Eagle3)。
* 其他模型，包括 `gpt-oss-120b` 和 `Qwen3`：查看 NVIDIA 的 [投机解码模块集合](https://huggingface.co/collections/nvidia/speculative-decoding-modules)。

```python
from tensorrt_llm.llmapi import Eagle3DecodingConfig

model = "meta-llama/Llama-3.1-8B-Instruct"
speculative_model = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"

speculative_config = Eagle3DecodingConfig(
    max_draft_len=3,
    speculative_model=speculative_model)

llm = LLM(model, speculative_config=speculative_config)
```

EAGLE 3 可以与[后缀自动机增强](#suffix-automaton-sa-enhancement)结合，提高重复内容上的接受率。详见下文 SA 章节。

> 💡 **AI Infra 视角**：EAGLE 系列是目前效果最好的投机解码方案之一。它和"小模型草稿"的区别：EAGLE 的草稿器不是独立模型，而是**挂在目标模型主干上的轻量模块**（用目标模型中间层的特征做预测）——比单独跑一个小模型更准（和主模型共享表征）。MTP（Multi-Token Prediction）是 DeepSeek 的同类方案，**两者都是"训练时就为投机解码预留的模块"**（训练成本换推理加速，这叫 training-time inference optimization）。

#### 动态树模式（Dynamic Tree Mode）

动态树模式为 EAGLE 3 启用树状草稿生成：草稿器在每一层展开多个候选 token，而不是单个 token。与线性草稿相比，这可以提高接受率，代价是每个生成步骤的额外计算。

要启用动态树模式，在 `Eagle3DecodingConfig` 上设置 `use_dynamic_tree=True` 并提供以下参数：

* `use_dynamic_tree`（`bool`）：启用动态树草稿生成。与 `eagle_choices`（静态树）互斥。
* `dynamic_tree_max_topK`（`int`）：每层每个节点最多展开的 token 数。
* `max_total_draft_tokens`（`int`，可选）：树的总草稿 token 预算。必须满足 `max_draft_len <= max_total_draft_tokens <= dynamic_tree_max_topK * max_draft_len`。未设置时默认为 `dynamic_tree_max_topK * max_draft_len`。

当 `use_dynamic_tree=True` 时，动态树 CUDA 缓冲区基于 `LLM` 的 `max_batch_size` 预分配，该值在内部传播，**不能**直接传给 `Eagle3DecodingConfig`。

```python
from tensorrt_llm.llmapi import Eagle3DecodingConfig

speculative_config = Eagle3DecodingConfig(
    max_draft_len=6,
    speculative_model="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    use_dynamic_tree=True,
    dynamic_tree_max_topK=10,
    max_total_draft_tokens=60,
)

llm = LLM("/path/to/target_model", speculative_config=speculative_config)
```

> 💡 **AI Infra 视角**：为什么树比链好？链式草稿只要第 2 个 token 猜错，后面全作废。树状草稿一次展开多条分支（第 1 层 10 个候选，第 2 层每分支再展开……），目标模型一次验证整棵树——**只要有一条分支猜对就算赚**。树结构 + 验证 = 用"一次前向验证更多 token"换接受率。代价是验证时的计算量（树越大，一次前向的 token 越多）。

```{note}
动态树模式目前**不支持**使用滑动窗口注意力或 MLA（多潜变量注意力）的模型，如 DeepSeek 和 gpt-oss 模型。
```

### NGram

NGram 方法是[这个 Prompt Lookup Decoding 算法](https://github.com/apoorvumang/prompt-lookup-decoding)的实现。

使用 NGram 算法时，TRT-LLM 会维护一个从 token 前缀到候选草稿序列的映射。例如，3-gram ["The ", " future ", " is"] 可以映射到草稿序列 [" bright", " because"]。前缀是来自 prompt 和目标模型生成的 token 序列。NGram 池和匹配过程可以用以下选项调优：

* `max_draft_len`：最大草稿候选长度。
* `max_matching_ngram_size`：与池中键匹配的最大 prompt 后缀长度。
* `is_public_pool`：为 true 时，所有请求共享一个 ngram 池。否则，每个请求有自己的 ngram 池。
* `is_keep_all`：为 true 时，草稿候选会永远保留在池中。否则，只保留最大的草稿候选。
* `is_use_oldest`：为 true 时，对于给定匹配总是提出最旧的草稿候选。否则使用最新的草稿候选。仅当 `is_keep_all == True` 时适用，因为 `is_keep_all == False` 意味着每个键永远只有一个值。

```python
from tensorrt_llm.llmapi import NGramDecodingConfig

speculative_config = NGramDecodingConfig(
    max_draft_len=3, max_matching_ngram_size=4, is_public_pool=True)

llm = LLM("/path/to/target_model", speculative_config=speculative_config, disable_overlap_scheduler=True)
```

> 💡 **AI Infra 视角**：NGram 是**无模型（model-free）**的草稿方案——不需要任何草稿模型！它的"猜测"完全靠文本统计：从 prompt 和已生成内容里找"前缀重复"（比如用户问题里出现过 "The future is..."，后面的 "bright" 大概率会重复出现）。适用于**重复性强的场景**（代码生成、翻译、检索内容拼接），对全新内容无效。**"投机解码不一定需要草稿模型"——这是它和 EAGLE/MTP 的显著区别**。

### MTP

MTP 由 DeepSeek 模型和其他自带原生 MTP 模块的架构（包括 Step-3.x）支持。MTP 可以用以下配置选项调优：

* `max_draft_len`：最大草稿候选长度。
* `num_nextn_predict_layers`：使用的 MTP 模块数量。目前必须等于 `max_draft_len`。
* `use_relaxed_acceptance_for_thinking`：为 true 时，对推理模型（reasoning model）在思考（thinking）阶段使用宽松解码。在此模式下，思考阶段的投机要求被放宽——如果草稿 token 出现在用 `relaxed_topk` 和 `relaxed_delta` 构造的候选集中，就可以被接受。
* `relaxed_topk`：从目标模型的 logits 中采样前 K 个 token，创建宽松解码的初始候选集。
* `relaxed_delta`：用于进一步过滤宽松解码的前 K 候选集。我们移除满足 `log(P(top 1 token)) - log(P(t)) > relaxed_delta` 的 token `t`。

```python
from tensorrt_llm.llmapi import MTPDecodingConfig

speculative_config = MTPDecodingConfig(max_draft_len=3)

llm = LLM("/path/to/deepseek_model", speculative_config=speculative_config)
```

MTP 可以与[后缀自动机增强](#suffix-automaton-sa-enhancement)结合，提高重复内容上的接受率。详见下文 SA 章节。

> 💡 **AI Infra 视角**：MTP 是 DeepSeek-V3/R1 训练时就内置的多 token 预测头（模型同时被训练"预测下一个 token"和"预测下 N 个 token"）。`relaxed_acceptance_for_thinking` 很巧妙：**推理模型（R1 这类）的"思考"阶段本来就不确定性高、草稿经常猜错**——与其严格拒绝，不如放松标准（候选集里有就收），提高思考阶段的加速比。**"按模型行为特征定制解码策略"**是高级调优思路。

### PARD

PARD（PARallel Draft，并行草稿）是一种与目标无关的投机解码方法，使用掩码 token（mask tokens）在一次前向传播中预测所有草稿 token。与 MTP 或 EAGLE 3 逐个生成草稿 token 不同，PARD 并行产生 K 个草稿 token。

参考：[PARD: Parallel Drafting for Speculative Decoding](https://arxiv.org/pdf/2504.18583)

* `max_draft_len`：最大草稿候选长度。
* `speculative_model`：PARD 草稿模型的路径或 HuggingFace 模型 ID。
* `mask_token_id`：用于并行预测的掩码 token 的 token ID。未设置时从草稿模型配置读取。

```python
from tensorrt_llm.llmapi import PARDDecodingConfig

speculative_config = PARDDecodingConfig(
    max_draft_len=4, speculative_model="/path/to/pard_model")

llm = LLM("/path/to/target_model", speculative_config=speculative_config)
```

PARD 可以与[后缀自动机增强](#suffix-automaton-sa-enhancement)结合，提高重复内容上的接受率。详见下文 SA 章节。

> 💡 **AI Infra 视角**：PARD 的"掩码 token"思路：把"要预测的 K 个位置"用特殊掩码 token 占位，一次性喂给草稿模型做并行预测（类似 BERT 的掩码预测，但用于生成）——草稿不再串行，一次前向出 K 个候选。分类：**依赖目标 vs 不依赖目标**（draft/target 和 EAGLE 可以用任意目标模型，MTP 依赖特定模型内置模块），PARD 属于不依赖目标。

### DFlash

DFlash 是一种依赖目标的投机解码方法，使用目标模型特定层的隐藏状态作为草稿模型中的交叉注意力上下文，并行预测多个草稿 token。

参考：[DFlash: Distilled Flash Speculative Decoding](https://arxiv.org/pdf/2602.06036)

* `max_draft_len`：最大草稿候选长度。
* `speculative_model`：DFlash 草稿模型的路径或 HuggingFace 模型 ID。
* `mask_token_id`：用于并行预测的掩码 token 的 token ID。未设置时从草稿模型配置读取。
* `target_layer_ids`：目标模型层索引列表，这些层的隐藏状态被捕获用于草稿模型中的交叉注意力。未设置时从草稿模型配置读取。

```python
from tensorrt_llm.llmapi import DFlashDecodingConfig

speculative_config = DFlashDecodingConfig(
    max_draft_len=4, speculative_model="/path/to/dflash_model")

llm = LLM("/path/to/target_model", speculative_config=speculative_config)
```

> 💡 **AI Infra 视角**：DFlash 比 EAGLE 更进一步：不只是共享主干特征，而是把目标模型的隐藏状态直接作为交叉注意力的 key/value 喂给草稿模型——草稿模型"看着"大模型当前在想什么来猜下一个词，猜测质量更高。这类方案的成本是**多模型间的数据传输**（目标模型的中间层激活要传给草稿模型）。投机解码的演进史：小模型草稿 → 共享主干（EAGLE）→ 交叉注意力（DFlash）→ 训练内置（MTP），**每一步都在提高"猜中率"**。

### 用户自定义草稿
可以通过 `UserProvidedDecodingConfig` 提供完全自定义的草稿方法，包含
* `max_draft_len`：最大草稿候选长度。
* `drafter`：实现 `prepare_draft_tokens` 方法的 `Drafter` 对象（见[开发者指南](speculative-decoding.md#developer-guide)第 7 节）
* `resource_manager`：可选的 `ResourceManager` 对象（见[开发者指南](speculative-decoding.md#developer-guide)第 4 节）

```python
from tensorrt_llm.llmapi import UserProvidedDecodingConfig

speculative_config = UserProvidedDecodingConfig(
    max_draft_len=3, drafter=MyDrafter())

llm = LLM("/path/to/target_model", speculative_config=speculative_config)
```

## 后缀自动机（SA）增强

后缀自动机（Suffix Automaton，SA）是一个**无模型**、基于 GPU 的模式匹配草稿增强器。它在已生成的 token 中寻找后缀匹配，当匹配足够长时提出草稿 token。SA 在匹配时非常准确（精确模式重复），而神经方法对全新内容更好——**两者结合，两全其美**。

> 💡 **AI Infra 视角**：SA 的直觉：LLM 生成大量重复性文本（代码里的样板、JSON 里的字段、对话里的问候语）。SA 用字符串匹配算法（后缀自动机）在"已经生成的内容"里找和当前后缀相同的旧片段，直接把旧片段后面的 token 作为草稿——**零模型开销，纯字符串查找**。它和 NGram 类似（无模型），但实现机制不同（自动机 vs n-gram 池）。生产里"重复率高的负载 + 投机解码"是标配组合。

SA 可以与以下投机解码技术结合：

* **MTP**（`MTPDecodingConfig`）
* **EAGLE 3**（`Eagle3DecodingConfig`）
* **PARD**（`PARDDecodingConfig`）

要启用 SA 组合，在投机配置上设置 `use_sa_spec=True`。`sa_spec_threshold` 参数控制覆盖神经草稿所需的最小后缀匹配长度（默认：4）。

```python
from tensorrt_llm.llmapi import Eagle3DecodingConfig

speculative_config = Eagle3DecodingConfig(
    max_draft_len=4,
    speculative_model="/path/to/eagle3_model",
    use_sa_spec=True,
    sa_spec_threshold=4)

llm = LLM("/path/to/target_model", speculative_config=speculative_config)
```

SA 也可以通过 `SADecodingConfig` 作为独立的投机解码技术使用：

```python
from tensorrt_llm.llmapi import SADecodingConfig

speculative_config = SADecodingConfig(max_draft_len=4)

llm = LLM("/path/to/target_model", speculative_config=speculative_config)
```

> 💡 **AI Infra 视角**：SA 的组合逻辑："谁猜得准用谁"——后缀匹配足够长时（≥ threshold）用 SA 的字符串候选（纯重复场景必中），否则用神经草稿（新内容更擅长）。**把两种互补机制组合，是工程上提升平均接受率的经典套路**。

## 与 `trtllm-bench` 和 `trtllm-serve` 一起使用

```{eval-rst}
.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-config-flag-alias
   :end-before: .. end-note-config-flag-alias
```

`trtllm-bench` 和 `trtllm-serve` 的投机解码选项都必须通过 `--config config.yaml` 指定。所有投机解码选项都可以在这个 YAML 文件中指定。额外的 `decoding_type` 选项用于指定投机的类型。可用选项：

* `MTP`
* `Eagle3`
* `NGram`
* `DraftTarget`
* `PARD`
* `DFlash`
* `SA`

> 注：PyTorch 后端只支持 `Eagle3`。`decoding_type: Eagle` 作为 `Eagle3` 的向后兼容别名被接受，但 EAGLE（v1/v2）草稿 checkpoint 不兼容。

其余参数名/合法值与快速开始章节中对应配置类相同。例如，YAML 配置可以是：

```yaml
# 使用 HuggingFace Hub 模型 ID（自动下载）
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 4
  speculative_model: yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
```

```yaml
# 或使用本地路径
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 4
  speculative_model: /path/to/draft/model
```

```yaml
# 动态树模式
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 6
  speculative_model: /path/to/eagle3_model
  use_dynamic_tree: true
  dynamic_tree_max_topK: 10
  max_total_draft_tokens: 60
  max_batch_size: 4
```

```yaml
# SA 组合：为任何受支持的技术启用后缀自动机增强
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 4
  speculative_model: /path/to/draft/model
  use_sa_spec: true
  sa_spec_threshold: 4
```

```{note}
字段名 `speculative_model_dir` 也可以作为 `speculative_config.speculative_model` 的别名使用。例如：

    speculative_config:
      decoding_type: Eagle3
      max_draft_len: 4
      speculative_model_dir: /path/to/draft/model
```
