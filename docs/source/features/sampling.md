<!--
  本文档为 TensorRT-LLM 官方 Sampling 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/features/sampling.md
-->

# 采样（Sampling）

PyTorch 后端支持多种多样的特性，如下表所示：

| 前向传播（Forward Pass） | 采样策略（Sampling Strategies） | 采样特性（Sampling Features） |
|--------------------|----------------------------------|--------------------------------|
| 无草稿（No drafting） | Greedy（贪心） | Guided Decoding（引导解码） |
| 草稿目标模型（Draft target model） | TopP | Plugging Logits Post-Processor（logits 后处理） |
| Eagle 3 | TopK | Temperature（温度） |
| Ngram | TopK + TopP | MinP |
| | Beam Search（束搜索） | Embedding / Logits Bias（偏置） |
| | Best of / n（可组合） | Stop criteria（停止条件） |
| | Rejection sampling（可组合） | Return Logits |
| | | Return LogProbs |
| | | TopK LogProbs |
| | | Penalties（惩罚项） |

> 💡 **AI Infra 视角**：先建立整体认知：**采样（sampling）= 决定"下一个 token 选谁"**。模型前向输出的是 logits（每个词一个分数），采样模块把它变成具体的 token。这张表的前两列（策略）管"怎么选"，第三列（特性）管"选的时候加什么约束"。逐行理解：
> - **Greedy**：永远选概率最大的 token——确定性强，但输出单调（容易重复）；
> - **TopK/TopP/Temperature/MinP**：给"选谁"加随机性——TopK 只从前 K 个里选，TopP 从累计概率达 p 的最小子集里选，temperature 缩放概率分布的尖锐程度（>1 变平更随机，<1 变尖更确定），MinP 过滤掉比最优 token 概率低 min_p 倍的 token；
> - **Beam Search**：同时维护多条候选序列（beam），最终返回最好的几条——质量优先，但显存和时间开销大，Chat 类模型一般不用；
> - **Best of n**：独立跑 n 次取最好——简单粗暴的质量提升法（OpenAI 的 `n` 参数就是这么实现的）。
> 生产经验：聊天场景主流用 temperature=0.x + top_p 组合；**要可复现（测试/评测）就 temperature=0（贪心）**。

## 一般用法

有两个可用的采样后端：

* Torch Sampler
* TRTLLM Sampler（已弃用）

Torch Sampler 是默认使用的，支持 TRTLLM Sampler 特性的超集。TRTLLM Sampler 将在 release 1.4 中移除。
可以通过以下方式显式指定使用哪个采样器：

```python
from tensorrt_llm import LLM

# 显式选择 TorchSampler
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8',
          sampler_type="TorchSampler")

# 显式选择 TRTLLMSampler
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8',
          sampler_type="TRTLLMSampler")
```

默认情况下，采样后端被选为 `auto`。这将为所有请求使用 Torch Sampler。

下面是一个使用基本采样参数的示例。这个示例准备了两个相同的 prompt，由于选择的采样参数不同，会得到不同结果：

```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8')
sampling_params = SamplingParams(
        temperature=1.0,
        top_k=8,
        top_p=0.5,
    )
llm.generate(["Hello, my name is",
            "Hello, my name is"], sampling_params)
```

也可以按 prompt 指定不同的采样参数：

```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8')
sampling_params_0 = SamplingParams(
        temperature=1.0,
        top_k=8,
        top_p=0.5,
    )
sampling_params_1 = SamplingParams(
        top_k=4,
    )
llm.generate(["Hello, my name is",
            "Hello, my name is"],
            [sampling_params_0,
            sampling_params_1])
```

> 💡 **AI Infra 视角**：注意最后这个例子：**同一个 batch 里的请求可以用不同的采样参数**。这在生产里很常见——比如一个服务里，聊天请求用 temperature=0.7，评测请求用 temperature=0。推理引擎要能支持"按请求粒度"的配置，而不是全局一套参数。另外，对服务端而言，**采样参数也是成本的一部分**：beam search 的显存和算力开销远大于 greedy，所以 Serving 层通常只开放"安全"参数子集给用户（如 OpenAI 兼容接口只暴露 temperature/top_p）。

### 模型生成配置默认值

PyTorch 后端可以使用模型 `generation_config.json` 中显式指定的兼容采样默认值。此行为是选择加入（opt-in）的：

```python
from tensorrt_llm import LLM

llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8',
          generation_config='auto')
```

对于 `trtllm-serve`，在命令行启用：

```bash
trtllm-serve nvidia/Llama-3.1-8B-Instruct-FP8 --generation-config auto
```

或在服务器 YAML 配置中：

```yaml
generation_config: auto
```

`generation_config` 选项有两种模式：

* `trtllm`（默认）保持 TRT-LLM 的采样行为和默认值。
* `auto` 从模型的 `generation_config.json` 加载支持的采样值。

在 `auto` 模式下，数值按以下顺序解析：

1. 请求中显式指定的值。
2. `generation_config.json` 中显式存在的值。
3. LLM API 或服务协议的现有默认值。

支持的字段有 `temperature`、`top_p`、`top_k`、`min_p`、
`repetition_penalty`、`no_repeat_ngram_size`、`length_penalty` 和
`early_stopping`（当其值为布尔或整数时）。Hugging Face Transformers 为 JSON 文件中缺失的字段合成的默认值不会被应用。

TRT-LLM 对 `eos_token_id`、BART 的 `forced_bos_token_id` 和 Whisper 抑制 token 的现有模型特定处理在两种模式下都保持生效。

> 💡 **AI Infra 视角**：`generation_config.json` 是 HuggingFace 模型自带的采样默认值文件（每个模型发布时定的）。`auto` 模式的语义：**请求没指定的参数，听模型的**——让模型作者的设计意图生效（比如某个模型官方推荐 temperature=0.6）。服务端做模型兼容层时这个机制很省心。

### 使用 Torch Sampler 时 LLM API 的采样行为

* 采样通过 `SamplingParams` 控制。

* 默认情况下（`temperature = top_p = top_k = None`），使用贪心采样（除非启用了 min-p 或 top-p decay，见下文）。使用 `generation_config='auto'` 时，模型的 `generation_config.json` 中显式指定的值会取代这些默认值；参见 [模型生成配置默认值](#model-generation-config-defaults)。

* 如果指定了 `temperature = 0`、`top_p = 0`、`top_k = 1` 和/或 `min_p = 1` 中的任何一个，则为贪心采样，与其余参数的值无关。

> 💡 **AI Infra 视角**：记这个等价关系：**temperature=0、top_k=1、top_p=0、min_p=1 都是贪心**。调试时如果发现"明明设了 top_p 怎么还是贪心"，先检查是不是这些值之一。OpenAI API 也这样：temperature=0 就是贪心。

* 否则，采样按指定的采样参数值进行，未指定的参数默认为 `top_k = 0`、`top_p = 1`、`min_p = 0`、`temperature = 1.0`：

  * logits 在应用 softmax 计算概率之前先按 `1/temperature` 缩放。采样根据这些概率进行。

  > 💡 **AI Infra 视角**：temperature 的数学含义：logits 除以 temperature 再 softmax。temperature < 1 时概率分布变"尖"（差距放大，更倾向高分 token）；> 1 时变"平"（更随机）。temperature → 0 时退化为 argmax（贪心）。这是所有采样策略的底层第一步。

  * 如果 `top_k = 0`（或 `top_k = vocab_size`）、`top_p = 1` 且 `min_p = 0`，则从整个词表中采样输出 token。

  * 如果指定了 `0 < min_p < 1`，采样被限制在概率至少为最可能 token 概率的 `min_p` 倍的 token 中（"min-p 采样"）。与 `top_k` 和/或 `top_p` 结合时，`min_p` 先应用。

  * 如果指定了 `1 < top_k < vocab_size`，采样被限制在概率最高的 `top_k` 个 token 中。

  * 如果指定了 `0 < top_p < 1.0`，采样被进一步限制在总概率大于 `top_p` 的最小高概率 token 子集中（"核采样 nucleus sampling"）。特别是，所选子集中概率最低的 token 的概率大于或等于任何未选中 token 的概率。与 `top_k` 结合时，`top_k` 选中的 token 的概率会重新缩放，使它们在应用 `top_p` 前总和为 1。

  * 实现不保证并列概率（tied probabilities）的任何特定处理方式。

  > 💡 **AI Infra 视角**：TopK vs TopP 的直觉：TopK 是"固定选前 K 个"——当分布很平（大量 token 概率接近）时 K 个可能太多或太少；TopP 是"动态选"——从最高概率往下加，加到累计概率超过 p 为止。所以 TopP（核采样）更自适应，业界更常用（HF 的 generation 默认就是 TopP）。组合用 top_k=40 + top_p=0.9 也是常见做法（先硬切，再动态切）。

* 支持 Top-P decay：如果指定 `top_p_decay < 1`，有效 `top_p` 会在每次采样 token 后乘以 `top_p_decay`，下限为 `top_p_min`（默认 `1e-6`），并且在采样到 token `top_p_reset_ids` 时重置为初始 `top_p`（默认 `-1`，永不匹配任何 token）。范围外的值（`top_p_decay` 或 `top_p_min` 在 `(0, 1]` 之外、`top_p_reset_ids` 为负）会被拒绝。

  * 活跃的 top-p decay 意味着即使 `top_p` 未指定或 `top_p = 1` 也进行 top-p 采样（初始 `top_p` 然后默认为 1）。然而，显式请求的贪心采样（`temperature = 0`、`top_p = 0` 和/或 `top_k = 1`）优先于 top-p decay。

  * Top-P decay 不支持与 beam search 或经由 Torch Sampler 路由草稿 token 的投机解码模式结合；此类请求会被拒绝。

* 正数 Min-P 不支持与单模型投机解码结合。此类请求在准入（admission）时被拒绝。

* 支持出现惩罚（occurrence penalties）：`repetition_penalty`、`presence_penalty` 和 `frequency_penalty` 用于阻止（或鼓励）模型重用已见过的 token。三者都在温度缩放之前重写 logits，由 prompt 加已生成内容的出现历史驱动。记 `c` 为该历史中 token 出现的次数：

  * `repetition_penalty`（默认 `1.0`）重新缩放每个 `c > 0` 的 token 的 logit：logit 为非负时除以惩罚值，为负时乘以惩罚值。两个分支对正负 logit 的移动方向一致，因此值 `> 1` 总是压低已见 token，值 `< 1` 总是抬高它。必须 `> 0`。

  * `presence_penalty`（默认 `0.0`）从每个 `c > 0` 的 token 中减去惩罚值本身。数额不依赖 `c`，所以它控制 token 是否重新出现，而不是出现的频率。

  * `frequency_penalty`（默认 `0.0`）减去惩罚值乘以 `c`，所以 token 被产生的次数越多，被压得越狠。

  * `prompt_ignore_length`（默认 `0`）将前 N 个 prompt token 排除在 presence 和 frequency 计数之外。被忽略的 token 仍计入 `repetition_penalty`。值 `<= 0` 无效，大于 prompt 长度的值会被截断到 prompt 长度。

  * 出现惩罚不支持与 beam search 结合；此类请求会被拒绝。

  > 💡 **AI Infra 视角**：三个惩罚项的区分（面试/写代码都要分清）：
  > - `repetition_penalty`（HuggingFace 系常用，如 1.1）：对**出现过的词**统一打压，管"同一种说法是否反复出现"；
  > - `presence_penalty`（OpenAI 系）：只要出现过就罚一次固定值——管"新词多不多"；
  > - `frequency_penalty`（OpenAI 系）：出现过几次罚几倍——管"高频词被压低多少"。
  > 它们的共同点：**在 logits 上直接加减乘**，不重新采样。服务端暴露这些参数时要注意别让用户把 temperature 和 penalties 同时设到"互相打架"的极端值。

* 如果指定 `no_repeat_ngram_size = n`，任何会重建序列中（含 prompt）已存在的 `n`-gram 的 token 都会被排除在采样之外。`None` 或 `0` 禁用此限制。

### 性能

Torch Sampler 利用 [FlashInfer](https://docs.flashinfer.ai/api/sampling.html) 提供的优化采样 kernel，这是 Torch Sampler 的必需依赖。采样器还尽可能使用[免排序实现](https://flashinfer.ai/2025/03/10/sampling.html)。这种优化不计算完整的 token 采样概率集合（在 top-k / top-p 掩码之后等），通常可以省略，除非用户要求或投机解码（拒绝采样）需要。

此外，Torch Sampler 在内部将具有兼容采样参数的请求分组批处理。当请求批次由采样策略高度异构的请求组成时（例如混合使用 greedy 和 top-p-after-top-k 采样的请求），这可以大大降低采样步骤的整体延迟。

> 💡 **AI Infra 视角**：两个性能细节值得学习：
> 1. **采样也要优化 kernel**：很多人以为采样只是"torch.argmax 一下"，其实 top-k/top-p 需要在词表维度（几万到十几万）上做排序/筛选——GPU kernel 化后（FlashInfer）比 CPU 实现快一个数量级。**推理链路上没有小事，采样也可能成为瓶颈**；
> 2. **按参数分组批处理**：不同采样参数的请求在 GPU 上无法共享同一个采样 kernel 调用，引擎把参数相同的请求凑成一组，减少 kernel 启动次数。这是"动态批处理"在采样阶段的具体应用。

## 高级采样模式（投机解码）

对于单模型投机解码（如 MTP-Eagle one-model），每个请求的高级采样器会在采样每个草稿/目标 token 前应用 `top_k` 掩码、温度 softmax 和 `top_p` 过滤。当部署固定了采样配置，使得某个过滤总是被禁用（`top_k = 0` / `top_k = vocab_size`，或 `top_p = 1`）时，该过滤的 kernel 就是纯开销。

`advanced_sampling_mode`（在 `DecodingBaseConfig` 上，因此任何投机配置都可用）允许你为固定的部署配置跳过这些冗余 kernel。只要被跳过的过滤器本来就已禁用，输出与 `FULL` 完全相同，因此这对高级用例来说是一种无损的吞吐优化：

| 模式 | `top_k` kernel | `top_p` kernel |
|---|---|---|
| `full`（默认） | 应用 | 应用 |
| `no_topk` | **跳过** | 应用 |
| `no_topp` | 应用 | **跳过** |
| `no_topk_no_topp` | **跳过** | **跳过** |

> 💡 **AI Infra 视角**：这体现了一个通用优化思路：**固定部署配置下的冗余 kernel 直接跳过**。生产环境里采样配置通常是固定的（比如只用 temperature），那 top-k/top-p 的 kernel 每次都在空转——用 `advanced_sampling_mode` 告诉引擎"别跑了"，无损提速。这类"按部署配置裁剪计算图"的思想（也是 CUDA Graph 的原理）在 AI Infra 处处可见。

注意事项：

* `full` 是默认且始终安全的；特化是选择加入（opt-in）的。
* `advanced_sampling_mode` 与 `use_rejection_sampling` 相互独立：每种模式都可以搭配拒绝采样开或关；该标志不再限制模式选择。
* `no_topp` 和 `no_topk_no_topp` 禁用 `top_p`，将采样器从融合的 `top_p_sampling_from_probs` 切换到更便宜的 `sampling_from_probs`；`no_topk` 保留 `top_p`。
* 贪心请求被原生处理（通过一个哨兵温度使 softmax 坍缩为 one-hot argmax），因此任何模式都支持贪心 + 采样混合批次，无需特判。
* `advanced_sampling_mode` 是部署时的选择；它*不*是 CUDA graph 键的一部分，因此不会增加额外的预热图。

```python
from tensorrt_llm.llmapi import MTPDecodingConfig

spec_config = MTPDecodingConfig(
    max_draft_len=3,
    advanced_sampling_mode="no_topk_no_topp",  # 仅温度的部署配置
)
```

## Beam search（束搜索）

Beam search 是一种解码策略，在文本生成期间维护多条候选序列（beams），探索不同的可能续写以找到更高质量的输出。与贪心解码或采样不同，beam search 同时考虑多个假设。

要启用 beam search，你必须：

1. 在 `SamplingParams` 对象中启用 `use_beam_search` 选项
2. 将 `LLM` 类中的 `max_beam_width` 参数设置为与 `SamplingParams` 中的 `best_of` 参数匹配

参数配置：
- `best_of`：控制生成期间处理的 beam 数（beam width）
- `n`：控制返回的输出序列数（可以小于 `best_of`）
- 如果省略 `best_of`，处理的 beam 数默认为 `n`
- `LLM` 类中的 `max_beam_width` 必须等于 `SamplingParams` 中的 `best_of`

下面的示例演示了 beam width 为 4、返回 top 3 序列的 beam search：

```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8',
          max_beam_width=4,   # 必须等于 SamplingParams.best_of
    )
sampling_params = SamplingParams(
        best_of=4,   # 必须等于 LLM.max_beam_width
        use_beam_search=True,
        n=3,         # 返回 top 3 序列
    )
llm.generate(["Hello, my name is",
            "Hello, my name is"], sampling_params)
```

> 💡 **AI Infra 视角**：beam search 的成本直觉：beam=4 意味着**同时推进 4 条序列**——4 倍的前向计算和 4 倍的 KV cache。所以它不是默认选项，而是"质量优先"场景（翻译、摘要评测）的专用工具。对服务端：`max_beam_width` 是构建时就要定死的（影响显存预留），改它要重新构建——这就是为什么上面强调"必须等于"。

## Logits processor

Logits processor 允许你在采样前修改网络产生的 logits，从而实现自定义的生成行为和约束。

要使用自定义 logits processor：

1. 创建一个继承自 [`LogitsProcessor`](source:tensorrt_llm/sampling_params.py#L48) 并实现 `__call__` 方法的自定义类
2. 将该类的实例传给 `SamplingParams` 的 `logits_processor` 参数

下面的示例演示了 logits 处理：

```python
import torch
from typing import List, Optional

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.sampling_params import LogitsProcessor

class MyCustomLogitsProcessor(LogitsProcessor):
    def __call__(self,
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int],
        client_id: Optional[int]
    ) -> None:
        # 在这里实现你的自定义原地 logits 处理逻辑
        logits *= logits

llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8')
sampling_params = SamplingParams(
        logits_processor=MyCustomLogitsProcessor()
    )
llm.generate(["Hello, my name is"], sampling_params)
```

> 💡 **AI Infra 视角**：logits processor 是"采样前最后一道编辑关卡"——温度、惩罚、top-k 掩码全在这一层实现。自定义用途：强制输出格式（JSON schema 的字符级约束）、屏蔽敏感词、禁止特定 token 序列等。生产中的引导解码（guided decoding，阶段 3 会讲）本质上也是在这层做约束。注意它是**原地修改（in-place）**的——直接在 logits 张量上改，避免拷贝开销。

更详细的 logits processor 示例见[这里](source:examples/llm-api/llm_logits_processor.py)。
