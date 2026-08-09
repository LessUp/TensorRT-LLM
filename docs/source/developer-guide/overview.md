<!--
  本文档为 TensorRT-LLM 官方 Architecture Overview 的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/developer-guide/overview.md
-->

# 架构总览

`LLM` 类是 TensorRT LLM 的核心入口，提供了简化的 `generate()` API 用于高效的大语言模型推理。这个抽象旨在简化用户体验，如 TinyLlama 示例所示：

```python
from tensorrt_llm import LLM

# 用指定模型初始化 LLM
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 用模型生成文本
output = llm.generate("Hello, my name is")
```

`LLM` 类自动管理必要的前处理和后处理步骤，包括分词（tokenization，将输入 prompt 编码为数值表示）和反分词（detokenization，将模型输出解码回人类可读的文本）。

> 💡 **AI Infra 视角**：tokenization/detokenization 是推理链路两端的"翻译官"：模型只认识 token id（整数序列），不认识文字。这是每次请求都必经的 CPU 开销，在高吞吐场景下也是优化点之一（比如 tokenizer 优化、缓存）。理解 `LLM` 对象 = 模型 + tokenizer + 执行器的组合，是读懂 TRT-LLM 所有代码的前提。

在内部，`LLM` 类在每个 rank（进程）上编排创建了一个独立的 `PyExecutor(Worker)` 进程。

![TensorRT LLM 架构总览](../media/TRTLLM_Architecture_Overview.png)

> 💡 **AI Infra 视角**："每个 rank 一个 PyExecutor 进程"是分布式推理的关键设计：
> - **rank** = 参与推理的一个 GPU 进程实例。用 8 张卡跑 TP=8 的模型，就会启动 8 个进程，每个进程一个 PyExecutor；
> - 主进程（用户进程）通过 IPC/队列把请求发给各 rank 的 executor，executor 之间再通过 NCCL（NVIDIA 的集合通信库）同步张量。
> - 这种"多进程 + 消息传递"的架构避免 Python 的 GIL 限制，也是 vLLM 等引擎的常见做法。

这个 `PyExecutor` 运行在一个连续的后台循环中，专为高效、异步地处理推理请求而设计。

> 💡 **AI Infra 视角**：**异步 + 后台循环** 是推理引擎与训练框架的显著区别之一。训练是"等这一批算完再算下一批"，而推理服务要持续接收新请求——所以引擎用事件循环不断"取请求 → 调度 → 执行 → 返回结果"。理解这个循环，就理解了 PyExecutor 的全部工作。

`PyExecutor` 的功能建立在几个关键组件之上：

- **`Scheduler`（调度器）**：负责判断在每个处理步骤中哪些活动请求可以执行。

> 💡 **AI Infra 视角**：调度器是推理引擎"排队系统"的化身。请求到达有先有后，GPU 每步只能处理一个 batch——调度器决定"这一步让谁上 GPU"。TRT-LLM 的默认调度是 In-Flight Batching（阶段 1 详讲），它能做到：新请求的 prefill 和正在进行的 decode 混在同一个 batch 里跑，让 GPU 永远不闲着。**调度策略是 AI Infra 面试的高频考点**，也是吞吐优化的第一杠杆。

- **`KVCacheManager`（KV 缓存管理器）**：负责 KV Cache 的分配、释放和维护。这是 Transformer 模型的关键优化：在自回归文本生成过程中存储先前计算过的 attention 键值（keys/values），显著提升生成性能。

> 💡 **AI Infra 视角**：为什么需要 KV Cache？自回归生成是"一次吐一个 token"：生成第 100 个 token 时，attention 需要看前 99 个 token 的 K/V。如果不缓存，每步都要重新算前 99 个 token 的 K/V——O(n²) 的重复计算。KV Cache 用显存换计算：把算过的 K/V 存起来，每步只算新 token 的 K/V，把复杂度降到 O(n)。
> 代价是显存：KV Cache 是推理时显存的最大开销之一（长上下文 + 大并发时尤其恐怖），所以有了 PagedAttention、块复用、KV Cache 量化等一整套优化（阶段 1 详讲）。

- **`ModelEngine`（模型引擎）**：负责在 GPU 硬件上加载并高效执行语言模型。

- **`Sampler`（采样器）**：接收 ModelEngine 的原始输出（logits），应用适当的采样策略（如 greedy、top-k、top-p、beam search）生成最终的输出 token。

> 💡 **AI Infra 视角**：为什么采样要单独成组件？因为模型输出的是"下一个 token 的概率分布"（logits），不是直接的结果。采样策略决定了从这个分布里怎么挑 token——贪心（greedy）取概率最大的，temperature/top-k/top-p 控制随机性。这是 LLM 和传统软件最大的差异之一：**同一个 prompt 两次调用结果可能不同**（当 temperature > 0 时）。生产中控制输出确定性（temperature=0）和吞吐（采样阶段也是 GPU 空闲的 CPU 时机）都是工程师的日常工作。

在后台循环的每次迭代中，`PyExecutor` 执行以下操作序列：

- **请求获取（Request Fetching）**：从内部请求队列中取出新的推理请求（如果有）。

- **调度（Scheduling）**：与 `Scheduler` 交互，识别并优先处理当前步骤中已就绪的请求。

- **资源准备（Resource Preparation）**：与 `KVCacheManager` 协调，确保为选中的请求分配必要的 KV Cache 资源。

- **模型执行（Model Execution）**：调用 `ModelEngine` 对已调度的请求执行一次前向传播（forward pass），预测下一个输出 token。

- **输出处理（Output Handling）**：更新进行中请求的部分输出，并最终确定已完成的请求结果，返回给用户。

> 💡 **AI Infra 视角**：把这 5 步连起来，就是一次"迭代"（iteration）——也就是 GPU 上的一次 forward。**一次迭代 = 一个 batch 的请求各生成 1 个 token**。整个服务就是一个循环：调度谁 → 备资源 → 算一步 → 收结果。性能优化的核心目标就是让这个循环每一步都快、GPU 每时每刻都在干活。

## 运行时优化

TensorRT LLM 通过集成一系列运行时优化来提升推理吞吐并降低延迟，包括 CUDA Graph、[Overlap Scheduler](../features/overlap-scheduler.md)、[投机解码](../features/speculative-decoding.md) 等。

### CUDA Graph

CUDA Graphs 大幅降低启动 GPU kernel 带来的 CPU 侧开销，这对基于 PyTorch 的推理尤其重要，因为 Python 主机端代码可能是瓶颈。通过将一串 CUDA 操作捕获为单个图（graph），整个序列可以用一次 API 调用启动，最大限度减少 CPU-GPU 同步和驱动开销。

> 💡 **AI Infra 视角**：为什么 Python 是瓶颈？每个 PyTorch 算子的执行都要走 "Python → 框架调度 → CUDA API → 驱动" 的链路，一个 transformer 层有几十上百个算子，每步生成都要重复这套开销——GPU 可能几十微秒干完活，CPU 花几百微秒发指令，GPU 就饿着等。CUDA Graph 把固定形状的一串 kernel 一次性捕获成图，之后每次只需一次 launch，把 CPU 开销降到 1/100。代价是：**图的输入形状必须固定**（所以 batch 大小要 padding，见下文）。这是 PyTorch 推理引擎的通用优化（vLLM 也有 CUDA Graph），面试必考。

为了提高这些缓存图的"命中率"，TensorRT LLM 采用 CUDA Graph padding。如果传入 batch 的大小与已捕获的图不匹配，就把它填充（padding）到最接近的、已有图支持的更大尺寸。虽然这会因计算"浪费的" token 带来少量开销，但通常比回退到较慢的 eager 模式执行更划算。这项优化影响显著，在某些模型和硬件上端到端吞吐可提升高达 22%。

> 💡 **AI Infra 视角**：padding 的思路：与其为每个 batch 大小都捕获一张图（显存和时间都爆炸），不如只捕获少量尺寸（如 1,2,4,8,16...）的图，来请求时向上取整。多算几个 padding token 的代价 << 回到 eager 模式的代价。类似的"用少量固定形状覆盖动态输入"思想在推理引擎中处处可见。

### Overlap Scheduler

Overlap Scheduler（重叠调度器）通过用 GPU 计算掩盖（hide）CPU 侧延迟，最大化 GPU 利用率。

关键策略是：立即启动下一步（n+1）的 GPU 工作，而不等待 CPU 处理完当前步骤（n）的结果。这样，GPU 执行下一批模型计算的同时，CPU 可以处理上一批的收尾任务（如检查停止条件或更新响应）。

这个并发执行流水线在 `PyExecutor` 的逻辑中展示：

```python
# 调度并启动当前步骤 (n) 的 GPU 工作
scheduled_batch, _, _ = self._schedule()
batch_outputs = self._forward_step(scheduled_batch, previous_tensors_device)
sample_state = self._sample_async(scheduled_batch, batch_outputs)

# GPU 忙碌期间，处理上一步 (n-1) 的 CPU 侧结果
if self.previous_batch is not None:
    self._process_previous_batch()
```

> 💡 **AI Infra 视角**：这是"软件流水线（software pipelining）"在推理服务中的应用：CPU 处理和 GPU 计算是两个不同的执行单元，把它们并行起来。代码里体现为：先把第 n 步的 GPU 工作全部 launch（异步），趁 GPU 在跑，CPU 回头处理第 n-1 步的采样/停止判断等 CPU 活。**代价是延迟多一步**（pipeline 初始填充），但换来 GPU 空闲时间大幅减少——吞吐收益大于延迟损失，所以默认开启。这是"以延迟换吞吐"的典型 trade-off，面试常考。

这种方法有效减少了 GPU 空闲时间，提高了整体硬件占用率。虽然它给流水线引入了一个额外的解码步骤，但由此带来的吞吐提升是值得的权衡。因此，Overlap Scheduler 在 TensorRT LLM 中默认启用。

## 视觉生成（Visual Generation）

对于基于扩散模型的视觉生成（图像/视频），TensorRT-LLM 提供了独立的 `VisualGen` API 和 `DiffusionExecutor`，拥有自己的流水线架构。参见[视觉生成](../models/visual-generation.md)特性文档。

## 模块级日志（Module-Level Logging）

TensorRT-LLM 日志消息包含一个定宽模块标签，用于标识消息来自哪个子系统：

```txt
[TRT-LLM] [I] [runtime ] Loading model weights...
[TRT-LLM] [W] [_torch  ] FlashAttention not available, falling back to default
[TRT-LLM] [I] [serve   ] Server listening on port 8000
```

> 💡 **AI Infra 视角**：会看日志是 AI Infra 的基本功。日志格式：级别（I=info, W=warning, S=...）+ 模块名。调试时经常遇到的情况：`_torch` 模块报 FlashAttention 不可用回退到默认实现——性能突然变差先查这类警告；`batchmgr` 模块的日志对排查调度问题最有用。生产环境排障的第一动作永远是"把对应模块的日志级别调到 debug 再看"。

### 模块缩写表

超过 8 个字符的模块名会缩写以适配定宽标签：

| 模块名 | 显示标签 |
|-------------|-------------|
| `_torch` | `_torch  ` |
| `batch_manager` | `batchmgr` |
| `common` | `common  ` |
| `cutlass_extensions` | `cutl_ext` |
| `deep_ep` | `deep_ep ` |
| `deep_gemm` | `deepgemm` |
| `executor` | `executor` |
| `flash_mla` | `flashmla` |
| `kernels` | `kernels ` |
| `layers` | `layers  ` |
| `runtime` | `runtime ` |
| `serve` | `serve   ` |
| `quantization` | `quantize` |
| `scaffolding` | `scaffold` |
| `auto_parallel` | `autoprll` |
| `visual_gen` | `vis_gen ` |

### 按模块设置日志级别

使用 `TLLM_LOG_LEVEL_BY_MODULE` 可以为不同模块设置不同日志级别，覆盖全局的 `TLLM_LOG_LEVEL`：

```bash
# 格式: "level:module1,module2;level:module3"
export TLLM_LOG_LEVEL=warning
export TLLM_LOG_LEVEL_BY_MODULE="debug:_torch,runtime;info:serve"
```

这个示例将全局级别设为 `warning`，但对 `_torch` 和 `runtime` 模块启用 `debug` 输出，对 `serve` 启用 `info`。合法级别：`trace`、`debug`、`verbose`、`info`、`warning`、`error`、`internal_error`。
