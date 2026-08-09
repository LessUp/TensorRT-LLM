<!--
  本文档为 TensorRT-LLM 官方 Overview 的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/overview.md
-->

(product-overview)=

# 总览

## 关于 TensorRT LLM

[TensorRT LLM](https://developer.nvidia.com/tensorrt) 是 NVIDIA 的开源库，用于在 NVIDIA GPU 上加速和优化最新大语言模型（LLM）的推理性能。

> 💡 **AI Infra 视角**：理解这个项目的定位：LLM 推理链路 = 模型权重 → 推理引擎（TRT-LLM）→ 服务框架（Triton/Dynamo）→ 应用。TRT-LLM 负责"把模型跑得最快"这层，是 AI Infra 中推理侧的核心组件之一。同类开源竞品：vLLM、SGLang、TensorRT-LLM（本家）、llama.cpp。面试时被问"你了解哪些推理引擎"可以答这一串，并对比各自特点（vLLM 生态好、SGLang 性能强、TRT-LLM 与 NVIDIA 硬件耦合最深）。

## 核心能力

### 🔥 **基于 PyTorch 架构**

TensorRT LLM 提供高层 Python [LLM API](./quick-start-guide.md#run-offline-inference-with-llm-api)，支持从单 GPU 到多 GPU、多节点的广泛推理部署形态。内置对多种并行策略和高级特性的支持。LLM API 与更广泛的推理生态无缝集成，包括 NVIDIA [Dynamo](https://github.com/ai-dynamo/dynamo) 和 [Triton Inference Server](https://github.com/triton-inference-server/server)。

TensorRT LLM 的设计目标是模块化和易于修改。其 PyTorch 原生架构让开发者可以实验运行时或扩展功能。多个流行模型已预定义，并可以使用[原生 PyTorch 代码](source:tensorrt_llm/_torch/models/modeling_deepseekv3.py)定制，便于按需适配系统。

### ⚡ **最先进的性能**

TensorRT LLM 在最新的 NVIDIA GPU 上提供突破性性能：

- **DeepSeek R1**：[在 Blackwell GPU 上创下世界纪录的推理性能](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/)
- **Llama 4 Maverick**：[在 B200 GPU 上打破 1,000 TPS/用户 的壁垒](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/)

> 💡 **AI Infra 视角**：TPS（tokens per second）是衡量推理服务吞吐的行业标准单位。实际工作中衡量推理服务性能的核心指标：
> - **吞吐（Throughput）**：单位时间生成的 token 数（TPS），或每秒处理的请求数（QPS）——服务器卖钱的指标
> - **延迟（Latency）**：单个请求从发出到首 token 的时间（TTFT, Time To First Token）和整体完成时间
> - 吞吐和延迟通常是矛盾的：批处理越大吞吐越高，但单请求延迟变差。AI Infra 的日常就是在两者间权衡（后续 perf 文档会详讲）。

### 🎯 **全面的模型支持**

TensorRT LLM 支持最新、最流行的 LLM 和 DiT 架构。查看[完整列表](./models/supported-models.md)。

- **语言模型**：GPT-OSS、Deepseek-R1/V3、Llama 3/4、Qwen2/3、Gemma 3、Phi 4...
- **多模态模型**：LLaVA-NeXT、Qwen2-VL、VILA、Llama 3.2 Vision...
- **[视觉生成](./models/visual-generation.md)模型**：FLUX、Wan2.1/2.2 用于图像和视频生成。

TensorRT LLM 致力于对最流行的模型提供 **Day 0** 支持。

> 💡 **AI Infra 视角**："Day 0 支持"是推理引擎厂商的军备竞赛：新模型（如 DeepSeek 开源）发布当天就能跑。对从业者的意义：新模型发布后，推理引擎是否支持决定了团队能多快上线服务。开源模型（Llama、Qwen、DeepSeek）是业内主线，所以引擎都优先支持它们。

### FP4 支持
[NVIDIA B200 GPU](https://www.nvidia.com/en-us/data-center/dgx-b200/) 与 TensorRT LLM 配合使用，可无缝加载新的 [FP4 格式](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/#what_is_nvfp4)模型权重，自动利用优化的 FP4 kernel 实现高效、准确的低精度推理。

### FP8 支持

在 NVIDIA H100 及更新的 GPU 上，TensorRT LLM 支持 [FP8 量化](./features/quantization.md)，与 16 位浮点相比，性能可翻倍、显存占用减半，而对模型准确率影响很小。

> 💡 **AI Infra 视角**：FP8/FP4 是"低精度推理"的核心手段——用更少的位存权重和计算，换来速度翻倍、显存减半。硬件在跟着走：H100（Hopper）原生支持 FP8，B200（Blackwell）原生支持 FP4。这是模型规模越大、显存越不够用的必然趋势。量化（quantization）相关的原理和落地细节在阶段 2 的 quantization.md 详解。

### 🚀 **高级优化与生产特性**
- **[飞行中批处理（In-Flight Batching）与 Paged Attention](./features/paged-attention-ifb-scheduler.md)**：飞行中批处理通过动态管理请求执行消除等待时间，将上下文（context）处理和生成（generation）阶段同时进行，最大化 GPU 利用率并降低延迟。
- **[多 GPU 多节点推理](./features/parallel-strategy.md)**：通过 Model Definition API 在多个 GPU 和节点上实现张量并行、流水线并行和专家并行的无缝分布式推理。
- **[高级量化](./features/quantization.md)**：
  - **FP4 量化**：在 NVIDIA B200 GPU 上原生支持，使用优化的 FP4 kernel
  - **FP8 量化**：在 NVIDIA H100 GPU 上利用 Hopper 架构自动转换
- **[投机解码（Speculative Decoding）](./features/speculative-decoding.md)**：包括 EAGLE、MTP 和 NGram 等多种算法
- **[KV Cache 管理](./features/kvcache.md)**：分页 KV cache，支持智能块复用和显存优化
- **[分块 Prefill（Chunked Prefill）](./features/paged-attention-ifb-scheduler.md)**：将长序列的上下文拆分为可管理的块，高效处理长序列
- **[LoRA 支持](./features/lora.md)**：支持 HuggingFace 和 NeMo 格式的多适配器，高效微调和适配
- **[Checkpoint 加载](./features/checkpoint-loading.md)**：从多种格式（HuggingFace、NeMo、自定义）灵活加载模型
- **[引导解码（Guided Decoding）](./features/guided-decoding.md)**：支持停止词、坏词和自定义约束的高级采样
- **[分离式服务（Disaggregated Serving，Beta）](./features/disagg-serving.md)**：将上下文（prefill）和生成（decode）阶段分离到不同的 GPU 上，实现最优资源利用

> 💡 **AI Infra 视角**：这段特性列表就是 TRT-LLM 的"能力清单"，也是 AI Infra 面试的高频考点，建议逐个展开学习（本学习路线后续文档会覆盖其中大部分）：
> - **Paged Attention + In-Flight Batching**：推理引擎的"心脏"，解决显存碎片化和 GPU 空闲问题（阶段 1）
> - **KV Cache 管理**：显存大头，管理好了才能装下更多并发请求（阶段 1）
> - **并行策略（TP/PP/EP）**：单卡放不下模型时怎么办（阶段 2）
> - **投机解码**：让生成"看起来"快几倍（阶段 2）
> - **分离式服务**：大规模生产部署的架构趋势（阶段 3）

## 你能用 TensorRT LLM 做什么？

无论你是在构建下一代 AI 应用、优化现有 LLM 部署，还是探索大语言模型技术的前沿，TensorRT LLM 都能提供生成式 AI 时代所需的工具、性能和灵活性。要开始使用，请参考 {ref}`quick-start-guide`。
