<!--
  本文档为 TensorRT-LLM 官方 Quick Start Guide 的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/quick-start-guide.md
-->

(quick-start-guide)=

# 快速开始指南

这是试用 TensorRT LLM 的起点。本指南让你快速完成环境配置，并通过 HTTP 请求使用 TensorRT LLM。

> 💡 **AI Infra 视角**：整个指南展示了 LLM 推理服务的两种典型使用方式，AI Infra 岗位日常都会碰到：
> - **在线服务（online serving）**：启动常驻服务进程，对外提供 HTTP/gRPC 接口，支持高并发请求——生产环境的主要形态；
> - **离线推理（offline inference）**：在 Python 脚本里直接调用引擎做推理（如批量评测、离线生成）——开发调试和批处理任务常用。
> 转行学习时建议先跑通这两个例子，获得"推理引擎到底怎么用"的第一手感受。

## 安装 TensorRT LLM

按照[安装指南](installation/installation-guide)配置 TensorRT LLM。最快的途径是从 NGC 拉取并运行预构建的 release 容器。

> 💡 **AI Infra 视角**：NGC（NVIDIA GPU Cloud）是英伟达的容器/镜像仓库（类似 Docker Hub）。生产环境里常见的安装形态：官方容器镜像（开箱即用，推荐）或 pip 安装（`pip install tensorrt-llm`，依赖本地已装好的 CUDA/TensorRT 环境）。注意 TRT-LLM 依赖特定 CUDA/TensorRT 版本，容器方式能避免环境地狱。

(deploy-with-trtllm-serve)=
## 用 trtllm-serve 部署在线服务

你可以使用 `trtllm-serve` 命令启动一个 OpenAI 兼容的服务器来与模型交互。
要启动服务器，在 Docker 容器内运行类似下面的命令：

```bash
trtllm-serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

你也可以部署预量化模型来提升性能。
运行以下命令前请确保你的 GPU 支持 FP8 量化：

```bash
trtllm-serve "nvidia/Qwen3-8B-FP8"
```

更多选项请浏览完整的[生成模型集合](https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer)，这些模型已使用 TensorRT Model Optimizer 量化和优化，可直接用于推理。

```{note}
如果你在 Docker 容器内运行 `trtllm-serve`，有两种方式发送 API 请求：
1. 暴露端口（如 8000），让外部可以访问容器内的服务器。
2. 打开新终端，用以下命令直接进入运行中的容器：
```bash
docker exec -it <container_id> bash
```

> 💡 **AI Infra 视角**：`trtllm-serve` 只需一个 HuggingFace 模型名就能拉起服务——它内部会自动完成"下载权重 → 加载 → 编译/优化 → 启动服务"整条链路。"OpenAI 兼容"是行业事实标准：各大推理服务（vLLM、Triton、SGLang 等）都实现 OpenAI 的 `/v1/chat/completions` 接口，这样上层应用（LangChain、各种 Agent 框架）无需改动就能切换底层引擎。你在面试和工作中都会反复听到这个词。

服务器启动后，你可以访问著名的 OpenAI 端点，如 `v1/chat/completions`。
在另一个终端中，可以使用类似下面的示例进行推理：

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages":[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Where is New York? Tell me in a single sentence."}],
        "max_tokens": 32,
        "temperature": 0
    }'
```

_示例输出_

```json
{
  "id": "chatcmpl-ef648e7489c040679d87ed12db5d3214",
  "object": "chat.completion",
  "created": 1741966075,
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "New York is a city in the northeastern United States, located on the eastern coast of the state of New York.",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 43,
    "total_tokens": 69,
    "completion_tokens": 26
  }
}
```

> 💡 **AI Infra 视角**：响应里的 `usage` 字段值得注意——`prompt_tokens`（输入 token 数）、`completion_tokens`（输出 token 数）、`total_tokens`（总数）。token 计数直接决定你的**计费和成本核算**，也是性能优化时衡量效果的基准。另外 `finish_reason: "stop"` 表示模型因为遇到结束符而正常停止生成（还有 `"length"`=达到 max_tokens、`"eos"` 等情况）。

详细的示例和命令语法，请参考 [trtllm-serve](commands/trtllm-serve/trtllm-serve.rst) 章节。

```{note}
使用 `trtllm-serve` 部署热门模型的预配置参数可以在我们的[部署指南](deployment-guide/index.rst)中找到。
```

## 使用 LLM API 进行离线推理

LLM API 是一个 Python API，旨在直接在 Python 中简化 TensorRT LLM 的设置和推理。只需指定一个 HuggingFace 仓库名或模型 checkpoint，即可完成模型优化。LLM API 通过一个 `LLM` 实例统一管理模型加载、优化和推理的整个流程。

下面是一个使用 LLM API 运行 TinyLlama 的简单示例。

```{literalinclude} ../../examples/llm-api/quickstart_example.py
    :language: python
    :linenos:
```

你也可以在 `LLM` 构造函数中直接加载预量化模型，如 [Hugging Face 上的量化 checkpoint](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4)。
想进一步了解 LLM API，请查看 [](llm-api/index) 和 [](examples/llm_api_examples)。

> 💡 **AI Infra 视角**：LLM API 的"一个对象管全部"设计（模型加载 + 优化 + 推理）是推理引擎的经典抽象。理解它背后的流程对调试很有帮助：
> 1. `LLM(model=...)` 构造时：下载/加载权重 → 按配置做并行切分、量化等优化 → 构建执行器；
> 2. `llm.generate(...)` 调用时：进入请求调度（scheduler）→ 前向推理（forward）→ 采样（sampling）→ 返回结果。
> 后续学习路线会沿着这条链路逐层深入（阶段 1 讲调度与 KV cache，阶段 4 讲 PyTorch 后端实现）。

## 使用 VisualGen API 进行离线推理

VisualGen API 为基于扩散模型的图像和视频生成提供了类似接口。下面是使用 Wan 2.1 生成视频的简单示例。

```{literalinclude} ../../examples/visual_gen/quickstart_example.py
    :language: python
    :linenos:
```

想进一步了解 VisualGen，请查看[视觉生成](models/visual-generation.md)文档和 [`examples/visual_gen/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/visual_gen)。

## 下一步

在本快速开始指南中，你已经：

- 学会了如何用 `trtllm-serve` 部署模型进行在线服务
- 探索了用 LLM API 进行 TensorRT LLM 离线推理

继续你的 TensorRT LLM 之旅，可以探索以下资源：

- **[安装指南](installation/index.rst)** - 不同平台的详细安装说明
- **[模型专属部署指南](deployment-guide/index.rst)** - 使用 TensorRT LLM 服务特定模型的说明
- **[部署指南](examples/llm_api_examples)** - 各种场景下部署 LLM 推理的全面示例
- **[模型支持](models/supported-models.md)** - 查看支持的模型以及如何添加新模型
- **CLI 参考** - 探索 TensorRT LLM 命令行工具：
  - [`trtllm-serve`](commands/trtllm-serve/trtllm-serve.rst) - 部署模型进行在线服务
  - [`trtllm-bench`](commands/trtllm-bench.rst) - 模型性能基准测试
  - [`trtllm-eval`](commands/trtllm-eval.rst) - 模型准确率评估
