<!--
  本文档为 TensorRT-LLM 官方 Qwen3 部署指南的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md
-->

# Qwen3 在 TensorRT LLM 上的部署指南 - Blackwell 与 Hopper 硬件

## 简介

这是在 TensorRT LLM 上运行 Qwen3 模型的功能性快速入门指南。它专注于开箱即用的工作配置和推荐的默认参数。更多性能优化和支持将在未来更新中推出。

> 💡 **AI Infra 视角**：这篇是"实战部署"的完整模板——**建议对照它跑通一次真实部署**。整个流程：启动容器 → 取官方推荐配置 → 启动服务器 → curl 测试 → 压测。注意这篇文档里反复出现的 Qwen3 命名：`Qwen3-30B-A3B` = 总参数 30B、激活参数 3B 的 MoE 模型（A3B = Active 3B），`235B-A22B` 同理。**模型名里的 AxxB 是 MoE 模型的激活参数**——它决定推理算力需求，总参数决定显存需求。

## 前置条件

* GPU：NVIDIA Blackwell 或 Hopper 架构
* OS：Linux
* 驱动：CUDA Driver 575 或更高
* 安装有 NVIDIA Container Toolkit 的 Docker
* Python3 和 python3-pip（可选，仅用于准确率评估）

> 💡 **AI Infra 视角**："Hopper 或 Blackwell"= H100/H200 或 B200 系列。为什么限定这两代？因为这篇指南的默认配置用了 FP8/FP4 量化（前面 quantization.md 讲过，FP8 需要 Hopper+，FP4 需要 Blackwell+）。**部署指南的硬件要求通常由量化方案决定**。

## 模型

* [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)
* [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)
* [Qwen3-235B-A22B-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8)
* [Qwen3-30B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4)
* [Qwen3-235B-A22B-NVFP4](https://huggingface.co/nvidia/Qwen3-235B-A22B-NVFP4)

> 💡 **AI Infra 视角**：注意模型清单的规律：同一模型有 BF16 原版、FP8 版、NVFP4 版（NVIDIA 预量化版）——**部署时按显存选型**：显存够用原版，紧张用 FP8，B200 上追求性能用 NVFP4。235B-A22B 的 FP8 版权重约 235GB×~1byte/param ≈ 240GB，需要多卡或大显存单卡（B200 180GB 也放不下，要 TP2）。

## 部署步骤

### 运行 Docker 容器

构建并运行 docker 容器。详情参见 [Docker 指南](../../../docker/README.md)。

```shell
cd TensorRT-LLM

make -C docker release_build IMAGE_TAG=qwen3-local

make -C docker release_run IMAGE_NAME=tensorrt_llm IMAGE_TAG=qwen3-local LOCAL_USER=1
```

> 💡 **AI Infra 视角**：官方容器是 TRT-LLM 的推荐运行方式（NGC 镜像 + 本地构建）。`release_build` 构建带本地代码的镜像（开发用），`release_run` 启动容器。生产部署可以直接 `docker pull nvcr.io/nvidia/tensorrt-llm:<版本>` 拉预构建镜像。

### 推荐的性能配置

我们在 [`examples/configs`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/configs) 目录中维护带推荐性能设置的 YAML 配置文件。这些配置文件在 TensorRT LLM 容器中的路径为 `/app/tensorrt_llm/examples/configs`。你可以开箱即用，或根据你的具体用例调整。

```shell
TRTLLM_DIR=/app/tensorrt_llm # 按你的环境需要修改
EXTRA_LLM_API_FILE=${TRTLLM_DIR}/examples/configs/curated/qwen3.yaml
```

> 💡 **AI Infra 视角**：**"官方维护的推荐配置"是 AI Infra 团队的宝贵资产**——每个模型+硬件组合的优化参数（TP 大小、KV cache 比例、CUDA graph batch 列表）都经过验证。AGENTS.md 里也强调：优先用 `examples/configs/database/` 的 pareto 优化配置，而不是自己手调。部署新模型的第一动作：**找官方有没有现成配置**。

注意：如果你本地没有源码，可以用下面下拉框中的代码手动创建 YAML 配置文件。

````{admonition} 显示代码
:class: dropdown

```{literalinclude} ../../../examples/configs/curated/qwen3.yaml
---
language: shell
prepend: |
  EXTRA_LLM_API_FILE=/tmp/config.yml

  cat << EOF > ${EXTRA_LLM_API_FILE}
append: EOF
---
```
````


### 启动 TensorRT LLM 服务器

下面是在容器内用 Qwen3 模型启动 TensorRT LLM 服务器的示例命令。

```shell
trtllm-serve Qwen/Qwen3-30B-A3B --host 0.0.0.0 --port 8000 --config ${EXTRA_LLM_API_FILE}
```

服务器启动后，客户端就可以向服务器发送 prompt 请求并接收结果。

### LLM API 选项（YAML 配置）

<!-- TODO: 本节在多个部署指南中重复；应合并到中心文件按需导入，或删除并链接到 LLM API 参考 -->

这些选项控制 TensorRT LLM 的行为，设置在通过 `--config` 参数传给 `trtllm-serve` 命令的 YAML 文件中。

> 💡 **AI Infra 视角**：下面每个参数在前面文档里都讲过——这里是它们的**实战含义**。把这节当作"参数速查表 + 部署建议"来读，全部掌握后你就具备了独立调优一个推理服务的知识基础。

#### `tensor_parallel_size`

* **描述：** 设置**张量并行**大小。通常应匹配你计划用于单个模型实例的 GPU 数量。

#### `moe_expert_parallel_size`

* **描述：** 为混合专家（MoE）模型设置**专家并行**大小。与 `tensor_parallel_size` 类似，一般应与使用的 GPU 数量匹配。此设置对非 MoE 模型无效。

> 💡 **AI Infra 视角**：注意 TP 和 MoE-EP 的语义：对 Qwen3（MoE）模型，配置通常同时设置 `tensor_parallel_size` 和 `moe_expert_parallel_size`（乘积=总卡数，见 parallel-strategy.md）。对非 MoE 模型（如 Llama），`moe_expert_parallel_size` 无效——**配置项与模型架构强相关**。

#### `kv_cache_free_gpu_memory_fraction`

* **描述：** 一个 `0.0` 到 `1.0` 之间的值，指定模型加载后为 KV cache 预留的空闲 GPU 显存比例。由于显存使用可能波动，这个缓冲有助于防止内存不足（OOM）错误。
* **建议：** 如果遇到 OOM 错误，尝试将此值降低到 `0.7` 或更低。

#### `max_batch_size`

* **描述：** 可以分组到单个 batch 进行处理的最大用户请求数。实际可达到的最大 batch 大小取决于总序列长度（输入 + 输出）。

#### `max_num_tokens`

* **描述：** 单个调度 batch 中允许的最大 token 总数（跨所有请求）。

#### `max_seq_len`

* **描述：** 单个请求的最大可能序列长度，包括输入和生成的输出 token。我们不会特别设置它。它将从模型配置推断。

> 💡 **AI Infra 视角**："我们不设置它，从模型配置推断"——因为 `max_seq_len` 默认取 `max_position_embeddings`（前面 paged-attention 文档讲过）。**默认值够用就别动**，是部署调参的稳健原则。

#### `trust_remote_code`
* **描述：** 允许 TensorRT LLM 从 Hugging Face 下载模型和 tokenizer。此标志直接传给 Hugging Face API。

> 💡 **AI Infra 视角**：`trust_remote_code=True` 意味着 HF 会执行模型仓库里的自定义 Python 代码（`modeling_*.py`）——**有安全风险**（供应链攻击），只对可信模型仓库开启。这也是为什么推理引擎通常自己实现模型结构（TRT-LLM 的 `tensorrt_llm/_torch/models/`），而不是直接跑 HF 的 modeling 代码。

#### `cuda_graph_config`

* **描述：** 配置 CUDA graphs 以优化性能的章节。

* **选项：**

  * `enable_padding`：如果为 `true`，输入 batch 会被 padding 到最近的 `cuda_graph_batch_size`。这可以显著提高性能。

    **默认值**：`false`

  * `batch_sizes`：将预捕获 CUDA graphs 的 batch 大小列表。

    **建议**：将其设置为覆盖生产环境中预期的 batch 大小范围。

> 💡 **AI Infra 视角**：CUDA graph 参数是"性能与显存的交换"：捕获的图越多（batch_sizes 列表越长），命中率越高，但每张图都要占显存（图和缓冲）。**生产建议：覆盖你实际流量会出现的 batch 范围**（比如 1~256），而不是越大越好。

#### `moe_config`

* **描述：** 混合专家（MoE）模型的配置。

* **选项：**

  * `backend`：MoE 操作使用的后端。

    **默认值**：`CUTLASS`

> 💡 **AI Infra 视角**：`moe_config.backend` 是 MoE kernel 的实现选择（CUTLASS/TRITON/TRTLLM）——不同 kernel 在不同 GPU/形状下各有优劣。默认 CUTLASS（NVIDIA 的模板库），调优时可以切换对比（前面 perf-overview 的示例配置里就有切换）。

完整的 YAML 配置可用选项列表参见 [`TorchLlmArgs` 类](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs)。

## 测试 API 端点

### 基本测试

在主机上打开一个新终端测试你刚启动的 TensorRT LLM 服务器。

你可以用以下命令查询服务器的健康/就绪状态：

```shell
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

当返回 `Status: 200` 时，服务器已准备好接受查询。注意第一次查询可能因为初始化和编译而耗时更长。

> 💡 **AI Infra 视角**：**冷启动 vs 预热**：第一次请求慢是正常的——模型权重加载、CUDA 图捕获、kernel 编译（JIT）都发生在启动阶段。**监控/告警要排除启动窗口，压测要先预热**（前面 perf-analysis 也强调过）。

TensorRT LLM 服务器启动并显示 `Application startup complete` 后，你可以向服务器发送请求。

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json"  -d '{
    "model": "Qwen/Qwen3-30B-A3B",
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95
}' -w "\n"
```

这是示例响应：

```json
{
  "id": "chatcmpl-abc123def456",
  "object": "chat.completion",
  "created": 1759022940,
  "model": "Qwen/Qwen3-30B-A3B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris. Paris is not only the capital but also the largest city in France, known for its rich history, culture, art, and iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 58,
    "total_tokens": 73
  }
}
```

> 💡 **AI Infra 视角**：注意请求里的 `temperature: 0.7, top_p: 0.95`——这就是 sampling.md 讲的采样参数在真实 API 中的用法。测试时如果响应不稳定（每次不同），正常——temperature>0 就是有随机性；**要验证功能用 temperature=0**。

### 故障排查提示

* 如果遇到 CUDA 内存不足错误，尝试减小 `max_batch_size`、`max_num_tokens` 或 `kv_cache_free_gpu_memory_fraction`。
* 确保你的模型 checkpoint 与预期格式兼容。
* 对于性能问题，在服务器运行时用 `nvidia-smi` 检查 GPU 利用率。
* 如果容器无法启动，验证 NVIDIA Container Toolkit 是否正确安装。
* 对于连接问题，确保服务器端口（本指南中是 `8000`）没有被其他应用占用。
* 对于 MoE 模型（Qwen3-30B-A3B、Qwen3-235B-A22B），确保正确配置 `moe_expert_parallel_size`。

> 💡 **AI Infra 视角**：这段排障提示是最实用的——OOM 的调整顺序：`max_batch_size`/`max_num_tokens`（降低并发容量）→ `kv_cache_free_gpu_memory_fraction`（少分点给 KV cache）。**"显存不够 → 先砍 KV cache 比例或并发，而不是砍模型精度"**——除非你确认精度要求可以放宽。

## 性能基准测试

要对 TensorRT LLM 服务器的性能做基准测试，可以利用内置的 `benchmark_serving.py` 脚本。为此，先创建一个包装脚本 `bench.sh`。

```shell
cat <<'EOF' > bench.sh
#!/usr/bin/env bash
set -euo pipefail

# 根据你要基准测试的 Qwen3 模型调整模型名
MODEL_NAME="Qwen/Qwen3-30B-A3B"

concurrency_list="1 2 4 8 16 32 64 128"
multi_round=5
isl=1024
osl=1024
result_dir=/tmp/qwen3_output

for concurrency in ${concurrency_list}; do
    num_prompts=$((concurrency * multi_round))
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model ${MODEL_NAME} \
        --backend openai \
        --dataset-name "random" \
        --random-input-len ${isl} \
        --random-output-len ${osl} \
        --random-prefix-len 0 \
        --random-ids \
        --num-prompts ${num_prompts} \
        --max-concurrency ${concurrency} \
        --ignore-eos \
        --tokenize-on-client \
        --percentile-metrics "ttft,tpot,itl,e2el"
done
EOF
chmod +x bench.sh
```

> 💡 **AI Infra 视角**：这个脚本是**并发压力扫描（concurrency sweep）**的标准模板——从 1 到 128 逐级加压，得到"吞吐-并发"曲线。逐项理解：
> - `max-concurrency`：模拟的并发用户数（在线基准的核心参数）；
> - `--ignore-eos`：忽略结束符，强制生成满 output-len（保证测量稳定）；
> - `--tokenize-on-client`：客户端分词（不占服务器 CPU）；
> - `--percentile-metrics "ttft,tpot,itl,e2el"`：输出 TTFT/TPOT/ITL/端到端延迟的百分位分布。
> **曲线解读**：并发低时吞吐线性涨（资源闲置），到拐点后涨不动（资源饱和），再高延迟爆炸。**SLO 决定你在曲线的哪个点运营**——这是容量规划的直觉基础。

要达到最大吞吐（启用 attention DP 时），需要扫描到 `concurrency = max_batch_size * num_gpus`。

> 💡 **AI Infra 视角**：为什么上限是 `max_batch_size × num_gpus`？attention DP 下每卡独立处理一部分请求，全部卡同时满负荷时并发 = 单卡 batch 上限 × 卡数。**"压测上限 = 配置的容量上限"**——超过这个值的并发只会排队，不会增加吞吐。

如果你想将结果保存到文件，添加以下选项。

```shell
--save-result \
--result-dir "${result_dir}" \
--result-filename "concurrency_${concurrency}.json"
```

更多基准测试选项参见 [benchmark_serving.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/benchmark_serving.py)

运行 `bench.sh` 开始服务基准测试。如果你运行上面 `bench.sh` 脚本中提到的所有并发度，这将花费很长时间。

```shell
./bench.sh
```
