<!--
  本文档为 TensorRT-LLM 官方 Benchmarking 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/developer-guide/perf-benchmarking.md
-->

(perf-benchmarking)=

# TensorRT LLM 基准测试


```{eval-rst}
.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-config-flag-alias
   :end-before: .. end-note-config-flag-alias
```

TensorRT LLM 提供 `trtllm-bench` CLI，这是一个打包好的基准测试工具，目标是让用户更容易复现我们官方发布的[性能总览](./perf-overview.md#throughput-measurements)。`trtllm-bench` 提供以下功能：

- 为多种模型和平台构建调优引擎进行基准测试的简化流程。
- 完全 Python 化的基准测试工作流。
- 能够对 TensorRT LLM 内的各种流程和特性进行基准测试。

> 💡 **AI Infra 视角**：`trtllm-bench` 是 AI Infra 工程师的"官方尺子"——所有官方性能数字都来自它。三个子命令：`prepare-dataset`（造数据）、`throughput`（吞吐）、`latency`（延迟）。学习路径建议：**先跑通一条完整命令**（生成数据集 → 跑 throughput → 读输出），你就掌握了性能评估的标准姿势。

TensorRT LLM 还通过 `trtllm-serve` 命令提供 OpenAI 兼容 API，启动一个支持以下端点的 OpenAI 兼容服务器：
- `/v1/models`
- `/v1/completions`
- `/v1/chat/completions`

以下指南主要聚焦于使用 `trtllm-bench` CLI 的基准测试。要对 OpenAI 兼容的 `trtllm-serve` 做基准测试，请参阅[使用 `trtllm-serve` 运行基准测试](../commands/trtllm-serve/run-benchmark-with-trtllm-serve.md)章节。

## 目录
- [TensorRT LLM 基准测试](#tensorrt-llm-benchmarking)
  - [目录](#table-of-contents)
  - [基准测试之前](#before-benchmarking)
    - [持久模式](#persistence-mode)
    - [GPU 时钟管理](#gpu-clock-management)
    - [设置功率限制](#set-power-limits)
    - [Boost 设置](#boost-settings)
  - [吞吐基准测试](#throughput-benchmarking)
    - [限制与注意事项](#limitations-and-caveats)
      - [已验证的网络](#validated-networks-for-benchmarking)
      - [支持的量化模式](#supported-quantization-modes)
    - [准备数据集](#preparing-a-dataset)
    - [使用 PyTorch 工作流运行](#running-with-the-pytorch-workflow)
      - [PyTorch 工作流中用 LoRA 适配器做基准测试](#benchmarking-with-lora-adapters-in-pytorch-workflow)
      - [PyTorch 工作流中运行多模态模型](#running-multi-modal-models-in-the-pytorch-workflow)
      - [PyTorch 流程中的量化](#quantization-in-the-pytorch-flow)
  - [在线服务基准测试](#online-serving-benchmarking)

要对 OpenAI 兼容的 `trtllm-serve` 做基准测试，请参阅[使用 `trtllm-serve` 运行基准测试](../commands/trtllm-serve/run-benchmark-with-trtllm-serve.md)章节。

## 基准测试之前

对于需要一致且可复现结果的严格基准测试，正确的 GPU 配置至关重要。这些设置有助于最大化 GPU 利用率、消除性能变异性，并确保测量的最佳条件。虽然正常运行并不严格要求，但我们建议在进行性能比较或发布基准结果时应用这些配置。

> 💡 **AI Infra 视角**：为什么要调 GPU 时钟？GPU 有**动态调频（DVFS）**——负载高时降频（过热保护）、负载低时升频。不同时刻的时钟频率不同，性能就不可复现。基准测试的黄金法则：**消除一切环境变量**。下面四步（持久模式、时钟、功耗、boost）就是"把 GPU 固定在最稳定的状态"。

### 持久模式（Persistence mode）

确保启用持久模式以保持 GPU 状态一致：
```shell
sudo nvidia-smi -pm 1
```

> 💡 **AI Infra 视角**：持久模式让 GPU 驱动常驻（而不是每次调用都启动新驱动进程）。基准时避免"第一个 kernel 特别慢"的冷启动效应。

### GPU 时钟管理

让 GPU 根据工作负载和温度动态调整时钟速度。将时钟锁定在最大频率看似有利，但有时会导致热降频（thermal throttling）和性能下降。使用以下命令重置 GPU 时钟：
```shell
sudo nvidia-smi -rgc
```

> 💡 **AI Infra 视角**：反直觉点：**锁最高频反而可能更慢**——因为过热触发降频保护，性能反而波动。让时钟"自由呼吸"但记录时钟状态，是更稳的做法。

### 设置功率限制

首先查询最大功率限制：
```shell
nvidia-smi -q -d POWER
```
然后将 GPU 配置为在最大功率限制下运行以获得一致的性能：
```shell
sudo nvidia-smi -pl <max_power_limit>
```

### Boost 设置

GPU 可能支持 boost 级别。首先查询可用的 boost 级别：
```shell
sudo nvidia-smi boost-slider -l
```
如果支持，使用可用级别之一启用 boost slider 以获得最大性能：
```shell
sudo nvidia-smi boost-slider --vboost <max_boost_slider>
```


## 吞吐基准测试

### 限制与注意事项

#### 已验证的网络

虽然 `trtllm-bench` 应该能运行 TensorRT LLM 支持的任何网络，但以下是经过广泛验证的列表，与
[性能总览](./perf-overview.md) 页面上的列表相同。

- [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf)
- [tiiuae/falcon-180B](https://huggingface.co/tiiuae/falcon-180B)
- [EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b)
- [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
- [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)
- [meta-llama/Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)
- [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
- [meta-llama/Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)
- [mistralai/Mixtral-8x7B-v0.1-Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1-Instruct)

```{tip}
`trtllm-bench` 可以自动从 Hugging Face Model Hub 下载模型。
把你的 token 导出到 `HF_TOKEN` 环境变量中。
```

#### 支持的量化模式

`trtllm-bench` 支持以下量化模式：

- None（不应用量化）
- `FP8`
- `NVFP4`

关于量化的更多信息，请参阅 [](../features/quantization.md) 以及
每个网络支持的量化方法[支持矩阵](../features/quantization.md#model-supported-matrix)。

```{tip}
虽然 TensorRT LLM 支持的量化模式比上面列出的更多，但 `trtllm-bench` 目前只配置了
较小的子集。
```

### 准备数据集

吞吐基准测试使用固定的 JSON schema 来指定请求。schema 定义如下：

| 键             | 必需 |     类型     | 描述                                     |
| :-------------- | :------: | :-----------: | :---------------------------------------------- |
| `task_id`       |    Y     |    String     | 请求的唯一标识符。              |
| `prompt`        |    N*    |    String     | 生成请求的输入文本。            |
| `input_ids`     |    Y*    | List[Integer] | 组成请求 prompt 的 token ID 列表。 |
| `output_tokens` |    Y     |    Integer    | 该请求生成的 token 数量。    |

```{tip}
\* 必须指定 `prompt` 或 `input_ids` 之一。但不能同时定义 prompt 和 token ID（`input_ids`）。
如果指定了 `input_ids`，请求生成时会忽略 `prompt` 条目。
```

> 💡 **AI Infra 视角**：注意 dataset 的精妙设计：每条请求**预先指定要生成多少个 token**（`output_tokens`）——这样引擎可以精确控制每条请求的生成长度，保证基准的可控性（否则有的请求生成 5 个 token 就停了，吞吐统计失真）。`input_ids` 直接给 token ID（用随机 ID 模拟任意内容），`prompt` 给文本（用于真实内容）。

参考以下基准测试的合法条目示例：

- 带人类可读 prompt 且无 logits 的条目。

  ```json
  {"task_id": 1, "prompt": "Generate an infinite response to the following: This is the song that never ends, it goes on and on my friend.", "output_tokens": 1000}
  {"task_id": 2, "prompt": "Generate an infinite response to the following: Na, na, na, na", "output_tokens": 1000}
  ```

- 包含 logits 的条目。

  ```json
  {"task_id":0,"input_ids":[863,22056,25603,11943,8932,13195,3132,25032,21747,22213],"output_tokens":128}
  {"task_id":1,"input_ids":[14480,13598,15585,6591,1252,8259,30990,26778,7063,30065,21764,11023,1418],"output_tokens":128}
  ```

```{tip}
每条目占一行。
为简化数据传递，每行是一个完整的 JSON 条目，这样基准测试器只需读取一行即可假定一个完整条目。创建数据集时，请确保每行都是完整的 JSON 条目。
```

> 💡 **AI Infra 视角**：**JSONL 格式**（JSON Lines）：每行一个完整 JSON 对象——大数据集的高效格式（可以流式读取、并行处理）。这是 AI 领域的标准数据格式（训练数据、评测数据、日志都是它）。

要准备合成数据集，可以使用 `benchmarks` 目录中提供的脚本。例如，要为 [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) 生成 1000 个请求、均匀 ISL/OSL 128/128 的合成数据集，运行：

```shell
trtllm-bench --model meta-llama/Llama-3.1-8B prepare-dataset --output /tmp/synthetic_128_128.txt token-norm-dist --input-mean 128 --output-mean 128 --input-stdev 0 --output-stdev 0 --num-requests 1000
```

### 使用 PyTorch 工作流运行

要基准测试 PyTorch 后端（`tensorrt_llm._torch`），使用下面的命令配合前几步[生成的数据集](#preparing-a-dataset)。`throughput` 基准通过针对 `--dataset` 提供的数据集调优后端来初始化后端（或其他上文描述的构建模式设置）。

注意 CUDA graph 默认启用。你可以用 `--config` 加 YAML 文件路径来添加额外的 pytorch 配置。更多细节请运行 `--help` 查看帮助文本。

```{tip}
下面的命令指定了 `--model_path` 选项。模型路径是可选的，仅在你想要运行本地
存储的 checkpoint 时使用。使用 `--model_path` 时，`--model` 仍然是必需的，用于报告
以及查找构建启发式参数。
```

```shell
trtllm-bench --model meta-llama/Llama-3.1-8B \
  --model_path /Ckpt/Path/To/Llama-3.1-8B \
  throughput \
  --dataset /tmp/synthetic_128_128.txt \
  --backend pytorch

# 示例输出
<snip verbose logging>
===========================================================
= PyTorch backend
===========================================================
Model:                  meta-llama/Llama-3.1-8B
Model Path:             /Ckpt/Path/To/Llama-3.1-8B
TensorRT LLM Version:   0.17.0
Dtype:                  bfloat16
KV Cache Dtype:         None
Quantization:           FP8

===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:                1
PP Size:                1
Max Runtime Batch Size: 2048
Max Runtime Tokens:     4096
Scheduling Policy:      Guaranteed No Evict
KV Memory Percentage:   90.00%
Issue Rate (req/sec):   7.6753E+14

===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Number of requests:             3000
Average Input Length (tokens):  128.0000
Average Output Length (tokens): 128.0000
Token Throughput (tokens/sec):  20685.5510
Request Throughput (req/sec):   161.6059
Total Latency (ms):             18563.6825

```

> 💡 **AI Infra 视角**：解读"WORLD + RUNTIME INFORMATION"块（这是引擎的"体检报告"）：
> - **Scheduling Policy: Guaranteed No Evict**——保证不驱逐：所有请求都会跑完（不因资源不足被踢掉）。另一种常见策略是 Max Utilization（优先喂饱 GPU，必要时驱逐）；
> - **KV Memory Percentage: 90%**——KV cache 吃掉 90% 空闲显存（kvcache.md 讲过）；
> - **Max Runtime Tokens: 4096**——每步 token 预算（max_num_tokens）。
> 这些字段就是前几篇文档讲的配置项在运行时的实际生效值——**看运行报告验证配置是否生效**，是排查"我配了怎么没效果"的第一动作。

启用流式（streaming）时，还会记录首 token 时间（TTFT）和 token 间延迟（ITL）指标。
```shell
trtllm-bench --model meta-llama/Llama-3.1-8B \
  --model_path /Ckpt/Path/To/Llama-3.1-8B \
  throughput \
  --dataset /tmp/synthetic_128_128.txt \
  --backend pytorch
```

> 💡 **AI Infra 视角**：**ITL（Inter-Token Latency）**是流式场景的关键指标——用户看到"打字速度"快不快。注意：**只有流式（streaming）模式才记录 TTFT/ITL**，因为非流式模式下引擎一次算完整个输出，这两个延迟没有意义。**评测在线体验必须开 streaming**。

另外，用户可以基准测试低延迟模式：
```shell
trtllm-bench --model meta-llama/Llama-3.1-8B \
  --model_path /Ckpt/Path/To/Llama-3.1-8B \
  latency \
  --dataset /tmp/synthetic_128_128.txt \
  --backend pytorch
```

> 💡 **AI Infra 视角**：`throughput` vs `latency` 的区别（必考）：
> - **throughput**：压力测试——请求疯狂灌入（Issue Rate 显示 1e14 req/s，就是"不设限"的意思），测**系统最多能产出多少**；
> - **latency**：单请求计时——请求一个个来（低并发），测**单个请求多快完成**。
> 生产优化往往要同时看两者：throughput 高但 latency 烂 = 排队太久；latency 好但 throughput 低 = 资源没吃满。

#### PyTorch 工作流中用 LoRA 适配器做基准测试

PyTorch 工作流支持用 LoRA（低秩适配）适配器做基准测试。这需要准备带 LoRA 元数据的数据集并配置 LoRA 设置。

**准备 LoRA 数据集**

使用 `trtllm-bench prepare-dataset` 加 LoRA 特定选项，生成带 LoRA 元数据的请求：

```shell
trtllm-bench \
  --model /path/to/tokenizer \
  prepare-dataset \
  --rand-task-id 0 1 \
  --lora-dir /path/to/loras \
  token-norm-dist \
  --num-requests 100 \
  --input-mean 128 \
  --output-mean 128 \
  --input-stdev 16 \
  --output-stdev 24 \
  > synthetic_lora_data.json
```

关键 LoRA 选项：
- `--lora-dir`：包含 LoRA 适配器子目录的父目录，子目录按 task ID 命名（例如 `0/`、`1/` 等）
- `--rand-task-id`：随机分配给请求的 LoRA task ID 范围
- `--task-id`：所有请求固定的 LoRA task ID（`--rand-task-id` 的替代）

生成的数据集将包含 LoRA 请求元数据。下面是单个请求数据条目的示例：

```json
{
  "task_id": 0,
  "input_ids": [3452, 88226, 102415, ...],
  "output_tokens": 152,
  "lora_request": {
    "lora_name": "lora_0",
    "lora_int_id": 0,
    "lora_path": "/path/to/loras/0"
  }
}
```

**LoRA 配置**

创建带 LoRA 配置的 `config.yaml` 文件：

```yaml
lora_config:
  lora_dir:
    - /path/to/loras/0
    - /path/to/loras/1
  max_lora_rank: 64
  lora_target_modules:
    - attn_q
    - attn_k
    - attn_v
  trtllm_modules_to_hf_modules:
    attn_q: q_proj
    attn_k: k_proj
    attn_v: v_proj
```

**运行 LoRA 基准测试**

```shell
trtllm-bench --model /path/to/base/model \
  throughput \
  --dataset synthetic_lora_data.json \
  --backend pytorch \
  --config config.yaml
```

```{note}
LoRA 目录结构应有按 task ID 命名的任务特定子目录（例如 `loras/0/`、`loras/1/`）。
每个子目录应包含该特定任务的 LoRA 适配器文件。
```

> 💡 **AI Infra 视角**：**LoRA 推理 = 一个基座模型 + 多个低成本适配器**（每个适配器只是很小的低秩矩阵）。服务端可以同时挂几十个 LoRA 适配器，一个模型服务多个客户定制版本——成本远低于每个客户一个模型。注意 LoRA 基准测的 `max_lora_rank`（适配器的秩，决定额外显存）和 `lora_target_modules`（要注入适配的层）。**"多租户 LoRA 服务"是 AI Infra 的常见业务模式**。

#### PyTorch 工作流中运行多模态模型

要在 PyTorch 工作流中基准测试多模态模型，可以遵循类似上面的方法。

首先，准备数据集：
```bash
trtllm-bench \
  --model Qwen/Qwen2-VL-2B-Instruct \
  prepare-dataset \
  --output mm_data.jsonl \
  real-dataset
  --dataset-name lmms-lab/MMMU \
  --dataset-split test \
  --dataset-image-key image \
  --dataset-prompt-key question \
  --num-requests 10 \
  --output-len-dist 128,5
```
它会将媒体文件下载到 `/tmp` 目录并准备带路径的数据集。注意 `prompt` 字段是文本而不是 tokenized ids。这是因为
多模态文件的 `prompt` 和媒体（图像/视频）由预处理器处理。

多模态数据集示例：
```
{"task_id":0,"prompt":"Brahma Industries sells vinyl replacement windows to home improvement retailers nationwide. The national sales manager believes that if they invest an additional $25,000 in advertising, they would increase sales volume by 10,000 units. <image 1> What is the total contribution margin?","media_paths":["/tmp/tmp9so41y3r.jpg"],"output_tokens":126}
{"task_id":1,"prompt":"Let us compute for the missing amounts under work in process inventory, what is the cost of goods manufactured? <image 1>","media_paths":["/tmp/tmpowsrb_f4.jpg"],"output_tokens":119}
{"task_id":2,"prompt":"Tsuji is reviewing the price of a 3-month Japanese yen/U.S. dollar currency futures contract, using the currency and interest rate data shown below. Because the 3-month Japanese interest rate has just increased to .50%, Itsuji recognizes that an arbitrage opportunity exists nd decides to borrow $1 million U.S. dollars to purchase Japanese yen. Calculate the yen arbitrage profit from Itsuji's strategy, using the following data: <image 1> ","media_paths":["/tmp/tmpxhdvasex.jpg"],"output_tokens":126}
...
```

运行基准测试：
```shell
trtllm-bench --model Qwen/Qwen2-VL-2B-Instruct \
  throughput \
  --dataset mm_data.jsonl \
  --backend pytorch \
  --num_requests 10 \
  --max_batch_size 4 \
  --modality image
```


示例输出：
```
===========================================================
= REQUEST DETAILS
===========================================================
Number of requests:             10
Number of concurrent requests:  5.3019
Average Input Length (tokens):  411.6000
Average Output Length (tokens): 128.7000
===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:                1
PP Size:                1
EP Size:                None
Max Runtime Batch Size: 4
Max Runtime Tokens:     12288
Scheduling Policy:      GUARANTEED_NO_EVICT
KV Memory Percentage:   90.00%
Issue Rate (req/sec):   1.4117E+17

===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                     1.4439
Total Output Throughput (tokens/sec):             185.8351
Per User Output Throughput (tokens/sec/user):     38.1959
Per GPU Output Throughput (tokens/sec/gpu):       185.8351
Total Token Throughput (tokens/sec):              780.1607
Total Latency (ms):                               6925.4963
Average request latency (ms):                     3671.8441

-- Request Latency Breakdown (ms) -----------------------

[Latency] P50    : 3936.3022
[Latency] P90    : 5514.4701
[Latency] P95    : 5514.4701
[Latency] P99    : 5514.4701
[Latency] MINIMUM: 2397.1047
[Latency] MAXIMUM: 5514.4701
[Latency] AVERAGE: 3671.8441

===========================================================
= DATASET DETAILS
===========================================================
Dataset Path:         /workspaces/tensorrt_llm/mm_data.jsonl
Number of Sequences:  10

-- Percentiles statistics ---------------------------------

        Input              Output           Seq. Length
-----------------------------------------------------------
MIN:   167.0000           119.0000           300.0000
MAX:  1059.0000           137.0000          1178.0000
AVG:   411.6000           128.7000           540.3000
P50:   299.0000           128.0000           427.0000
P90:  1059.0000           137.0000          1178.0000
P95:  1059.0000           137.0000          1178.0000
P99:  1059.0000           137.0000          1178.0000
===========================================================
```

> 💡 **AI Infra 视角**：多模态基准的两个注意点：一是 `prepare-dataset real-dataset`（真实数据集模式，从 HF 拉 MMMU 等评测集）——**真实数据集用于验证功能，合成数据集用于测性能**，两者用途分明；二是**图片会转成 token**（视觉编码器把图像切成 patch 变成序列），所以图像请求的"输入长度"比纯文本长得多（411 vs 128）。**视觉模型的推理成本中，图像预处理（vision encoder）占比不小**——多模态服务的算力评估要算上它。

**注意事项与限制**：
- 目前只支持图像数据集。
- 多模态数据集必须使用 `--output-len-dist` 参数。
- 准备阶段不使用 tokenizer，但它仍是必需参数。
- 由于图像在模型运行时被转换为 token，`trtllm-bench` 在设置执行参数时对最大输入序列长度使用默认的大值。
  你也可以用 `--max_input_len` 参数指定适合你用例的其他值来修改此行为。

#### PyTorch 流程中的量化

要使用 PyTorch 流程运行量化基准测试，你需要使用预量化的
checkpoint。对于 Llama-3.1 模型，TensorRT LLM 通过 HuggingFace 提供以下 checkpoint：

- [`nvidia/Llama-3.1-8B-Instruct-FP8`](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8)
- [`nvidia/Llama-3.1-70B-Instruct-FP8`](https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-FP8)
- [`nvidia/Llama-3.1-405B-Instruct-FP8`](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP8)

要了解更多如何量化你自己的 checkpoint，请参考 ModelOpt [文档](https://nvidia.github.io/Model-Optimizer/deployment/3_unified_hf.html)。

`trtllm-bench` 利用上述预量化 checkpoint 中的 `hf_quant_config.json` 文件。该配置
文件存在于用 [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) 量化的 checkpoint 中，
描述该 checkpoint 编译时使用的计算和 KV cache 量化。例如，从上面的 checkpoint：

```json
{
    "producer": {
        "name": "modelopt",
        "version": "0.23.0rc1"
    },
    "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": null
    }
}
```

> 💡 **AI Infra 视角**：`hf_quant_config.json` 是量化 checkpoint 的"身份证"——记录了量化算法（`quant_algo`）和 KV cache 量化算法（`kv_cache_quant_algo`）。**引擎就是靠读这个文件知道模型是什么量化格式的**。如果 KV cache 量化是 null（未指定），引擎会按下面的表自动选。

上面的 checkpoint 量化为以 `FP8` 计算精度运行，默认无 KV cache 量化（完整
`FP16` 缓存）。运行 `trtllm-bench throughput` 时，如果 `kv_cache_quant_algo` 指定为 `null`，基准测试器会自动选择最适合
checkpoint 计算精度的 KV cache 量化，否则将强制匹配指定的非 null KV cache 量化。当 checkpoint 未指定 KV cache 量化算法时，`trtllm-bench` 遵循以下映射：

| Checkpoint 计算量化 | Checkpoint KV Cache 量化 | `trtllm-bench` | 说明 |
| - | - | - | - |
| `null` | `null` | `null` | 这种情况不存在量化配置。 |
| `FP8` | `FP8` | `FP8` | 与 checkpoint 匹配 |
| `FP8` | `null` | `FP8` | 由基准测试设置为 `FP8` |
| `NVFP4` | `null` | `FP8` | 由基准测试设置为 `FP8` |

> 💡 **AI Infra 视角**：最后一行值得注意：**NVFP4 权重的 checkpoint，KV cache 自动用 FP8**（NVFP4 KV cache 需要专门离线量化，见 quantization.md；FP8 是运行时量化，开箱即用）。这就是"自动选型"逻辑：KV cache 量化选一个"能直接用的最优"。

如果你想强制 KV cache 量化，可以在 YAML 文件中指定以下内容来强制精度（当 checkpoint 精度为 `null` 时）：

```yaml
kv_cache_config:
  dtype: fp8
```

```{tip}
`kv_cache_config.dtype` 的两个合法值是 `auto` 和 `fp8`。
```

## 在线服务基准测试

TensorRT LLM 通过 `trtllm-serve` 命令提供 OpenAI 兼容 API，并用 `tensorrt_llm.serve.scripts.benchmark_serving` 包对在线服务器做基准测试。另外，[AIPerf](https://github.com/ai-dynamo/aiperf) 是一个全面的基准测试工具，也可以测量 `trtllm-serve` 启动的 OpenAI 兼容服务器的性能。

> 💡 **AI Infra 视角**：**离线基准 vs 在线基准**的区别：`trtllm-bench throughput` 是"引擎极限测试"（本地客户端疯狂灌请求）；在线基准是"真实服务测试"——通过 HTTP 接口、模拟真实用户的到达模式（如泊松分布）、测量端到端延迟。**生产验收用在线基准**，因为要验证的是"用户实际体验"。AIPerf（NVIDIA Dynamo 生态的工具）支持 ShareGPT 等真实工作负载回放。

要对 OpenAI 兼容的 `trtllm-serve` 做基准测试，请参阅[使用 `trtllm-serve` 运行基准测试](../commands/trtllm-serve/run-benchmark-with-trtllm-serve.md)章节。
