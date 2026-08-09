<!--
  本文档为 TensorRT-LLM 官方 Performance Overview 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/developer-guide/perf-overview.md
-->

(perf-overview)=

# 性能总览

本文档汇总了 TensorRT-LLM 在一系列关键模型和多款 GPU 上的性能测量结果。

下面表格中的数据仅作为参考点，帮助用户验证观察到的性能。
*不应*被视为 TensorRT-LLM 可以提供的峰值性能。

并非所有配置都在所有 GPU 上测试过。

我们尽量保持命令简单以方便复现，许多选项保持默认设置。
调整 batch 大小、并行配置和其他选项可能根据你的情况带来性能提升。

> 💡 **AI Infra 视角**：读这张表的姿势很重要：**性能数字是"配置 × 硬件 × 模型"共同作用的结果**——同一模型，FP4 比 FP8 快，B200 比 H200 快，TP 配置不同结果也不同。这份文档最大的学习价值不在数字本身，而在：
> 1. **ISL/OSL**（Input/Output Sequence Length）——输入/输出序列长度是定义基准场景的坐标轴；
> 2. **DEP4/DEP8/TP1** 等配置标记——"注意力 DP × 张量并行"的组合；
> 3. **复现命令**——`trtllm-bench` 的完整用法，这是 AI Infra 日常工作的标准工具。

关于 DeepSeek R1 的性能，请看我们的[性能指南](../blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md)

关于使用 `trtllm-bench` 进行基准测试的更多信息，参见 NVIDIA [博客文章](https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/)。

## 吞吐量测量

下表展示了本地推理客户端以高速率（消息间无延迟）向系统喂入请求时的性能数据，展示最大负载下的吞吐场景。报告的指标是 `每 GPU 输出吞吐（tokens/sec/GPU）`。

> 💡 **AI Infra 视角**：理解"输出吞吐"：只统计**生成的 token**（输出），不统计输入 token——因为输入 token 是"给定"的（prefill 成本），输出 token 才是服务的"产出"。"每 GPU"归一化让不同规模部署可比。表里的数字量级：单卡每秒几千到几万 token，这是 LLM 推理引擎的典型量级。

下面的性能数字是使用本文档描述的步骤收集的。

测试使用的模型权重通过 [ModelOpt](https://nvidia.github.io/Model-Optimizer/) 量化，并由 NVIDIA 发布在 [Model Optimizer HuggingFace 集合](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4) 上。

RTX 6000 Pro Blackwell Server Edition 的数据现在也包含在性能总览中。RTX 6000 系统可以在 LLM 工作负载中受益于启用流水线并行（PP），因此我们为这款 GPU 在各种 TP × PP 组合下增加了几项新基准。这些数据显示在每个网络的单独表格中。

### 硬件
测试使用了以下 GPU 型号：
- H100 SXM 80GB（DGX H100）
- H200 SXM 141GB（DGX H200）
- B200 180GB（DGX B200）
- GB200 192GB（GB200 NVL72）
- RTX 6000 Pro Blackwell Server Edition

其他硬件型号可能有不同的 TDP、显存带宽、核心数量或其他特性，导致在这些工作负载上的性能差异。

> 💡 **AI Infra 视角**：**GPU 型号速查**（写简历、选型都要会）：A100（Ampere，40/80GB）→ H100（Hopper，80GB，FP8 时代）→ H200（Hopper 大显存版，141GB）→ B200（Blackwell，180GB，FP4 时代）→ GB200（Blackwell 机柜级，192GB，NVLink 高速互联）。演进主线：**显存越来越大、精度越来越低（FP16→FP8→FP4）、互联越来越快**——因为模型越来越大。

### FP4 模型

```text
nvidia/DeepSeek-R1-0528-NVFP4-v2
nvidia/Qwen3-235B-A22B-FP4
nvidia/Qwen3-30B-A3B-FP4
nvidia/Llama-3.3-70B-Instruct-FP4
nvidia/Llama-4-Maverick-17B-128E-Instruct-NVFP4
```

### FP8 模型

```text
deepseek-ai/DeepSeek-R1-0528
nvidia/Qwen3-235B-A22B-FP8
nvidia/Llama-3.3-70B-Instruct-FP8
nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8
```

# 性能汇总 - 所有网络

## 单位

所有性能值都以 `每 GPU 每秒输出 token 数` 为单位，其中 `输出 token` 包括第一个及之后所有生成的 token（输入 token 不包含在内）。

表中数据取自 `trtllm-bench` 报告的 `Per GPU Output Throughput (tps/gpu)` 指标。
trtllm-bench 报告的指标计算见数据类 [reporting.py](../../../tensorrt_llm/bench/dataclasses/reporting.py#L570) 和 [statistics.py](../../../tensorrt_llm/bench/dataclasses/statistics.py#L188)。

## 目录

- [Deepseek R1 0528](#deepseek-r1-0528)
- [GPT-OSS 120B](#gpt-oss-120b)
- [GPT-OSS 20B](#gpt-oss-20b)
- [LLaMA v3.3 70B](#llama-v33-70b)
  - [LLaMA v3.3 70B - RTX 6000 Pro Blackwell Server Edition](#llama-v33-70b-rtx-configurations)
- [LLaMA v4 Maverick](#llama-v4-maverick)
- [Qwen3 235B A22B](#qwen3-235b-a22b)
  - [Qwen3 235B A22B - RTX 6000 Pro Blackwell Server Edition](#qwen3-235b-a22b-rtx-configurations)
- [Qwen3 30B A3B](#qwen3-30b-a3b)
  - [Qwen3 30B A3B - RTX 6000 Pro Blackwell Server Edition](#qwen3-30b-a3b-rtx-configurations)

---

<a id="deepseek-r1-0528"></a>

# Deepseek R1 0528

| 序列长度（ISL/OSL） | B200<br/>DEP4 (FP4) | GB200<br/>DEP4 (FP4) | H200<br/>DEP8 (FP8) |
|---|---|---|---|
| 1000/1000 | 6,463 | 6,939 | 1,627 |
| 1024/1024 | 6,430 | 6,924 | 1,620 |
| 1024/8192 | 3,862 | 4,379 | 1,218 |
| 1024/32768 | 1,451 | 1,465 | 438 |
| 8192/1024 | 1,168 | 1,192 | |

单位：`每 GPU 每秒输出 token 数`

> 💡 **AI Infra 视角**：读表训练（以 DeepSeek R1 为例）：
> - **同硬件（B200）下 ISL/OSL 从 1K/1K 到 1K/32K，吞吐从 6463 掉到 1451**——输出越长，KV cache 越大，每步算得越久，吞吐骤降。这就是"长输出场景性能差"的直观证据；
> - **B200 FP4 (6463) vs H200 FP8 (1627) ≈ 4 倍差距**——新硬件 + 低精度叠加的效果；
> - 8192/1024（长输入短输出）比 1024/8192 还慢——长输入的 prefill 成本巨大（O(L²) attention）。**这就是为什么长上下文场景都要上分块 prefill、稀疏注意力等技术**。

---

<a id="gpt-oss-120b"></a>

# GPT-OSS 120B

| 序列长度（ISL/OSL） | B200<br/>DEP2 (FP4) | GB200<br/>TP1 (FP4) | H200<br/>TP1 (FP8) | H100<br/>DEP4 (FP8) |
|---|---|---|---|---|
| 1000/1000 | 25,943 | 27,198 | 6,868 | 4,685 |
| 1024/1024 | 25,870 | 26,609 | 6,798 | 4,715 |
| 1024/8192 | 17,289 | 14,800 | 3,543 | |
| 1024/32768 | 6,279 | 5,556 | | 1,177 |
| 8192/1024 | 6,111 | 6,835 | 1,828 | 1,169 |
| 32768/1024 | 1,392 | 1,645 | 519 | 333 |

单位：`每 GPU 每秒输出 token 数`

---

<a id="gpt-oss-20b"></a>

# GPT-OSS 20B

| 序列长度（ISL/OSL） | B200<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) | H200<br/>TP1 (FP8) | H100<br/>TP1 (FP8) |
|---|---|---|---|---|
| 1000/1000 | 53,812 | 55,823 | 13,858 | 11,557 |
| 1024/1024 | 53,491 | 56,528 | 13,890 | 11,403 |
| 1024/8192 | 34,702 | 38,100 | 12,743 | 8,617 |
| 1024/32768 | 14,589 | 16,463 | | |
| 8192/1024 | 11,904 | 12,941 | 4,015 | 3,366 |
| 32768/1024 | 2,645 | 2,905 | 915 | 785 |

单位：`每 GPU 每秒输出 token 数`

> 💡 **AI Infra 视角**：GPT-OSS-20B 在 B200 上达到 5.3 万 token/s/GPU——比 120B 快一倍多，因为**模型小一半、激活计算少一半**（20B 是稀疏 MoE，激活参数只有 3.6B）。对比同一模型的不同显存配置（TP1 vs DEP2）可以看出注意力 DP 的效果。**"参数激活量（active parameters）决定速度"**：MoE 模型的推理速度主要由激活参数决定，而不是总参数。

---

<a id="llama-v33-70b"></a>

# LLaMA v3.3 70B

| 序列长度（ISL/OSL） | B200<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) | H200<br/>TP2 (FP8) | H100<br/>TP2 (FP8) |
|---|---|---|---|---|
| 1000/1000 | 6,920 | 7,769 | 2,587 | 2,209 |
| 1024/1024 | 6,842 | 7,751 | 2,582 | |
| 1024/8192 | 3,242 | 3,805 | 2,009 | |
| 8192/1024 | 1,362 | 1,491 | 537 | 398 |
| 32768/1024 | 274 | 302 | 120 | |

单位：`每 GPU 每秒输出 token 数`

---

<a id="llama-v33-70b-rtx-configurations"></a>

# LLaMA v3.3 70B - RTX 6000 Pro Blackwell Server Edition

*展示张量并行（TP）和流水线并行（PP）配置*

| 序列长度（ISL/OSL） | **1 GPU**<br/>TP1,PP1 (FP4) | **2 GPUs**<br/>TP1,PP2 (FP4) |
|---|---|---|
| 1000/1000 | 1,724 | 1,901 |
| 1024/1024 | 1,708 | 1,887 |
| 8192/1024 | 296 | 327 |
| 32768/1024 | | 67 |

单位：`每 GPU 每秒输出 token 数`

> 💡 **AI Infra 视角**：RTX 6000 的 TP1,PP2 例子展示了**显存不足时的应对**：单卡放不下 70B FP4 权重（或者显存被 KV cache 挤爆），就用 2 卡 PP 分摊。注意 PP2 的吞吐（1901）只比 PP1（1724）高 10%——**PP 有气泡开销，两卡不是翻倍**。加卡的收益递减效应是并行调优的重要认知。

---

<a id="llama-v4-maverick"></a>

# LLaMA v4 Maverick

| 序列长度（ISL/OSL） | B200<br/>DEP4 (FP4) | GB200<br/>DEP4 (FP4) | H200<br/>DEP8 (FP8) |
|---|---|---|---|
| 1000/1000 | 11,337 | 11,828 | 4,146 |
| 1024/1024 | 11,227 | 11,905 | 4,180 |
| 1024/8192 | 5,174 | 5,508 | 1,157 |
| 1024/32768 | 2,204 | 2,300 | 679 |
| 8192/1024 | 3,279 | 3,444 | 1,276 |
| 32768/1024 | 859 | 963 | |

单位：`每 GPU 每秒输出 token 数`

---

<a id="qwen3-235b-a22b"></a>

# Qwen3 235B A22B

| 序列长度（ISL/OSL） | B200<br/>DEP4 (FP4) | GB200<br/>DEP4 (FP4) | H200<br/>DEP4 (FP8) | H100<br/>DEP8 (FP8) |
|---|---|---|---|---|
| 1000/1000 | 5,764 | 6,172 | 3,288 | 1,932 |
| 1024/1024 | 5,756 | 5,862 | 3,268 | 1,935 |
| 1024/8192 | 3,389 | 3,423 | 1,417 | 873 |
| 1024/32768 | 1,255 | | | |
| 8192/1024 | 1,410 | 1,464 | 627 | |
| 32768/1024 | 319 | 333 | 134 | |

单位：`每 GPU 每秒输出 token 数`

---

<a id="qwen3-235b-a22b-rtx-configurations"></a>

# Qwen3 235B A22B - RTX 6000 Pro Blackwell Server Edition

*展示张量并行（TP）和流水线并行（PP）配置*

| 序列长度（ISL/OSL） | **4 GPUs**<br/>DEP2,PP2 (FP4) | **8 GPUs**<br/>DEP8,PP1 (FP4) |
|---|---|---|
| 1000/1000 | 1,731 | 969 |
| 1024/1024 | 1,732 | 963 |
| 1024/8192 | 644 | 711 |
| 32768/1024 | 70 | |

单位：`每 GPU 每秒输出 token 数`

> 💡 **AI Infra 视角**：这个例子很有意思：235B 模型在 RTX 6000 上，**4 卡（1731）反而比 8 卡（969）快**！为什么？DEP8 意味着 8 卡做 attention DP——每卡只有 1/8 的请求，attention 计算密度太低，GPU 空转，而权重通信开销照付。**加卡不一定变快——并行度超过拐点后，通信/空闲开销吃掉收益**。这是面试加分点：并行不是越多越好。

---

<a id="qwen3-30b-a3b"></a>

# Qwen3 30B A3B

| 序列长度（ISL/OSL） | B200<br/>TP1 (FP4) | GB200<br/>TP1 (FP4) |
|---|---|---|
| 1000/1000 | 26,971 | 22,856 |
| 1024/1024 | 26,611 | 22,201 |
| 1024/8192 | 13,497 | 14,272 |
| 1024/32768 | 4,494 | 4,925 |
| 8192/1024 | 5,735 | 6,201 |
| 32768/1024 | 1,265 | 1,380 |

单位：`每 GPU 每秒输出 token 数`

---

<a id="qwen3-30b-a3b-rtx-configurations"></a>

# Qwen3 30B A3B - RTX 6000 Pro Blackwell Server Edition

*展示张量并行（TP）和流水线并行（PP）配置*

| 序列长度（ISL/OSL） | **2 GPUs**<br/>DEP2,PP1 (FP4) | **4 GPUs**<br/>DEP2,PP2 (FP4) | **8 GPUs**<br/>DEP8,PP1 (FP4) | **1 GPU**<br/>TP1,PP1 (FP4) |
|---|---|---|---|---|
| 1000/1000 | 8,409 | 7,059 | 3,985 | 9,938 |
| 1024/1024 | | 7,019 | | 9,755 |
| 1024/8192 | 3,577 | | 2,406 | 3,621 |
| 8192/1024 | | 1,416 | | 1,914 |
| 32768/1024 | | | 180 | 374 |

单位：`每 GPU 每秒输出 token 数`

> 💡 **AI Infra 视角**：再次印证：Qwen3-30B-A3B 单卡（9938）比 2 卡（8409）、4 卡（7059）、8 卡（3985）都快！**模型小（激活仅 3B）时单卡就是最优**——多卡只是把通信开销加进来。**并行只在"单卡放不下"或"单卡算不动"时才需要**（parallel-strategy.md 开头的两句话）。这个认知对选型极其重要。

---



## 复现基准结果

```{note}
只有上表展示的模型支持此工作流。
```

下表是基准测试过程中使用的命令参考。关于此基准测试工作流的更详细描述，见[基准测试套件文档](./perf-benchmarking.md)。

### 命令总览

测试使用 PyTorch 后端执行——此工作流不需要构建引擎。

| 阶段 | 描述 | 命令 |
| :- | - | - |
| [数据集](#preparing-a-dataset) | 创建合成数据集 | `trtllm-bench --model $model_name prepare-dataset --output $dataset_file token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0` |
| [运行](#running-the-benchmark) | 用数据集运行基准测试 | `trtllm-bench --model $model_name throughput --dataset $dataset_file --backend pytorch --config $llm_options` |

> 💡 **AI Infra 视角**：**"不需要构建引擎"是 PyTorch 后端的卖点**——传统 TRT-LLM 要先把模型编译成 TensorRT engine（耗时数分钟到数小时），PyTorch 后端加载即跑，benchmark 迭代快得多。生产上"能不能快速迭代"决定优化效率。

### 变量

| 名称 | 描述 |
| :- | - |
| `$isl` | 基准输入序列长度。 |
| `$osl` | 基准输出序列长度。 |
| `$tp_size` | 运行基准的张量并行度。 |
| `$pp_size` | 运行基准的流水线并行度。 |
| `$ep_size` | 运行基准的专家并行度。 |
| `$model_name` | HuggingFace 模型名，如 meta-llama/Llama-2-7b-hf，或本地权重目录路径。 |
| `$dataset_file` | `trtllm-bench prepare-dataset` 生成的数据集文件路径。 |
| `$num_requests` | 生成数据集时的请求数量。 |
| `$seq_len` | ISL + OSL 的序列长度。 |
| `$llm_options` | （可选）包含 LLM API 附加选项的 yaml 文件。 |

### 准备数据集

要准备数据集，使用 `trtllm-bench prepare-dataset` 子命令。
要生成合成数据集，运行以下命令：

```shell
trtllm-bench --model $model_name prepare-dataset --output $dataset_file token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0
```

该命令会生成一个文本文件（路径由 `$dataset_file` 指定），其中所有请求具有相同的输入/输出序列长度组合。脚本通过 tokenizer 获取词表大小并从中随机采样 token ID，生成完全随机的序列。上面的命令中，由于输入和输出序列的标准差都设为 0，所有请求的长度都相同。

> 💡 **AI Infra 视角**：为什么用**随机 token 的合成数据**而不是真实数据？因为基准测试要测的是"引擎的性能上限"，不是"模型回答质量"——随机 token 完全消除内容对性能的影响（缓存命中率、生成内容等），让结果只反映引擎能力。真实生产调优时再换真实流量（用 `replay` 真实数据集）。

对于每个输入和输出序列长度组合，下表详细列出了使用的 `$num_requests`。较短的输入和输出长度使用了更多请求，以保证系统进入稳定状态（steady state），因为请求进出系统的速率更快。对于较长的输入/输出序列长度，请求在系统中停留更久，因此需要较少的请求就能达到稳定状态。

| 输入长度 | 输出长度 | 请求数量 |
|--------------|---------------|---------------------|
| 1024         | 1024          | 3000                |
| 8192         | 1024          | 1500                |
| 1024         | 8192          | 1500                |
| 32768        | 1024          | 1000                |
| 1024         | 32768         | 1000                |

### 运行基准测试

要使用生成的数据集运行基准测试，直接使用 `trtllm-bench throughput` 子命令。基准测试器会运行一个离线最大吞吐场景，所有请求快速连续排队。你只需提供模型名（HuggingFace 引用或本地模型路径）、[生成的数据集](#preparing-a-dataset)，以及包含任何所需 LLM API 附加选项的文件（详见 [tensorrt_llm/llmapi/llm_args.py:LlmArgs](source:tensorrt_llm/llmapi/llm_args.py)）。

对于稠密 / 非 MoE 模型：
```shell
trtllm-bench --tp $tp_size --pp $pp_size --model $model_name throughput --dataset $dataset_file --backend pytorch --config $llm_options
```
Llama 3.3

`llm_options.yml`
```yaml
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192]
```

对于 MoE 模型：

```shell
trtllm-bench --tp $tp_size --pp $pp_size --ep $ep_size --model $model_name throughput --dataset $dataset_file --backend pytorch --config $llm_options
```

GPT-OSS：

`llm_options.yml`
```yaml
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192]
enable_attention_dp: true
kv_cache_config:
  dtype: fp8
  # Hopper: use auto
moe_config:
  backend: CUTLASS
  # Hopper: use TRITON
```

DeepSeek R1：

`llm_options.yml`
```yaml
attention_dp_config:
  batching_wait_iters: 0
  enable_balance: true
  timeout_iters: 60
enable_attention_dp: true
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192]
moe_config:
  backend: CUTLASS
kv_cache_config:
  dtype: fp8
```

Qwen3 MoE、Llama4 Maverick：

`llm_options.yml`
```yaml
enable_attention_dp: true
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192]
```

> 💡 **AI Infra 视角**：注意配置里的三个高频项，理解它们的用意：
> - **`cuda_graph_config.enable_padding: true` + batch_sizes 列表**：CUDA Graph 的"形状集合"（架构文档讲过）——把 batch 1 到 8192 的图都捕获，任何 batch 都能命中；
> - **`enable_attention_dp: true`**：attention 层数据并行（parallel-strategy.md 讲过）——基准场景是**高并发**，attention DP 是标配；
> - **`kv_cache_config.dtype: fp8`**：KV cache 降精度省显存。
> 这些配置合在一起代表"高并发基准的标准配方"。

在许多情况下，我们还会通过给基准命令加 `--kv_cache_free_gpu_mem_fraction 0.95` 来提高 KV cache 百分比。这让我们能获得比默认的 `0.90` 更好的性能。如果遇到内存不足（OOM）错误，我们回退到 `0.90` 或更低。

基准测试完成后，结果会打印到终端。例如：

```shell
===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                     43.2089
Total Output Throughput (tokens/sec):             5530.7382
Per User Output Throughput (tokens/sec/user):     2.0563
Per GPU Output Throughput (tokens/sec/gpu):       5530.7382
Total Token Throughput (tokens/sec):              94022.5497
Total Latency (ms):                               115716.9214
Average request latency (ms):                     75903.4456
Per User Output Speed [1/TPOT] (tokens/sec/user): 5.4656
Average time-to-first-token [TTFT] (ms):          52667.0339
Average time-per-output-token [TPOT] (ms):        182.9639

-- Per-Request Time-per-Output-Token [TPOT] Breakdown (ms)

[TPOT] MINIMUM: 32.8005
[TPOT] MAXIMUM: 208.4667
[TPOT] AVERAGE: 182.9639
[TPOT] P50    : 204.0463
[TPOT] P90    : 206.3863
[TPOT] P95    : 206.5064
[TPOT] P99    : 206.5821

-- Per-Request Time-to-First-Token [TTFT] Breakdown (ms)

[TTFT] MINIMUM: 3914.7621
[TTFT] MAXIMUM: 107501.2487
[TTFT] AVERAGE: 52667.0339
[TTFT] P50    : 52269.7072
[TTFT] P90    : 96583.7187
[TTFT] P95    : 101978.4566
[TTFT] P99    : 106563.4497

-- Request Latency Breakdown (ms) -----------------------

[Latency] P50    : 78509.2102
[Latency] P90    : 110804.0017
[Latency] P95    : 111302.9101
[Latency] P99    : 111618.2158
[Latency] MINIMUM: 24189.0838
[Latency] MAXIMUM: 111668.0964
[Latency] AVERAGE: 75903.4456
```

> 💡 **AI Infra 视角**：**必须学会读这份输出**——这是 AI Infra 日常基准测试的"体检报告"：
> - `Request Throughput`：每秒完成的请求数（QPS）；
> - `Total Output Throughput`：每秒生成的所有输出 token 数（含所有请求）；
> - `Per GPU Output Throughput`：归一化到每卡（跨部署比较用）；
> - `TTFT / TPOT`：前文讲过的两个延迟指标，这里给出分布（P50/P90/P95/P99）——**SLO 承诺的是 P95/P99，不是平均**；
> - 观察：本例 TPOT 的 P50~P99 接近（204~206ms），而 TTFT 的 P50~P99 差 2 倍（52s~106s）——**TTFT 的尾部恶化通常来自排队**（prefill 排队），TPOT 稳定说明 decode 执行健康。
> 调试流程：吞吐不达标 → 看是算力（kernel）瓶颈还是排队（调度）瓶颈 → 针对性调参。

> [!WARNING] 在某些情况下，基准测试器可能什么都不打印。这种行为通常意味着基准测试遇到了内存不足（OOM）问题。尝试用 `--kv_cache_free_gpu_mem_fraction` 选项降低 KV cache 百分比来减少内存使用。
