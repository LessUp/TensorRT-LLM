<!--
  本文档为 TensorRT-LLM 官方 Performance Analysis 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/developer-guide/perf-analysis.md
-->

(perf-analysis)=

# 性能分析

NVIDIA Nsight Systems 在应用层面的报告信息量非常大。指标采样能力随着世代不断增强，在计时分析和用 NVIDIA Nsight Compute 做 kernel 级深潜之间提供了一个很好的中间地带。

> 💡 **AI Infra 视角**：性能分析的"工具金字塔"（必须知道）：
> - **顶层：计时/日志**——引擎自带的耗时统计（如 trtllm-bench 的输出），秒级粗粒度；
> - **中层：Nsight Systems（nsys）**——整个应用的时序图：每个 kernel 何时启动、何时结束、GPU 何时空闲、CPU 在忙什么。**回答"时间都去哪了"**，本页主角；
> - **底层：Nsight Compute（ncu）**——单个 kernel 内部的微观分析：占用率、显存带宽、瓶颈类型。**回答"这个 kernel 为什么慢"**。
> 正确姿势：先 nsys 找到瓶颈 kernel，再用 ncu 分析它——**从粗到细，不要上来就 ncu**。

考虑到大语言模型（LLM）可能运行时间很长，且模型在单次推理或一次二进制执行中可能经历多样的工作负载，NVIDIA 为 TensorRT LLM 增加了特性，以充分利用 Nsight Systems 的能力。本文档概述这些特性，并提供如何最好地利用它们来理解你的应用。

## 特性说明

主要功能：
  * 依赖开关 CUDA profiler 运行时 API。
  * （仅 PyTorch 工作流）开关 PyTorch profiler。
  * 提供一种了解用户可能想要聚焦哪些区域的方法。

开关 CUDA profiler 运行时 API：
  * 允许用户确切知道被分析的区域对应什么。
  * 产生更小的后处理文件（如用于指标提取等）。

（仅 PyTorch 工作流）开关 PyTorch profiler：
  * 帮助用户分析模型中的性能分解。
  * 产生更小的后处理文件（如用于指标提取等）。

> 💡 **AI Infra 视角**：核心思路是**精确圈定分析范围**：LLM 推理跑几小时，只 profile 第 100~150 次迭代（稳定运行区间），而不是全程记录——文件小、噪音少、定位准。`TLLM_PROFILE_START_STOP=A-B` 就是"只在这几轮迭代内打开 profiler"的开关。**"分析窗口要精准"是所有 profiling 的第一原则**。

## 与 NVIDIA Nsight Systems 启动配合

完整选项请查阅 Nsight Systems 用户指南。

在 PyTorch 工作流上，默认提供基础 NVTX 标记。在 C++/TensorRT 工作流上，调用 `scripts/build_wheel.py` 编译脚本时追加 `--nvtx`，并干净地重新构建代码。

> 💡 **AI Infra 视角**：**NVTX**（NVIDIA Tools Extension）是"给代码贴标签"的机制——在时间线上给每个代码段命名（如"调度"、"forward"、"采样"），nsys 时间线上就能直接看到每个阶段花了多久。TRT-LLM 默认就在关键位置贴了 NVTX 标记，所以开箱即可看到结构化时间线。**看懂 nsys 时间线的第一步：认识 NVTX 标记**。

### 只采集特定迭代

为减小 Nsight Systems profile 文件大小，并确保只采集特定迭代，设置环境变量 `TLLM_PROFILE_START_STOP=A-B`，并在 `nsys profile` 命令后追加 `-c cudaProfilerApi`。

> 💡 **AI Infra 视角**：`-c cudaProfilerApi` 是"按需采集"模式：profiler 默认不记录，等应用调用 CUDA profiler API 开始/停止才记录。TRT-LLM 在每次迭代的边界调用这些 API——配合 `TLLM_PROFILE_START_STOP`，实现"只录 100~150 轮"。

### 启用更多 NVTX 标记用于调试

设置环境变量 `TLLM_NVTX_DEBUG=1`。

### 启用垃圾回收（GC）NVTX 标记

设置环境变量 `TLLM_PROFILE_RECORD_GC=1`。

> 💡 **AI Infra 视角**：为什么要记录 GC？Python 的垃圾回收会在**任意时刻**暂停执行（stop-the-world）——如果时间线上某处 GPU 空闲又找不到原因，可能就是 GC 在跑。给 GC 贴上 NVTX 标记后，时间线上直接可见。**"异常空闲先排除 GC/同步点"是 PyTorch 推理排障的基本功**。

### 在 NVTX 标记中启用 GIL 信息

在 Nsys 的 "-t" 选项中追加 "python-gil"。

> 💡 **AI Infra 视角**：**GIL**（Global Interpreter Lock）是 Python 的全局锁——同一时刻只有一个线程能执行 Python 字节码。多线程代码如果大量时间花在"等 GIL"，说明 Python 层并行度没起来。时间线上看 GIL 状态能直接判断"CPU 并行是否被 GIL 卡死"。

## 与 PyTorch profiler 配合（仅 PyTorch 工作流）

### 收集 PyTorch profiler 结果

1. 设置环境变量 `TLLM_PROFILE_START_STOP=A-B` 指定要收集的迭代范围。
2. 设置环境变量 `TLLM_TORCH_PROFILE_TRACE=<path>`，结果将保存到 `<path>`。

对于 VisualGen，改用 `TLLM_PROFILE_VISUAL_GEN_START_STOP`。数字范围选择
每个请求的去噪（denoise）步骤，而 `predenoise`、`postdenoise` 和 `all` 选择
对应的生成阶段。例如：

```bash
TLLM_PROFILE_VISUAL_GEN_START_STOP=0-4 \
TLLM_TORCH_PROFILE_TRACE=/tmp/visual-gen-trace.json \
python examples/visual_gen/quickstart_example.py
```

每个进程将其 trace 写入 rank 特定的路径，如
`/tmp/visual-gen-trace-rank-0.json`。如果一个进程捕获了多个
窗口，后续的 trace 会加窗口后缀，如
`/tmp/visual-gen-trace-rank-0-window-1.json`。

这些范围使用现有的 VisualGen CUDA/Nsight 边界。`all` 捕获
从文本编码到 VAE 解码的完整请求。`predenoise`
捕获文本编码、latent 准备和去噪循环设置；
`postdenoise` 捕获 VAE 解码和剩余的请求工作。

有两个契约值得了解：数字范围永远不会超出它选择的去噪循环
（超过最后一步的停止索引会在最后一步处关闭），
并且在每个请求运行多个去噪循环的流水线上，per-loop
模式适用于每个循环——数字范围每个循环写一个 trace，而
`postdenoise` 在第一个循环后（而不是最后一个）布防。所有窗口在
流水线的推理入口点打开，因此 executor 侧的请求准备在
窗口之外，即使它计入报告的 `generation` 延迟。

有关每种模式的具体细节，请参见
`tensorrt_llm/_torch/visual_gen/profiler.py` 中的 `parse_profile_range`。

### 可视化 PyTorch profiler 结果

使用 [chrome://tracing/](chrome://tracing/) 检查保存的 profile。

> 💡 **AI Infra 视角**：PyTorch profiler（torch.profiler）给的是**算子级**时间线（每个 PyTorch 算子/ kernel 的耗时和调用栈），chrome://tracing 打开 JSON 后可以看到每层每个算子的耗时分解——**回答"模型前向里哪一层最慢"**。nsys 管整体、torch.profiler 管模型内部，两者配合使用。

## 示例

与 MPI 相关的完整选项请查阅 Nsight Systems 用户指南。

### 在 `trtllm-bench`/`trtllm-serve` 运行中分析特定迭代

假设我们要分析 `trtllm-bench`/`trtllm-serve` 运行的第 100 到 150 次迭代，并希望为调试收集尽可能多的信息，如 GIL、调试 NVTX 标记等：

```bash
#!/bin/bash

# 为基准测试准备数据集
trtllm-bench --model ${MODEL_PATH} \
    prepare-dataset \
    --output dataset.txt \
    token-norm-dist \
    --num-requests=${NUM_SAMPLES} \
    --input-mean=1000 --output-mean=1000 --input-stdev=0 --output-stdev=0

# 基准测试并分析
TLLM_PROFILE_START_STOP=100-150 nsys profile \
  -o trace -f true \
  -t 'cuda,nvtx,python-gil' -c cudaProfilerApi \
  --cuda-graph-trace node \
  -e TLLM_PROFILE_RECORD_GC=1,TLLM_LLMAPI_ENABLE_NVTX=1,TLLM_TORCH_PROFILE_TRACE=trace.json \
  --trace-fork-before-exec=true \
  trtllm-bench \ # 或 trtllm-serve 命令
    --model deepseek-ai/DeepSeek-V3 \
    --model_path ${MODEL_PATH} \
    throughput \
    --dataset /tmp/dataset.txt --warmup 0 \
    --backend pytorch \
    --streaming
```

Nsight Systems 报告将保存到 `trace.nsys-rep`。用 NVIDIA Nsight Systems 应用程序打开。

PyTorch profiler 结果将保存到 `trace.json`。使用 [chrome://tracing/](chrome://tracing/) 检查保存的 profile。

> 💡 **AI Infra 视角**：这就是完整的"生产性能采样命令"——建议直接收藏作为模板。逐项理解：
> - `-t 'cuda,nvtx,python-gil'`：记录 CUDA kernel + NVTX 标记 + GIL 状态；
> - `-c cudaProfilerApi`：按需采集（配合 TLLM_PROFILE_START_STOP）；
> - `--cuda-graph-trace node`：CUDA Graph 的 node 级追踪（能看 graph 内部每个 kernel）；
> - `--trace-fork-before-exec=true`：多进程（TP/PP）时每个 rank 都单独 trace；
> - `--streaming`：流式输出（便于观察 TTFT）。
> 采样出的 `trace.nsys-rep` 时间线上，你应该能认出：调度阶段（CPU）、N 个 forward kernel（GPU 繁忙区）、采样阶段、GC/GIL 事件——结合前面文档学的迭代流程，做"阶段耗时分解"。

## MoE 专家负载均衡分析（Perfect Router）

对于混合专家（MoE）模型，性能会因 token 如何路由到专家而显著变化。专家负载不均衡会导致某些 GPU 过载而其他 GPU 利用不足，从而吞吐次优。

TensorRT-LLM 提供 `ENABLE_PERFECT_ROUTER` 环境变量来帮助分析和隔离专家负载均衡问题与 kernel 性能问题。

> 💡 **AI Infra 视角**：这是一个非常聪明的**实验设计**——"替换变量法"：想知道"负载不均衡"到底拖累了多少性能？把路由器换成"完美均衡"的假路由器跑一遍，对比两次吞吐。差值 = 负载不均衡的代价。**这是性能归因的经典方法论：控制变量，把怀疑的变量替换成理想值**。

### 它的作用

启用后，此特性**绕过学习的路由器（learned router）**，用预先计算的、完美负载均衡的路由 logits 替代它。这创造了一个理想化场景：token 在所有专家和 GPU 间均匀分布。

关键行为：
- 学习的门控/路由器仍然计算（保持计时真实）
- 门控输出被**丢弃**，替换为理想的均衡 logits
- logits 为常见 batch 大小预先计算并缓存，以最小化开销
- 适用于所有 MoE 后端（CUTLASS、TRTLLM、TRITON）

```{warning}
此特性**仅用于性能分析**。它会产生**不正确的模型输出**，因为学习的路由器决策被丢弃了。生产推理中绝不要使用。
```

### 何时使用

当你想做以下事情时使用 `ENABLE_PERFECT_ROUTER`：

1. **建立性能上限**：测量专家负载完美均衡时 MoE 吞吐的理论最佳情况。

2. **隔离路由瓶颈**：比较有/无完美路由的性能，确定学习的路由器是否导致负载不均衡问题。

3. **测试不同的负载均衡策略**：在实现自定义路由逻辑之前，验证 MoE kernel 和通信模式在均衡负载下行为正确。

4. **基准测试 kernel 效率**：消除路由可变性，获得一致、可复现的 kernel 性能测量。

### 如何启用

运行工作负载前设置环境变量。它同时适用于 `trtllm-bench` 和 `trtllm-serve`：

```bash
export ENABLE_PERFECT_ROUTER=1
```

### 示例工作流

```bash
# 第 1 步：用正常（学习）路由做基准测试
trtllm-bench ...
# 或
trtllm-serve ...

# 第 2 步：用完美路由做基准测试（上限）
ENABLE_PERFECT_ROUTER=1 trtllm-bench ...
# 或
ENABLE_PERFECT_ROUTER=1 trtllm-serve ...

# 第 3 步：比较吞吐数字
# 如果完美路由提升超过 10%，说明路由不均衡问题显著
```

### 解读结果

| 场景 | 解读 |
|----------|----------------|
| 有/无完美路由性能相似 | 路由器负载均衡不是瓶颈；把优化精力放在别处 |
| 完美路由显著提升 | 学习的路由器导致负载不均衡；考虑路由器优化或负载均衡策略 |

> 💡 **AI Infra 视角**：">10% 提升"这个阈值是经验值：路由不均衡通常至少带来 10% 的损失（热门专家效应）。**如果完美路由没有显著提升，就别在负载均衡上浪费时间**——把精力投到 kernel 或调度上。这就是"用数据决定优化方向"的工作方式，面试时讲这种"分析→归因→决策"流程很加分。

### 支持的模型

```{note}
此特性目前需要模型特定的集成。支持完美路由的管道必须添加到每个 MoE 模型实现中。如果你需要此特性但模型还不支持，你需要按照现有实现中的模式添加集成。
```

```{note}
完美路由器 logits 是专门为 `RenormalizeMoeRoutingMethod`（先 TopK，再 Softmax）设计的。使用其他路由方法的模型，如 `DefaultMoeRoutingMethod` 或 `DeepSeekV3MoeRoutingMethod`，需要调整 logit 生成逻辑以匹配其路由行为。
```

目前支持：
- GPT-OSS（使用 `RenormalizeMoeRoutingMethod`）
- DeepSeek-V3 / DeepSeek-R1（使用 `DeepSeekV3MoeRoutingMethod`）
