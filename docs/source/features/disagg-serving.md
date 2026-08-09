<!--
  本文档为 TensorRT-LLM 官方 Disaggregated Serving 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/features/disagg-serving.md
-->

# 分离式服务（Disaggregated Serving）

- [动机](#Motivation)
- [KV Cache 交换](#KV-Cache-Exchange)
  - [多后端支持](#Multi-backend-Support)
  - [NIXL 后端配置](#nixl-backend-configuration)
  - [重叠优化](#Overlap-Optimization)
  - [缓存布局转换](#Cache-Layout-Transformation)
  - [唯一全局请求 ID](#Unique-Global-Request-ID)
- [用法](#Usage)
  - [Dynamo](#Dynamo)
  - [trtllm-serve](#trtllm-serve)
  - [多实例](#multiple-instances)
- [环境变量](#Environment-Variables)
- [故障排查与 FAQ](#Troubleshooting-and-FAQ)

## 动机

LLM 推理有两个阶段：上下文（context/prefill）和生成（generation/decode）阶段。上下文阶段计算 prompt token 的 KV cache，而生成阶段使用缓存值逐个生成 token。这两个阶段的计算特性不同。

> 💡 **AI Infra 视角**：两个阶段的"性格"完全不同（这是全文的理论基础）：
> - **prefill**：**计算密集 + 大 batch**——一次性算完整个 prompt 的 attention，可以并行度拉满（矩阵乘大），但响应慢（可能几秒）；
> - **decode**：**显存带宽密集 + 小计算**——每步只算 1 个 token 的 attention，矩阵乘很小，主要瓶颈是把巨大的 KV cache 读进来。
> 两者对硬件、并行策略、batch 大小的最优解都不同——**把两个阶段绑在同一个 GPU 上，本质是让两个性格不合的人合租**。

LLM 推理请求有两种服务方式：

* **聚合式 LLM 服务**（aggregated serving，本技术博客中也称为 in-flight batching 或 IFB），上下文和生成阶段在同一个 GPU 上运行。
* **分离式 LLM 服务**（disaggregated serving），上下文和生成阶段在不同的 GPU 上运行。

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture1.png" width="640" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 1. 聚合式 LLM 服务的执行时间线</em></sub></p>

在聚合式 LLM 服务中，上下文和生成阶段共享相同的 GPU 资源和并行策略。这可能导致干扰（interference）：上下文处理会延迟 token 生成，增加 token 间延迟（TPOT）并降低交互性。图 1 展示了聚合式 LLM 服务的执行时间线。聚合式服务还迫使两个阶段使用单一 GPU 类型和并行配置，尽管它们的计算需求不同。结果是，优化一个指标（如首 token 时间 TTFT）往往以牺牲另一个指标（如 TPOT）为代价。

> 💡 **AI Infra 视角**：**"干扰"是聚合式服务的核心痛点**：一个 32K 长 prompt 的 prefill 进来，会瞬间占满 GPU——正在生成的用户立刻卡顿（TPOT 飙升）。这就是为什么"长输入场景下服务质量不稳定"。**TTFT 和 TPOT 的矛盾**：想压低 TTFT（prefill 优先），decode 就受害；想保 TPOT（decode 优先），prefill 排队 TTFT 变长。聚合式服务只能在两者之间做跷跷板。

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture2.png" width="580" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 2. 分离式 LLM 服务的执行时间线</em></sub></p>

分离式服务通过解耦两个阶段解决这些挑战：每个阶段运行在独立的 GPU 池上，并使用不同的并行策略。这种分离消除了上下文和生成阶段之间的干扰，如图 2 所示，并支持对 TTFT 和 TPOT 独立优化。虽然分离会带来将 KV cache 块从上下文 GPU 传输到生成 GPU 的开销，但优势可能很大——特别是对于**长输入序列 + 中等输出长度**的工作负载，这种场景下干扰最严重。

> 💡 **AI Infra 视角**：分离式服务的成本收益分析（面试加分点）：
> - **成本**：KV cache 要从 prefill 卡传到 decode 卡（网络/互联带宽开销）+ 部署复杂度上升（两套池子、编排器）；
> - **收益**：① TTFT/TPOT 独立优化（prefill 池专注压 TTFT，decode 池专注压 TPOT）；② 资源按需分配（prefill 卡和 decode 卡 1:2 甚至 1:3 配比，因为 decode 阶段长得多）；③ 长输入不干扰生成。
> - **最佳场景**：长输入 + 中等输出（如 RAG 应用、Agent 多轮）——prefill 占比高、干扰严重，分离收益最大。
> **注意：分离式服务不是新概念**——这是从传统 Web 架构（前端/后端分离、读写分离）借来的"职责分离"思想。业界 2024 年起主流化（DeepSeek 的部署架构也用）。

你也可以参考[这篇论文](https://arxiv.org/pdf/2506.05508)了解分离式服务的原理和设计考虑的更多细节。

## KV Cache 交换

### 多后端支持

在 TensorRT-LLM 中，KV cache 交换模块与 KV cache 管理器和底层通信库是模块化解耦的，如图 3 所示。KV cache 交换模块负责缓存的高效发送和接收、及时释放缓存空间，以及在交换过程中执行缓存布局转换。目前，主流通信协议——MPI、UCX 和 NIXL——都被 TensorRT-LLM 支持，底层通信协议利用 RDMA / NVLink。目前我们推荐使用 UCX 和 NIXL 后端，因为我们正在它们之上添加动态扩缩容机制——具体来说，就是动态节点加入和离开。这让客户可以根据流量需求调整负载，或在上下文和生成角色之间动态切换。

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture6.png" width="890" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 3. KV cache 交换架构</em></sub></p>

> 💡 **AI Infra 视角**：KV cache 传输走的是什么链路？
> - **节点内**：NVLink（GPU 到 GPU 直连，几百 GB/s）——快；
> - **节点间**：RDMA over InfiniBand（网卡到网卡直通，绕过 CPU，200~400 Gb/s）——也很快。
> 注意 RDMA 的关键特性：**数据不经过 CPU 和内核**，网卡直接从 GPU 显存搬数据（GPU Direct RDMA）——省掉两次拷贝。传输协议层：MPI（老）、UCX（通用中间层）、NIXL（NVIDIA 新自研）。**"谁来做零拷贝传输"是高性能分布式系统的核心工程问题**。

### NIXL 后端配置

NIXL 支持多种底层通信后端用于分离式服务中的 KV cache 交换。后端可以通过 `TRTLLM_NIXL_KVCACHE_BACKEND` 环境变量配置。

**支持的 NIXL 后端：**
- **UCX**（默认）
- **LIBFABRIC**（从 v0.16.0 起可用）

如果指定了不支持的后端，NIXL 会自动回退到 UCX。

详细的设置说明和配置示例，请参考[分离式服务示例文档](../../../examples/disaggregated/README.md)。

### 重叠优化

为优化分离式服务的整体性能，TensorRT LLM 将 KV cache 传输与多个独立请求的计算重叠。当一个请求在发送或接收其 KV cache 块时，其他请求可以继续计算，如图 4 所示。此外，如果上下文和生成实例每个实例使用多个 GPU，不同 GPU 组之间的 KV cache 传输可以并行进行。

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture7.png" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 4. KV cache 交换时序图</em></sub></p>

> 💡 **AI Infra 视角**：传输开销被"藏"起来的手段：多个请求错开传输（请求 A 传 KV 时，请求 B 在算）——和 Overlap Scheduler 的 CPU/GPU 重叠是同一思想的不同层面。**"凡是等待都可以被重叠掩盖"是高性能系统的通用原则**。

### 缓存布局转换

为最小化 KV cache 传输延迟，TensorRT LLM 目前使用设备显存之间的直接传输进行缓存交换。KV cache 传输支持上下文和生成阶段使用**不同的并行策略**。这种情况下，需要仔细编排 KV cache 块映射。图 5 以上下文阶段 TP2、生成阶段 PP2 为例说明。

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture8.png" width="680" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 5. KV cache 布局转换</em></sub></p>

> 💡 **AI Infra 视角**：这是分离式服务最"工程硬核"的部分：**prefill 池用 TP2 切分（每卡存一半的 KV 头），decode 池用 PP2 切分（每卡存一半的层）**——prefill 卡上的 KV 块布局和 decode 卡上需要的布局完全不同！传输时必须重新映射/重排（把 TP 切分的块重新组织成 PP 需要的层切分）。**"异构并行之间的数据重排"**是分布式推理的常见难题。

KV cache 传输所需的优化因场景而异：单节点多 GPU、多节点多 GPU、或不同的 GPU 型号。为适应这种情况，TensorRT LLM 提供了一组环境变量供不同环境选择。详见以下章节[环境变量](#Environment-Variables)。

### 唯一全局请求 ID

一个请求的上下文和生成阶段必须共享同一个请求 ID：ctx↔gen 的 KV cache 传输以它作为键，所以碰撞（两个在途请求 ID 相同）会破坏传输。这个共享 ID 携带在 `DisaggregatedParams.disagg_request_id` 上。

分离式服务器自己生成这个 ID，采用 **snowflake** 格式——一个自包含的 64 位正整数，无需跨进程协调即可保证唯一。位布局：

```
[ 0 (1 bit) | timestamp_ms (39 bits) | node_id (8 bits) | process_id (6 bits) | counter (10 bits) ]
```

- `node_id`（0–255）标识节点（默认取 MAC 地址的哈希；可在分离式配置中通过 `node_id` 覆盖）。
- `process_id`（0–63）标识该节点上的编排进程。在 [coordinator + worker 集群](#coordinator-and-worker-fleet) 中，每个集群 worker 获得不同的值，因此同一节点上的 worker 不会在同一毫秒内生成相同的 ID。它由 `TRTLLM_DISAGG_WORKER_PROCESS_ID` 环境变量设置（launcher 自动为每个 worker 分配）。
- `(node_id, process_id)` 组合因此使 ID 在所有编排进程间唯一，无需共享计数器或额外的网络往返——每个 worker 在本地铸造自己的 ID。

> 💡 **AI Infra 视角**：**Snowflake ID 算法**是分布式系统生成全局唯一 ID 的经典方案（Twitter 发明）：时间戳 + 机器 ID + 序号 三段拼一个 64 位整数。**"无协调生成全局唯一 ID"**是分布式系统的基础设施问题——snowflake 比"连数据库取号"快得多（不需要网络往返），比"随机 UUID"更紧凑有序。面试分布式系统时这是必知概念。

全局分离式 ID 占用 `[1 << 40, 2**63)` 区间；worker 本地和预热请求 ID 占用不相交的 `[0, 1 << 40)` 区间，两者永不碰撞。如果客户端自带正的 `disagg_request_id`，该值按原样使用，必须全局唯一；未设置时，服务器按上面的 snowflake 格式铸造 ID。

## 用法

### Dynamo

第一种方法使用 [Dynamo](https://github.com/ai-dynamo/dynamo)，一个专为 LLM 工作负载开发的数据中心级推理服务器。Dynamo 引入了其他方法没有的几项高级特性，包括解耦的前处理和后处理 worker，在高并发条件下特别有用。使用 Dynamo 的分离式 LLM 推理工作流如图 7 所示。

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture4.png" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 7. Dynamo 与分离式服务集成</em></sub></p>

在 Dynamo 工作流中，请求首先由前/后处理 worker 处理，然后查询一个智能路由器，确定将请求路由到哪个最优的 decode worker。根据 KV cache 块的可用性，decoder worker 可能跳过 prefill 阶段，或将请求转发给 prefill worker。一旦 prefill worker 处理完 prompt，KV cache 块可以从 prefill worker 发送到 decoder worker，使用上图中称为 ctx_params 的元数据。

> 💡 **AI Infra 视角**：Dynamo 的路由逻辑非常聪明——**"如果 KV cache 已经命中，decode worker 直接开干，跳过 prefill"**：这是 KV cache 前缀复用 + 分离式服务的组合拳（命中缓存的请求 = 免 prefill）。这也解释了为什么 Dynamo 要做"智能路由"：路由决策（去哪个 decode worker）要结合 KV cache 命中情况。

Dynamo 还内置支持 Kubernetes 部署、监控和指标收集。开发团队正在积极推进动态实例扩缩容，进一步增强其在生产环境中的适用性。

关于如何将 Dynamo 与 TensorRT-LLM 一起使用，请参考[此文档](https://docs.nvidia.com/dynamo/backends/tensor-rt-llm)。

### trtllm-serve

评估 TensorRT LLM 分离式推理的第二种方法是用 `trtllm-serve` 为每个上下文和生成实例启动独立的 OpenAI 兼容服务器。另外还会用 `trtllm-serve` 启动一个"分离式（disaggregated）"服务器，作为编排器（orchestrator），接收客户端请求并通过 OpenAI REST API 分发给适当的上下文和生成服务器。图 6 展示了使用这种方法时的分离式服务工作流。当上下文实例完成与 prompt 关联的 KV 块生成后，它向分离式服务器返回一个响应。该响应包含 prompt token、第一个生成的 token 以及与上下文请求和上下文实例相关的元数据。这些元数据称为上下文参数（图 6 中的 `ctx_params`）。生成实例随后使用这些参数与上下文实例建立通信并取回与该请求关联的 KV cache 块。

```{eval-rst}
.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-config-flag-alias
   :end-before: .. end-note-config-flag-alias
```

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog5_Picture3.png" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 6. `trtllm-serve` 与分离式服务集成</em></sub></p>

> 💡 **AI Infra 视角**：用 trtllm-serve 搭建分离式服务的结构图（3 层）：
> ```
> 客户端 → 分离式编排器(:8000) → context 服务器(:8001/:8002) ──KV cache 传输──→ generation 服务器(:8003)
> ```
> 编排器把请求标记为"context-only"发给 prefill 池、把后续请求标记为"generation-only"发给 decode 池；KV cache 传输是**服务器之间直接进行**的（不经过编排器）——编排器只管路由，不管数据搬运。**控制面和数据面分离**，这是分布式系统的经典架构原则。

要以分离模式运行 TRT-LLM，你必须先用 `trtllm-serve` 启动上下文（prefill）和生成（decode）服务器。

我们使用 `cache_transceiver_config` 配置来设置分离式服务，包含以下参数：

```yaml
cache_transceiver_config:
  backend: <str>
  max_tokens_in_buffer: <int>
```

`backend` 指定传输 kvCache 的通信后端，合法选项包括 `DEFAULT`、`UCX`、`NIXL` 和 `MPI`。默认后端是 NIXL。

注意：NIXL 支持通过 `TRTLLM_NIXL_KVCACHE_BACKEND` 环境变量配置的多个底层后端：
- `UCX`（默认）
- `LIBFABRIC`（从 v0.16.0 起可用）

`max_tokens_in_buffer` 定义 kvCache 传输的缓冲区大小，建议将此值设置为大于或等于所有请求的最大 ISL（输入序列长度），以获得最佳性能。

> 💡 **AI Infra 视角**：`max_tokens_in_buffer` 的直觉：传输缓冲区要能装下"一整条请求的 KV"（最大输入长度对应的 KV 量）——装不下就得把一条 KV 拆多次传，慢。**"缓冲要够大，一次搬完"**。这个参数和显存预算有关（缓冲占显存），调大要确认显存够。

例如，你可以这样启动两个上下文服务器和一个生成服务器：

```

# 生成 context_config.yml
# 上下文服务器禁用重叠调度器，因为分离式上下文服务器尚不支持
echo -e "disable_overlap_scheduler: True\ncache_transceiver_config:\n  backend: UCX\n  max_tokens_in_buffer: 2048" > context_config.yml

# 启动上下文服务器
CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8001 --backend pytorch --config ./context_config.yml &> log_ctx_0 &
CUDA_VISIBLE_DEVICES=1 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8002 --backend pytorch --config ./context_config.yml &> log_ctx_1 &

# 生成 gen_config.yml
echo -e "cache_transceiver_config:\n  backend: UCX\n  max_tokens_in_buffer: 2048" > gen_config.yml

# 启动生成服务器
CUDA_VISIBLE_DEVICES=2 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8003 --backend pytorch --config ./gen_config.yml &> log_gen_0 &
```
上下文和生成服务器启动后，你可以启动分离式
服务器，它将接受来自客户端的请求，并在上下文和生成服务器之间做编排。分离式服务器可以这样启动：

```
trtllm-serve disaggregated -c disagg_config.yaml
```
其中 `disagg_config.yaml` 包含上下文和生成服务器的信息。对于当前示例，
它看起来像这样：
```
hostname: localhost
port: 8000
backend: pytorch
context_servers:
  num_instances: 2
  urls:
      - "localhost:8001"
      - "localhost:8002"
generation_servers:
  num_instances: 1
  urls:
      - "localhost:8003"
```

> 💡 **AI Infra 视角**：注意这个配置的比例：**2 个 context 实例 : 1 个 generation 实例**。为什么 context 要更多？因为 prefill 计算密度高（同样的 GPU 时间产出更多 token），处理得快；而 decode 慢（每 token 一步步来），一个 decode 实例能"接住"两个 prefill 实例的产出。**prefill:decode 的实例配比是分离式部署调优的核心参数**（需要按工作负载实验确定）。

将请求路由到上下文服务器时，分离式服务器会把请求标记为"context-only"以跳过生成阶段。同样，
将请求路由到生成服务器时，分离式服务器会把请求标记为"generation-only"以跳过上下文阶段。

配置还接受一个可选字段来调优 HTTP 监听器：

- `server_keep_alive_timeout`（int，默认 `10`）——HTTP keep-alive 超时（秒），应用于面向客户端的监听器，以及协调器在进程内运行时其监听器（见 [Coordinator 和 Worker 集群](#coordinator-and-worker-fleet)）。当客户端持有大型空闲连接池并遇到"已复用连接上 `Connection reset by peer`"时，把它调大（例如 `3600`）：服务器先关闭空闲连接会让客户端留下半关闭的 socket，下次请求时失败。

> 💡 **AI Infra 视角**：`Connection reset by peer` 是生产环境最常见的坑之一：HTTP keep-alive 复用连接，服务器按超时先关掉空闲连接，客户端不知道还在用——下次请求打在死连接上。**长连接池场景要把 keep-alive 超时调到大于客户端空闲时间**。

然后客户端可以向 `localhost:8000` 的分离式服务器发送请求，这是一个 OpenAI 兼容端点。例如，你可以用 curl 向分离式服务器发送请求：
```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "NVIDIA is a great company because",
        "max_tokens": 16,
        "temperature": 0
    }' -w "\n"
```

#### 在 SLURM 集群上启动分离式服务器

请参考 [分离式推理基准脚本](../../../examples/disaggregated/slurm)。

### 多实例

要增加最大并发而无需更多 GPU 节点，你可以在不同节点上部署多个分离式服务器实例，每个实例管理相同的上下文/生成服务器。当单个分离式服务器成为性能瓶颈或用尽临时端口时，这很有帮助。

示例（两节点部署）：

- **节点 A**
  - 上下文服务器：`node-a:8001`
  - 生成服务器：`node-b:8002`
  - 分离式编排器端点：`node-a:8000`
- **节点 B**
  - 上下文服务器：`node-a:8001`
  - 生成服务器：`node-b:8002`
  - 分离式编排器端点：`node-b:8000`
- **客户端入口**
  - 发送请求或用负载均衡器转发到 `node-a:8000` 和 `node-b:8000`

> 💡 **AI Infra 视角**：多实例方案本质是**无状态编排器的水平扩展**——编排器不持有状态（KV 在各服务器上），所以可以多部署几个、负载均衡分发。**"无状态 → 可水平扩展"**是分布式系统设计的金律。

### Coordinator 和 Worker 集群

单个分离式服务器进程本身是单线程的编排器，可能成为吞吐瓶颈（它终止每个客户端连接、运行路由、并代理 ctx→gen 跳转）。为了在单节点上扩展编排器而不部署多个独立实例，`trtllm-serve disaggregated` 可以在共享的 **coordinator（协调器）** 后面运行一组**无状态**的分离式服务器 worker 进程。

> 💡 **AI Infra 视角**：单线程编排器为什么是瓶颈？它处理的是"每请求的每次转发"——TCP 连接处理、HTTP 解析、路由决策都是 CPU 活。流量大时单进程就饱和了（Python 的 GIL 更是雪上加霜）。解法：**协调器/worker 分离**——路由状态集中在协调器，多个 worker 进程分担连接。

两个角色分工如下：

- **Coordinator（协调器）** — 单个进程，拥有所有集群状态：ctx/gen 路由器、worker 就绪状态，以及（对于 KV cache 感知路由器）唯一的 ZMQ 事件摄取端点。它暴露一个内部协调 API（`/select`、`/finish`、`/cluster_info`、`/health`）。
- **集群 worker（Fleet workers）** — `num_workers` 个无状态分离式服务器，通过 `SO_REUSEPORT` 共享公共端口（每个 worker 是绑定同一端口的独立进程，内核按 4 元组哈希在它们之间负载均衡入站连接）。每个 worker 持有一个轻量级委托客户端：它在本地计算路由键（如块哈希），并通过 HTTP 将放置决策委托给协调器。Worker 不持有路由状态，因此无论哪个 worker 终止连接，路由始终保持全局一致。每个 worker 还会为[全局请求 ID](#unique-global-request-id) 获得不同的 `process_id`。

> 💡 **AI Infra 视角**：`SO_REUSEPORT` 是 Linux 的经典技巧：多个进程绑定同一端口，内核自动分发新连接——**零代码水平扩展 TCP 服务**。路由状态集中在协调器（"谁决定放哪"是全局一致性问题），连接处理分散到 workers（"谁来接客"是本地问题）。**有状态集中、无状态分散**是分布式系统架构的标准答案。

这由分离式配置中的两个字段控制：

- `num_workers`（int，默认 `1`）——在公共端口上运行的分离式服务器 worker 进程数。
- `disagg_coordinator_url`（str，可选）——已运行的 coordinator 的 URL。设置后，本进程**不启动** coordinator，其 worker 集群委托给外部 coordinator。

三种拓扑：

| `num_workers` | `disagg_coordinator_url` | 行为 |
|---------------|--------------------------|----------|
| `1` | 未设置 | 单个自包含服务器，进程内 coordinator（默认；与前面示例相同）。 |
| `> 1` | 未设置 | 本进程启动一个**隐式** coordinator（在 `port - 1` 上），并在公共端口上运行 `num_workers` 个委托服务器。 |
| 任意 | 已设置 | 本进程**不启动** coordinator；`num_workers` 个委托服务器指向外部 `disagg_coordinator_url`。 |

```{note}
worker 集群对*有状态*路由器（`kv_cache_aware`、`conversation`）最有用，因为放置决策必须全局一致——该决策被委托给 coordinator。使用*无状态*路由器（`round_robin`、`load_balancing`）时，每个 worker 直接在本地放置，不会发生 coordinator 往返。
```

#### 示例：隐式 coordinator + 4 worker 集群

在 [trtllm-serve](#trtllm-serve) 示例的 `disagg_config.yaml` 中加上 `num_workers` 和路由器类型：

```yaml
hostname: localhost
port: 8000
backend: pytorch
# 在端口 8000 上运行 4 个无状态分离式服务器 worker，隐式
# coordinator 在进程内启动于端口 7999（port - 1）。
num_workers: 4
context_servers:
  num_instances: 2
  urls:
      - "localhost:8001"
      - "localhost:8002"
  router:
    type: kv_cache_aware
generation_servers:
  num_instances: 1
  urls:
      - "localhost:8003"
  router:
    type: kv_cache_aware
```

和之前一样启动——coordinator 和 worker 集群会自动为你启动：

```bash
trtllm-serve disaggregated -c disagg_config.yaml
```

客户端仍然向公共端点（`localhost:8000`）发送请求；worker 集群透明地把路由委托给 coordinator。

#### 示例：外部 coordinator

要把 worker 集群指向已在别处运行的 coordinator（例如在节点间共享的一个），设置 `disagg_coordinator_url` 并省略本进程中的 coordinator：

```yaml
hostname: localhost
port: 8000
backend: pytorch
num_workers: 4
disagg_coordinator_url: "http://coordinator-host:7999"
context_servers:
  num_instances: 2
  urls:
      - "localhost:8001"
      - "localhost:8002"
  router:
    type: kv_cache_aware
generation_servers:
  num_instances: 1
  urls:
      - "localhost:8003"
  router:
    type: kv_cache_aware
```

```{note}
集群 worker 在 coordinator 不可达时会快速失败：启动时它用有界重试（最多 `--server_start_timeout` 秒）探测 coordinator 的 `/cluster_info`，如果失败就以错误退出，而不是启动后对每个请求返回 `Cluster is not ready`。
```

## 环境变量

TRT-LLM 使用一些环境变量控制分离式服务的行为。

* `TRTLLM_NIXL_KVCACHE_BACKEND`：使用 NIXL 作为 cache transceiver 后端时，此变量指定 NIXL 的底层通信后端。合法选项：
  - `UCX`（默认）
  - `LIBFABRIC`（从 v0.16.0 起可用）
  - 如果指定了不支持的值，NIXL 会自动回退到 UCX

* `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP`：设为 `1` 时，generationExecutor 不会将 KV cache 传输与模型推理重叠。默认值为 `0`。

* `TRTLLM_ENABLE_KVCACHE_RECEIVE_PARALLEL`：当生成 rank 从单个上下文实例中的多个上下文 rank 接收 KV cache 时，它会依次从每个 rank 接收 KV cache。设为 `1` 时，生成 rank 并行接收一个上下文实例内各 rank 的 KV cache。默认值为 `0`。

* `TRTLLM_REQUEST_KV_CACHE_CONCURRENT`：设为 `1` 时，generationExecutor 为每个上下文 executor 准备独立资源来接收 KV cache，从不同上下文 executor 收到 KV cache 的请求会被并发处理。设为 `0` 时，生成 executor 复用同一资源串行处理每个请求的 KV cache 传输，减少 KV cache 传输使用的资源，从而降低内存耗尽的风险。默认值为 `0`。

* `TRTLLM_TRY_ZCOPY_FOR_KVCACHE_TRANSFER`：TRT-LLM 通常在发送 KV cache 前把非连续数据复制到临时缓冲区。设为 `1` 时，TRT-LLM 尝试直接传输每个 KV cache 块，消除额外复制。默认值为 `0`。

> 💡 **AI Infra 视角**：zcopy（零拷贝）是高性能传输的核心追求：**每次拷贝都花时间、占带宽**。KV cache 是分页的（块散落各处），直接传每块（zcopy）省掉"先拼成连续 buffer 再传"的拷贝，但要小心每块传输的启动开销。**"拷贝 vs 启动开销"的权衡**是 RDMA 编程的经典问题。

* `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE`：默认情况下，TRT-LLM 使用 `stream-ordered memory allocator` 分配临时缓冲区。如果此环境变量设为 #Size，TRT-LLM 将使用 `cudaMalloc` 分配大小为 #Size 的缓冲区用于 KV cache 传输。默认值为 `512MB`。用户可以设置 `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE=1GB` 用 `cudaMalloc` 分配 1 GB 缓冲区用于 KV cache 传输。

* `TRTLLM_KVCACHE_TRANSFER_USE_ASYNC_BUFFER`：设为 `1` 时，TRT-LLM 使用 `cudaMallocAsync` 分配 KV cache 传输缓冲区。默认值为 `0`。此环境变量仅在 `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE` 大于 0 时生效。

* `TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM`：最大并发 KV cache 发送数。默认值为 `1`。此环境变量仅在 `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE` 大于 0 时生效。

还有一些其他有用的环境变量，可能在遇到故障或性能问题时有所帮助。

* `NCCL_GRAPH_MIXING_SUPPORT`：TensorRT-LLM 现在默认关闭公共 NCCL communicator 的 graph mixing 支持，以减少 CUDA graph 捕获的 NCCL 操作的启动开销。这假设 communicator 不会被并行 graph 启动或未捕获的 NCCL 调用在 graph 启动未完成时使用。如果你的工作负载需要，设置 `NCCL_GRAPH_MIXING_SUPPORT=1` 恢复 NCCL 默认的 graph mixing 行为。更多细节见 [NCCL_GRAPH_MIXING_SUPPORT 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-mixing-support)。

> 💡 **AI Infra 视角**：这个环境变量展示了 CUDA Graph 与 NCCL 的**兼容性冲突**：CUDA Graph 要求"同样的 kernel 序列"可重复捕获执行，而 NCCL 的 graph mixing 允许在 graph 执行中混入新操作——两者理念冲突，TRT-LLM 默认关掉 mixing 以省启动开销。**"优化 A 与优化 B 互斥时的取舍"**——这种兼容性矩阵知识是排查疑难杂症时最值钱的。

* `UCX_MAX_RNDV_RAILS`：默认值 2 时，UCX 尝试为每次 Rendezvous（RNDV）传输使用每 GPU 两个 InfiniBand（IB）网卡设备。当上下文和生成实例都启用张量和专家并行（TEP）时，多个 TP rank 可能并发传输 KV cache。由于每个 TP rank 最多使用两个网卡设备，某些网卡设备可能被多个 GPU 共享，导致争用和吞吐下降。此时设置 `UCX_MAX_RNDV_RAILS=1` 可以减少争用。

> 💡 **AI Infra 视角**："rails"是 IB 网络的术语——并行网卡通道。默认用 2 条通道快，但多 rank 抢网卡时反而互相干扰（都想要同 2 条）——降成 1 条减少争抢。**"并行度越高越好"的例外：共享资源上的并行会互相拖累**。

## 故障排查与 FAQ

### 常见 FAQ

*问：TRT-LLM 中分离式服务的限制是什么？*

答：目前只支持 decoder-only 模型和 beam width 为 1。此外，模型每层的 KV cache 必须是同质的（homogeneous），具有相同的数据类型和相同的 attention 头数。

*问：使用 TRT 后端时，用于分离式服务的引擎与其他引擎有什么不同吗？*

答：没有。构建引擎的参数没有特殊要求。

*问：使用 TRT 后端时，上下文和生成实例使用的引擎需要相同吗？*

答：不需要。上下文和生成实例使用的引擎可以不同，它们的并行度可以是异构的，即 TP、PP 可以不同，TRT-LLM 会处理 KV cache 的异构性。

> 💡 **AI Infra 视角**：这条 FAQ 价值很高：**prefill 池和 decode 池可以独立选择并行配置**（prefill 用大 TP 求快、decode 用不同配置），引擎不必相同——这正是分离式服务"独立优化"的体现。KV cache 布局转换（上文讲过）就是为异构并行准备的。

*问：TRT-LLM 服务器实例能同时处理 context-only 和 generation-only 请求吗？*

答：可以，但不推荐。TRT-LLM 没有为实例同时处理混合的 context-only 和 generation-only 请求实现最优调度。最好在独立的服务器组上运行 context-only 和 generation-only 请求。

*问：TRT-LLM 的分离式服务支持多 GPU 和多节点吗？*

答：支持，建议不同的服务器实例使用不同的 GPU。我们支持在同一节点或不同节点上运行上下文和生成服务器。`CUDA_VISIBLE_DEVICES` 环境变量可用于控制每个实例使用哪些 GPU。

### 调试 FAQ

*问：即使设置了 `TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC`，NIXL 为什么还是无法使用 LIBFABRIC 后端？*

答：TensorRT-LLM 容器默认不包含 NIXL LIBFABRIC 插件。你需要：

1. **重新构建 NIXL**：先安装 libfabric 和 hwloc，然后按上面的安装说明重新构建 NIXL
2. **使用预编译插件**：如果你有兼容的 `libplugin_LIBFABRIC.so`，设置 `NIXL_PLUGINS_DIR` 指向其目录

详细的安装和配置说明请参阅[分离式服务示例文档](../../../examples/disaggregated/README.md)。

*问：如何处理错误 `Disaggregated serving is not enabled, please check the configuration?`*

答：请设置 `CacheTransceiverConfig` 的 `backendType`。
```cpp
ExecutorConfig executorConfig{...};

executorConfig.setCacheTransceiverConfig(texec::CacheTransceiverConfig(BackendType::DEFAULT));
```
*问：TRT-LLM 支持节点间 KV Cache 传输使用 GPU direct RDMA 吗？*

答：支持，TRT-LLM 支持节点间 KV cache 传输使用 GPU direct RDMA。

*问：如何调试疑似与重叠 NCCL graph 操作相关的挂起？*

答：TensorRT-LLM 默认关闭公共 NCCL communicator 的 graph mixing 支持。要检查挂起是否与 NCCL graph mixing 支持有关，设置 `NCCL_GRAPH_MIXING_SUPPORT=1` 恢复 NCCL 默认的 graph mixing 行为。

*问：什么导致 kvCache 传输的带宽大幅波动，尤其是服务初始化后的前几个请求？*

答：executor 之间用于 kvCache 传输的通信是动态建立的。连接建立过程有显著开销，这解释了服务启动后最初几个请求观察到的 kvCache 传输带宽明显较低。这个较低带宽反映了连接建立开销的计入。进行基准测试时，建议执行预热阶段（warm-up）以确保性能测量准确。

> 💡 **AI Infra 视角**：**"跑基准必须先预热"**——连接建立、CUDA 图捕获、内存池分配都发生在启动初期，前几个请求的耗时不能代表稳态性能。所有基准测试的通用准则：先跑几轮热身，再取稳定段的数据。

*问：当我的服务器运行在不同的 NVLink 域时，一些服务器挂起或性能较低。如何修复？*

答：NVLink 域可以用 `nvidia-smi -q` 在 `Fabric.ClusterUUID` 字段中找到。当你的服务器位于不同的 NVLink 域时，可以调整几个 UCX 环境变量：

* `UCX_CUDA_IPC_ENABLE_MNNVL`：设为 `n`。这也可以减少类似 `UCX  ERROR   cuMemImportFromShareableHandle failed: invalid resource handle` 的 UCX 超时错误消息，虽然这些错误不一定导致你的 trtllm-serve 失败。

* `UCX_NET_DEVICES`：检查此变量是否设置正确，或取消设置让 UCX 使用所有可用设备。

* `UCX_RNDV_SCHEME`：在 GB200 上设置为 `get_zcopy` 或 `put_zcopy` 以获得更好性能。默认值为 `auto`。
