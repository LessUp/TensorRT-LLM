<!--
  本文档为 TensorRT-LLM 官方 Parallelism 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/features/parallel-strategy.md
-->

# TensorRT LLM 中的并行

在以下情况出现时，跨多个 GPU 的并行就变得必要：
* 模型无法装进单张 GPU 的显存，或
* 单张 GPU 无法达到期望的性能。

TensorRT LLM 支持多种并行策略，可在单节点和多节点上部署：
* **张量并行（Tensor Parallel，TP）** - 将模型权重切分到多个 GPU
* **流水线并行（Pipeline Parallel，PP）** - 将模型层分布到多个 GPU
* **数据并行（Data Parallel，DP）** - 为不同请求在多个 GPU 上复制模型
* **专家并行（Expert Parallel，EP）** - 为 MoE 模型将专家分布到多个 GPU
* **上下文并行（Context Parallel，CP）** - 将上下文处理分布到多个 GPU
* **广域专家并行（Wide Expert Parallel，Wide-EP）** - 针对大规模 MoE 模型的带负载均衡的高级 EP

> 💡 **AI Infra 视角**：并行策略是 AI Infra 最核心的知识域之一（面试必考，实际部署天天用）。先用一张"记忆图"理解六种并行：
> - **TP（张量并行）**：**把一层切开**——一个矩阵乘法拆成多卡各算一部分，卡之间频繁通信（每层都要 all-reduce）。适合小 batch、单卡放不下；
> - **PP（流水线并行）**：**把层切开**——GPU0 算第 1~10 层，GPU1 算第 11~20 层……数据像流水线一样流过各卡。通信少（只传激活值），但有流水线气泡（bubble，卡空闲等待）；
> - **DP（数据并行）**：**把请求切开**——每卡一份完整模型，各处理不同的请求，互不通信（推理时）。最"省心"，但显存翻倍（每卡都要整份权重）；
> - **EP（专家并行）**：**把 MoE 的专家切开**——每个专家完整放在一张卡上，token 按路由结果发给对应卡。通信是 all-to-all；
> - **CP（上下文并行）**：**把长序列切开**——一个超长请求的 token 分到多卡一起算（attention 的 ring 结构）；
> - **Wide-EP**：EP 的进阶版，解决"热门专家"负载不均问题。
> 现实部署几乎总是组合使用（如 TP×EP、TP×PP），组合时要注意各维度乘积 = GPU 总数。

## 并行策略概览

### 张量并行（TP）
张量并行将模型权重切分到多个 GPU。每张 GPU 持有部分权重，处理相同的输入 token，结果通过通信合并。

**最适合：** 小 batch 大小、显存受限场景

> 💡 **AI Infra 视角**：TP 的通信成本是它的命门：每层 attention/FFN 之后都要一次 all-reduce（把各卡的部分结果加起来）。batch 越小，通信占比越大——所以"TP 适合小 batch"。另一个重要事实：TP 规模通常不超过单节点内的卡数（8 卡），因为跨节点通信走网卡，太慢。

### 流水线并行（PP）
流水线并行将模型的不同层分布到多个 GPU。每张 GPU 处理部分层，激活值（activations）在 GPU 之间传递。

**最适合：** 单张 GPU 显存放不下的大模型

> 💡 **AI Infra 视角**：PP 的经典问题叫**气泡（bubble）**：GPU0 算第 1 步时 GPU1 在空等（它还没收到数据）。层数越多、micro-batch 切得越细，气泡越小。因为通信只在层边界发生一次（比 TP 每层都通信省得多），PP 可以跨节点。**TP 管"层内"，PP 管"层间"**——两者组合是百亿参数模型的标准方案。

### 数据并行（DP）
数据并行在多个 GPU 上复制整个模型。每张 GPU 独立处理不同的请求。

**最适合：** 大 batch 大小、高吞吐场景

> 💡 **AI Infra 视角**：推理的 DP 与训练的 DP 不同：训练要梯度同步，推理**完全不需要通信**（每卡各算各的请求）——所以"推理的 DP"本质就是"同一模型跑多份实例，请求负载均衡地分发"（vLLM 管这叫多实例/复制）。注意本页后面的"attention DP"是另一种东西（更细粒度，见下）。

### 专家并行（EP）
专家并行专为混合专家（MoE）模型设计，不同的专家分布到不同的 GPU。

**最适合：** 专家数量多的 MoE 模型

### 上下文并行（CP）
上下文并行将长序列的处理分布到多个 GPU。

**最适合：** 长上下文场景

### 广域专家并行（Wide-EP）
Wide-EP 是专家并行的进阶形式，通过智能负载均衡和专家复制解决大规模 MoE 模型中固有的工作负载不平衡问题。

**最适合：** DeepSeek-V3/R1、LLaMA4、Qwen3 等大规模 MoE 模型

## 模块级并行指南

### Attention 模块

TensorRT LLM 支持两种 attention 模块策略：

- **张量并行（TP）** — 最适合小 batch 大小
- **数据并行（DP）** — 最适合大 batch 大小

> 💡 **AI Infra 视角**：为什么 attention 的 TP/DP 选择取决于 batch 大小？attention 的计算 = 每个 query 和所有 KV 做点积——**batch 越大，attention 的计算密度越高**（KV 被更多 query 复用）。所以：
> - 小 batch 时 attention 计算量小、通信开销占比高 → TP 更亏 → 用 DP（复制权重，零通信）；
> - 大 batch 时 attention 算得多、TP 的通信被摊薄 → TP 合算。
> 这就是 `enable_attention_dp` 的由来：**attention 层用 DP、其他层用 TP 的混合方案**。

#### 张量并行（TP）

* attention kernel 前后的 GEMM 权重在 GPU 间均匀切分，attention 的 `num_heads` 也同样切分。
* 例外：
  1. **DeepSeek-R1**：`fused_A` GEMM *不*切分。
  2. **GQA / MQA / MLA**：如果 `num_heads < tensor_parallel_size`，KV cache 会在每张 GPU 上**复制**。

> 💡 **AI Infra 视角**：最后一个例外的逻辑：KV 头数（如 8）少于 TP 卡数（如 16）时，没法把 8 个 KV 头切到 16 张卡上——所以 KV 在每张卡上复制一份（显存多点，但能算）。**切分维度必须能被 TP 大小整除**，这是并行切分的铁律。GQA 模型的 KV 头数往往较小（Llama 3 只有 8 个），TP 大时这个例外就触发。

#### 数据并行（DP）

* 所有 GEMM 权重在每张 GPU 上**复制**。
* KV cache 被**分区（partitioned）**，因为不同的用户请求被路由到不同的 DP rank。

#### 如何启用 Attention 并行

要使用 `trtllm-serve` 部署上述并行策略的模型，或用 `trtllm-bench` 跑基准测试，请创建名为 `parallel_config.yaml` 的 YAML 配置文件：

```bash
cat <<EOF > parallel_config.yaml
# TP-8
tensor_parallel_size: 8
enable_attention_dp: false    # 默认
# DP-8
tensor_parallel_size: 8
enable_attention_dp: true
EOF
```

然后为 `trtllm-serve` 或 `trtllm-bench` 设置 `--config parallel_config.yaml`。

> 💡 **AI Infra 视角**：注意 `enable_attention_dp: true` 时 `tensor_parallel_size` 仍是 8——它的意思是"8 卡中，attention 层按 DP 方式（复制权重）跑，非 attention 层按 TP 方式（切分权重）跑"。**一个并行配置是逐层生效的**，这是并行系统设计的关键抽象：不同算子选最合适的并行方式。

### FFN 模块

#### 稠密模型（Dense Models）

稠密模型的 FFN 层支持张量并行。

#### 混合专家（Mixture of Experts，MoE）

MoE 用多个专家替换单个 FFN。路由器（router）为每个 token 选择 top-k 专家，并将对应的隐藏状态分发过去。

TensorRT LLM 支持三种 MoE 执行模式：

* **TP** - 每个专家的权重矩阵被切分到所有 GPU。每张 GPU 都能看到所有 token。
* **EP** - 每个专家的完整权重放在一张 GPU 上。每张 GPU 只看到路由到其本地专家的 token。
* **混合 ETP** - 每张 GPU 存储部分专家（EP），并对这些权重进一步切分（TP），在负载均衡和 kernel 效率之间取得平衡。

> 💡 **AI Infra 视角**：MoE 的 TP vs EP 是必考点，理解核心差异：
> - **TP 的代价**：每个 token 都要经过**所有专家**（专家权重被切到所有卡上），MoE 的稀疏性优势（只算少数专家）在 TP 下被浪费——所有卡都在忙，但每个 token 只用到部分专家权重；
> - **EP 的代价**：token 被路由到专家所在的卡——**all-to-all 通信**（每张卡给其他每张卡都发数据），通信量大但计算精准。
> 业界趋势：MoE 模型（DeepSeek-V3 有 256 个专家）用 EP，且 EP 现在主要靠 NVLink 的高带宽 all-to-all 撑起来。**"MoE 为什么用 EP 不用 TP"是面试高频题**。

#### 如何启用 MoE 并行

要使用 `trtllm-serve` 部署上述并行策略的模型，或用 `trtllm-bench` 跑基准测试，请创建名为 `parallel_config.yaml` 的 YAML 配置文件：

```bash
cat <<EOF > parallel_config.yaml
# 仅 TP
tensor_parallel_size: 8
moe_tensor_parallel_size: 8

# 仅 EP
tensor_parallel_size: 8
moe_expert_parallel_size: 8

# 混合（TP-4 × EP-2）
tensor_parallel_size: 8      # 4 × 2
moe_tensor_parallel_size: 4
moe_expert_parallel_size: 2
EOF
```
```{note}
`moe_tensor_parallel_size` 和 `moe_expert_parallel_size` 的乘积必须等于 `tensor_parallel_size`。
```

## 广域专家并行（Wide-EP）

广域专家并行（Wide-EP）是 TensorRT LLM 针对大规模 MoE 模型推理的进阶方案。它通过智能负载均衡和专家复制策略解决传统专家并行的挑战。

### Wide-EP 的动机

DeepSeek-V3/R1、LLaMA4 和 Qwen3 等大规模 MoE 模型采用细粒度专家设计，引入了新的挑战：

- **专家权重的显存需求高**
- **稀疏执行模式带来的固有专家级负载不均衡**
- **分布式专家并行中的通信开销**
- **热门专家问题**：某些专家收到的 token 明显多于其他专家

> 💡 **AI Infra 视角**："热门专家（hot expert）"问题是 MoE 推理的核心难点：路由不是均匀的——某些专家（如"数学推理"专家）收到的 token 特别多。如果每个专家只放一张卡，热门专家所在的卡会忙死、其他卡闲着。这就像数据库的**热点分片问题**，解法思路也类似：**复制热门分片（专家复制）+ 动态重新分片（在线负载均衡）**。

### Wide-EP 的关键特性

#### 1. 专家复制与负载均衡
Wide-EP 引入了**专家槽位（expert slots）**的概念，它与具体专家解耦。这允许：
- 热门专家在多个 GPU 上拥有多个副本
- 基于工作负载模式动态放置专家
- 离线（offline）和在线（online）两种负载均衡策略

#### 2. 自定义 EP 通信 kernel
- 针对 NVIDIA GB200 多节点 NVLink（MNNVL）优化
- 高效的 all-to-all 通信，用于专家分发（dispatch）和合并（combine）
- 相比传统 EP 降低通信开销

#### 3. 专家并行负载均衡器（EPLB）
- **离线 EPLB**：基于历史工作负载统计预先计算专家放置
- **在线 EPLB**：动态专家放置，适应实时流量模式
- 逐层（layer-wise）权重再分配，最小化推理中断

### 架构总览

Wide-EP 将**专家**和**槽位**的概念分离：
- **专家（Expert）**：从模型视角看的概念（例如 Expert 0、Expert 1 等）
- **槽位（Slot）**：从模型引擎视角看的概念（例如 Slot 0、Slot 1 等）

系统维护一张路由表，将专家 ID 映射到槽位 ID，该表可以由负载均衡策略更新。

> 💡 **AI Infra 视角**：专家/槽位分离是"间接层（indirection）"思想：模型说"我要 Expert 7"，引擎查表发现 Expert 7 在 Slot 3 和 Slot 9 有两个副本。这样负载均衡器可以自由移动/复制专家而无需修改模型。**"加一层间接层解决动态重配置问题"是系统设计的通用手法**——和 KV cache 里"块→物理位置"的映射是同一思路。

### 最佳实践

1. **生产部署先上离线 EPLB**（工作负载模式已知时）
2. **动态工作负载或流量模式频繁变化时用在线 EPLB**
3. **监控专家统计信息**，了解工作负载分布
4. **根据显存约束和 EP 大小调整 max_num_tokens**
5. **用代表性数据集测试**，验证负载均衡的有效性

### 参考资料

- [技术博客：Scaling Expert Parallelism in TensorRT LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.md)
- [DeepSeek-V3 论文](https://arxiv.org/abs/2412.19437)
- [EPLB 实现](https://github.com/deepseek-ai/EPLB)

详细的实现示例和高级用法，请参见：
- [`examples/wide_ep/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/)：完整的 Wide-EP 示例
- [`examples/wide_ep/ep_load_balancer/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/ep_load_balancer/)：负载均衡工具
- [`examples/wide_ep/slurm_scripts/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/slurm_scripts/)：集群部署脚本
