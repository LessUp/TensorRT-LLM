# TensorRT-LLM 中文学习路线（AI Infra 转行用）

> 本仓库的 25 篇核心文档已翻译为中文，并在恰当位置插入了 **「💡 AI Infra 视角」** 讲解块（全库共 220+ 处），用于解释设计动机、业界对比、面试考点和实战注意点。
> 翻译日期：2026-08-07。英文原文随时可用 `git checkout HEAD -- <文件路径>` 恢复。

## 学习顺序（5 个阶段，约 25 篇）

### 阶段 0 · 项目整体认识（先建立全局地图）
| 文档 | 核心收获 |
|------|---------|
| `README.md` | 项目全貌、生态位置（TRT-LLM / Dynamo / Triton 的关系） |
| `docs/source/overview.md` | 核心能力清单 = AI Infra 面试考点清单 |
| `docs/source/quick-start-guide.md` | 在线服务 vs 离线推理两种形态 |
| `docs/source/developer-guide/overview.md` | **PyExecutor 架构**：Scheduler / KVCacheManager / ModelEngine / Sampler + CUDA Graph、Overlap Scheduler |

### 阶段 1 · LLM 推理核心机制（面试必考）
| 文档 | 核心收获 |
|------|---------|
| `docs/source/features/kvcache.md` | KV Cache 原理、块/池、前缀复用、驱逐策略 |
| `docs/source/features/paged-attention-ifb-scheduler.md` | **核心中的核心**：In-Flight Batching、三个容量参数、调度可视化 |
| `docs/source/features/overlap-scheduler.md` | CPU/GPU 重叠执行 |
| `docs/source/features/sampling.md` | 采样策略全解（temperature/top-k/top-p/penalties/beam） |
| `docs/source/features/attention.md` | MHA/MQA/GQA、FlashAttention、三个 attention 后端 |
| `docs/source/features/long-sequence.md` | 长序列三招：分块 prefill / 分块注意力 / 滑动窗口 |

### 阶段 2 · 性能与扩展（AI Infra 吃饭的本事）
| 文档 | 核心收获 |
|------|---------|
| `docs/source/features/parallel-strategy.md` | TP/PP/DP/EP/CP/Wide-EP 六种并行 |
| `docs/source/features/quantization.md` | FP8/FP4、GPTQ/AWQ、支持矩阵 |
| `docs/source/features/speculative-decoding.md` | 投机解码原理 + 六种算法 |
| `docs/source/developer-guide/perf-overview.md` | 性能指标（TTFT/TPOT/tps）、读性能表的方法 |
| `docs/source/developer-guide/perf-benchmarking.md` | **trtllm-bench 完整用法**、基准测试方法论 |
| `docs/source/developer-guide/perf-analysis.md` | nsys/ncu 性能分析、Perfect Router 归因法 |

### 阶段 3 · 高级服务架构（生产级）
| 文档 | 核心收获 |
|------|---------|
| `docs/source/features/disagg-serving.md` | prefill/decode 分离式服务、KV 交换、snowflake ID |
| `docs/source/features/checkpoint-loading.md` | 四组件插件架构（Loader/Config/Weight/Mapper） |
| `docs/source/features/guided-decoding.md` | 结构化输出（JSON/正则/EBNF/函数调用） |
| `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md` | **实战部署模板**：容器→配置→serve→压测全流程 |

### 阶段 4 · 深入源码（PyTorch 后端）
| 文档 | 核心收获 |
|------|---------|
| `docs/source/torch/arch_overview.md` | PyExecutor 组件与代码文件对应关系 |
| `docs/source/torch/scheduler.md` | 两步调度（容量→微批）+ 自定义调度器 |
| `docs/source/torch/kv_cache_manager.md` | KVCacheManager 接口层（与调度器/引擎的契约） |
| `docs/source/torch/attention.md` | 注意力后端实现指南（与 features/attention.md 配套） |
| `docs/source/torch/adding_new_model.md` | **给引擎加模型的标准流程**（配置→定义→加载→注册） |

### 附：C++ 核心代码中文讲解
以下文件头部（及关键方法处）已插入中文讲解注释（英文原注释保留）：
- `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp` — 四种调度策略
- `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` — V1 KV cache 管理
- `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.cpp`、`kvCacheManager.cpp` — V2 管理器
- `cpp/tensorrt_llm/batch_manager/kvCacheTransferManager.cpp` — 分离式 KV 传输
- `cpp/tensorrt_llm/batch_manager/createNewDecoderRequests.cpp` — 解码收尾流程
- `cpp/tensorrt_llm/batch_manager/decoderBuffers.cpp` — 解码缓冲区

## 阅读建议

1. **按阶段顺序读**，每阶段之间可以停下来动手验证（跑 quick-start、跑一遍 trtllm-bench）
2. **讲解块是重点**：`💡 AI Infra 视角` 块总结了文档背后的"为什么"和面试考点，比正文更值得反复看
3. **术语保留英文**：KV Cache、PagedAttention、TP/PP/EP 等按行业惯例不翻译，首次出现有「中文（英文）」标注
4. **对照源码**：阶段 4 的文档都标注了源码文件路径，读文档时打开对应代码文件对照
5. **英文原文恢复**：`git checkout HEAD -- <文件>` 可随时恢复单个文件

## 配套工具

- `scripts/check_translation_links.py` — 校验翻译后文档的链接是否与原文一致（翻译质量保障工具）
