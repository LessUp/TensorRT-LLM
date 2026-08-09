<!--
  本文档为 TensorRT-LLM 官方 README 的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  原始英文内容可从 git 历史恢复：git checkout HEAD -- README.md
  术语约定：LLM API、KV Cache、TP/PP/EP/CP、prefill/decode 等行业术语保留英文。
-->

<div align="center">

TensorRT LLM
===========================
<h4>TensorRT LLM 通过面向常见算子（attention、GEMM、MoE 等）的专用 kernel、高效的运行时（runtime）以及可定制的 Python 框架，为 LLM 和 Visual Gen 模型的推理提供深度优化。</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/TensorRT-LLM/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/TensorRT-LLM)
[![python](https://img.shields.io/badge/python-3.12-green)](https://www.python.org/downloads/release/python-3123/)
[![python](https://img.shields.io/badge/python-3.10-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-13.2.1-green)](https://developer.nvidia.com/cuda-downloads)
[![torch](https://img.shields.io/badge/torch-2.11.0-green)](https://pytorch.org)
[![version](https://img.shields.io/badge/release-1.3.0rc24-green)](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/version.py)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/NVIDIA/TensorRT-LLM/blob/main/LICENSE)

[架构](https://nvidia.github.io/TensorRT-LLM/developer-guide/overview.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[性能](https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-overview.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[示例](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[文档](https://nvidia.github.io/TensorRT-LLM/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Roadmap](https://github.com/NVIDIA/TensorRT-LLM/issues?q=is%3Aissue%20state%3Aopen%20label%3Aroadmap)

---
<div align="left">

## 技术博客

<!-- 使用 github markdown 链接指向最新博客（文档站点尚未构建）。文档构建更新后，应更新为网页链接。 -->

* [07/17] DeepSeek-V4 on NVIDIA Blackwell: Model-Specific and Agentic-Workload Optimizations in TensorRT LLM
✨ [➡️ link](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md)

* [07/01] Scaling Video Generation Across NVL72 Rack with TensorRT-LLM
✨ [➡️ link](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog25_Scaling_Video_Generation_Across_NVL72_Rack_with_TensorRT-LLM.md)

* [05/15] Joint Optimization of Agent Applications and TensorRT-LLM
✨ [➡️ link](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog23_Joint_Optimization_of_Agent_Applications_and_TensorRT-LLM.md)

* [04/03] Tuning CUDA Graph Batch Sizes for Higher Output Throughput
✨ [➡️ link](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.md)

* [04/03] DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog19_DWDP_Distributed_Weight_Data_Parallelism_for_High_Performance_LLM_Inference_on_NVL72.html)


* [03/16] Optimizing MoE Communication with One-Sided AlltoAll Over NVLink
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.html)

* [03/04] Sparse Attention in TensorRT LLM
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog17_Sparse_Attention_in_TensorRT-LLM.html)

* [02/06] Accelerating Long-Context Inference with Skip Softmax Attention
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.html)

* [01/09] Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs)

<details close>
<summary>往期博客</summary>
* [10/13] Scaling Expert Parallelism in TensorRT LLM (Part 3: Pushing the Performance Boundary)
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.html)

* [09/26] Inference Time Compute Implementation in TensorRT LLM
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog13_Inference_Time_Compute_Implementation_in_TensorRT-LLM.html)

* [09/19] Combining Guided Decoding and Speculative Decoding: Making CPU and GPU Cooperate Seamlessly
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog12_Combining_Guided_Decoding_and_Speculative_Decoding.html)

* [08/29] ADP Balance Strategy
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog10_ADP_Balance_Strategy.html)

* [08/05] Running a High-Performance GPT-OSS-120B Inference Server with TensorRT LLM
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog9_Deploying_GPT_OSS_on_TRTLLM.html)

* [08/01] Scaling Expert Parallelism in TensorRT LLM (Part 2: Performance Status and Optimization)
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog8_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.html)

* [07/26] N-Gram Speculative Decoding in TensorRT LLM
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog7_NGram_performance_Analysis_And_Auto_Enablement.html)

* [06/19] Disaggregated Serving in TensorRT LLM
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)

* [06/05] Scaling Expert Parallelism in TensorRT LLM (Part 1: Design and Implementation of Large-scale EP)
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.html)

* [05/30] Optimizing DeepSeek R1 Throughput on NVIDIA Blackwell GPUs: A Deep Dive for Developers
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.html)

* [05/23] DeepSeek R1 MTP Implementation and Optimization
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog2_DeepSeek_R1_MTP_Implementation_and_Optimization.html)

* [05/16] Pushing Latency Boundaries: Optimizing DeepSeek-R1 Performance on NVIDIA B200 GPUs
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html)
</details>

## 最新动态
* [04/03] 🎨 TensorRT LLM 现已支持扩散模型（diffusion models）用于视觉生成 [➡️ link](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/visual-generation.md)

<details close>
<summary>往期动态</summary>

* [08/05] 🌟 TensorRT LLM 对 OpenAI 最新开源权重模型提供 Day-0 支持：GPT-OSS-120B [➡️ link](https://huggingface.co/openai/gpt-oss-120b) 和 GPT-OSS-20B [➡️ link](https://huggingface.co/openai/gpt-oss-20b)

* [07/15] 🌟 TensorRT LLM 对 LG AI Research 最新模型 EXAONE 4.0 提供 Day-0 支持 [➡️ link](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B)

* [05/22] Blackwell 凭借 Meta 的 Llama 4 Maverick 打破 1000 TPS/用户 的壁垒
✨ [➡️ link](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/)

* [04/10] TensorRT LLM DeepSeek R1 性能基准测试最佳实践已发布。
✨ [➡️ link](https://nvidia.github.io/TensorRT-LLM/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.html)

* [04/05] TensorRT LLM 在 B200 GPU 上运行 Llama 4 可达每秒 40,000+ tokens！

![L4_perf](https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/media/l4_launch_perf.png)


* [03/22] TensorRT LLM 现已完全开源，开发工作已迁移到 GitHub！
* [03/18]  🚀🚀 NVIDIA Blackwell 凭借 TensorRT LLM 实现创纪录的 DeepSeek-R1 推理性能 [➡️ Link](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/)
* [02/28] 🌟 NAVER Place 使用 TensorRT LLM 优化基于 SLM 的垂直服务 [➡️ Link](https://developer.nvidia.com/blog/spotlight-naver-place-optimizes-slm-based-vertical-services-with-nvidia-tensorrt-llm/)

* [02/25] 🌟 DeepSeek-R1 现已针对 Blackwell 优化 [➡️ Link](https://huggingface.co/nvidia/DeepSeek-R1-FP4)

* [02/20] 查看 [完整指南](https://www.nvidia.com/en-us/solutions/ai/inference/balancing-cost-latency-and-performance-ebook/?ncid=so-twit-348956&linkId=100000341423615)，了解如何以最低成本获得优秀的准确率、高吞吐和低延迟。

* [02/18] 在 @AWS EKS 上解锁带自动扩缩容的 #LLM 推理 ✨ [➡️ link](https://aws.amazon.com/blogs/hpc/scaling-your-llm-inference-workloads-multi-node-deployment-with-tensorrt-llm-and-triton-on-amazon-eks/)

* [02/12] 🦸⚡ 使用 DeepSeek-R1 和 Inference Time Scaling 自动化 GPU kernel 生成
[➡️ link](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/?ncid=so-twit-997075&linkId=100000338909937)

* [02/12] 🌟 Scaling Laws 如何驱动更强大、更智能的 AI
[➡️ link](https://blogs.nvidia.com/blog/ai-scaling-laws/?ncid=so-link-889273&linkId=100000338837832)

* [2025/01/25] Nvidia 将 AI 重心转向推理成本与效率 [➡️ link](https://www.fierceelectronics.com/ai/nvidia-moves-ai-focus-inference-cost-efficiency?linkId=100000332985606)

* [2025/01/24] 🏎️ 使用 NVIDIA 全栈解决方案优化 AI 推理性能 [➡️ link](https://developer.nvidia.com/blog/optimize-ai-inference-performance-with-nvidia-full-stack-solutions/?ncid=so-twit-400810&linkId=100000332621049)

* [2025/01/23] 🚀 快速、低成本的推理是 AI 盈利的关键 [➡️ link](https://blogs.nvidia.com/blog/ai-inference-platform/?ncid=so-twit-693236-vt04&linkId=100000332307804)

* [2025/01/16] 介绍 TensorRT LLM 中的新 KV Cache 复用优化 [➡️ link](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/?ncid=so-twit-363876&linkId=100000330323229)

* [2025/01/14] 📣 Bing 向 LLM/SLM 模型过渡：用 TensorRT LLM 优化搜索 [➡️ link](https://blogs.bing.com/search-quality-insights/December-2024/Bing-s-Transition-to-LLM-SLM-Models-Optimizing-Search-with-TensorRT-LLM)

* [2025/01/04] ⚡借助 TensorRT LLM 投机解码（Speculative Decoding）将 Llama 3.3 70B 推理吞吐提升 3 倍
[➡️ link](https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/)

* [2024/12/10] ⚡ AI at Meta 的 Llama 3.3 70B 由 TensorRT-LLM 加速。🌟 在推理、数学、指令跟随和工具调用方面与 Llama 3.1 405B 相当的最先进模型。探索预览版
[➡️ link](https://build.nvidia.com/meta/llama-3_3-70b-instruct)

* [2024/12/03] 🌟 将 AI 推理吞吐最高提升 3.6 倍。我们现已支持投机解码，可将 token 吞吐提升 3 倍，由 NVIDIA TensorRT-LLM 提供。⚡在技术深潜文章中了解详情
[➡️ link](https://nvda.ws/3ZCZTzD)

* [2024/12/02] 正在为性能关键型应用部署 ONNX 模型？试试 NVIDIA Nsight Deep Learning Designer ⚡ 直观易用的 GUI，与 NVIDIA TensorRT 紧密集成，提供：
✅ ONNX 模型图的可视化
✅ 快速调整模型架构与参数
✅ 使用 ORT 或 TensorRT 进行详细的性能分析
✅ 轻松构建 TensorRT engines
[➡️ link](https://developer.nvidia.com/nsight-dl-designer?ncid=so-link-485689&linkId=100000315016072)

* [2024/11/26] 📣 为 Jetson AGX Orin 推出 TensorRT LLM，通过 TensorRT LLM 仓库的 v0.12.0-jetson 分支在 JetPack 6.1 中提供初步支持，让 Jetson AGX Orin 上的部署更加容易。✅ 预编译的 TensorRT LLM wheels 和容器，便于集成 ✅ 全面的指南和文档助你上手
[➡️ link](https://forums.developer.nvidia.com/t/tensorrt-llm-for-jetson/313227?linkId=100000312718869)

* [2024/11/21] NVIDIA TensorRT LLM 多块注意力（Multiblock Attention）在 NVIDIA HGX H200 上为长序列将吞吐提升 3 倍以上
[➡️ link](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-multiblock-attention-boosts-throughput-by-more-than-3x-for-long-sequence-lengths-on-nvidia-hgx-h200/)

* [2024/11/19] Llama 3.2 全栈优化在 NVIDIA GPU 上释放高性能
[➡️ link](https://developer.nvidia.com/blog/llama-3-2-full-stack-optimizations-unlock-high-performance-on-nvidia-gpus/?ncid=so-link-721194)

* [2024/11/09] 🚀🚀🚀 借助 NVSwitch 和 TensorRT LLM MultiShot 将 AllReduce 提速 3 倍
[➡️ link](https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/)

* [2024/11/09] ✨ NVIDIA 推动 LG AI Research 的 AI 模型发展 🙌
[➡️ link](https://blogs.nvidia.co.kr/blog/nvidia-lg-ai-research/)

* [2024/11/02] 🌟🌟🌟 NVIDIA 与 LlamaIndex 开发者大赛
🙌 参加即有机会赢取包括 NVIDIA® GeForce RTX™ 4080 SUPER GPU、DLI 积分等大奖🙌
[➡️ link](https://developer.nvidia.com/llamaindex-developer-contest)

* [2024/10/28] 🏎️🏎️🏎️ NVIDIA GH200 Superchip 在多轮交互中与 Llama 模型配合，将推理速度提升 2 倍
[➡️ link](https://developer.nvidia.com/blog/nvidia-gh200-superchip-accelerates-inference-by-2x-in-multiturn-interactions-with-llama-models/)

* [2024/10/22] 新的 📝 分步指南，教你如何
✅ 使用 NVIDIA TensorRT-LLM 优化 LLM，
✅ 使用 Triton Inference Server 部署优化后的模型，
✅ 在 Kubernetes 环境中对 LLM 部署进行自动扩缩容。
🙌 技术深潜：
[➡️ link](https://nvda.ws/3YgI8UT)

* [2024/10/07] 🚀🚀🚀使用 NVIDIA 加速库优化 Microsoft Bing 视觉搜索
[➡️ link](https://developer.nvidia.com/blog/optimizing-microsoft-bing-visual-search-with-nvidia-accelerated-libraries/)

* [2024/09/29] 🌟 AI at Meta PyTorch + TensorRT v2.4 🌟 ⚡TensorRT 10.1 ⚡PyTorch 2.4 ⚡CUDA 12.4 ⚡Python 3.12
[➡️ link](https://github.com/pytorch/TensorRT/releases/tag/v2.4.0)

* [2024/09/17] ✨ NVIDIA TensorRT LLM Meetup
[➡️ link](https://drive.google.com/file/d/1RR8GqC-QbuaKuHj82rZcXb3MS20SWo6F/view?usp=share_link)

* [2024/09/17] ✨ Accelerating LLM Inference at Databricks with TensorRT-LLM
[➡️ link](https://drive.google.com/file/d/1NeSmrLaWRJAY1rxD9lJmzpB9rzr38j8j/view?usp=sharing)

* [2024/09/17] ✨ TensorRT LLM @ Baseten
[➡️ link](https://drive.google.com/file/d/1Y7L2jqW-aRmt31mCdqhwvGMmCSOzBUjG/view?usp=share_link)

* [2024/09/04] 🏎️🏎️🏎️ 使用 BentoML 调优 TensorRT LLM 以获得最佳服务性能的最佳实践
[➡️ link](https://www.bentoml.com/blog/tuning-tensor-rt-llm-for-optimal-serving-with-bentoml)


* [2024/08/20] 🏎️ 使用 #Model Optimizer 优化 SDXL ⏱️⚡ 🏁 缓存扩散 🏁 量化感知训练 🏁 QLoRA 🏁 #Python 3.12
[➡️ link](https://developer.nvidia.com/blog/nvidia-tensorrt-model-optimizer-v0-15-boosts-inference-performance-and-expands-model-support/)

* [2024/08/13] 🐍 使用 #Mamba ⚡ #TensorRT #LLM 实现 DIY 代码补全，速度惊人 🤖 NIM 简化部署 ☁️ 随处部署
[➡️ link](https://developer.nvidia.com/blog/revolutionizing-code-completion-with-codestral-mamba-the-next-gen-coding-llm/)

* [2024/08/06] 🗫 多语言挑战接受 🗫
🤖 #TensorRT #LLM 提升希伯来语、印尼语和越南语等低资源语言性能 ⚡[➡️ link](https://developer.nvidia.com/blog/accelerating-hebrew-llm-performance-with-nvidia-tensorrt-llm/?linkId=100000278659647)

* [2024/07/30] 介绍 🍊 @SliceXAI ELM Turbo 🤖 训练一次 ELM ⚡ #TensorRT #LLM 优化 ☁️ 随处部署
[➡️ link](https://developer.nvidia.com/blog/supercharging-llama-3-1-across-nvidia-platforms)

* [2024/07/23] 👀 @AIatMeta Llama 3.1 405B 在 16K 个 NVIDIA H100 上训练 - 推理由 #TensorRT #LLM 优化 ⚡
🦙 400 tok/s - 每节点
🦙 37 tok/s - 每用户
🦙 1 节点推理
[➡️ link](https://developer.nvidia.com/blog/supercharging-llama-3-1-across-nvidia-platforms)

* [2024/07/09] 使用 #TensorRT #LLM 推理最大化 @meta #Llama3 多语言性能的清单：
✅ 多语言
✅ NIM
✅ LoRA 适配器[➡️ 技术博客](https://developer.nvidia.com/blog/deploy-multilingual-llms-with-nvidia-nim/)

* [2024/07/02] 让 @MistralAI MoE token 飞起来 📈 🚀 #Mixtral 8x7B 搭配 #TensorRT #LLM 在 #H100 上。
[➡️ 技术博客](https://developer.nvidia.com/blog/achieving-high-mixtral-8x7b-performance-with-nvidia-h100-tensor-core-gpus-and-tensorrt-llm?ncid=so-twit-928467)

* [2024/06/24] 借助 #TensorRT #LLM 增强，@upstage.ai 的 solar-10.7B-instruct 已准备好通过我们的 API 目录为你的开发者项目提供支持 🏎️. ✨[➡️ link](https://build.nvidia.com/upstage/solar-10_7b-instruct?snippet_tab=Try )

* [2024/06/18] CYMI: 🤩 Stable Diffusion 3 上周发布 🎊 🏎️ 使用 #TensorRT INT8 量化加速你的 SD3 [➡️ link](https://build.nvidia.com/upstage/solar-10_7b-instruct?snippet_tab=Try )

* [2024/06/18] 🧰 使用 TensorRT 部署 ComfyUI？这是你的配置指南 [➡️ link](https://github.com/comfyanonymous/ComfyUI_TensorRT)

* [2024/06/11] ✨#TensorRT 权重剥离引擎（Weight-Stripped Engines）✨
面向认真开发者的技术深潜 ✅压缩率 +99% ✅1 套权重 → ** 个 GPU ✅0 性能损失 ✅** 模型…LLM、CNN 等。[➡️ link](https://developer.nvidia.com/blog/maximum-performance-and-minimum-footprint-for-ai-apps-with-nvidia-tensorrt-weight-stripped-engines/)

* [2024/06/04] ✨ #TensorRT 与 GeForce #RTX 解锁 ComfyUI SD 超级英雄之力 🦸⚡ 🎥 演示: [➡️ link](https://youtu.be/64QEVfbPHyg)
📗 DIY notebook: [➡️ link](https://console.brev.dev/launchable/deploy?userID=2x2sil999&orgID=ktj33l4xj&name=ComfyUI_TensorRT&instance=L4%40g2-standard-4%3Anvidia-l4%3A1&diskStorage=500&cloudID=GCP&baseImage=docker.io%2Fpytorch%2Fpytorch%3A2.2.0-cuda12.1-cudnn8-runtime&ports=ComfUI%3A8188&file=https%3A%2F%2Fgithub.com%2Fbrevdev%2Fnotebooks%2Fblob%2Fmain%2Ftensorrt-comfyui.ipynb&launchableID=env-2hQX3n7ae5mq3NjNZ32DfAG0tJf)

* [2024/05/28] ✨#TensorRT ResNet-50 权重剥离 ✨ ✅压缩率 +99%
✅1 套权重 → ** 个 GPU\ ✅0 性能损失 ✅** 模型…LLM、CNN 等
👀 📚 DIY [➡️ link](https://console.brev.dev/launchable/deploy?userID=2x2sil999&orgID=ktj33l4xj&launchableID=env-2h6bym7h5GFNho3vpWQQeUYMwTM&instance=L4%40g6.xlarge&diskStorage=500&cloudID=devplane-brev-1&baseImage=nvcr.io%2Fnvidia%2Ftensorrt%3A24.05-py3&file=https%3A%2F%2Fgithub.com%2FNVIDIA%2FTensorRT%2Fblob%2Frelease%2F10.0%2Fsamples%2Fpython%2Fsample_weight_stripping%2Fnotebooks%2Fweight_stripping.ipynb&name=tensorrt_weight_stripping_resnet50)

* [2024/05/21] ✨@modal_labs 已备好基于 #TensorRT #LLM 的无服务器 @AIatMeta Llama 3 代码 ✨👀 📚 精彩的 Modal 手册：
Serverless TensorRT LLM (LLaMA 3 8B) | Modal Docs [➡️ link](https://modal.com/docs/examples/trtllm_llama)

* [2024/05/08] NVIDIA Model Optimizer -- #TensorRT 生态的最新成员，一个包含后训练与训练中模型优化技术的库，涵盖 ✅量化 ✅稀疏化 ✅QAT [➡️ blog](https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-tensorrt-model-optimizer-now-publicly-available/)

* [2024/05/07] 🦙🦙🦙 每秒 24,000 tokens 🛫Meta Llama 3 借助 #TensorRT #LLM 腾飞 📚[➡️ link](https://blogs.nvidia.com/blog/meta-llama3-inference-acceleration/)

* [2024/02/06] [🚀 使用 TRT-LLM 中最先进的量化技术加速推理](./docs/source/blogs/quantization-in-TRT-LLM.md)
* [2024/01/30] [新 XQA-kernel 在同一延迟预算内提供 2.4 倍的 Llama-70B 吞吐](./docs/source/blogs/XQA-kernel.md)
* [2023/12/04] [Falcon-180B 单张 H200 GPU、INT4 AWQ，Llama-70B 比 A100 快 6.7 倍](./docs/source/blogs/Falcon180B-H200.md)
* [2023/11/27] [SageMaker LMI 现已支持 TensorRT LLM - 与上一版本相比吞吐提升 60%](https://aws.amazon.com/blogs/machine-learning/boost-inference-performance-for-llms-with-new-amazon-sagemaker-containers/)
* [2023/11/13] [H200 在 Llama2-13B 上达到每秒近 12,000 tokens](./docs/source/blogs/H200launch.md)
* [2023/10/22] [🚀 使用 TensorRT LLM 和 LlamaIndex 在 Windows 上实现 RAG 🦙](https://github.com/NVIDIA/trt-llm-rag-windows#readme)
* [2023/10/19] 入门指南 - [使用 NVIDIA TensorRT-LLM 优化大语言模型推理，现已公开发布
](https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/)
* [2023/10/17] [借助适用于 Windows 的 TensorRT LLM，在 RTX 上将大语言模型提速最高 4 倍
](https://blogs.nvidia.com/blog/2023/10/17/tensorrt-llm-windows-stable-diffusion-rtx/)

</details>

## TensorRT LLM 概览

TensorRT LLM 是一个开源库，用于优化 LLM 和 Visual Gen（视觉生成）推理。它提供了业界领先的优化手段，包括面向常见推理算子的专用 kernel（attention、GEMM、MoE 等）、算法级的运行时优化（Prefill-Decode 分离、广域专家并行 Wide Expert Parallelism、投机解码 Speculative Decoding 等），以及更多技术，让推理在 NVIDIA GPU 上高效运行。

> 💡 **AI Infra 视角**：这段话值得仔细读。它点名了 AI Infra（LLM 推理服务）的三大优化方向，也是面试常问的"推理优化手段有哪些"的标准答案骨架：
> 1. **算子级（kernel 级）**：为 attention、GEMM、MoE 写专用 CUDA kernel，利用 Tensor Core、共享内存、flash-attention 等技巧压榨硬件；
> 2. **算法级（运行时级）**：Prefill-Decode 分离（把长 prompt 处理和 token 生成分开部署，避免互相干扰）、专家并行（MoE 模型的专家分散到多卡）、投机解码（用小模型草拟、大模型验证，加速生成）；
> 3. 后续文档（阶段 1、2）会逐个展开这些概念。

TensorRT LLM [基于 PyTorch 架构构建](https://github.com/NVIDIA/TensorRT-LLM/blob/release/1.1/docs/source/developer-guide/overview.md)，提供高层 Python [LLM API](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html#llm-api)，支持从单 GPU 到多 GPU、多节点的广泛推理部署形态。它内置了对多种并行策略和高级特性的支持。LLM API 与更广泛的推理生态无缝集成，包括 NVIDIA [Dynamo](https://github.com/ai-dynamo/dynamo) 和 [Triton Inference Server](https://github.com/triton-inference-server/server)。

> 💡 **AI Infra 视角**：理解 TRT-LLM 在生态中的位置很重要——它不是孤立的推理引擎，而是 NVIDIA 推理软件栈的核心一环：
> - **上游**：PyTorch 训练出的模型权重（HF 格式）→ 经过 TRT-LLM 优化/编译；
> - **下游**：Triton Inference Server（通用推理服务框架，负责 HTTP/gRPC 接口、并发管理、模型生命周期）和 Dynamo（数据中心级分布式推理调度框架）负责对外提供服务；
> - **同类对比**：vLLM 是开源社区最火的同类引擎（Python/GPU 推理服务），TRT-LLM 的优势在于 NVIDIA 全栈优化（TensorRT 编译 + 专用 kernel），劣势是上手门槛更高。转行 AI Infra，两个引擎最好都了解。

TensorRT LLM 的设计目标是模块化和易于修改。其 PyTorch 原生架构让开发者可以实验运行时或扩展功能。多个流行模型已预定义，并可以使用[原生 PyTorch 代码](./tensorrt_llm/_torch/models/modeling_deepseekv3.py)定制，便于按需适配系统。

> 💡 **AI Infra 视角**：注意"模块化、可修改"这个卖点。早期的 TRT-LLM 是把模型编译成 TensorRT engine（类似 C++ 编译成二进制），黑盒难调试；现在基于 PyTorch 的架构允许直接读 Python 代码理解推理过程——这也是你学习本仓库的最佳入口：`tensorrt_llm/_torch/models/` 下的 `modeling_*.py` 就是每个模型的前向推理实现。

## 快速开始

开始使用 TensorRT-LLM，请参阅我们的文档：

- [快速开始指南](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
    - [运行 DeepSeek](./examples/models/core/deepseek_v3)
- [安装指南](https://nvidia.github.io/TensorRT-LLM/installation/index.html)
- [支持的硬件、模型及其他软件](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)
- [性能基准测试](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/benchmarking-default-performance.html#benchmarking-with-trtllm-bench)
- [发布说明](https://nvidia.github.io/TensorRT-LLM/release-notes.html)

## 弃用政策（Deprecation Policy）

弃用（Deprecation）用于告知开发者某些 API 和工具不再推荐使用。从 1.0 版本开始，TensorRT LLM 采用以下弃用政策：

1. 弃用通知
  - 弃用说明记录在发布说明（Release Notes）中。
  - 被弃用的 API、方法、类或参数会在源码中注明弃用时间。
  - 如果被使用，弃用的方法、类或参数会发出运行时弃用警告。
2. 迁移期
  - TensorRT LLM 在弃用后提供 3 个月的迁移期。
  - 迁移期内，被弃用的 API、工具或参数仍可正常工作，但会触发警告。
3. 弃用范围
  - 完全弃用：整个 API/方法/类被标记为待移除。
  - 部分弃用：如果只有 API/方法的某些参数被弃用（例如 `LLM.generate(param1, param2)` 中的 param1），方法本身仍可用，但被弃用的参数将在未来版本中移除。
4. 迁移期结束后移除
  - 3 个月迁移期结束后，被弃用的 API、工具或参数将按照语义化版本（semantic versioning）规则移除（大版本升级可能包含破坏性变更）。

> 💡 **AI Infra 视角**：这是库开发者的惯例——给用户缓冲期避免 API 突变。你学习时如果遇到"deprecated"警告，知道这是正常现象即可；在公司里维护推理框架时，也遵循类似的"先警告、后移除"节奏。

## 遥测数据收集（Telemetry）

TensorRT-LLM 默认收集匿名遥测数据。这些数据用于汇总分析使用模式、确定工程优先级。
**这些数据无法追溯到任何个人用户。** 不会收集任何提示词（prompts）、输出、模型权重、模型路径、tokenizer 路径、用户身份信息、原始自由格式配置字符串或持久标识符。部署标识符是临时的，每次部署随机生成，不与用户关联。我们收集的数据包括：

- 入口点（例如：LLM API、CLI、serve 命令）
- 部署时长（通过周期性心跳）
- GPU 型号（SKU）、数量、显存和 CUDA 版本
- 模型架构类名（例如：`LlamaForCausalLM`）
- 并行配置（TP/PP/CP/MoE-EP/MoE-TP 规模）、量化算法、dtype、KV cache dtype
- 系统信息（OS 平台、Python 版本、CPU 架构、CPU 核数）
- TRT-LLM 版本和后端
- 特性摘要标志（LoRA、投机解码、前缀缓存、CUDA graphs、分块上下文、数据并行）
- 分离式服务（disaggregated serving）元数据（角色和部署 ID）
- 选定的 LLM API 配置值：并行度、dtype、KV cache、调度器、CUDA graph 和编译设置
- 捕获诊断信息：schema 校验和（用于溯源）、捕获字段数量、以及是否有自由格式值被跳过

遥测在 CI 和测试环境中自动禁用。

### 关闭遥测数据收集

可以使用以下任一方式禁用遥测数据收集：

- **环境变量**：设置 `TRTLLM_NO_USAGE_STATS=1`、`DO_NOT_TRACK=1` 或 `TELEMETRY_DISABLED=true`
- **基于文件**：创建文件 `~/.config/trtllm/do_not_track`
- **Python API**：向 `LLM()` 传入 `TelemetryConfig(disabled=True)`
- **CLI 参数**：在 `trtllm-serve`、`trtllm-bench` 或 `trtllm-eval` 上使用 `--no-telemetry`

遥测收集代码完全开源，可在
[`tensorrt_llm/usage/`](./tensorrt_llm/usage/) 审计。关于具体收集内容的逐字段
参考，请参阅 [schema 文档](./tensorrt_llm/usage/schemas/README.md)。

> 💡 **AI Infra 视角**：遥测是商业推理框架的常见做法（vLLM、Triton 也都有）。作为从业者要注意两点：一是企业部署时按上面四种方式之一关掉；二是"并行配置、KV cache dtype"这些字段说明官方想知道用户怎么用框架——这类信息直接指导他们优化方向的优先级。

## 有用链接
- [Hugging Face 上的量化模型](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4)：不断增长的量化（如 FP8、FP4）和优化 LLM 集合，包括 [DeepSeek FP4](https://huggingface.co/nvidia/DeepSeek-R1-FP4)，可直接用于 TensorRT LLM 快速推理。
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)：数据中心规模的分布式推理服务框架，与 TensorRT LLM 无缝配合。
- [AutoDeploy](https://nvidia.github.io/TensorRT-LLM/features/auto_deploy/auto-deploy.html)：TensorRT LLM 的 beta 后端，用于简化并加速 PyTorch 模型的部署。
- [微信讨论群](https://github.com/NVIDIA/TensorRT-LLM/issues/5359)：TensorRT LLM 问答与新闻的实时交流渠道。
