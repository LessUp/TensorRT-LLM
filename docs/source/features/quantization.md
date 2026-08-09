<!--
  本文档为 TensorRT-LLM 官方 Quantization 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/features/quantization.md
-->

# 量化（Quantization）

## TensorRT LLM 中的量化

量化是一种通过将模型的权重和/或激活从高精度浮点数（如 BF16）转换为更低精度的数据类型（如 INT8、FP8 或 FP4）来减少显存占用和计算成本的技术。

> 💡 **AI Infra 视角**：量化是 AI Infra 的"显存和速度放大器"。先建立核心直觉：
> - **为什么能量化**：神经网络权重分布通常集中在某个范围，用低精度近似时损失不大（尤其推理时）；
> - **省什么**：显存（权重减半/减到 1/4）+ 计算（低精度矩阵乘在 Tensor Core 上更快）+ 带宽（读权重更快）；
> - **代价**：精度损失（通常可控，几 pt 的准确率下降换 2~4 倍性能提升）。
> 格式速记：FP8 = 8 位浮点（H100 起支持），FP4 = 4 位（B200 起），W4A16 表示"权重 4 位、激活 16 位"，W4A8 表示"权重 4 位、激活 8 位"。

TensorRT LLM 提供多种量化方案（recipe）来优化 LLM 推理。这些方案大致可分为：

* FP4
* FP8 Per Tensor（逐张量）
* FP8 Block Scaling（块缩放）
* FP8 Rowwise（逐行）
* FP8 KV Cache
* NVFP4 KV Cache
* W4A16 GPTQ
* W4A8 GPTQ
* W4A16 AWQ
* W4A8 AWQ

> 💡 **AI Infra 视角**：术语解释（面试会问）：
> - **Per Tensor / Rowwise / Block Scaling**：量化时的**缩放粒度**——整层一个 scale（最简单，精度差）、每行一个 scale（常用）、每 128 个元素一个 scale（最细，精度最好）。粒度越细精度越高，但 scale 存储和计算开销越大；
> - **GPTQ / AWQ**：两种主流**权重量化算法**（训练后量化，PTQ）——GPTQ 用"误差最小化"逐个修正权重，AWQ 根据激活分布保护重要权重通道。它们都是离线把 FP16 权重转成 4 位权重，推理引擎直接加载；
> - **KV Cache 量化**：前面 kvcache.md 讲过，把缓存也降精度省显存。
> 名称规则：W4A16 意思是权重（Weight）4 位、激活（Activation）16 位。

## 用法

默认的 PyTorch 后端在最新的 Blackwell 和 Hopper GPU 上支持 FP4 和 FP8 量化。

### 运行预量化模型

TensorRT LLM 可以直接运行使用 [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) 生成的[预量化模型](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4)。

```python
from tensorrt_llm import LLM
llm = LLM(model='nvidia/Llama-3.1-8B-Instruct-FP8')
llm.generate("Hello, my name is")
```

> 💡 **AI Infra 视角**：预量化模型是"开箱即用"的：NVIDIA 已经把流行模型量好放在 HF 上（模型名带 -FP8/-FP4 后缀）。生产上省掉自己量化的环节，直接 `LLM(model='nvidia/xxx-FP8')`。**注意：FP8 权重模型必须跑在支持 FP8 的 GPU 上**（H100 及以上）。

#### FP8 KV Cache

```{note}
TensorRT LLM 允许你手动启用 FP8 KV cache，即使 checkpoint 默认没有启用。
```

下面是设置 FP8 KV Cache 选项的示例：

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
llm = LLM(model='/path/to/model',
          kv_cache_config=KvCacheConfig(dtype='fp8'))
llm.generate("Hello, my name is")
```

> 💡 **AI Infra 视角**：这个例子展示了**权重量化和 KV cache 量化是独立的两件事**——权重是 FP16 也能给 KV cache 单独开 FP8。因为量化时机不同：权重是离线量好的，KV cache 是运行时量化的（写入时量化、读取时反量化，见 attention.md 的讲解）。**显存不够时的第一梯队手段：KV cache 降精度**。

#### NVFP4 KV Cache

要启用 NVFP4 KV cache，需要用 ModelOpt 做离线量化。请按照下面的章节操作。
量化完成后，可以通过以下方式设置 NVFP4 KV cache 选项：

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
llm = LLM(model='/path/to/model',
          kv_cache_config=KvCacheConfig(dtype='nvfp4'))
llm.generate("Hello, my name is")
```

> 💡 **AI Infra 视角**：NVFP4 是 NVIDIA 为 Blackwell 设计的 4 位浮点格式（比普通的 FP4 多了微缩放技巧，精度更好）。**注意它比 FP8 KV cache 多一步"离线量化"**——KV cache 量化格式需要配套的量化模型权重（FP8 权重 + NVFP4 缓存），不能随便配。**量化的"配方"（权重格式 × KV 格式）是组合拳**，不能随意混搭。

### 用 ModelOpt 离线量化

如果在 [Hugging Face Hub](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4) 上没有可用的预量化模型，你可以用 ModelOpt 离线量化。

按照这个分步指南量化模型：

```bash
git clone https://github.com/NVIDIA/Model-Optimizer.git
cd Model-Optimizer/examples/llm_ptq
scripts/huggingface_example.sh --model <huggingface_model_card> --quant fp8
```

#### NVFP4 KV Cache

要生成 NVFP4 KV cache 的 checkpoint：

```bash
git clone https://github.com/NVIDIA/Model-Optimizer.git
cd Model-Optimizer/examples/llm_ptq
scripts/huggingface_example.sh --model <huggingface_model_card> --quant fp8 --kv_cache_quant nvfp4
```

注意，目前 TRT-LLM 在启用 NVFP4 KV cache 时只支持 FP8 权重/激活量化。因此，这里必须使用 `--quant fp8`。

## 模型支持矩阵

| 模型          |  NVFP4  | MXFP4  | FP8(per tensor)| FP8(block scaling) | FP8(rowwise) | FP8 KV Cache | NVFP4 KV Cache | W4A8 AWQ  | W4A16 AWQ | W4A8 GPTQ  | W4A16 GPTQ |
| :------------- | :---:   | :---:  | :---: | :---: | :---: | :---: |:---:| :-------: | :-------: | :--------: | :--------: |
| BERT           |   .     |   .    |   .   |   .   |   .   |   Y   |  .  |     .     |     .     |     .      |     .      |
| DeepSeek-R1    |   Y     |   .    |   .   |   Y   |   .   |   Y   |  .  |     .     |     .     |     .      |     .      |
| EXAONE         |   .     |   .    |   Y   |   .   |   .   |   Y   |  .  |     Y     |     Y     |     .      |     .      |
| Gemma 3        |   .     |   .    |   Y   |   .   |   .   |   Y   |  .  |     Y     |     Y     |     .      |     .      |
| GPT-OSS        |   .     |   Y    |   .   |   .   |   .   |   Y   |  .  |     .     |     .     |     .      |     .      |
| LLaMA          |   Y     |   .    |   Y   |   .   |   .   |   Y   |  .  |     .     |     Y     |     .      |     Y      |
| LLaMA-v2       |   Y     |   .    |   Y   |   .   |   .   |   Y   |  Y  |     Y     |     Y     |     .      |     Y      |
| LLaMA 3        |   .     |   .    |   .   |   .   |   Y   |   Y   |  Y  |     Y     |     .     |     .      |     .      |
| LLaMA 4        |   Y     |   .    |   Y   |   .   |   .   |   Y   |  .  |     .     |     .     |     .      |     .      |
| Mistral        |   .     |   .    |   Y   |   .   |   .   |   Y   |  .  |     .     |     Y     |     .      |     .      |
| Mixtral        |   Y     |   .    |   Y   |   .   |   .   |   Y   |  .  |     .     |     .     |     .      |     .      |
| Phi            |   .     |   .    |   .   |   .   |   .   |   Y   |  .  |     Y     |     .     |     .      |     .      |
| Qwen           |   .     |   .    |   .   |   .   |   .   |   Y   |  .  |     Y     |     Y     |     .      |     Y      |
| Qwen-2/2.5     |   Y     |   .    |   Y   |   .   |   .   |   Y   |  .  |     Y     |     Y     |     .      |     Y      |
| Qwen-3         |   Y     |   .    |   Y   |   .   |   .   |   Y   |  Y  |     .     |     Y     |     .      |     Y      |
| BLIP2-OPT      |   .     |   .    |   .   |   .   |   .   |   Y   |  .  |     .     |     .     |     .      |     .      |
| BLIP2-T5       |   .     |   .    |   .   |   .   |   .   |   Y   |  .  |     .     |     .     |     .      |     .      |
| LLaVA          |   .     |   .    |   Y   |   .   |   .   |   Y   |  .  |     .     |     Y     |     .      |     Y      |
| VILA           |   .     |   .    |   Y   |   .   |   .   |   Y   |  .  |     .     |     Y     |     .      |     Y      |
| Nougat         |   .     |   .    |   .   |   .   |   .   |   Y   |  .  |     .     |     .     |     .      |     .      |

> 💡 **AI Infra 视角**：读表方法：每个单元格 = "这个模型 × 这种量化方式是否被支持"。注意规律——**FP8 KV Cache 几乎人人支持**（通用优化），但 FP4/NVFP4 只支持新模型（需要配套 kernel 和验证）。选型时先看这张表，别选了个模型不支持的量化方式（会报错或回退）。表在变（随版本更新），以仓库最新文档为准。

```{note}
多模态模型的视觉组件（BLIP2-OPT/BLIP2-T5/LLaVA/VILA/Nougat）默认使用 FP16。
语言组件决定给定的多模态模型支持哪些量化方法。
```

## 硬件支持矩阵

| 模型          |  NVFP4  | MXFP4  | FP8(per tensor)| FP8(block scaling) | FP8(rowwise) | FP8 KV Cache | NVFP4 KV Cache | W4A8 AWQ  | W4A16 AWQ | W4A8 GPTQ  | W4A16 GPTQ |
| :------------- | :---:   | :---:  | :---: | :---: | :---: | :---: | :---: | :-------: | :-------: | :--------: | :--------: |
| Blackwell(sm120)       |   Y     |   Y    |   Y   |   .   |   .   |   Y   |   .   |     .     |     .     |     .      |     .      |
| Blackwell(sm100/103)       |   Y     |   Y    |   Y   |   Y   |   .   |   Y   |   Y   |     Y     |     Y     |     Y      |     Y      |
| Hopper           |   .     |   .    |   Y   |   Y   |   Y   |   Y   |   .   |     Y     |     Y     |     Y      |     Y      |
| Ada Lovelace          |   .     |   .    |   Y   |   .   |   .   |   Y   |   .   |     Y     |     Y     |     Y      |     Y      |
| Ampere         |   .     |   .    |   .   |   .   |   .   |   Y   |   .   |     .     |     Y     |     .      |     Y      |

> 💡 **AI Infra 视角**：**量化能力是"硬件 × 软件"共同决定的**——kernels 必须针对某代 GPU 的 Tensor Core 指令编写。读这张表：Hopper（H100）是 FP8 时代主力；Blackwell（B200）解锁 FP4/NVFP4；Ampere（A100）老当益壮但只支持 FP8 KV cache 和 W4A16。部署选型时：**先确认目标 GPU 支持什么，再选量化方案**。

```{note}
sm100/103 的 FP8 块缩放 GEMM kernel 使用 MXFP8 配方（E4M3 激活/权重和 UE8M0 激活/权重缩放），与 SM90 的 FP8 配方（E4M3 激活/权重和 FP32 激活/权重缩放）略有不同。
```

> 💡 **AI Infra 视角**：FP8 有两种子格式：**E4M3**（4 位指数 + 3 位尾数，动态范围大、精度低）和 **E5M2**（5 位指数 + 2 位尾数，范围更大、精度更低）。MXFP8 是"微缩放"生态的一部分——用额外的 UE8M0 缩放字节包住 FP8，精度更高。**同一叫 FP8，不同代硬件/不同 kernel 的细节并不相同**，这是阅读硬件文档时要注意的。

## 快速链接

- [ModelOpt 预量化模型](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4)
- [ModelOpt 支持矩阵](https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html)
