<!--
  本文档为 TensorRT-LLM 官方 Adding a New Model 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/torch/adding_new_model.md
-->

# 在 PyTorch 后端添加新模型

## 目录
1. [简介](#introduction)
2. [前置条件](#prerequisites)
3. [分步指南](#step-by-step-guide)
    1. [模型配置](#model-configuration)
    2. [模型定义](#model-definition)
    3. [权重加载](#weight-loading)
    4. [模型注册](#model-registration)
        1. [核心模型](#core-models)
        2. [树外模型（Out-of-Tree）](#out-of-tree-models)

## 简介

本指南提供在 PyTorch 后端添加新模型的分步过程。

> 💡 **AI Infra 视角**：这篇是"给引擎加模型"的标准流程——**AI Infra 工程师的日常高频任务**（新模型发布就要接入）。四步：配置类 → 模型类 → 权重加载 → 注册。全部掌握后，你就能理解引擎是怎么做到"新模型 Day-0 支持"的。**建议对照 `tensorrt_llm/_torch/models/modeling_llama.py` 读本文**——它就是"标准答案"。

## 前置条件

开始之前，请确保你具备：
- 可用的 TensorRT-LLM 安装。按照这些[说明](../installation/build-from-source.md)。

## 分步指南

### 模型配置

假设你想支持一个名为 `MyModel` 的新模型。如果该模型已在 HuggingFace 的 transformers 中支持，你应该移植 PyTorch modeling 代码并复用 HuggingFace 的配置类。例如，我们的 `tensorrt_llm/_torch/models/modeling_llama.py` 改编自 HuggingFace 的 [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)；在 modeling 代码中，我们复用了配置类：

```python
from transformers import LlamaConfig
```

如果模型没有在 HuggingFace 的 transformers 中注册，你需要在 `configuration_mymodel.py` 中按照 HuggingFace 的 [configuration_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py) 定义配置类：

```python
from transformers.configuration_utils import PretrainedConfig

class MyConfig(PretrainedConfig):
    def __init__(self, ...):
        ...
```

> 💡 **AI Infra 视角**：**"复用 HF 配置类"是引擎兼容 HF 生态的关键**——用户从 HF 下载的模型自带 `config.json`，引擎用同一个配置类解析它，天然兼容。这就是为什么 AGENTS.md 里说"模型架构由 HF config 的 architectures 字段自动发现"。

### 模型定义

删除任何不必要的代码（例如训练专用代码），然后重写一些 PyTorch 模块。对于一个典型的 Transformer decoder 模型，你需要像这样实现 `modeling_mymodel.py`：

```python
from typing import Optional

import torch
from torch import nn
from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer

from configuration_mymodel import MyConfig


class MyAttention(Attention):
    def __init__(self, model_config: ModelConfig[MyConfig], layer_idx: Optional[int] = None):
        # 使用 model_config 初始化 Attention 模块
        super().__init__(...)


class MyDecoderLayer(DecoderLayer):
    def __init__(self, model_config: ModelConfig[MyConfig], layer_idx: int):
        super().__init__()
        # 使用 model_config 初始化子模块
        self.input_layernorm = ...
        self.self_attn = MyAttention(model_config, layer_idx)
        self.post_attention_layernorm = ...
        self.mlp = ...

    def forward(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata, **kwargs):
        # 定义单个 decoder 层的前向计算
        ...


class MyModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[MyConfig]):
        super().__init__(model_config)
        # 使用 model_config 初始化子模块
        self.embed_tokens = ...
        self.layers = nn.ModuleList([
            MyDecoderLayer(model_config, layer_idx) for layer_idx in range(model_config.pretrained_config.num_hidden_layers)
        ])

    def forward(self,
                attn_metadata: AttentionMetadata,
                input_ids: Optional[torch.IntTensor] = None,
                position_ids: Optional[torch.IntTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None):
        # 定义模型的前向计算
        ...


class MyModelForCausalLM(DecoderModelForCausalLM[MyModel, MyConfig]):
    def __init__(self, model_config: ModelConfig[MyConfig]):
        super().__init__(MyModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)
```

> 💡 **AI Infra 视角**：注意模型定义的继承结构——`MyAttention` 继承引擎的 `Attention` 模块（不是自己写 attention！），这样自动获得全部注意力后端支持（attention.md 讲的那些）；`DecoderLayer`/`DecoderModel`/`DecoderModelForCausalLM` 基类提供通用脚手架（权重加载、注册等）。**"继承引擎模块 = 白拿全部运行时能力"**——这就是为什么添加新模型可以很快。

注意，`MyAttention` 继承自我们的 `Attention` 模块（在 `tensorrt_llm/_torch/modules/attention.py`），因此注意力计算与我们的 PyTorch 运行时兼容。与此相关，模块输入也应适配：

- `attn_metadata` 存储来自批处理输入和 KV cache 的元数据，供注意力后端使用。它由运行时创建并传递，模型开发者需要确保 `attn_metadata` 被正确传递给注意力模块。
- 输入张量（即 `input_ids`、`position_ids`、`hidden_states`）采用打包模式（packed mode）。第一维对应一个 batch 中的 token 数。

> 💡 **AI Infra 视角**：这两条约束是"运行时的契约"：① `attn_metadata` 必须层层传递（它是 attention kernel 的"说明书"）；② 输入是打包的（第一维是 token 数而不是 batch 数——IFB 无 padding 的要求，前文讲过）。**模型代码必须遵守运行时契约，否则推理结果错乱**。

另外，`MyDecoderLayer`、`MyModel` 和 `MyModelForCausalLM` 分别是 `DecoderLayer`、`DecoderModel` 和 `DecoderModelForCausalLM` 的子类。基类定义了接口，并提供定义模型层、加载权重等的通用脚手架。

可选地，你可以用我们的实现替换原生 PyTorch 模块以启用特性或获得更高性能：
- `Linear`（在 `tensorrt_llm/_torch/modules/linear.py`）：启用张量并行和量化。
- `Embedding`（在 `tensorrt_llm/_torch/modules/embedding.py`）：为嵌入启用张量并行。
- `RotaryEmbedding`（在 `tensorrt_llm/_torch/modules/rotary_embedding.py`）：启用高性能旋转嵌入。
- `RMSNorm`（在 `tensorrt_llm/_torch/modules/rms_norm.py`）：启用高性能 RMS norm。

> 💡 **AI Infra 视角**：**"可选替换"列表是性能关键**：原生 PyTorch 的 `nn.Linear` 不支持张量并行切分和量化——换成引擎的 `Linear` 模块后自动获得 TP/FP8 支持。**不换 = 能跑但单卡慢；换了 = 获得全部分布式/量化能力**。接入新模型时的性能调优基本都在这里。

具体参考请查看 `tensorrt_llm/_torch/models/modeling_llama.py`。

### 权重加载

基类 `DecoderModelForCausalLM` 提供了 `load_weights` 方法，从 checkpoint 文件加载权重并分配给模型中的对应层。但是，如果默认方法不适用于 `MyModelForCausalLM`，你需要实现自己的 `load_weights`：

```python
class MyModelForCausalLM(DecoderModelForCausalLM[MyModel, MyConfig]):

    def load_weights(self, weights: dict):
        # 定义权重加载逻辑
        ...
```

例如，Huggingface 的 LLaMA 模型对 Q/K/V 投影使用三个独立 linear 层，导致 checkpoint 中有三个权重张量：

```python
>>> weights
{
    ...,
    "model.layers.0.self_attn.q_proj.weight": torch.Tensor([hidden_size, hidden_size]),
    "model.layers.0.self_attn.k_proj.weight": torch.Tensor([hidden_size, hidden_size]),
    "model.layers.0.self_attn.v_proj.weight": torch.Tensor([hidden_size, hidden_size]),
    ...,
}
```

然而，我们的 LLaMA 模型把这三层融合成一个 linear 层：

```python
>>> llama.model.layers[0].self_attn.qkv_proj.weight.data
torch.Tensor([hidden_size * 3, hidden_size])
```

> 💡 **AI Infra 视角**：这就是 checkpoint-loading.md 里 WeightMapper 的职责在具体模型上的体现：**HF 的 q/k/v 三个张量 → 引擎的 qkv 一个张量**（融合后节省 kernel 启动和显存带宽）。`load_weights` 要做：拼接 + 改名 + （TP 时）切分 + （量化时）加 scale。**"模型能加载"和"模型加载对"是两个层次**——后者是无数 bug 的来源（维度、顺序、scale 错位）。

因此，`load_weights` 需要从原始 checkpoint 收集三个权重张量，拼接它们，并分配给融合的 linear 层。考虑到张量并行和量化，这个过程会更复杂。我们建议在实现模型级 `load_weights` 方法时调用预定义的模块级 `load_weights`（如 `Linear` 和 `Embedding`）。

总之，`load_weights` 应处理 `MyModelForCausalLM` 与 checkpoint 加载权重之间的任何差异，以便 `MyModelForCausalLM` 能够执行与原始模型等价的前向计算。

### 模型注册

需要注册新模型，以便 PyTorch 运行时能够识别它。注册可以简单地通过为 `MyModelForCausalLM` 添加 `register_auto_model` 装饰器完成：

```python
from tensorrt_llm._torch.models.modeling_utils import register_auto_model

@register_auto_model("MyModelForCausalLM")
class MyModelForCausalLM(DecoderModelForCausalLM[MyModel, MyConfig]):
    def __init__(self, model_config: ModelConfig[MyConfig]):
       ...
```

> 💡 **AI Infra 视角**：注册 = 告诉引擎"HF 配置里出现 `architectures: [MyModelForCausalLM]` 时，用我这个类"——**注册表机制**（automodel.py 的 auto-discovery，AGENTS.md 讲过）。这也是引擎支持新模型后"Day-0 加载"的实现原理。

#### 核心模型（Core Models）

要将新模型添加到核心模型中，`modeling_mymodel.py`（可能还有 `configuration_mymodel.py`）应放在 `tensorrt_llm/_torch/models`。然后，你需要在 `tensorrt_llm/_torch/models/__init__.py` 中导入 modeling 代码：

```python
from .modeling_mymodel import MyModelForCausalLM

__all__ = [
    ...,
    "MyModelForCausalLM",
]
```

#### 树外模型（Out-of-Tree Models）

或者，你可以将新模型注册为树外模型，这样无需改动 TensorRT LLM 代码库即可使用新模型。为此，将 `modeling_mymodel.py`（可能还有 `configuration_mymodel.py`）放在你的工作目录中，并在脚本中导入 modeling 代码：

```python
from tensorrt_llm import LLM
import modeling_mymodel

def main():
    llm = LLM(...)

if __name__ == '__main__':
    main()
```

> 💡 **AI Infra 视角**：**Out-of-tree 模式是引擎"可扩展性"的设计**：个人/公司可以给私有模型写建模代码，放在自己的仓库里，import 一下就生效——不用 fork 引擎、不用提 PR。这就是**"引擎内核稳定，扩展靠注册"**的生态哲学。对比：核心模型（in-tree）要过 PR 评审，适合通用模型。

我们在 `examples/llm-api/out_of_tree_example` 提供了一个树外建模示例。模型在 `modeling_opt.py` 中实现，你可以这样运行示例：

```bash
python examples/llm-api/out_of_tree_example/main.py
```
