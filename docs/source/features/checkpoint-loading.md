<!--
  本文档为 TensorRT-LLM 官方 Checkpoint Loading 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/features/checkpoint-loading.md
-->

# Checkpoint 加载

PyTorch 后端提供了一个灵活且可扩展的基础设施，用于从不同格式（如 HuggingFace，HF）加载模型 checkpoint。该系统允许你通过实现所需组件（如 checkpoint 的权重加载器、映射器和配置解析器）从各种来源（如 HuggingFace 或自定义格式）加载模型。

> 💡 **AI Infra 视角**：先理解"checkpoint"是什么：**模型训练结束后保存的权重文件**（HF 格式是主流——`config.json` + 一堆 `.safetensors` 分片文件）。推理引擎要做的三件事：① 读配置（多少层、多少头）；② 读权重；③ **把 HF 的命名/布局转成自己引擎的命名/布局**。第三步是坑最多的——不同引擎对参数名、张量形状、合并方式（如 QKV 融合）的要求不同。**"权重格式转换"是每个推理引擎都要解决的第一工程问题**。

## 目录
1. [总览](#overview)
2. [核心组件](#core-components)
3. [内置 Checkpoint 格式](#built-in-checkpoint-formats)
4. [使用 Checkpoint 加载器](#using-checkpoint-loaders)
5. [创建自定义 Checkpoint 加载器](#creating-custom-checkpoint-loaders)

## 总览

checkpoint 加载设计围绕插件式架构构建，分为四个独立组件：

- **Checkpoint Loaders（加载器）**：编排特定格式的加载过程
- **Config Loaders（配置加载器）**：处理模型配置的解析和验证
- **Weight Loaders（权重加载器）**：管理从存储加载模型权重到内存的实际过程
- **Weight Mappers（权重映射器）**：将加载的权重映射和转换为 TensorRT LLM 模型的定义

> 💡 **AI Infra 视角**：四个组件的分工，用一个类比理解：**搬家**——ConfigLoader 看房子的图纸（模型结构），WeightLoader 把家具从旧屋搬出来（读文件），WeightMapper 按新房布局摆家具（改名字、改形状、合并拆分），CheckpointLoader 是搬家公司总调度。**"四个组件可独立替换"是插件架构的核心价值**：支持新格式时，能复用的组件直接用，只写差异部分。

这种模块化设计便于扩展以支持新的 checkpoint 格式，同时保持向后兼容性和性能优化。通过将 checkpoint 加载组件分成四个子组件，任何用户都可以复用相关的现有工作，同时引入自己的自定义 checkpoint 组件。

如果想要支持一种新的 checkpoint 格式，必须实现全部四个组件。
同样，如果该格式与已支持的框架（如 HF）共享某些组件，则只需实现自定义特有的组件。

## 核心组件

### BaseCheckpointLoader

`BaseCheckpointLoader` 是所有 checkpoint 加载操作符的核心基础接口。无论底层 checkpoint 格式如何，它都提供统一的 API。该接口负责持有并暴露加载和解析过程所需的全部对象。

**关键方法：**
- `load_config(checkpoint_dir, **kwargs)`：加载并返回一个 `ModelConfig` 对象
- `load_weights(checkpoint_dir, mapping, **kwargs)`：加载并返回权重字典
- `get_initialized_weight_mapper(model, config)`：返回模型运行时初始化的权重映射器
- `cleanup()`：释放资源并清理内部状态

### BaseConfigLoader

负责从 checkpoint 目录加载模型配置并将其解析为 TRTLLM `ModelConfig`：

```python
from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader

class CustomConfigLoader(BaseConfigLoader):
    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        # 从你的自定义格式加载并解析配置
        pretrained_config = self._get_pretrained_config(checkpoint_dir, **kwargs)

        return ModelConfig(pretrained_config=pretrained_config,
                            ...)

    def _get_pretrained_config(self, checkpoint_dir, **kwargs):
        ...

```

### BaseWeightLoader

处理从存储加载模型权重：

```python
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader

class CustomWeightLoader(BaseWeightLoader):
    def load_weights(self, checkpoint_dir: str, mapping: Mapping) -> dict[str, Any]:
        # 从你的自定义格式加载权重
        # 返回参数名到张量的映射字典
        return weights_dict
```

### BaseWeightMapper

在不同命名约定之间转换权重，并应用模型特定的转换，将其转成 TRTLLM 模型的对象。

> 💡 **AI Infra 视角**：WeightMapper 是四个组件里最容易踩坑的。HF 里 `q_proj.weight`、`k_proj.weight`、`v_proj.weight` 是三个独立张量，TRT-LLM 里可能是融合的 `qkv_proj.weight`（一个拼接张量）——映射器要负责**拼接/拆分 + 重命名**。还有 transposed 布局（HF 用 [out, in]，TRT-LLM 可能要 [in, out]）、scale 因子等。**做推理引擎最常改的代码就是 weight mapper**。

## 内置 Checkpoint 格式

### HuggingFace 格式

目前，HF checkpoint 加载器是主要的内置格式，支持：

- **权重加载**（`.safetensors/.bin/.pth`）- 从磁盘加载 HF 兼容权重
- **配置解析器** - 将 HF 存储的配置信息解析为 TRTLLM `ModelConfig` 对象
- **权重映射** - 将 HF 权重转换为 TRTLLM 兼容表示

> 💡 **AI Infra 视角**：`.safetensors` 是 HF 社区的现代权重格式（安全、快速、支持内存映射），`.bin` 是 PyTorch 旧格式（pickle 序列化，有安全隐患）。**选格式时优先 safetensors**——这也是业界共识。

### ModelExpress (MX) 加载路径

PyTorch 后端可以使用 ModelExpress (MX) 进行点对点（peer-to-peer）权重传输，
从正在运行的 TensorRT-LLM 源实例获取权重，之后才回退到 Hugging Face
checkpoint 加载。选择 MX 不需要 MX 特有的磁盘 checkpoint，也不需要转换 Hugging Face checkpoint。安装、MX
服务部署和配置细节，请参见
[ModelExpress (MX) Checkpoint 加载](./model-express.md)。

> 💡 **AI Infra 视角**：MX 是"零磁盘权重加载"方案：**权重直接从另一台运行中的实例的显存/内存通过 RDMA 传过来**，而不是读磁盘文件。适用场景：大规模集群快速拉起新实例（不用每台都从对象存储下载几百 GB 权重）、权重在实例间流动。**"启动时间"是大规模推理集群的隐性成本**——从磁盘/网络加载 700GB 权重可能要好几分钟，MX 可以大幅缩短。

## 使用 Checkpoint 加载器

### 基本用法

有两种主要方式触发 checkpoint 加载对象的使用。

第一种方式，通过 llm-api，如下例所示：

```python
from tensorrt_llm import LLM

hf_model_dir = "llama-models-v2/llama-v2-13b-hf"

llm = LLM(model=hf_model_dir)
```

在这个例子中，默认会选择 `HfCheckpointLoader`。

要显式设置 checkpoint 加载器，你需要调用所需 checkpoint 特定的加载器

```python
from tensorrt_llm import LLM
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader

hf_model_dir = "llama-models-v2/llama-v2-13b-hf"

llm = LLM(model=hf_model_dir,
          checkpoint_loader=HfCheckpointLoader())
```

同样，如果你想使用一个基础实现的 checkpoint 加载器，但配合特定的子组件，可以根据需要提供任何特定的子组件

```python
from tensorrt_llm import LLM
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader

hf_model_dir = "llama-models-v2/llama-v2-13b-hf"

llm = LLM(model=hf_model_dir,
          checkpoint_loader=HfCheckpointLoader(weight_loader=MyCustomWeightLoader()))
```

在第二种方式中，可以直接使用 checkpoint 加载的组件。

```python
from tensorrt_llm._torch.models.checkpoints.hf.gemma3_weight_mapper import \
    Gemma3HfWeightMapper
from tensorrt_llm._torch.models.modeling_gemma3 import Gemma3ForCausalLM

gemma3 = Gemma3ForCausalLM(model_config)
weight_mapper = Gemma3HfWeightMapper()
weight_mapper.init_model_and_config(gemma3, model_config)
gemma3.load_weights(hf_gemma3.state_dict(), weight_mapper)
```

> 💡 **AI Infra 视角**：注意 `LLM(model=...)` 的自动发现机制：用户只给一个路径，引擎靠 `@register_checkpoint_loader` 装饰器注册的加载器**按格式自动匹配**。这也是 AGENTS.md 里讲的"auto-discovery 模式"（模型自注册）在 checkpoint 层的复用。**约定优于配置 + 注册表**是这类框架的通用设计。

## 创建自定义 Checkpoint 加载器

要支持新的 checkpoint 格式，你需要实现全部四个组件。本节为每个组件提供最小模板。

### 何时创建自定义组件

- **全新格式**：支持完全新的 checkpoint 格式时，实现全部四个组件
- **自定义权重存储**：只有你有独特的权重存储格式（如自定义二进制格式、数据库存储等）时才需要实现自定义权重加载器
- **自定义配置**：只有当现有解析器无法解析你的配置格式时才需要实现自定义配置加载器
- **自定义权重映射**：只有当你的模型有独特的、checkpoint 特有的权重命名或转换需求时才需要实现自定义权重映射器

> 💡 **AI Infra 视角**：这段"何时创建"清单体现了**只写差异**的工程原则：新格式和 HF 格式 80% 相同，就只写那 20% 的差异组件。**复用优先**——这也是为什么插件架构比"整个重写"好。

### 第 1 步：创建 Checkpoint 加载器

```python
from typing import Optional
from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_loader

@register_checkpoint_loader("CUSTOM_FORMAT")
class CustomCheckpointLoader(BaseCheckpointLoader):
    def __init__(self,
                 *,
                 weight_loader: Optional[BaseWeightLoader] = None,
                 weight_mapper: Optional[BaseWeightMapper] = None,
                 config_loader: Optional[BaseConfigLoader] = None):
        self._weight_loader = weight_loader or self.get_default_weight_loader()
        self._config_loader = config_loader or self.get_default_config_loader()
        self._weight_mapper = weight_mapper
        self._checkpoint_format = "CUSTOM_FORMAT"

    def get_default_weight_loader(self) -> BaseWeightLoader:
        return CustomWeightLoader()

    def get_default_config_loader(self) -> BaseConfigLoader:
        return CustomConfigLoader()
```

### 第 2 步：创建 Checkpoint 权重加载器

```python
from typing import Any
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_weight_loader

@register_checkpoint_weight_loader("CUSTOM_FORMAT")
class CustomWeightLoader(BaseWeightLoader):
    def load_weights(self, checkpoint_dir: str, mapping: Mapping, **kwargs) -> dict[str, Any]:
        """
        从你的自定义格式加载权重。
        参数:
            checkpoint_dir: 包含 checkpoint 文件的目录
            mapping: 包含分布式配置的映射对象。
            **kwargs: 附加加载参数
        返回:
            参数名到张量的映射字典
        """
        weights = {}

        # 在这里实现你的自定义权重加载逻辑
        # 示例:
        # - 从自定义二进制文件加载
        # - 从数据库加载
        # - 从压缩归档加载
        # - 应用自定义预处理

        return weights
```

### 第 3 步：创建 Checkpoint 配置加载器

```python
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader
from tensorrt_llm._torch.models.modeling_utils import register_config_loader

@register_config_loader("CUSTOM_FORMAT")
class CustomConfigLoader(BaseConfigLoader):
    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        """
        从你的自定义格式加载并解析配置。
        参数:
            checkpoint_dir: 包含配置文件的目录
            **kwargs: 附加加载参数
        返回:
            包含解析后配置的 ModelConfig 对象
        """
        # 加载你的自定义配置格式
        # 示例:
        # - 解析 YAML/TOML 文件
        # - 从专有格式转换

        pretrained_config = self._load_pretrained_config(checkpoint_dir, **kwargs)

        return ModelConfig(
            pretrained_config=pretrained_config,
            # 按需添加其他 ModelConfig 参数
        )

    def _load_pretrained_config(self, checkpoint_dir: str, **kwargs):
        """从你的自定义格式加载原始配置。"""
        pass
```

### 第 4 步：创建 Checkpoint 权重映射器

```python
from torch import nn
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper

@register_mapper("CUSTOM_FORMAT")
class CustomWeightMapper(BaseWeightMapper):
    def __init__(self):
        super().__init__()
        # 定义任何权重转换回调
        self._callbacks = [
            # 添加你的自定义权重转换函数
            # self._custom_transform_function,
        ]

    def map_weights(self) -> None:
        """
        定义源权重名和目标权重名之间的映射。
        """
        self.mapping.update({
            # 将源名称映射到目标名称
            # 'target_module_name': ['source_param1', 'source_param2'],
            # 示例: 'qkv_proj': ['q_proj', 'k_proj', 'v_proj']
        })

    def apply_callbacks(self, module: nn.Module, module_name: str,
                        module_names_breakdown: list[str],
                        weights: dict) -> list[dict]:
        """
        为需要特殊处理的模块应用权重转换。
        参数:
            module: 目标模块
            module_name: 正在处理的特定模块名
            module_names_breakdown: 模块路径组件
            weights: 源权重字典
        返回:
            转换后的权重字典列表
        """
        module_weights = []

        for new_name in self._mapping[module_name]:
            # 过滤此特定参数的权重
            fw = self.filter_weights(
                '.'.join(module_names_breakdown + [new_name]), weights)

            # 应用转换回调
            for callback in self._callbacks:
                fw = callback(module, new_name, fw)

            module_weights.append(fw)

        return module_weights

    def should_skip_module(self, module_name: str) -> bool:
        """
        定义加载期间应跳过哪些模块。
        """
        # 根据你的需求添加跳过特定模块的逻辑
        # 示例:
        # - 跳过 LoRA 特定模块
        # - 跳过临时/辅助模块

        return super().should_skip_module(module_name)
```

> 💡 **AI Infra 视角**：这个模板值得逐行读懂，它展示了权重映射的完整模式：`map_weights` 定义"目标模块 ← 哪些源参数"（如 `qkv_proj ← [q_proj, k_proj, v_proj]`）；`apply_callbacks` 做实际转换（拼接、重排、变形）；`should_skip_module` 跳过不需要的模块。**回调（callbacks）机制**让特殊转换（如量化 scale 的计算）可以按模块注入，而不用改通用逻辑。

注意：创建自定义 mapper 时，你可以定义 checkpoint 格式特定的 mapper。例如：

```python
@register_mapper("CUSTOM_FORMAT")
class CustomWeightMapper(BaseWeightMapper)
```

或者，你可以定义 checkpoint 模型特定的 mapper。例如：

```python
@register_mapper("CUSTOM_FORMAT", "Gemma3ForCausalLM")
class CustomWeightMapper(BaseWeightMapper)
```

通过设置模型名，注册的 mapper 将与特定模型关联。
