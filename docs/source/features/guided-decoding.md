<!--
  本文档为 TensorRT-LLM 官方 Guided Decoding 文档的中文翻译版（AI 翻译，翻译日期 2026-08-07）。
  英文原文可从 git 历史恢复：git checkout HEAD -- docs/source/features/guided-decoding.md
-->

# 引导解码（Guided Decoding）

引导解码（也称受约束解码 constrained decoding、结构化生成 structured generation）保证 LLM 输出符合用户指定的语法（例如 JSON schema、[正则表达式](https://en.wikipedia.org/wiki/Regular_expression) 或 [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) 文法）。

> 💡 **AI Infra 视角**：为什么需要引导解码？LLM 是"概率地输出 token"，**不保证格式**——让它输出 JSON 它可能给多行注释，让它输出数字它可能给 "大约 100 万"。而生产应用（API 后端、Agent 工具调用、数据库查询生成）要求**格式必须合法**。两种实现路线：
> - **事后修补（reparsing）**：生成完再解析/重试——浪费 token、延迟高、不保证成功；
> - **引导解码（guided decoding）**：**采样时就把不合法 token 屏蔽掉**——每步只从"符合语法的 token"里采样，输出天然合法（也叫 grammar-constrained sampling）。
> 实现原理：把语法（JSON schema/正则/EBNF）编译成一个**token 级状态机**，每步根据当前状态算出"下一个允许的 token 集合"，采样时只允许这些。XGrammar（MLC 出品）的核心就是这种**grammar → 状态机 → 掩码**的流水线，且做了缓存和并行优化。

TensorRT LLM 支持两个文法后端：
* [XGrammar](https://github.com/mlc-ai/xgrammar/blob/v0.1.21/python/xgrammar/matcher.py#L341-L350)：支持 JSON schema、正则表达式、EBNF 和[结构化标签（structural tag）](https://xgrammar.mlc.ai/docs/structural_tag/structural_tag_api.html)。
* [LLGuidance](https://github.com/guidance-ai/llguidance/blob/v1.1.1/python/llguidance/_lib.pyi#L363-L366)：支持 JSON schema、正则表达式、EBNF。

> 💡 **AI Infra 视角**：两个后端的区别：XGrammar（MLC AI 开源）功能更全（支持 structural tag，且对新格式适配快）；LLGuidance（Microsoft guidance 生态）专注 JSON/正则/EBNF。**生产选型看场景**：需要函数调用标签控制选 XGrammar，常规 JSON 输出两者都行。注意引导解码有性能开销（每步计算掩码），短输出场景开销可忽略，长结构化输出要考虑。

## 在线 API：`trtllm-serve`

如果你使用 `trtllm-serve`，在 YAML 配置文件中用 `xgrammar` 或 `llguidance` 指定 `guided_decoding_backend`，并通过 `--config` 传入。例如：

```{eval-rst}
.. include:: ../_includes/note_sections.rst
   :start-after: .. start-note-config-flag-alias
   :end-before: .. end-note-config-flag-alias
```

```bash
cat > config.yaml <<EOF
guided_decoding_backend: xgrammar
EOF

trtllm-serve nvidia/Llama-3.1-8B-Instruct-FP8 --config config.yaml
```

你应该会看到类似下面的日志，表示文法后端已成功启用。

```txt
......
[TRT-LLM] [I] Guided decoder initialized with backend: GuidedDecodingBackend.XGRAMMAR
......
```

### JSON Schema

定义 JSON schema 并在创建 OpenAI chat completion 请求时通过 `response_format` 传入。或者，JSON schema 可以用 [pydantic](https://docs.pydantic.dev/latest/) 创建。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

json_schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "pattern": "^[\\w]+$"
        },
        "population": {
            "type": "integer"
        },
    },
    "required": ["name", "population"],
}
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": "Give me the information of the capital of France in the JSON format.",
    },
]
chat_completion = client.chat.completions.create(
    model="nvidia/Llama-3.1-8B-Instruct-FP8",
    messages=messages,
    max_completion_tokens=256,
    response_format={
        "type": "json",
        "schema": json_schema
    },
)

message = chat_completion.choices[0].message
print(message.content)
```

输出看起来像：
```txt
{
    "name": "Paris",
    "population": 2145200
}
```

> 💡 **AI Infra 视角**：**JSON mode 是 AI 应用对接的刚需**——Agent 框架、结构化数据抽取、RAG 后处理全都要。注意 response 里连缩进都合法（schema 只约束结构）。**OpenAI 兼容 API 的 `response_format` 字段**已被各引擎（vLLM、SGLang 等都支持）当作事实标准——作为引擎开发者要兼容这个字段的语义。

### 正则表达式

定义正则表达式并在创建 OpenAI chat completion 请求时通过 `response_format` 传入。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": "What is the capital of France?",
    },
]
chat_completion = client.chat.completions.create(
    model="nvidia/Llama-3.1-8B-Instruct-FP8",
    messages=messages,
    max_completion_tokens=256,
    response_format={
        "type": "regex",
        "regex": "(Paris|London)"
    },
)

message = chat_completion.choices[0].message
print(message.content)
```

输出看起来像：
```txt
Paris
```

### EBNF 文法

定义 EBNF 文法并在创建 OpenAI chat completion 请求时通过 `response_format` 传入。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

ebnf_grammar = """root ::= description
city ::= "London" | "Paris" | "Berlin" | "Rome"
description ::= city " is " status
status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""
messages = [
    {
        "role": "system",
        "content": "You are a helpful geography bot."
    },
    {
        "role": "user",
        "content": "Give me the information of the capital of France.",
    },
]
chat_completion = client.chat.completions.create(
    model="nvidia/Llama-3.1-8B-Instruct-FP8",
    messages=messages,
    max_completion_tokens=256,
    response_format={
        "type": "ebnf",
        "ebnf": ebnf_grammar
    },
)

message = chat_completion.choices[0].message
print(message.content)
```

输出看起来像：
```txt
Paris is the capital of France
```

> 💡 **AI Infra 视角**：EBNF 是比 JSON schema/正则更通用的**文法描述语言**（编程语言语法就是用 EBNF 描述的）。注意这个例子演示了引导解码的"递归约束"：`description ::= city " is " status` 定义了完整的句子结构，模型输出的每个 token 都必须符合这个结构。**"输出长度受限的填空式生成"（如 SQL 查询、公式、代码片段）用 EBNF 最灵活**。

### 结构化标签（Structural tag）

定义结构化标签并在创建 OpenAI chat completion 请求时通过 `response_format` 传入。

结构化标签只由 `xgrammar` 后端支持。它是表达 LLM 输出约束的强大而灵活的工具。完整的教程请看[结构化标签用法](https://xgrammar.mlc.ai/docs/structural_tag/structural_tag_api.html)。下面是一个为 `Llama-3.1-8B-Instruct` 自定义函数调用格式的示例。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

tool_get_current_weather = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for, e.g. 'San Francisco'",
                },
                "state": {
                    "type": "string",
                    "description": "the two-letter abbreviation for the state that the city is in, e.g. 'CA' which would mean 'California'",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city", "state", "unit"],
        },
    },
}

tool_get_current_date = {
    "type": "function",
    "function": {
        "name": "get_current_date",
        "description": "Get the current date and time for a given timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
                }
            },
            "required": ["timezone"],
        },
    },
}

system_prompt = f"""# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search
You have access to the following functions:
Use the function 'get_current_weather' to: Get the current weather in a given location
{tool_get_current_weather["function"]}
Use the function 'get_current_date' to: Get the current date and time for a given timezone
{tool_get_current_date["function"]}
If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where
start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`
Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query
You are a helpful assistant."""
user_prompt = "You are in New York. Please get the current date and time, and the weather."

messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": user_prompt,
    },
]

chat_completion = client.chat.completions.create(
    model="nvidia/Llama-3.1-8B-Instruct-FP8",
    messages=messages,
    max_completion_tokens=256,
    response_format={
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<function="],
            "tags": [
                {
                    "begin": "<function=get_current_weather>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": tool_get_current_weather["function"]["parameters"]
                    },
                    "end": "</function>",
                },
                {
                    "begin": "<function=get_current_date>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": tool_get_current_date["function"]["parameters"]
                    },
                    "end": "</function>",
                },
            ],
        },
    },
)

message = chat_completion.choices[0].message
print(message.content)
```

输出看起来像：
```txt
<function=get_current_date>{"timezone": "America/New_York"}</function>
<function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function>
```

> 💡 **AI Infra 视角**：structural tag 是**函数调用（function calling / tool use）场景**的利器——Agent 应用的核心需求。原理：定义"触发词"（`<function=`）和每个函数的"开始标签 → JSON 参数 schema → 结束标签"。模型一旦输出 `<function=`，后面的内容就**强制**符合对应函数的 JSON schema——保证 Agent 框架解析函数调用永不失败。
> 对比 OpenAI 原生 function calling：它靠模型训练对齐，输出仍可能格式错误；structural tag 是**硬约束**，100% 合法。**"结构化输出保证"是现代 Agent 基础设施（如 OpenAI Responses API、MCP）的底层需求**。

## 离线 API：LLM API

如果你使用 LLM API，在创建 LLM 实例时用 `xgrammar` 或 `llguidance` 指定 `guided_decoding_backend` 来启用引导解码。例如：

```python
from tensorrt_llm import LLM

llm = LLM("nvidia/Llama-3.1-8B-Instruct-FP8", guided_decoding_backend="xgrammar")
```

### JSON Schema

创建带 `json` 字段（指定 JSON schema）的 `GuidedDecodingParams`，用它创建 `SamplingParams`，然后传给 `llm.generate` 或 `llm.generate_async`。或者，JSON schema 可以用 [pydantic](https://docs.pydantic.dev/latest/) 创建。

```python
from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import SamplingParams, GuidedDecodingParams

if __name__ == "__main__":
    llm = LLM("nvidia/Llama-3.1-8B-Instruct-FP8", guided_decoding_backend="xgrammar")

    json_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[\\w]+$"
            },
            "population": {
                "type": "integer"
            },
        },
        "required": ["name", "population"],
    }
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Give me the information of the capital of France in the JSON format.",
        },
    ]
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = llm.generate(
        prompt,
        sampling_params=SamplingParams(max_tokens=256, guided_decoding=GuidedDecodingParams(json=json_schema)),
    )
    print(output.outputs[0].text)
```

输出看起来像：
```txt
{
  "name": "Paris",
  "population": 2145206
}
```

> 💡 **AI Infra 视角**：注意 LLM API 的用法结构：`GuidedDecodingParams(json=...)` 是**采样参数的一部分**（SamplingParams 里），说明引导解码在实现上就是采样阶段的一个约束器（sampling.md 里 logits processor 那一层的近亲）。API 设计启示：**引导解码是"按请求粒度"的能力**——同一个服务里，一个请求要 JSON、另一个请求自由生成，互不影响。

### 正则表达式

创建带 `regex` 字段（指定正则表达式）的 `GuidedDecodingParams`，用它创建 `SamplingParams`，然后传给 `llm.generate` 或 `llm.generate_async`。

```python
from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import SamplingParams, GuidedDecodingParams

if __name__ == "__main__":
    llm = LLM("nvidia/Llama-3.1-8B-Instruct-FP8", guided_decoding_backend="xgrammar")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
    ]
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = llm.generate(
        prompt,
        sampling_params=SamplingParams(max_tokens=256, guided_decoding=GuidedDecodingParams(regex="(Paris|London)")),
    )
    print(output.outputs[0].text)
```

输出看起来像：
```txt
Paris
```

### EBNF 文法

创建带 `grammar` 字段（指定 EBNF 文法）的 `GuidedDecodingParams`，用它创建 `SamplingParams`，然后传给 `llm.generate` 或 `llm.generate_async`。

```python
from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import SamplingParams, GuidedDecodingParams

if __name__ == "__main__":
    llm = LLM("nvidia/Llama-3.1-8B-Instruct-FP8", guided_decoding_backend="xgrammar")

    ebnf_grammar = """root ::= description
city ::= "London" | "Paris" | "Berlin" | "Rome"
description ::= city " is " status
status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful geography bot."
        },
        {
            "role": "user",
            "content": "Give me the information of the capital of France.",
        },
    ]
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = llm.generate(
        prompt,
        sampling_params=SamplingParams(max_tokens=256, guided_decoding=GuidedDecodingParams(grammar=ebnf_grammar)),
    )
    print(output.outputs[0].text)
```

输出看起来像：
```txt
Paris is the capital of France
```

### 结构化标签

创建带 `structural_tag` 字段（指定结构化标签字符串）的 `GuidedDecodingParams`，用它创建 `SamplingParams`，然后传给 `llm.generate` 或 `llm.generate_async`。

结构化标签只由 `xgrammar` 后端支持。它是表达 LLM 输出约束的强大而灵活的工具。完整的教程请看[结构化标签用法](https://xgrammar.mlc.ai/docs/structural_tag/structural_tag_api.html)。下面是一个为 `Llama-3.1-8B-Instruct` 自定义函数调用格式的示例。

```python
import json
from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import SamplingParams, GuidedDecodingParams

if __name__ == "__main__":
    llm = LLM("nvidia/Llama-3.1-8B-Instruct-FP8", guided_decoding_backend="xgrammar")

    tool_get_current_weather = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "the two-letter abbreviation for the state that the city is in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }

    tool_get_current_date = {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date and time for a given timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
                    }
                },
                "required": ["timezone"],
            },
        },
    }

    system_prompt = f"""# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search
You have access to the following functions:
Use the function 'get_current_weather' to: Get the current weather in a given location
{tool_get_current_weather["function"]}
Use the function 'get_current_date' to: Get the current date and time for a given timezone
{tool_get_current_date["function"]}
If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where
start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`
Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query
You are a helpful assistant."""
    user_prompt = "You are in New York. Please get the current date and time, and the weather."
    structural_tag = {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<function="],
            "tags": [
                {
                    "begin": "<function=get_current_weather>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": tool_get_current_weather["function"]["parameters"]
                    },
                    "end": "</function>",
                },
                {
                    "begin": "<function=get_current_date>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": tool_get_current_date["function"]["parameters"]
                    },
                    "end": "</function>",
                },
            ],
        },
    }

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = llm.generate(
        prompt,
        sampling_params=SamplingParams(max_tokens=256, guided_decoding=GuidedDecodingParams(structural_tag=json.dumps(structural_tag))),
    )
    print(output.outputs[0].text)
```

输出看起来像：
```txt
<function=get_current_date>{"timezone": "America/New_York"}</function>
<function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function>
```

> 💡 **AI Infra 视角**：读完这篇你应能总结引导解码的完整使用姿势：在线服务 = `guided_decoding_backend` 配置 + `response_format` 请求参数；离线 = 构造时指定 backend + `GuidedDecodingParams`。支持的约束类型由后端决定（JSON schema 通用，structural tag 只有 XGrammar）。**面试/工作中"怎么保证 LLM 输出格式合法"的标准答案就是引导解码**——它比 prompt 指令（"please output JSON"）+ 事后重试（retry）可靠得多。
