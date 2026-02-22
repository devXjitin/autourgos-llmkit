# Autourgos LLM Kit — OpenAI Provider

The **OpenAI** provider for Autourgos LLM Kit enables seamless integration with OpenAI's frontier models. It supports both standard conversational models (GPT-4o) and advanced reasoning models (o3, o1).

**Official API Documentation:** [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

---

## Supported Models

| Model Name | Model ID | Max Context | Key Capabilities |
|---|---|---|---|
| **GPT-5.2** | `gpt-5.2` | High | Best for coding and agentic tasks |
| **GPT-5.2 Pro** | `gpt-5.2-pro` | High | Smartest and most precise model |
| **GPT-5 Mini** | `gpt-5-mini` | High | Faster, cheaper version of GPT-5 |
| **o4-mini** | `o4-mini` | 200k | Fast, cost-efficient reasoning model |
| **o3** | `o3` | 200k | Reasoning model for complex tasks |
| **GPT-4.1** | `gpt-4.1` | 128k | Smartest non-reasoning model |
| **GPT-4o** | `gpt-4o` | 128k | Fast, multimodal, flexible |

---

## Pricing

*Prices are per 1 million tokens. See [Official Pricing](https://openai.com/api/pricing/) for details.*

| Model | Input Price | Output Price | Cached Input |
|---|---|---|---|
| **GPT-5.2** | $1.75 | $14.00 | $0.175 |
| **GPT-5.2 Pro** | $21.00 | $168.00 | - |
| **GPT-5 Mini** | $0.25 | $2.00 | $0.025 |
| **o4-mini** | $4.00 | $16.00 | $1.00 |
| **GPT-4.1** | $3.00 | $12.00 | $0.75 |
| **GPT-4.1 Mini** | $0.80 | $3.20 | $0.20 |

---

## Installation & Configuration

Install the provider SDK as an optional extra:

```bash
pip install autourgos-llmkit[openai]
```

Set your API key using the environment variable:

```bash
export OPENAI_API_KEY="your-api-key"
```

---

## Usage

### Function-based API (Metadata included)

```python
from autourgos.llmkit.Openai import openai_llm

response = openai_llm(
    prompt="Explain P vs NP.",
    model="o4-mini",
    temperature=1.0 # Reasoning models typically use fixed temperature
)

print(response["content"])  # The generated text
print(response["usage"])    # Token usage dictionary ({'input_tokens': X, 'output_tokens': Y, 'total_tokens': Z})
```

### Class-based API (Text content only)

```python
from autourgos.llmkit.Openai import OpenAILLM

llm = OpenAILLM(
    model="gpt-5.2",
    temperature=0.2
)

# Returns only the generated string
text = llm.generate_response("Write a python script.")
print(text)
```

### Streaming API

```python
from autourgos.llmkit.Openai import openai_llm_stream

for chunk in openai_llm_stream(
    prompt="Generate a long explainer...",
    model="gpt-5-mini"
):
    print(chunk, end='', flush=True)
```

---

## Parameters

Both the function `openai_llm` and the class `OpenAILLM` accept the following configuration parameters:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | `str` | Yes | - | The OpenAI model ID to use |
| `api_key` | `str` | No | `None` | Explicit API key (overrides `OPENAI_API_KEY` env var) |
| `temperature` | `float` | No | `None` | Controls randomness (0.0 to 2.0). *Note: `o1` and `o3-mini` only support temperature=1.* |
| `top_p` | `float` | No | `None` | Nucleus sampling probability limit |
| `max_tokens` | `int` | No | `None` | Maximum number of output tokens |
| `frequency_penalty`| `float`| No | `None` | Penalizes new tokens based on existing frequency (-2.0 to 2.0) |
| `presence_penalty` | `float`| No | `None` | Penalizes new tokens based on presence (-2.0 to 2.0) |
| `max_retries` | `int` | No | `3` | Number of times to retry failed requests (non-streaming only) |
| `timeout` | `float`| No | `30.0` | Request timeout in seconds |
| `backoff_factor`| `float`| No | `0.5` | Exponential backoff base for retries |

---

## Error Handling

The module provides typed exceptions under the base `OpenAILLMError`:

- `OpenAILLMImportError`: The `openai` SDK is not installed.
- `OpenAILLMAPIError`: The SDK threw an API error (authentication, quota limits).
- `OpenAILLMResponseError`: The response format was missing choices or text blocks.
