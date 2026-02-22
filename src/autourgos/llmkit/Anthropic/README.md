# Autourgos LLM Kit — Anthropic Provider

The **Anthropic** provider for Autourgos LLM Kit enables seamless integration with the Claude family of models. Developed by [Anthropic](https://www.anthropic.com/), Claude models are known for high capability in reasoning, coding, and following complex instructions, built with safety via Constitutional AI.

**Official API Documentation:** [Anthropic API Reference](https://docs.anthropic.com/en/docs/)

---

## 🚀 Supported Models (2025 releases)

| Model Name | Model ID | Max Input Tokens | Max Output Tokens | Key Capabilities |
|---|---|---|---|---|
| **Claude Opus 4.5** | `claude-opus-4.5-20251124` | 200,000 | 32,000 | Ultimate reasoning, deep research, coding |
| **Claude Sonnet 4.5** | `claude-sonnet-4.5-20250929` | 200,000 | 64,000 | Frontier coding, agentic orchestration |
| **Claude Haiku 4.5** | `claude-haiku-4.5-20251015` | 200,000 | 8,192 | Ultra-fast responses, cost-efficiency |
| **Claude 3.5 Sonnet** | `claude-3-5-sonnet-20241022` | 200,000 | 8,192 | General-purpose high performance, vision |
| **Claude 3.5 Haiku** | `claude-3-5-haiku-20241022` | 200,000 | 8,192 | Low latency, high throughput |

---

## 📦 Installation & Configuration

Install the provider SDK as an optional extra:

```bash
pip install autourgos-llmkit[anthropic]
```

Set your API key using the environment variable:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

---

## 💻 Usage

### Function-based API (Metadata included)

```python
from autourgos.llmkit.Anthropic import anthropic_llm

response = anthropic_llm(
    prompt="Explain quantum computing in one sentence.",
    model="claude-sonnet-4.5-20250929",
    max_tokens=1024,
    temperature=0.7
)

print(response["content"])  # The generated text
print(response["usage"])    # Token usage dictionary ({'input_tokens': X, 'output_tokens': Y, 'total_tokens': Z})
```

### Class-based API (Text content only)

```python
from autourgos.llmkit.Anthropic import AnthropicLLM

llm = AnthropicLLM(
    model="claude-opus-4.5-20251124",
    max_tokens=2048
)

# Returns only the generated string
text = llm.generate_response("Write a python script to reverse a linked list.")
print(text)
```

### Streaming API

```python
from autourgos.llmkit.Anthropic import anthropic_llm_stream

for chunk in anthropic_llm_stream(
    prompt="Write a very long story...",
    model="claude-haiku-4.5-20251015",
    max_tokens=4000
):
    print(chunk, end='', flush=True)
```

---

## ⚙️ Parameters

Both the function `anthropic_llm` and the class `AnthropicLLM` accept the following configuration parameters:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | `str` | Yes | - | The Claude model ID to use |
| `max_tokens` | `int` | Yes* | 1024 | Maximum number of tokens to generate. Anthropic strictly requires this. |
| `api_key` | `str` | No | `None` | Explicit API key (overrides `ANTHROPIC_API_KEY` env var) |
| `temperature` | `float` | No | `None` | Controls randomness (0.0 to 1.0) |
| `top_p` | `float` | No | `None` | Nucleus sampling probability limit |
| `top_k` | `int` | No | `None` | Top-K sampling token limit |
| `max_retries` | `int` | No | `3` | Number of times to retry failed requests (non-streaming only) |
| `timeout` | `float`| No | `30.0` | Request timeout in seconds |
| `backoff_factor`| `float`| No | `0.5` | Exponential backoff base for retries |

*\* Default fallback value of `1024` is applied if not provided.*

---

## ⚠️ Error Handling

The module provides typed exceptions under the base `AnthropicLLMError`:

- `AnthropicLLMImportError`: The `anthropic` SDK is not installed.
- `AnthropicLLMAPIError`: The SDK threw an API error (authentication, limits, server errors).
- `AnthropicLLMResponseError`: The response format was totally unexpected or missing text blocks.
