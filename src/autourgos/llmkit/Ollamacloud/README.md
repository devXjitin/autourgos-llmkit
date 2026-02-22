# Autourgos LLM Kit — Ollama Cloud Provider

The **Ollama Cloud** provider for Autourgos LLM Kit connects to Ollama's hosted API. It provides a convenient way to access powerful open-source models like Llama, Mistral, and Gemma via an OpenAI-compatible interface, without managing local infrastructure.

**Official Documentation:** [Ollama API Documentation](https://ollama.ai/docs)

---

## Supported Models

Models must be available on your target Ollama Cloud host. Common cloud-native models include:

| Model Name | Model ID | Key Capabilities |
|---|---|---|
| **Llama 3.3** | `llama3.3` | High-quality general instruction tuned |
| **Qwen 3.5** | `qwen3.5` | Vision-language, sparse mixture-of-experts |
| **DeepSeek V3.2** | `deepseek-v3.2` | High efficiency, superior reasoning |
| **Phi-4** | `phi4` | Small, highly efficient reasoning |
| **Mistral Large** | `mistral-large` | Strong multilingual and coding capabilities |
| **Gemma 2** | `gemma2` | Lightweight, open-weight research model |

---

## Pricing

Ollama Cloud offers tiered subscription plans:

| Plan | Price | Features |
|---|---|---|
| **Free** | $0 / mo | Access to open models, limited cloud usage |
| **Pro** | $20 / mo | Run multiple cloud models, more usage, 3 private models |
| **Max** | $100 / mo | Run 5+ cloud models, 5x usage, 5 private models |

*See [Official Pricing](https://ollama.com/pricing) for details.*

---

## Installation & Configuration

Install the provider SDK as an optional extra. Because this uses the official Ollama Python client:

```bash
pip install autourgos-llmkit[ollama]
```

Set your API key using the environment variable:

```bash
export OLLAMA_API_KEY="your-api-key"
```

---

## Usage

### Function-based API (Metadata included)

```python
from autourgos.llmkit.Ollamacloud import ollama_cloud_llm

response = ollama_cloud_llm(
    prompt="Explain black holes.",
    model="llama3.3",
    # host="https://api.ollama.com", # Optional: Specify if using a custom endpoint
    temperature=0.7
)

print(response["content"])  # The generated text
print(response["usage"])    # Token usage dictionary ({'input_tokens': X, 'output_tokens': Y, 'total_tokens': Z})
```

### Class-based API (Text content only)

```python
from autourgos.llmkit.Ollamacloud import OllamaCloudLLM

llm = OllamaCloudLLM(
    model="phi4",
    temperature=0.2
)

# Returns only the generated string
text = llm.generate_response("Write a bash script.")
print(text)
```

### Streaming API

```python
from autourgos.llmkit.Ollamacloud import ollama_cloud_llm_stream

for chunk in ollama_cloud_llm_stream(
    prompt="Generate a long text sequence...",
    model="deepseek-v3.2"
):
    print(chunk, end='', flush=True)
```

---

## Parameters

Both the function `ollama_cloud_llm` and the class `OllamaCloudLLM` accept the following configuration parameters:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | `str` | Yes | - | The Ollama model ID to use (e.g., `llama3.3`) |
| `host` | `str` | No | `"https://ollama.com"` | Explicit host URL. |
| `api_key` | `str` | No | `None` | Explicit API key (overrides `OLLAMA_API_KEY` env var) |
| `temperature` | `float` | No | `None` | Controls randomness (0.0 to 1.0) |
| `top_p` | `float` | No | `None` | Nucleus sampling probability limit |
| `max_tokens` | `int` | No | `None` | Maximum number of output tokens |
| `max_retries` | `int` | No | `3` | Number of times to retry failed requests (non-streaming only) |
| `timeout` | `float`| No | `30.0` | Request timeout in seconds |
| `backoff_factor`| `float`| No | `0.5` | Exponential backoff base for retries |

---

## Error Handling

The module provides typed exceptions under the base `OllamaCloudLLMError`:

- `OllamaCloudLLMImportError`: The `ollama` SDK is not installed.
- `OllamaCloudLLMAPIError`: The SDK threw an API error (authentication, bad gateway).
- `OllamaCloudLLMResponseError`: The response format was unexpected.
