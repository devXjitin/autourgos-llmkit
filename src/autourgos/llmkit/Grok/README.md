# Autourgos LLM Kit — xAI Grok Provider

The **xAI Grok** provider connects to Grok API endpoints via the standard OpenAI-compatible client. Grok models offer powerful reasoning capabilities, massive context windows, and specialized real-time knowledge.

**Official API Documentation:** [xAI Grok API Reference](https://docs.x.ai/)

---

## Supported Models

| Model Name | Model ID | Capabilities |
|---|---|---|
| **Grok 3** | `grok-3` | Strong reasoning model, successor to Grok 2 |
| **Grok 3 Mini** | `grok-3-mini` | Efficient reasoning model, balanced performance/cost |
| **Grok 2** | `grok-2-latest` | Previous stable flagship model (aliased to latest version) |
| **Grok 2 Vision** | `grok-2-vision-latest` | Multimodal support (images) |
| **Grok Beta** | `grok-beta` | Preview of upcoming features (aliased to latest beta) |

*Note: `grok-4` (Reasoning Model) is available in early access and may have restrictions on certain parameters (e.g., `temperature`, `frequency_penalty`).*

---

## Pricing

*Prices are per 1 million tokens unless otherwise noted. See [Official Pricing](https://docs.x.ai/developers/models) for details.*

| Item | Cost |
|---|---|
| **Input Tokens** | Varies by model |
| **Output Tokens** | Varies by model |
| **Web Search Tool** | $5.00 / 1k searches |
| **X Search Tool** | $5.00 / 1k searches |
| **Collections Search** | $2.50 / 1k searches |
| **Voice Agent** | $0.05 / minute |
| **Batch API** | 50% discount on standard token rates |

---

## Installation & Configuration

Install the provider SDK as an optional extra (we use the exact same SDK as OpenAI):

```bash
pip install autourgos-llmkit[openai]
```

Set your API key using the environment variable:

```bash
export XAI_API_KEY="your-api-key"
```

---

## Usage

### Function-based API (Metadata included)

```python
from autourgos.llmkit.Grok import grok_llm

response = grok_llm(
    prompt="Explain quantum mechanics.",
    model="grok-3",
    temperature=0.7
)

print(response["content"])  # The generated text
print(response["usage"])    # Token usage dictionary ({'input_tokens': X, 'output_tokens': Y, 'total_tokens': Z})
```

### Class-based API (Text content only)

```python
from autourgos.llmkit.Grok import GrokLLM

llm = GrokLLM(
    model="grok-3-mini",
    temperature=0.5
)

# Returns only the generated string
text = llm.generate_response("Write a React component.")
print(text)
```

### Streaming API

```python
from autourgos.llmkit.Grok import grok_llm_stream

for chunk in grok_llm_stream(
    prompt="Generate a long text sequence...",
    model="grok-2-latest"
):
    print(chunk, end='', flush=True)
```

---

## Parameters

Both the function `grok_llm` and the class `GrokLLM` accept the following configuration parameters:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | `str` | Yes | - | The Grok model ID to use (e.g., `grok-3`) |
| `api_key` | `str` | No | `None` | Explicit API key (overrides `XAI_API_KEY` env var) |
| `temperature` | `float` | No | `None` | Controls randomness (0.0 to 2.0). *Not supported by reasoning models like `grok-4`* |
| `top_p` | `float` | No | `None` | Nucleus sampling probability limit |
| `max_tokens` | `int` | No | `None` | Maximum number of output tokens |
| `frequency_penalty`| `float`| No | `None` | Penalizes new tokens based on existing frequency (-2.0 to 2.0). *Not supported by reasoning models* |
| `presence_penalty` | `float`| No | `None` | Penalizes new tokens based on presence (-2.0 to 2.0). *Not supported by reasoning models* |
| `max_retries` | `int` | No | `3` | Number of times to retry failed requests (non-streaming only) |
| `timeout` | `float`| No | `30.0` | Request timeout in seconds |
| `backoff_factor`| `float`| No | `0.5` | Exponential backoff base for retries |

---

## Error Handling

The module provides typed exceptions under the base `GrokLLMError`:

- `GrokLLMImportError`: The `openai` SDK is not installed.
- `GrokLLMAPIError`: The SDK threw an API error.
- `GrokLLMResponseError`: The response format was unexpected.
