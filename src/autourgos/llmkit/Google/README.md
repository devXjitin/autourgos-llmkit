# Autourgos LLM Kit — Google Gemini Provider

The **Google Gemini** provider for Autourgos LLM Kit allows integration with Google's natively multimodal flagship AI models. Gemini models handle text, audio, images, video, and code natively, offering massive context windows (up to 2M tokens) and deep reasoning capabilities.

**Official API Documentation:** [Google Gemini API Reference](https://ai.google.dev/api)

---

## Supported Models

| Model Name | Model ID | Max Input Tokens | Max Output Tokens | Key Capabilities |
|---|---|---|---|---|
| **Gemini 3.1 Pro** *(Preview)* | `gemini-3.1-pro-preview` | 2,000,000 | 65,536 | Advanced intelligence, complex problem-solving, agentic capabilities |
| **Gemini 3 Pro** *(Preview)* | `gemini-3-pro-preview` | 2,000,000 | 65,536 | State-of-the-art reasoning, multimodal understanding |
| **Gemini 3 Flash** *(Preview)* | `gemini-3-flash-preview` | 1,000,000 | 65,536 | Frontier-class performance, high speed, low cost |
| **Gemini 2.5 Pro** | `gemini-2.5-pro` | 2,000,000 | 65,536 | Advanced reasoning for complex tasks |
| **Gemini 2.5 Flash** | `gemini-2.5-flash` | 1,000,000 | 65,536 | Price-performance optimized, low-latency reasoning |
| **Gemini 2.5 Flash-Lite** | `gemini-2.5-flash-lite` | 1,000,000 | 65,536 | Fastest and most budget-friendly multimodal model |

---

## Pricing (Paid Tier)

*Prices are per 1 million tokens unless otherwise noted. A free tier is available for most models with lower rate limits.*

| Model | Input Price (Text/Image/Video) | Output Price | Context Caching (Storage) |
|---|---|---|---|
| **Gemini 3.1 Pro** | $2.00 (≤200k) / $4.00 (>200k) | $12.00 (≤200k) / $18.00 (>200k) | $4.50 / hr |
| **Gemini 3 Pro** | $2.00 (≤200k) / $4.00 (>200k) | $12.00 (≤200k) / $18.00 (>200k) | $4.50 / hr |
| **Gemini 3 Flash** | $0.50 | $3.00 | $1.00 / hr |
| **Gemini 2.5 Pro** | $1.25 (≤200k) / $2.50 (>200k) | $10.00 (≤200k) / $15.00 (>200k) | $4.50 / hr |
| **Gemini 2.5 Flash** | $0.30 | $2.50 | $1.00 / hr |
| **Gemini 2.5 Flash-Lite** | $0.10 | $0.40 | $1.00 / hr |

*Note: Audio inputs typically cost more (e.g., $1.00 for Flash models).*
*See [Official Pricing](https://ai.google.dev/pricing) for full details.*

---

## Installation & Configuration

Install the provider SDK as an optional extra:

```bash
pip install autourgos-llmkit[google]
```

Set your API key using the environment variable:

```bash
export GOOGLE_API_KEY="your-api-key"
```

---

## Usage

### Function-based API (Metadata included)

```python
from autourgos.llmkit.Google import google_llm

response = google_llm(
    prompt="Explain quantum computing.",
    model="gemini-3-flash-preview",
    temperature=0.7
)

print(response["content"])  # The generated text
print(response["usage"])    # Token usage dictionary ({'input_tokens': X, 'output_tokens': Y, 'total_tokens': Z})
```

### Class-based API (Text content only)

```python
from autourgos.llmkit.Google import GoogleLLM

llm = GoogleLLM(
    model="gemini-3-flash-preview",
    temperature=0.2
)

# Returns only the generated string
text = llm.generate_response("Write a Python script to interact with BigQuery.")
print(text)
```

### Streaming API

```python
from autourgos.llmkit.Google import google_llm_stream

for chunk in google_llm_stream(
    prompt="Write a long explanatory article...",
    model="gemini-3-flash-preview"
):
    print(chunk, end='', flush=True)
```

---

## Parameters

Both the function `google_llm` and the class `GoogleLLM` accept the following configuration parameters:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | `str` | Yes | - | The Gemini model ID to use |
| `api_key` | `str` | No | `None` | Explicit API key (overrides `GOOGLE_API_KEY` env var) |
| `temperature` | `float` | No | `None` | Controls randomness (0.0 to 2.0 for Gemini) |
| `top_p` | `float` | No | `None` | Nucleus sampling probability limit |
| `top_k` | `int` | No | `None` | Top-K sampling token limit |
| `max_tokens` | `int` | No | `None` | Maximum number of output tokens |
| `max_retries` | `int` | No | `3` | Number of times to retry failed requests (non-streaming only) |
| `timeout` | `float`| No | `30.0` | Request timeout in seconds |
| `backoff_factor`| `float`| No | `0.5` | Exponential backoff base for retries |

---

## Error Handling

The module provides typed exceptions under the base `GoogleLLMError`:

- `GoogleLLMImportError`: The `google-generativeai` SDK is not installed.
- `GoogleLLMAPIError`: The SDK threw an API error (e.g. `google.api_core.exceptions.GoogleAPIError`).
- `GoogleLLMResponseError`: The response format was totally unexpected or missing parts.
