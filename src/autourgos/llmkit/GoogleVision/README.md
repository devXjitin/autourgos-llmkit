# Autourgos LLM Kit — Google Gemini Vision Provider

The **Google Gemini Vision** provider for Autourgos LLM Kit enables multimodal interaction with Google's Gemini models, allowing you to process images alongside text prompts.

**Official API Documentation:** [Google Gemini API Reference](https://ai.google.dev/api)

---

## Supported Models (Multimodal)

Most modern Gemini models are natively multimodal.

| Model Name | Model ID | Multimodal Capabilities |
|---|---|---|
| **Gemini 3.1 Pro** *(Preview)* | `gemini-3.1-pro-preview` | Advanced intelligence, complex visual reasoning |
| **Gemini 3 Pro** *(Preview)* | `gemini-3-pro-preview` | State-of-the-art reasoning over visual inputs |
| **Gemini 3 Flash** *(Preview)* | `gemini-3-flash-preview` | Fast image processing and analysis |
| **Gemini 2.5 Pro** | `gemini-2.5-pro` | High performance on visual QA and document understanding |
| **Gemini 2.5 Flash** | `gemini-2.5-flash` | Cost-effective for high-volume image tasks |

---

## Pricing (Paid Tier)

*Multimodal inputs (images, video) are generally charged per token or at a flat rate per image equivalent depending on the model.*

| Model | Input Price (Text/Image/Video) | Output Price |
|---|---|---|
| **Gemini 3.1 Pro** | $2.00 - $4.00 / 1M tokens | $12.00 - $18.00 / 1M tokens |
| **Gemini 3 Pro** | $2.00 - $4.00 / 1M tokens | $12.00 - $18.00 / 1M tokens |
| **Gemini 3 Flash** | $0.50 / 1M tokens | $3.00 / 1M tokens |
| **Gemini 2.5 Pro** | $1.25 - $2.50 / 1M tokens | $10.00 - $15.00 / 1M tokens |
| **Gemini 2.5 Flash** | $0.30 / 1M tokens | $2.50 / 1M tokens |

*See [Official Pricing](https://ai.google.dev/pricing) for detailed breakdown of image token costs.*

---

## Installation & Configuration

Install the provider SDK (same as standard Google provider):

```bash
pip install autourgos-llmkit[google]
```

Ensure Pillow is installed for image handling:

```bash
pip install Pillow
```

Set your API key:

```bash
export GOOGLE_API_KEY="your-api-key"
```

---

## Usage

### Vision API (Images + Text)

```python
from autourgos.llmkit.GoogleVision import google_vision_llm

# Pass image paths or PIL Image objects
response = google_vision_llm(
    prompt="Describe this image in detail.",
    images=["path/to/image.jpg"],
    model="gemini-3-flash-preview"
)

print(response["content"])
```

### Streaming with Images

```python
from autourgos.llmkit.GoogleVision import google_vision_llm_stream

# Streaming response for visual question answering
for chunk in google_vision_llm_stream(
    prompt="What is happening in this photo?",
    images=["path/to/photo.png"], 
    model="gemini-3-flash-preview"
):
    print(chunk, end="", flush=True)
```

### Class-Based Usage

```python
from autourgos.llmkit.GoogleVision import GoogleVisionLLM

llm = GoogleVisionLLM(model="gemini-3-flash-preview")
response = llm.generate_response(
    prompt="Extract text from this image.",
    images="path/to/document.jpg"
)
print(response)
```

---

## Parameters

Both the function `google_vision_llm` and the class `GoogleVisionLLM` accept the following configuration parameters:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `images` | `Union[str, List[str], Any, List[Any]]` | Yes (for vision) | - | Image input(s). Can be file path(s) or PIL Image object(s). |
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

- `GoogleLLMImportError`: The `google-generativeai` SDK or `Pillow` library is not installed.
- `GoogleLLMAPIError`: The SDK threw an API error (e.g. `google.api_core.exceptions.GoogleAPIError`).
- `GoogleLLMResponseError`: The response format was totally unexpected or missing parts.
