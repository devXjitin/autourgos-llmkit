# Autourgos LLM Kit — Ollama Local Provider

The **Ollama Local** provider for Autourgos LLM Kit connects directly to a local, self-hosted instance of Ollama. It enables completely private, secure, offline AI inference with zero data leaving your machine.

**Official Documentation:** [Ollama Local Server Docs](https://ollama.ai/docs)

---

## 🚀 Supported Models

You can run any model available in the [Ollama Library](https://ollama.ai/library). You must `ollama pull <model>` before making API requests to it. Common examples:

| Model Name | Model ID | Key Capabilities |
|---|---|---|
| **Llama 3.3 (70B/8B)** | `llama3.3` | State of the art open-weight model |
| **Gemma 2 (27B/9B/2B)** | `gemma2` | Lightweight models built by Google |
| **Phi-4** | `phi4` | Microsoft's highly efficient reasoning model |
| **DeepSeek R1** | `deepseek-r1` | Powerful reasoning, locally served |
| **Mistral** | `mistral` | Standard fast and versatile model |

---

## 📦 Installation & Configuration

Install the provider SDK as an optional extra (requires the official `ollama` Python client):

```bash
pip install autourgos-llmkit[ollama]
```

No API key is required. Ensure your local Ollama server is running (defaults to `http://localhost:11434`).

---

## 💻 Usage

### Function-based API (Metadata included)

```python
from autourgos.llmkit.Ollamalocal import ollama_local_llm

response = ollama_local_llm(
    prompt="Explain quantum physics locally.",
    model="llama3.3",
    host="http://localhost:11434",
    temperature=0.7
)

print(response["content"])  # The generated text
print(response["usage"])    # Token usage if available
```

### Class-based API (Text content only)

```python
from autourgos.llmkit.Ollamalocal import OllamaLocalLLM

llm = OllamaLocalLLM(
    model="phi4",
    host="http://localhost:11434"
)

# Returns only the generated string
text = llm.generate_response("Write a bash script.")
print(text)
```

### Streaming API

```python
from autourgos.llmkit.Ollamalocal import ollama_local_llm_stream

for chunk in ollama_local_llm_stream(
    prompt="Generate a long text sequence...",
    model="llama3.3"
):
    print(chunk, end='', flush=True)
```

---

## ⚙️ Parameters

Both the function `ollama_local_llm` and the class `OllamaLocalLLM` accept the following configuration parameters:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | `str` | Yes | - | The exact local model tag to use |
| `host` | `str` | No | `"http://localhost:11434"` | Host address and port of your Ollama runtime |
| `temperature` | `float` | No | `None` | Controls randomness (0.0 to 1.0) |
| `top_p` | `float` | No | `None` | Nucleus sampling probability limit |
| `max_tokens` | `int` | No | `None` | Maximum number of output tokens (Note: Ollama treats this loosely) |
| `max_retries` | `int` | No | `3` | Number of times to retry failed requests (useful if model is slow to load into VRAM) |
| `timeout` | `float`| No | `30.0` | Request timeout in seconds |
| `backoff_factor`| `float`| No | `0.5` | Exponential backoff base for retries |

---

## ⚠️ Error Handling

The module provides typed exceptions under the base `OllamaLocalLLMError`:

- `OllamaLocalLLMImportError`: The `ollama` SDK is not installed.
- `OllamaLocalLLMAPIError`: The SDK threw an API error (usually connection refused because Ollama isn't running).
- `OllamaLocalLLMResponseError`: The response format was unexpected.
