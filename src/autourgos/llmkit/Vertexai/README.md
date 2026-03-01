# Google Vertex AI Provider

Google [Vertex AI](https://cloud.google.com/vertex-ai) is a fully managed machine learning platform that provides access to **all** models available on the Vertex AI Model Garden — including first-party Google models (Gemini) and third-party models (Anthropic Claude, Meta Llama, Mistral, Moonshot Kimi, DeepSeek, and more) — through a single, unified OpenAI-compatible endpoint. **Zero pip dependencies.** This provider uses only Python built-in modules (`urllib`, `json`, `subprocess`). Authentication is handled via `gcloud auth print-access-token` or a manually supplied Bearer token.

## Supported Models

The full list of models is available in the [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden). Here are some popular examples:

| Model Name | Model ID | Input Types | Output Type | Max Input Tokens | Max Output Tokens | Key Capabilities | Best Use Case |
|-----------|----------|-------------|-------------|------------------|-------------------|------------------|---------------|
| **Gemini 2.5 Pro** | `google/gemini-2.5-pro-preview-06-05` | Text, Image, Audio, Video | Text | 1M+ | 65k | Advanced reasoning, multimodal | Complex agents |
| **Gemini 2.0 Flash** | `google/gemini-2.0-flash` | Text, Image, Audio, Video | Text | 1M+ | 8k | Fast, efficient multimodal | High-throughput tasks |
| **DeepSeek R1** | `deepseek/deepseek-r1` | Text | Text | 128k | 128k | Strong reasoning (MaaS) | Complex coding & math |
| **DeepSeek V3** | `deepseek/deepseek-v3` | Text | Text | 128k | 128k | Efficient general purpose (MaaS) | Chat & generation |
| **Mistral Large 2** | `mistralai/mistral-large-2411` | Text | Text | 128k | 128k | Multilingual, coding | Enterprise tasks |
| **Kimi K2 Thinking** | `moonshotai/kimi-k2-thinking` | Text | Text | 128k | 128k | Deep thinking & reasoning | Logic puzzles, planning |
| **Gemma 2** | `google/gemma-2-27b-it` | Text | Text | 8k | 8k | Open weights, efficient | Local/Cloud serving |
| **Claude 3.5 Sonnet** | `anthropic/claude-3-5-sonnet` | Text, Image | Text | 200k | 8k | Coding, nuances | Complex instructions |

*Note: For Partner Models (DeepSeek, Mistral, Kimi), ensure you have enabled the model in Vertex AI Model Garden. Some models may require a specific regional endpoint.*

## Installation

No extra pip packages are needed. Just ensure the **Google Cloud SDK** (`gcloud`) is installed and authenticated:

```bash
# Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## Configuration

### Authentication (pick one)

```python
import os

# Option 1 — Environment variable
os.environ["VERTEX_AI_ACCESS_TOKEN"] = "ya29.xxxxx..."

# Option 2 — Automatic via gcloud CLI (no code needed)
# The provider will run `gcloud auth print-access-token` automatically.

# Option 3 — Pass token explicitly
llm = VertexAILLM(model="...", access_token="ya29.xxxxx...")
```

### Project ID (pick one)

```python
import os

# Option 1 — Environment variable
os.environ["VERTEX_AI_PROJECT_ID"] = "my-gcp-project"
# or
os.environ["GOOGLE_CLOUD_PROJECT"] = "my-gcp-project"

# Option 2 — Pass explicitly
llm = VertexAILLM(model="...", project_id="my-gcp-project")
```

## Usage

### Class-based API

#### Standard Response

```python
from autourgos.llmkit.Vertexai import VertexAILLM

llm = VertexAILLM(
    model="moonshotai/kimi-k2-thinking",
    project_id="gen-lang-client-0152852093",
    region="us-central1",         # Partner models often require specific regions (e.g. us-central1)
    temperature=0.6,
    max_tokens=8192
)

# Returns only the text content (for backward compatibility)
response = llm.generate_response("Hello, tell me about yourself!")
print(response)
```

#### Streaming Response with DeepSeek

```python
from autourgos.llmkit.Vertexai import VertexAILLM

llm = VertexAILLM(
    model="deepseek/deepseek-r1",
    project_id="your-project-id",
    region="us-central1"
)

# Stream the response in real-time
for chunk in llm.generate_response_stream("Explain quantum entanglement"):
    print(chunk, end='', flush=True)
```

#### Using a Custom Endpoint ID

If you have deployed a model to a specific endpoint (not using the shared MaaS API), you can specify `endpoint_id`.

```python
llm = VertexAILLM(
    model="meta/llama3-405b-instruct-maas",
    project_id="your-project-id",
    region="us-central1",
    endpoint_id="1234567890" # Your specific Vertex AI Endpoint ID
)
```

### Function-based API

#### Standard Response

```python
from autourgos.llmkit.Vertexai import vertexai_llm

response = vertexai_llm(
    prompt="Explain quantum computing",
    model="anthropic/claude-sonnet-4@20250514",
    project_id="gen-lang-client-01528520XX",
    region="global",
    temperature=0.7,
    max_tokens=2048
)

# Response is a dictionary with metadata
print(response["content"])        # The generated text
print(response["model"])          # Model used
print(response["usage"])          # Token usage information

# Example output:
# {
#     "content": "Quantum computing is...",
#     "model": "anthropic/claude-sonnet-4@20250514",
#     "usage": {
#         "input_tokens": 15,
#         "output_tokens": 450,
#         "total_tokens": 465
#     }
# }
```

#### Streaming Response

```python
from autourgos.llmkit.Vertexai import vertexai_llm_stream

# Stream text chunks as they're generated
for chunk in vertexai_llm_stream(
    prompt="Explain machine learning",
    model="meta/llama-4-maverick-17b-128e-instruct-maas",
    project_id="gen-lang-client-01528520XX",
    region="us-central1",
    max_tokens=256
):
    print(chunk, end='', flush=True)
```

## Parameters

### `vertexai_llm()` and `VertexAILLM` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Model identifier on Vertex AI Model Garden |
| `project_id` | str | None | GCP project ID (falls back to env var) |
| `region` | str | "global" | GCP region |
| `endpoint` | str | None | Override API hostname (auto-derived from region) |
| `access_token` | str | None | Bearer token (falls back to env var / gcloud CLI) |
| `temperature` | float | None | Randomness (0.0-2.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `max_tokens` | int | None | Maximum output tokens |
| `max_retries` | int | 3 | Retry attempts (non-streaming only) |
| `timeout` | float | 60.0 | Request timeout (seconds) |
| `backoff_factor` | float | 0.5 | Exponential backoff base (non-streaming only) |

### `vertexai_llm_stream()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Model identifier on Vertex AI Model Garden |
| `project_id` | str | None | GCP project ID (falls back to env var) |
| `region` | str | "global" | GCP region |
| `endpoint` | str | None | Override API hostname (auto-derived from region) |
| `access_token` | str | None | Bearer token (falls back to env var / gcloud CLI) |
| `temperature` | float | None | Randomness (0.0-2.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `max_tokens` | int | None | Maximum output tokens |
| `timeout` | float | 60.0 | Request timeout (seconds) |

**Note**: Streaming functions do not support `max_retries` or `backoff_factor` parameters.

### Return Values

#### `vertexai_llm()` Function

Returns a dictionary with the following structure:

```python
{
    "content": str,              # The generated text
    "model": str,                # Model identifier used
    "usage": {
        "input_tokens": int,     # Tokens in the input prompt (may be None)
        "output_tokens": int,    # Tokens in the generated output (may be None)
        "total_tokens": int      # Total tokens used (may be None)
    }
}
```

#### `VertexAILLM.generate_response()` Method

Returns only the generated text as a string (for backward compatibility).

#### `vertexai_llm_stream()` and `VertexAILLM.generate_response_stream()`

Yields text chunks as strings. Token usage information is not available in streaming mode.

## Regions & Endpoints

| Region | Endpoint |
|--------|----------|
| `global` (default) | `aiplatform.googleapis.com` |
| `us-central1` | `us-central1-aiplatform.googleapis.com` |
| `europe-west4` | `europe-west4-aiplatform.googleapis.com` |
| `asia-southeast1` | `asia-southeast1-aiplatform.googleapis.com` |
| Custom | Pass `endpoint="your-custom-host.example.com"` |

## Equivalent curl Command

The provider is equivalent to this curl command:

```bash
ENDPOINT="aiplatform.googleapis.com"
REGION="global"
PROJECT_ID="gen-lang-client-0152852093"

curl \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://${ENDPOINT}/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/openapi/chat/completions" \
  -d '{
    "model": "moonshotai/kimi-k2-thinking-maas",
    "stream": true,
    "max_tokens": 8192,
    "temperature": 0.6,
    "top_p": 0.95,
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]
  }'
```

## Error Handling

```python
from autourgos.llmkit.Vertexai import (
    vertexai_llm,
    VertexAILLM,
    VertexAILLMError,
    VertexAILLMAuthError,
    VertexAILLMAPIError,
    VertexAILLMResponseError
)

try:
    # Function-based: Returns dict with metadata
    response = vertexai_llm(
        prompt="Hello!",
        model="google/gemini-2.0-flash",
        project_id="my-project"
    )
    print(f"Content: {response['content']}")
    print(f"Tokens used: {response['usage']['total_tokens']}")
    
    # Class-based: Returns string only
    llm = VertexAILLM(model="google/gemini-2.0-flash", project_id="my-project")
    text = llm.generate_response("Hello!")
    print(text)
    
    # Streaming response
    for chunk in llm.generate_response_stream("Tell me a story"):
        print(chunk, end='', flush=True)
        
except VertexAILLMAuthError as e:
    print(f"Authentication failed: {e}")
except VertexAILLMAPIError as e:
    print(f"API request failed: {e}")
except VertexAILLMResponseError as e:
    print(f"Invalid response: {e}")
except VertexAILLMError as e:
    print(f"General error: {e}")
```

## Features

- **Model Garden Access**: Use any model on Vertex AI — first-party and third-party
- **Zero Dependencies**: Uses only Python built-in modules (`urllib`, `json`, `subprocess`)
- **Automatic Authentication**: Falls back to `gcloud CLI` when no token is provided
- **Multi-Region Support**: Global and regional endpoints with auto-derived hostnames
- **OpenAI-Compatible**: Standard chat completions API format
- **Streaming Responses**: Real-time response streaming with `vertexai_llm_stream()` and `generate_response_stream()`
- **Retry Logic**: Automatic retry with exponential backoff for non-streaming requests
- **Error Handling**: Comprehensive exception hierarchy including authentication errors
- **Type Safety**: Full type hints for better IDE support
- **Detailed Error Messages**: HTTP response body included in error messages for easy debugging

## Additional Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Model Garden](https://console.cloud.google.com/vertex-ai/model-garden)
- [OpenAI-Compatible API Reference](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-vertex-using-openai-library)
- [Pricing](https://cloud.google.com/vertex-ai/pricing)
- [Quotas & Limits](https://cloud.google.com/vertex-ai/docs/quotas)

## Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under a Proprietary License - see the `LICENSE` file in the root directory for details.

## Support

For issues, questions, or contributions:

- **Issues**: [GitHub Issues](https://github.com/devXjitin/Autourgos-llmkit/issues)
- **Documentation**: [Full Documentation](https://github.com/devXjitin/Autourgos-llmkit)
- **Community**: [Discussions](https://github.com/devXjitin/Autourgos-llmkit/discussions)

## Author

Built by **Autourgos developers**
