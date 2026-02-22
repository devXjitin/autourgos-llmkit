# Autourgos LLM Kit — Azure (Microsoft) Provider

The **Azure** provider for Autourgos LLM Kit connects securely to your enterprise Microsoft Azure AI deployments. It supports both **Azure OpenAI Service** and **Azure AI Foundry (Models as a Service)**, allowing access to models from OpenAI, Microsoft (Phi), DeepSeek, Meta (Llama), Mistral, and more.

**Official API Documentation:** 
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)
- [Azure AI Foundry Model Catalog](https://azure.microsoft.com/en-us/products/ai-model-catalog/)

---

## 🚀 Supported Models

### Azure OpenAI Service
- **GPT-4o / GPT-4o-mini**
- **o1 / o3-mini**
- **GPT-4 / GPT-3.5-turbo** (legacy)

### Azure AI Foundry (Serverless / MaaS)
- **Microsoft Phi-4 / Phi-3.5** (Mini, Small, Medium, Vision, MoE)
- **DeepSeek R1 / V3** (Global, Regional, DataZone)
- **Meta Llama 3.3 / 3.1**
- **Mistral Large / Small**
- **Cohere Command R+**
- **xAI Grok** (via MaaS)

*Note: In Azure, you target a **Deployment Name** (or Endpoint) rather than just a model ID.*

---

## 💰 Pricing

*Prices are estimates based on US East regions. See [Official Pricing](https://azure.microsoft.com/pricing/details/ai-foundry-models/) for real-time data.*

### Microsoft Phi (Serverless)
| Model | Input (per 1k) | Output (per 1k) |
|---|---|---|
| **Phi-4** | $0.000125 | $0.0005 |
| **Phi-4 Mini** | $0.000075 | $0.0003 |
| **Phi-3.5 Mini** | $0.00013 | $0.00052 |

### DeepSeek (Serverless)
| Model | Input (per 1M) | Output (per 1M) |
|---|---|---|
| **DeepSeek V3** | $0.58 - $1.25 | $1.68 - $5.00 |
| **DeepSeek R1** | $1.35 - $1.49 | $5.40 - $5.94 |

### Azure OpenAI (Standard)
| Model | Input (per 1M) | Output (per 1M) |
|---|---|---|
| **GPT-4o** | $2.50 | $10.00 |
| **GPT-4o-mini** | $0.15 | $0.60 |

---

## 📦 Installation & Configuration

Install the provider SDK as an optional extra (uses the standard `openai` library which handles Azure routing):

```bash
pip install autourgos-llmkit[openai]
```

Set your API key and Endpoint using environment variables:

```bash
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
```

---

## 💻 Usage

### Azure OpenAI Service (Standard)

```python
from autourgos.llmkit.Microsoft import azure_llm

response = azure_llm(
    prompt="Explain Azure architecture.",
    deployment_name="my-gpt-4o-deployment",  # Azure OpenAI Deployment Name
    azure_endpoint="https://my-resource.openai.azure.com/",
    api_key="my-azure-key",
    api_version="2024-02-15-preview"
)
print(response["content"])
```

### Azure AI Foundry MaaS (DeepSeek / Phi / Llama)

For "Models as a Service" (Serverless) endpoints, use the specific endpoint URL from Azure AI Studio. The provider automatically detects MaaS endpoints.

```python
from autourgos.llmkit.Microsoft import azure_llm

# Example for DeepSeek R1 or Phi-4
response = azure_llm(
    prompt="Write a Python script for data analysis.",
    deployment_name="deepseek-r1", # Or whatever your deployment is named
    azure_endpoint="https://DeepSeek-R1-xyzw.eastus.models.ai.azure.com/", # MaaS Endpoint
    api_key="my-maas-key"
)

print(response["content"])
```

### Class-based API

```python
from autourgos.llmkit.Microsoft import AzureLLM

# Initialize for Phi-4
llm = AzureLLM(
    deployment_name="phi-4",
    azure_endpoint="https://Phi-4-abcd.eastus.models.ai.azure.com/",
    api_key="my-key"
)

text = llm.generate_response("Explain quantum entanglement.")
print(text)
```

### Streaming API

```python
from autourgos.llmkit.Microsoft import azure_llm_stream

for chunk in azure_llm_stream(
    prompt="Generate a long cloud migration plan...",
    deployment_name="my-gpt-4o-deployment"
):
    print(chunk, end='', flush=True)
```

---

## Parameters

Both the function `azure_llm` and the class `AzureLLM` accept the following configuration parameters:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `deployment_name` | `str` | Yes | - | The name of the deployment inside your Azure AI Studio |
| `azure_endpoint` | `str` | No | `None` | Endpoint URL (e.g. `https://resource.openai.azure.com/` or `https://model.models.ai.azure.com/`) |
| `api_key` | `str` | No | `None` | Explicit API key (overrides `AZURE_OPENAI_API_KEY` env var) |
| `api_version` | `str` | No | `"2024-02-15-preview"` | The Azure API Version (ignored for MaaS endpoints) |
| `temperature` | `float` | No | `None` | Controls randomness (0.0 to 2.0) |
| `top_p` | `float` | No | `None` | Nucleus sampling probability limit |
| `max_tokens` | `int` | No | `None` | Maximum number of output tokens |
| `frequency_penalty`| `float`| No | `None` | Penalizes new tokens based on existing frequency (-2.0 to 2.0) |
| `presence_penalty` | `float`| No | `None` | Penalizes new tokens based on presence (-2.0 to 2.0) |
| `max_retries` | `int` | No | `3` | Number of times to retry failed requests (non-streaming only) |
| `timeout` | `float`| No | `30.0` | Request timeout in seconds |
| `backoff_factor`| `float`| No | `0.5` | Exponential backoff base for retries |

---

## Error Handling

The module provides typed exceptions under the base `AzureLLMError`:

- `AzureLLMImportError`: The `openai` SDK is not installed.
- `AzureLLMAPIError`: The SDK threw an API error (authentication, endpoint mismatch).
- `AzureLLMResponseError`: The response format was unexpected.
