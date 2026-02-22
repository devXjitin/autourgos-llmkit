# Autourgos LLM Kit

<div align="center">

![Python Version](https://img.shields.io/pypi/pyversions/autourgos-llmkit)
![License](https://img.shields.io/github/license/devXjitin/autourgos-llmkit)
![Status](https://img.shields.io/badge/status-production--ready-green)

**The lightweight, unified interface for state-of-the-art LLMs.**  
*Connect to OpenAI, Gemini, Claude, Grok, Azure, Vertex AI, and Ollama with a single, consistent API.*

[**Installation**](#installation) · [**Quick Start**](#quick-start) · [**Providers**](#supported-providers) · [**Documentation**](#advanced-usage)

</div>

---

## 💡Why Autourgos LLM Kit?

Building AI agents requires flexibility. Today you might use **GPT-4o** for reasoning, tomorrow **Gemini 3** for long context, and **DeepSeek R1** for cost-efficiency. Switching providers usually means rewriting client code, handling different error types, and managing disparate retry logic.

**Autourgos LLM Kit** solves this without the bloat. Unlike heavy frameworks (LangChain, LlamaIndex) that enforce specific cognitive architectures, this library focuses solely on the **connectivity layer**.

*   **Unified Interface**: One `generate_response()` signature for all providers.
*   **Zero Lock-in**: Switch from OpenAI to Vertex AI to Ollama just by changing a config string.
*   **Production Hardened**: Built-in exponential backoff, retries, and standardized exception handling.
*   **Type Safe**: Fully typed for modern Python development.
*   **Lightweight**: Modular dependencies—install only the SDKs you need.

---

## 📦 Installation

Install the core package (lightweight, no heavy dependencies):

```bash
pip install autourgos-llmkit
```

Install provider-specific dependencies as needed to keep your build slim:

```bash
# Individual providers
pip install autourgos-llmkit[google]      # For Gemini
pip install autourgos-llmkit[openai]      # For GPT, Grok, Azure
pip install autourgos-llmkit[anthropic]   # For Claude
pip install autourgos-llmkit[ollama]      # For Ollama

# Or install everything
pip install autourgos-llmkit[all]
```

> **Note**: `Vertex AI` support uses the Python standard library and requires no extra pip packages, just the Google Cloud CLI.

---

## 🚀 Quick Start

### 1. The Unified `init_llm` Factory

The easiest way to get started is using the factory function. This allows you to drive your model selection purely via configuration.

```python
import os
from autourgos.llmkit import init_llm

# 1. Setup API Keys (or use .env file)
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["GOOGLE_API_KEY"] = "AIza..."

# 2. Initialize a provider
# Change 'provider' to 'google', 'anthropic', 'grok', 'ollama', etc.
llm = init_llm(
    provider="openai", 
    model="gpt-4o", 
    temperature=0.7
)

# 3. Generate text
response = llm.generate_response("Explain the concept of 'Agentic Workflow' in one sentence.")
print(f"Response: {response}")

# 4. Stream text (Real-time)
print("Streaming: ", end="")
for chunk in llm.generate_response_stream("List 3 benefits of Python."):
    print(chunk, end="", flush=True)
print()
```

---

## 🔌 Supported Providers

| Provider | Key Class | Env Variable | Capabilities |
|----------|-----------|--------------|--------------|
| **OpenAI** | `OpenAILLM` | `OPENAI_API_KEY` | Frontier models (`gpt-4o`, `o1`), Function Calling, JSON mode. |
| **Google Gemini** | `GoogleLLM` | `GOOGLE_API_KEY` | Massive context (2M+ tokens), Native Multimodal, `gemini-1.5-pro`. |
| **Google Vision** | `GoogleVisionLLM` | `GOOGLE_API_KEY` | Specialized image analysis and reasoning. |
| **Anthropic** | `AnthropicLLM` | `ANTHROPIC_API_KEY` | High reliability, `claude-3-5-sonnet`. |
| **xAI Grok** | `GrokLLM` | `XAI_API_KEY` | `grok-3`, `grok-2`. Access via OpenAI-compatible endpoint. |
| **Azure OpenAI** | `AzureLLM` | `AZURE_OPENAI_API_KEY` | Enterprise `gpt-4o` deployments. |
| **Azure Foundry** | `AzureLLM` | `AZURE_OPENAI_API_KEY` | Serverless MaaS: `deepseek-r1`, `phi-4`, `llama-3.3`. |
| **Vertex AI** | `VertexAILLM` | `VERTEX_AI_ACCESS_TOKEN` | Google Cloud infrastructure. Supports Gemini + Model Garden (Mistral, Llama). |
| **Ollama** | `OllamaCloudLLM` | `OLLAMA_API_KEY` | Local or Cloud open-source models (`llama3`, `mistral`). |

---

## 🛠️ Advanced Usage

### Multimodal (Vision) Requests

Process images effortlessly using the Vision-capable providers (Google, OpenAI, Anthropic).

```python
from autourgos.llmkit.GoogleVision import GoogleVisionLLM

llm = GoogleVisionLLM(model="gemini-1.5-flash")

# Pass local file paths or PIL Image objects
response = llm.generate_response(
    prompt="Extract all the text visible in this image.",
    images=["./receipt.jpg"]
)
print(response)
```

### Direct Class Instantiation (Type-Safe)

For strict typing and IDE autocompletion, instantiate provider classes directly.

```python
from autourgos.llmkit.Microsoft import AzureLLM

# Connect to a DeepSeek R1 endpoint on Azure Foundry
llm = AzureLLM(
    deployment_name="deepseek-r1",
    azure_endpoint="https://DeepSeek-R1-xyzw.eastus.models.ai.azure.com/",
    api_key="your-azure-key"
)

text = llm.generate_response("Solve this complex logic puzzle.")
```

### Standardized Error Handling

Don't catch 5 different `RateLimitError` exceptions. Autourgos unifies them.

```python
from autourgos.llmkit.Openai import OpenAILLM, OpenAILLMAPIError, OpenAILLMAuthError

try:
    llm = OpenAILLM(model="gpt-4o", api_key="invalid-key")
    llm.generate_response("Hello")
except OpenAILLMAuthError:
    print("Please check your API credentials.")
except OpenAILLMAPIError as e:
    print(f"Provider returned an error: {e}")
```

### Vertex AI (No Pip Dependencies)

Vertex AI is unique because it often requires complex auth libraries. We implemented it using raw `urllib` and `gcloud` CLI integration, so you can run it in restricted environments without installing the heavy `google-cloud-aiplatform` SDK.

```bash
# Just authenticate via CLI
gcloud auth login
```

```python
from autourgos.llmkit.Vertexai import VertexAILLM

# Automatically picks up credentials from gcloud
llm = VertexAILLM(
    model="google/gemini-2.0-flash",
    project_id="your-gcp-project-id",
    region="us-central1"
)
```

---

<div align="center">
  <sub>Built with ❤️ by the Autourgos Team</sub>
</div>
