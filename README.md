# 🤖 inflow-unified-ai

[![PyPI version](https://badge.fury.io/py/inflow-unified-ai.svg)](https://pypi.org/project/inflow-unified-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/inflow-unified-ai)](https://pepy.tech/project/inflow-unified-ai)

**Unified AI Abstraction Layer** for seamless integration with multiple AI providers. Built by [iNextLabs](https://inextlabs.com) for enterprise AI applications.

## ✨ Features

- **🔌 Multi-Provider Support**: Azure OpenAI, Anthropic Claude, Google Gemini, vLLM
- **🧠 Smart Model Detection**: Automatic parameter handling for 40+ models including GPT-4.1, GPT-5, o1, o3 series
- **🔄 Unified API**: Single interface for all providers - write once, use anywhere
- **📡 Streaming Support**: First-class async streaming with time-to-first-token metrics
- **🛡️ Enterprise Resilience**: Built-in retry logic with exponential backoff via Tenacity
- **📊 Token Tracking**: Comprehensive usage tracking across all providers
- **⚡ Async-First**: Full async/await support with sync wrappers
- **🎯 Structured Output**: JSON Schema support for type-safe responses

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install inflow-unified-ai
```

### From TestPyPI (Pre-release)

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ inflow-unified-ai
```

### From Source

```bash
git clone https://github.com/iNextLabs/inflow-unified-ai.git
cd inflow-unified-ai
pip install -e .
```

### With Development Dependencies

```bash
pip install inflow-unified-ai[dev]
```

## 🚀 Quick Start

### Basic Usage

```python
from inflow_unified_ai import AIClient, Message, MessageRole

# Initialize with Azure OpenAI
client = AIClient(
    provider="azure_openai",
    api_key="your-api-key",
    endpoint="https://your-resource.openai.azure.com"
)

# Simple completion
response = client.generate(
    model="gpt-4o",
    messages=[
        Message(role=MessageRole.USER, content="What is 2 + 2?")
    ]
)

print(response.content)  # "4"
print(response.usage.total_tokens)  # Token usage tracking
```

### Async Usage

```python
import asyncio
from inflow_unified_ai import AIClient, Message, MessageRole

async def main():
    client = AIClient(
        provider="azure_openai",
        api_key="your-api-key",
        endpoint="https://your-resource.openai.azure.com"
    )

    # Async completion
    response = await client.agenerate(
        model="gpt-4.1",
        messages=[Message(role=MessageRole.USER, content="Hello!")]
    )
    print(response.content)

asyncio.run(main())
```

### Streaming

```python
import asyncio
from inflow_unified_ai import AIClient, Message, MessageRole

async def stream_response():
    client = AIClient(
        provider="azure_openai",
        api_key="your-api-key",
        endpoint="https://your-resource.openai.azure.com"
    )

    async for chunk in client.astream(
        model="gpt-4o",
        messages=[Message(role=MessageRole.USER, content="Tell me a story")]
    ):
        print(chunk.content, end="", flush=True)

asyncio.run(stream_response())
```

### Structured Output (JSON Schema)

```python
from inflow_unified_ai import AIClient, Message, MessageRole, ResponseFormat

client = AIClient(
    provider="azure_openai",
    api_key="your-api-key",
    endpoint="https://your-resource.openai.azure.com"
)

response = client.generate(
    model="gpt-4o",
    messages=[Message(role=MessageRole.USER, content="Generate a company profile")],
    response_format=ResponseFormat(
        type="json_schema",
        json_schema={
            "name": "company_profile",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "industry": {"type": "string"},
                    "employee_count": {"type": "integer"}
                },
                "required": ["name", "industry", "employee_count"]
            }
        }
    )
)

import json
data = json.loads(response.content)
print(data)  # {"name": "TechCorp", "industry": "Technology", "employee_count": 500}
```

## 🧠 Supported Models

### Azure OpenAI

| Model Family | Type | Temperature | Streaming | Notes |
|--------------|------|-------------|-----------|-------|
| **GPT-4o** | Chat | ✅ | ✅ | Vision capable |
| **GPT-4o-mini** | Chat | ✅ | ✅ | Cost-effective |
| **GPT-4.1** | Chat | ✅ | ✅ | Latest non-reasoning |
| **GPT-4.1-mini** | Chat | ✅ | ✅ | Efficient |
| **GPT-4.1-nano** | Chat | ✅ | ✅ | Ultra-efficient |
| **o1** | Reasoning | ❌ | ❌ | Deep reasoning |
| **o3** | Reasoning | ❌ | ✅ | Advanced reasoning |
| **o3-mini** | Reasoning | ❌ | ✅ | Fast reasoning |
| **GPT-5** | Reasoning | ❌ | ✅ | Next-gen reasoning |
| **GPT-5-mini** | Reasoning | ❌ | ✅ | Efficient reasoning |
| **GPT-5-nano** | Reasoning | ❌ | ✅ | Ultra-fast reasoning |
| **GPT-5.1-chat** | Reasoning | ❌ | ✅ | Chat-optimized |
| **GPT-5.2** | Reasoning | ❌ | ✅ | Advanced reasoning |
| **GPT-5.2-chat** | Reasoning | ❌ | ✅ | Chat-optimized reasoning |
| **model-router** | Router | ✅ | ✅ | Intelligent model routing |

### Other Providers

| Provider | Models | Status |
|----------|--------|--------|
| **Anthropic** | Claude 3, Claude 3.5 | ✅ Ready |
| **Google** | Gemini Pro, Gemini Ultra | ✅ Ready |
| **vLLM** | Self-hosted models | ✅ Ready |

## ⚙️ Configuration

### Environment Variables

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-key

# Google Gemini
GOOGLE_API_KEY=your-google-key
```

### Programmatic Configuration

```python
from inflow_unified_ai import AIClient

# Azure OpenAI
client = AIClient(
    provider="azure_openai",
    api_key="your-key",
    endpoint="https://your-resource.openai.azure.com",
    api_version="2024-12-01-preview"  # Optional
)

# Anthropic
client = AIClient(
    provider="anthropic",
    api_key="your-anthropic-key"
)

# Google Gemini
client = AIClient(
    provider="gemini",
    api_key="your-google-key"
)
```

## 🛡️ Resilience & Retry

Built-in retry logic with exponential backoff:

```python
from inflow_unified_ai import AIClient, RetryConfig

client = AIClient(
    provider="azure_openai",
    api_key="your-key",
    endpoint="https://your-resource.openai.azure.com",
    retry_config=RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0
    )
)
```

## 📊 Token Usage Tracking

```python
response = client.generate(
    model="gpt-4o",
    messages=[Message(role=MessageRole.USER, content="Hello!")]
)

print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

## 🔧 Advanced Features

### Model Capabilities Detection

```python
from inflow_unified_ai import get_model_capabilities, is_reasoning_model

# Check model capabilities
caps = get_model_capabilities("gpt-5")
print(caps.supports_temperature)  # False
print(caps.is_reasoning_model)    # True
print(caps.supports_streaming)    # True

# Quick check for reasoning models
if is_reasoning_model("o3-mini"):
    print("This is a reasoning model - no temperature support")
```

### Custom Provider Registration

```python
from inflow_unified_ai import ModelFactory, LLMProvider

# Register a custom provider
@ModelFactory.register("my_provider")
class MyCustomProvider(LLMProvider):
    async def agenerate(self, request):
        # Your implementation
        pass
```

## 📋 API Reference

### AIClient

| Method | Description |
|--------|-------------|
| `generate()` | Synchronous completion |
| `agenerate()` | Async completion |
| `stream()` | Synchronous streaming |
| `astream()` | Async streaming |
| `chat()` | Convenience method for chat |
| `achat()` | Async chat |

### Response Objects

```python
# CompletionResponse
response.content      # str - The generated text
response.model        # str - Model used
response.usage        # Usage - Token counts
response.finish_reason # str - Why generation stopped

# Usage
response.usage.prompt_tokens      # int
response.usage.completion_tokens  # int
response.usage.total_tokens       # int
```

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| **0.1.0** | 2025-01-13 | Initial release with Azure OpenAI, Anthropic, Gemini, vLLM support. 40+ models including GPT-5.x, o3, model-router |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=inflow_unified_ai --cov-report=html

# Run specific test file
pytest tests/test_azure_openai.py -v
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- **PyPI**: [https://pypi.org/project/inflow-unified-ai/](https://pypi.org/project/inflow-unified-ai/)
- **GitHub**: [https://github.com/iNextLabs/inflow-unified-ai](https://github.com/iNextLabs/inflow-unified-ai)
- **Documentation**: [https://docs.inextlabs.com/inflow-unified-ai](https://docs.inextlabs.com/inflow-unified-ai)
- **Issues**: [https://github.com/iNextLabs/inflow-unified-ai/issues](https://github.com/iNextLabs/inflow-unified-ai/issues)

## 🏢 About iNextLabs

[iNextLabs](https://inextlabs.com) builds enterprise AI solutions including:
- **DocsAI** - Intelligent document processing
- **InsightsAI** - Business analytics and insights  
- **EngageAI** - Customer engagement automation
- **inflow-unified-ai** - Unified AI abstraction layer

---

Made with ❤️ by [iNextLabs](https://inextlabs.com)
