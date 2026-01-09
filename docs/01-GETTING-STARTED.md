# Getting Started

## What is inflow-unified-ai?

**inflow-unified-ai** is a Unified AI Abstraction Layer that provides a single, consistent API to interact with multiple AI providers:

- **Azure OpenAI** - GPT-4o, GPT-4.1, GPT-5, o1, o3 series
- **Anthropic** - Claude 3, Claude 3.5 series
- **Google** - Gemini Pro, Gemini Ultra
- **vLLM** - Self-hosted open-source models

## Why Use This Package?

1. **Write Once, Use Anywhere** - Same code works with any provider
2. **Smart Model Detection** - Automatically handles reasoning vs chat models
3. **Built-in Resilience** - Retry logic with exponential backoff
4. **Token Tracking** - Comprehensive usage tracking across providers
5. **Streaming Support** - First-class async streaming with TTFT metrics

## Installation

### From PyPI (Production)

```bash
pip install inflow-unified-ai
```

### From TestPyPI (Testing/Preview)

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ inflow-unified-ai
```

### From Source (Development)

```bash
git clone https://github.com/kajasherif/inflow-unified-ai.git
cd inflow-unified-ai
pip install -e ".[dev]"
```

## Quick Start

### 1. Basic Completion

```python
from inflow_unified_ai import AIClient, Message, MessageRole

# Initialize client
client = AIClient(
    provider="azure_openai",
    api_key="your-api-key",
    endpoint="https://your-resource.openai.azure.com"
)

# Generate response
response = client.generate(
    model="gpt-4o",
    messages=[
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="What is Python?")
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.content)
print(f"Tokens used: {response.usage.total_tokens}")
```

### 2. Async Completion

```python
import asyncio
from inflow_unified_ai import AIClient, Message, MessageRole

async def main():
    client = AIClient(
        provider="azure_openai",
        api_key="your-api-key",
        endpoint="https://your-resource.openai.azure.com"
    )
    
    response = await client.agenerate(
        model="gpt-4o",
        messages=[Message(role=MessageRole.USER, content="Hello!")]
    )
    
    print(response.content)

asyncio.run(main())
```

### 3. Streaming

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

### 4. Reasoning Models (o1, o3, GPT-5)

Reasoning models work differently - they don''t support temperature or system messages:

```python
from inflow_unified_ai import AIClient, Message, MessageRole

client = AIClient(
    provider="azure_openai",
    api_key="your-api-key",
    endpoint="https://your-resource.openai.azure.com"
)

# Use reasoning_effort instead of temperature
response = client.generate(
    model="o3",
    messages=[Message(role=MessageRole.USER, content="Solve this math problem: 15 * 17")],
    reasoning_effort="medium",  # low, medium, high
    max_tokens=1000
)

print(response.content)
```

### 5. Structured Output (JSON Schema)

```python
from inflow_unified_ai import AIClient, Message, MessageRole
from inflow_unified_ai.models.requests import ResponseFormat

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
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "industry": {"type": "string"},
                    "employee_count": {"type": "integer"}
                },
                "required": ["name", "industry", "employee_count"],
                "additionalProperties": False
            }
        }
    )
)

import json
data = json.loads(response.content)
print(data)  # {"name": "TechCorp", "industry": "Technology", "employee_count": 500}
```

## Environment Variables

You can also configure the client using environment variables:

```bash
# .env file
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

```python
import os
from dotenv import load_dotenv
from inflow_unified_ai import AIClient

load_dotenv()

client = AIClient(
    provider="azure_openai",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
```

## Next Steps

- [Package Architecture](./02-PACKAGE-ARCHITECTURE.md) - Understand the package structure
- [API Reference](./07-API-REFERENCE.md) - Complete API documentation
- [Development Guide](./03-DEVELOPMENT-GUIDE.md) - How to contribute
