# API Reference

## Overview

This document provides complete API documentation for the inflow-unified-ai package.

## Installation

```bash
# From PyPI (when published)
pip install inflow-unified-ai

# From TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ inflow-unified-ai

# For development
pip install -e ".[dev]"
```

---

## AIClient

The main entry point for interacting with AI models.

### Import

```python
from inflow_unified_ai import AIClient
```

### Constructor

```python
AIClient(
    provider: BaseProvider,
    model: Optional[AIModel] = None,
    **kwargs
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider` | `BaseProvider` | Yes | The AI provider instance |
| `model` | `AIModel` | No | The model to use (optional) |
| `**kwargs` | `Any` | No | Additional configuration options |

### Methods

#### generate()

Generate a response from the AI model.

```python
async def generate(
    messages: List[BaseMessage],
    **kwargs
) -> str
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `List[BaseMessage]` | List of messages to send |
| `**kwargs` | `Any` | Provider-specific options |

**Returns**: `str` - The generated response text

**Example**:
```python
from inflow_unified_ai import AIClient, TextMessage, AzureOpenAIProvider

provider = AzureOpenAIProvider(
    api_key="your-key",
    endpoint="your-endpoint",
    deployment="gpt-4"
)
client = AIClient(provider=provider)

messages = [TextMessage(role="user", content="Hello!")]
response = await client.generate(messages)
print(response)
```

#### generate_with_images()

Generate a response with image inputs.

```python
async def generate_with_images(
    messages: List[BaseMessage],
    **kwargs
) -> str
```

**Example**:
```python
from inflow_unified_ai import ImageMessage

messages = [
    ImageMessage(
        role="user",
        content="What's in this image?",
        image_url="https://example.com/image.jpg"
    )
]
response = await client.generate_with_images(messages)
```

#### stream()

Stream responses from the AI model.

```python
async def stream(
    messages: List[BaseMessage],
    **kwargs
) -> AsyncGenerator[str, None]
```

**Example**:
```python
messages = [TextMessage(role="user", content="Tell me a story")]
async for chunk in client.stream(messages):
    print(chunk, end="", flush=True)
```

---

## Messages

### BaseMessage

Abstract base class for all message types.

```python
from inflow_unified_ai import BaseMessage
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `role` | `str` | Message role: "user", "assistant", or "system" |
| `content` | `str` | Message content |

### TextMessage

Standard text message.

```python
from inflow_unified_ai import TextMessage
```

#### Constructor

```python
TextMessage(
    role: str = "user",
    content: str = ""
)
```

**Example**:
```python
# User message
user_msg = TextMessage(role="user", content="What is AI?")

# System message
system_msg = TextMessage(role="system", content="You are a helpful assistant.")

# Assistant message
assistant_msg = TextMessage(role="assistant", content="AI stands for...")
```

### ImageMessage

Message with image content.

```python
from inflow_unified_ai import ImageMessage
```

#### Constructor

```python
ImageMessage(
    role: str = "user",
    content: str = "",
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `role` | `str` | Message role |
| `content` | `str` | Text content accompanying the image |
| `image_url` | `str` | URL of the image |
| `image_base64` | `str` | Base64-encoded image data |

**Example**:
```python
# With URL
msg = ImageMessage(
    role="user",
    content="Describe this image",
    image_url="https://example.com/photo.jpg"
)

# With base64
import base64
with open("image.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

msg = ImageMessage(
    role="user",
    content="What's in this picture?",
    image_base64=img_data
)
```

---

## Models

### AIModel

Represents an AI model with its capabilities.

```python
from inflow_unified_ai import AIModel
```

#### Constructor

```python
AIModel(
    model_id: str,
    provider: str,
    family: ModelFamily,
    capabilities: ModelCapabilities,
    context_window: int = 4096,
    max_output_tokens: int = 4096
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | Unique model identifier |
| `provider` | `str` | Provider name (e.g., "azure", "openai") |
| `family` | `ModelFamily` | Model family enum value |
| `capabilities` | `ModelCapabilities` | Model capabilities |
| `context_window` | `int` | Maximum context length |
| `max_output_tokens` | `int` | Maximum output length |

**Example**:
```python
from inflow_unified_ai import AIModel, ModelFamily, ModelCapabilities

model = AIModel(
    model_id="gpt-4-custom",
    provider="azure",
    family=ModelFamily.GPT_4O,
    capabilities=ModelCapabilities(
        supports_vision=True,
        supports_streaming=True,
        supports_function_calling=True
    ),
    context_window=128000,
    max_output_tokens=4096
)
```

### ModelFamily

Enum of supported model families.

```python
from inflow_unified_ai import ModelFamily
```

| Value | Description |
|-------|-------------|
| `ModelFamily.GPT_4O` | GPT-4o models |
| `ModelFamily.GPT_4O_MINI` | GPT-4o-mini models |
| `ModelFamily.GPT_41` | GPT-4.1 models |
| `ModelFamily.GPT_41_MINI` | GPT-4.1-mini models |
| `ModelFamily.GPT_41_NANO` | GPT-4.1-nano models |
| `ModelFamily.O1` | o1 reasoning models |
| `ModelFamily.O1_MINI` | o1-mini models |
| `ModelFamily.O1_PRO` | o1-pro models |
| `ModelFamily.O3` | o3 models |
| `ModelFamily.O3_MINI` | o3-mini models |
| `ModelFamily.O4_MINI` | o4-mini models |
| `ModelFamily.CLAUDE_3_OPUS` | Claude 3 Opus |
| `ModelFamily.CLAUDE_3_SONNET` | Claude 3 Sonnet |
| `ModelFamily.CLAUDE_3_HAIKU` | Claude 3 Haiku |
| `ModelFamily.CLAUDE_35_SONNET` | Claude 3.5 Sonnet |
| `ModelFamily.CLAUDE_35_HAIKU` | Claude 3.5 Haiku |
| `ModelFamily.CLAUDE_4_OPUS` | Claude 4 Opus |
| `ModelFamily.CLAUDE_4_SONNET` | Claude 4 Sonnet |
| `ModelFamily.GEMINI_PRO` | Gemini Pro |
| `ModelFamily.GEMINI_ULTRA` | Gemini Ultra |
| `ModelFamily.GEMINI_2_FLASH` | Gemini 2 Flash |
| `ModelFamily.GEMINI_25_PRO` | Gemini 2.5 Pro |
| `ModelFamily.GEMINI_25_FLASH` | Gemini 2.5 Flash |

### ModelCapabilities

Defines what a model can do.

```python
from inflow_unified_ai import ModelCapabilities
```

#### Constructor

```python
ModelCapabilities(
    supports_vision: bool = False,
    supports_streaming: bool = True,
    supports_function_calling: bool = False,
    supports_json_mode: bool = False,
    supports_system_message: bool = True,
    supports_tools: bool = False
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `supports_vision` | `bool` | `False` | Can process images |
| `supports_streaming` | `bool` | `True` | Supports streaming responses |
| `supports_function_calling` | `bool` | `False` | Supports function calling |
| `supports_json_mode` | `bool` | `False` | Supports JSON output mode |
| `supports_system_message` | `bool` | `True` | Supports system messages |
| `supports_tools` | `bool` | `False` | Supports tools API |

---

## Model Registry

### ModelRegistry

Manages registered AI models.

```python
from inflow_unified_ai import ModelRegistry
```

#### Methods

##### get_model()

Get a model by its ID.

```python
@staticmethod
def get_model(model_id: str) -> Optional[AIModel]
```

**Example**:
```python
model = ModelRegistry.get_model("gpt-4o")
if model:
    print(f"Context window: {model.context_window}")
```

##### list_models()

List all registered models.

```python
@staticmethod
def list_models() -> List[AIModel]
```

**Example**:
```python
models = ModelRegistry.list_models()
for model in models:
    print(f"{model.model_id}: {model.family}")
```

##### register_model()

Register a custom model.

```python
@staticmethod
def register_model(model: AIModel) -> None
```

**Example**:
```python
custom_model = AIModel(
    model_id="my-fine-tuned-model",
    provider="azure",
    family=ModelFamily.GPT_4O,
    capabilities=ModelCapabilities(supports_vision=True)
)
ModelRegistry.register_model(custom_model)
```

##### filter_by_capability()

Filter models by capability.

```python
@staticmethod
def filter_by_capability(capability: str) -> List[AIModel]
```

**Example**:
```python
# Get all vision-capable models
vision_models = ModelRegistry.filter_by_capability("supports_vision")
```

---

## Providers

### BaseProvider

Abstract base class for all providers.

```python
from inflow_unified_ai.providers import BaseProvider
```

All providers must implement:
- `async generate(messages, **kwargs) -> str`
- `async stream(messages, **kwargs) -> AsyncGenerator`

### AzureOpenAIProvider

Provider for Azure OpenAI Service.

```python
from inflow_unified_ai import AzureOpenAIProvider
```

#### Constructor

```python
AzureOpenAIProvider(
    api_key: str,
    endpoint: str,
    deployment: str,
    api_version: str = "2024-02-01"
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | `str` | Azure OpenAI API key |
| `endpoint` | `str` | Azure OpenAI endpoint URL |
| `deployment` | `str` | Deployment name |
| `api_version` | `str` | API version |

**Example**:
```python
provider = AzureOpenAIProvider(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment="gpt-4o"
)
```

### OpenAIProvider

Provider for OpenAI API.

```python
from inflow_unified_ai import OpenAIProvider
```

#### Constructor

```python
OpenAIProvider(
    api_key: str,
    model: str = "gpt-4o",
    base_url: Optional[str] = None
)
```

**Example**:
```python
provider = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o"
)
```

### AnthropicProvider

Provider for Anthropic Claude models.

```python
from inflow_unified_ai import AnthropicProvider
```

#### Constructor

```python
AnthropicProvider(
    api_key: str,
    model: str = "claude-3-opus-20240229"
)
```

**Example**:
```python
provider = AnthropicProvider(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-5-sonnet-20241022"
)
```

### GoogleProvider

Provider for Google Gemini models.

```python
from inflow_unified_ai import GoogleProvider
```

#### Constructor

```python
GoogleProvider(
    api_key: str,
    model: str = "gemini-pro"
)
```

**Example**:
```python
provider = GoogleProvider(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.0-flash"
)
```

---

## Complete Example

```python
import asyncio
import os
from inflow_unified_ai import (
    AIClient,
    AzureOpenAIProvider,
    TextMessage,
    ImageMessage,
    ModelRegistry
)

async def main():
    # Initialize provider
    provider = AzureOpenAIProvider(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment="gpt-4o"
    )
    
    # Create client
    client = AIClient(provider=provider)
    
    # Text generation
    messages = [
        TextMessage(role="system", content="You are a helpful assistant."),
        TextMessage(role="user", content="Explain quantum computing in simple terms.")
    ]
    response = await client.generate(messages)
    print("Response:", response)
    
    # Streaming
    print("\nStreaming:")
    async for chunk in client.stream(messages):
        print(chunk, end="", flush=True)
    print()
    
    # Vision (if model supports it)
    model = ModelRegistry.get_model("gpt-4o")
    if model and model.capabilities.supports_vision:
        vision_messages = [
            ImageMessage(
                role="user",
                content="What's in this image?",
                image_url="https://example.com/image.jpg"
            )
        ]
        vision_response = await client.generate_with_images(vision_messages)
        print("Vision response:", vision_response)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Error Handling

```python
from inflow_unified_ai.exceptions import (
    ProviderError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError
)

try:
    response = await client.generate(messages)
except AuthenticationError as e:
    print(f"Invalid API key: {e}")
except RateLimitError as e:
    print(f"Rate limited, retry after: {e.retry_after}")
except ProviderError as e:
    print(f"Provider error: {e}")
```

---

## Type Definitions

All types are fully typed with Python type hints. For IDE support, ensure you have:

```bash
pip install types-requests mypy
```

Run type checking:

```bash
mypy your_code.py
```
