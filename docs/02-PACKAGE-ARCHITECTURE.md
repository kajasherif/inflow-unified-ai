# Package Architecture

## Directory Structure

```
inflow-unified-ai/
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # CI/CD pipeline configuration
├── docs/                        # Documentation
├── src/
│   └── inflow_unified_ai/       # Main package source
│       ├── __init__.py          # Package exports
│       ├── client.py            # AIClient main class
│       ├── models/              # Data models
│       │   ├── __init__.py
│       │   ├── capabilities.py  # Model capabilities registry
│       │   ├── requests.py      # Request models
│       │   └── responses.py     # Response models
│       ├── providers/           # AI provider implementations
│       │   ├── __init__.py
│       │   ├── base.py          # Base provider interface
│       │   ├── factory.py       # Provider factory
│       │   ├── azure_openai.py  # Azure OpenAI provider
│       │   ├── anthropic.py     # Anthropic Claude provider
│       │   ├── gemini.py        # Google Gemini provider
│       │   └── vllm.py          # vLLM provider
│       ├── prompts/             # Prompt management
│       │   ├── __init__.py
│       │   └── manager.py       # Prompt template manager
│       └── resilience/          # Resilience patterns
│           ├── __init__.py
│           └── retry.py         # Retry logic with backoff
├── tests/                       # Unit tests
│   ├── __init__.py
│   └── test_package.py          # Package tests
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
├── pyproject.toml               # Package configuration
├── pytest.ini                   # Pytest configuration
├── README.md                    # Package README
├── comprehensive_test.py        # Integration tests
└── demo.py                      # Demo script
```

## Core Components

### 1. AIClient (`client.py`)

The main entry point for all AI operations.

```python
class AIClient:
    """Unified AI client for multiple providers."""
    
    def __init__(
        self,
        provider: str,           # "azure_openai", "anthropic", "gemini", "vllm"
        api_key: str,
        endpoint: str = None,
        api_version: str = None,
        **kwargs
    )
    
    # Synchronous methods
    def generate(self, model, messages, **kwargs) -> CompletionResponse
    def stream(self, model, messages, **kwargs) -> Iterator[CompletionChunk]
    def chat(self, model, messages, **kwargs) -> CompletionResponse
    
    # Asynchronous methods
    async def agenerate(self, model, messages, **kwargs) -> CompletionResponse
    async def astream(self, model, messages, **kwargs) -> AsyncIterator[CompletionChunk]
    async def achat(self, model, messages, **kwargs) -> CompletionResponse
```

### 2. Models (`models/`)

#### `requests.py` - Request Models

```python
class Message:
    role: MessageRole      # USER, ASSISTANT, SYSTEM
    content: str
    name: Optional[str]

class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class CompletionRequest:
    model: str
    messages: List[Message]
    temperature: Optional[float]
    max_tokens: Optional[int]
    reasoning_effort: Optional[str]   # For reasoning models
    response_format: Optional[ResponseFormat]
    stream: bool

class ResponseFormat:
    type: str              # "json_schema"
    json_schema: dict      # Schema definition
```

#### `responses.py` - Response Models

```python
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionResponse:
    content: str
    model: str
    usage: Usage
    finish_reason: str

class CompletionChunk:
    content: str
    model: str
    usage: Optional[Usage]
    finish_reason: Optional[str]
```

#### `capabilities.py` - Model Capabilities

Contains the registry of all supported models and their capabilities:

```python
class ModelCapabilities:
    supports_streaming: bool
    supports_structured_output: bool
    supports_temperature: bool
    supports_system_message: bool
    max_tokens: int
    context_window: int

# Registry of 40+ models
MODEL_CAPABILITIES_REGISTRY = {
    "gpt-4o": ModelCapabilities(...),
    "gpt-4.1": ModelCapabilities(...),
    "o1": ModelCapabilities(...),
    "o3": ModelCapabilities(...),
    # ... more models
}

# Helper functions
def get_model_capabilities(model: str) -> ModelCapabilities
def is_reasoning_model(model: str) -> bool
```

### 3. Providers (`providers/`)

#### Base Provider Interface (`base.py`)

```python
class LLMProvider(ABC):
    """Abstract base class for all AI providers."""
    
    @abstractmethod
    async def agenerate(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion asynchronously."""
        pass
    
    @abstractmethod
    async def astream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        """Stream a completion asynchronously."""
        pass

class ProviderError(Exception):
    """Base exception for provider errors."""
    pass
```

#### Azure OpenAI Provider (`azure_openai.py`)

The most feature-complete provider with support for:
- All GPT-4.x and GPT-5.x models
- Reasoning models (o1, o3 series)
- Structured output with JSON schema
- Streaming with token usage tracking
- Automatic parameter adjustment for reasoning models

#### Provider Factory (`factory.py`)

```python
class ModelFactory:
    """Factory for creating provider instances."""
    
    @staticmethod
    def create(provider_name: str, **kwargs) -> LLMProvider:
        """Create a provider instance."""
        pass
    
    @staticmethod
    def register(name: str):
        """Decorator to register a custom provider."""
        pass
```

### 4. Resilience (`resilience/`)

#### Retry Logic (`retry.py`)

```python
class RetryConfig:
    max_attempts: int = 3
    min_wait: float = 1.0
    max_wait: float = 60.0
    exponential_base: float = 2.0

def with_retry(config: RetryConfig = None):
    """Decorator for adding retry logic to functions."""
    pass
```

## Data Flow

```
User Code
    │
    ▼
AIClient.generate()
    │
    ├─> Validate parameters
    ├─> Get model capabilities
    ├─> Adjust parameters for model type
    │
    ▼
ModelFactory.create()
    │
    ▼
Provider.agenerate()
    │
    ├─> Build API request
    ├─> Apply retry logic
    ├─> Call provider API
    │
    ▼
CompletionResponse
    │
    ▼
User receives response
```

## Model Support Matrix

### Reasoning Models (No Temperature)
| Model | Streaming | Structured Output |
|-------|-----------|-------------------|
| o1 | ❌ | ✅ |
| o3 | ✅ | ✅ |
| o3-mini | ✅ | ✅ |
| gpt-5 | ✅ | ✅ |
| gpt-5-mini | ✅ | ✅ |
| gpt-5-nano | ✅ | ✅ |
| gpt-5.1-chat | ✅ | ✅ |
| gpt-5.2 | ✅ | ✅ |
| gpt-5.2-chat | ✅ | ✅ |

### Chat Models (Temperature Supported)
| Model | Streaming | Structured Output |
|-------|-----------|-------------------|
| gpt-4o | ✅ | ✅ |
| gpt-4o-mini | ✅ | ✅ |
| gpt-4.1 | ✅ | ✅ |
| gpt-4.1-mini | ✅ | ✅ |
| gpt-4.1-nano | ✅ | ✅ |
| model-router | ✅ | ✅ |

## Adding New Components

### Adding a New Provider

1. Create a new file in `src/inflow_unified_ai/providers/`
2. Implement the `LLMProvider` interface
3. Register in `providers/__init__.py`
4. Add to factory in `factory.py`

See [Development Guide](./03-DEVELOPMENT-GUIDE.md) for details.

### Adding a New Model

1. Add model capabilities to `models/capabilities.py`
2. Add to `MODEL_CAPABILITIES_REGISTRY`
3. Add tests for the new model

See [Development Guide](./03-DEVELOPMENT-GUIDE.md) for details.
