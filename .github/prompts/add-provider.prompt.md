---
mode: agent
description: Add a new AI provider to the inflow-unified-ai package
tools: ['codebase', 'editFiles', 'search', 'runCommands', 'problems', 'usages']
---

# Add New AI Provider

## Purpose
Step-by-step guide to add a new AI provider to the inflow-unified-ai package.

## Required Information
- Provider name (e.g., "Mistral", "Cohere")
- API client library (e.g., `mistralai`, `cohere`)
- Authentication method (API key, OAuth, etc.)
- Available models and capabilities

## Implementation Steps

### Step 1: Create Provider File

Create `src/inflow_unified_ai/providers/{provider_name}.py`:

```python
"""${input:provider_name} Provider implementation."""
from typing import List, AsyncGenerator, Optional
import os

from .base import BaseProvider
from ..models.messages import BaseMessage, TextMessage, ImageMessage


class ${input:provider_class}Provider(BaseProvider):
    """Provider for ${input:provider_name} AI Service.
    
    Args:
        api_key: API key for authentication
        model: Model name to use
        **kwargs: Additional provider-specific options
    
    Example:
        ```python
        provider = ${input:provider_class}Provider(
            api_key=os.getenv("${input:env_var}"),
            model="${input:default_model}"
        )
        client = AIClient(provider=provider)
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "${input:default_model}",
        **kwargs
    ):
        self.api_key = api_key or os.getenv("${input:env_var}")
        self.model = model
        self._client = None
        self._kwargs = kwargs

    def _get_client(self):
        """Lazy initialization of API client."""
        if self._client is None:
            # Import and initialize the client
            # from ${input:client_lib} import Client
            # self._client = Client(api_key=self.api_key)
            pass
        return self._client

    async def generate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> str:
        """Generate a response from ${input:provider_name}.
        
        Args:
            messages: List of messages to send
            **kwargs: Additional generation options
            
        Returns:
            Generated response text
        """
        client = self._get_client()
        
        # Convert messages to provider format
        formatted_messages = self._format_messages(messages)
        
        # Make API call
        # response = await client.chat(
        #     model=self.model,
        #     messages=formatted_messages,
        #     **kwargs
        # )
        
        # return response.content
        raise NotImplementedError("Implement API call")

    async def stream(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream responses from ${input:provider_name}.
        
        Args:
            messages: List of messages to send
            **kwargs: Additional generation options
            
        Yields:
            Response text chunks
        """
        client = self._get_client()
        formatted_messages = self._format_messages(messages)
        
        # Make streaming API call
        # async for chunk in client.chat_stream(...):
        #     yield chunk.content
        raise NotImplementedError("Implement streaming")
        yield ""

    def _format_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert BaseMessage list to provider format."""
        formatted = []
        for msg in messages:
            if isinstance(msg, TextMessage):
                formatted.append({
                    "role": msg.role,
                    "content": msg.content
                })
            elif isinstance(msg, ImageMessage):
                # Handle image messages if supported
                formatted.append({
                    "role": msg.role,
                    "content": msg.content,
                    # Add image handling
                })
        return formatted
```

### Step 2: Export from providers/__init__.py

Add to `src/inflow_unified_ai/providers/__init__.py`:

```python
from .${input:provider_file} import ${input:provider_class}Provider

__all__ = [
    # ... existing exports
    "${input:provider_class}Provider",
]
```

### Step 3: Export from main __init__.py

Add to `src/inflow_unified_ai/__init__.py`:

```python
from .providers.${input:provider_file} import ${input:provider_class}Provider

__all__ = [
    # ... existing exports
    "${input:provider_class}Provider",
]
```

### Step 4: Add dependency to pyproject.toml (if needed)

```toml
[project.optional-dependencies]
${input:provider_name_lower} = ["${input:client_lib}>=1.0.0"]
```

### Step 5: Create Tests

Create `tests/test_${input:provider_file}.py`:

```python
"""Tests for ${input:provider_class}Provider."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from inflow_unified_ai import ${input:provider_class}Provider, TextMessage


def test_provider_import():
    """Test that provider can be imported."""
    from inflow_unified_ai import ${input:provider_class}Provider
    assert ${input:provider_class}Provider is not None


def test_provider_initialization():
    """Test provider initialization."""
    provider = ${input:provider_class}Provider(
        api_key="test-key",
        model="test-model"
    )
    assert provider.api_key == "test-key"
    assert provider.model == "test-model"


@pytest.mark.asyncio
async def test_generate_with_mock():
    """Test generate with mocked client."""
    provider = ${input:provider_class}Provider(api_key="test-key")
    
    # Mock the client
    with patch.object(provider, '_get_client') as mock_client:
        mock_client.return_value.chat = AsyncMock(
            return_value=MagicMock(content="Hello!")
        )
        
        messages = [TextMessage(role="user", content="Hi")]
        # response = await provider.generate(messages)
        # assert response == "Hello!"
```

### Step 6: Update Documentation

Add to `docs/07-API-REFERENCE.md`:

```markdown
### ${input:provider_class}Provider

Provider for ${input:provider_name} models.

\```python
from inflow_unified_ai import ${input:provider_class}Provider
\```

#### Constructor

\```python
${input:provider_class}Provider(
    api_key: str,
    model: str = "${input:default_model}"
)
\```

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | `str` | ${input:provider_name} API key |
| `model` | `str` | Model name |

**Example**:
\```python
provider = ${input:provider_class}Provider(
    api_key=os.getenv("${input:env_var}"),
    model="${input:default_model}"
)
\```
```

## Checklist

- [ ] Provider file created
- [ ] Exported from `providers/__init__.py`
- [ ] Exported from main `__init__.py`
- [ ] Dependencies added (if needed)
- [ ] Unit tests created
- [ ] Documentation updated
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Lint passes: `ruff check src/`
