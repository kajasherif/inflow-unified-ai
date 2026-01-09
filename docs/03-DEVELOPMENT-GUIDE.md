# Development Guide

## Prerequisites

- Python 3.10 or higher
- Git
- pip or uv package manager

## Setting Up Development Environment

### 1. Clone the Repository

```bash
git clone https://github.com/kajasherif/inflow-unified-ai.git
cd inflow-unified-ai
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv .venv

# Activate on Windows
.\.venv\Scripts\Activate.ps1

# Activate on Linux/Mac
source .venv/bin/activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs:
- The package in editable mode
- Testing tools (pytest, pytest-asyncio, pytest-cov)
- Linting tools (ruff, mypy)
- Build tools (build, twine)

### 4. Set Up Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your API keys
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
# AZURE_OPENAI_API_KEY_1=your-primary-key
# AZURE_OPENAI_API_KEY_2=your-secondary-key
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=inflow_unified_ai --cov-report=html

# Run specific test
pytest tests/test_package.py::TestModelCapabilities -v
```

### Linting and Formatting

```bash
# Check for lint errors
ruff check src/

# Auto-fix lint errors
ruff check src/ --fix

# Format code
ruff format src/

# Type checking
mypy src/inflow_unified_ai --ignore-missing-imports
```

### Building the Package

```bash
# Build wheel and tarball
python -m build

# Check package quality
twine check dist/*
```

## Adding a New Model

### Step 1: Add Model Capabilities

Edit `src/inflow_unified_ai/models/capabilities.py`:

```python
# Add to MODEL_CAPABILITIES_REGISTRY
"gpt-6": ModelCapabilities(
    family=ModelFamily.GPT_6,
    supports_streaming=True,
    supports_structured_output=True,
    supports_temperature=True,  # False for reasoning models
    supports_system_message=True,
    max_output_tokens=16384,
    context_window=128000,
),
```

### Step 2: Add Model Family (if new)

```python
class ModelFamily(Enum):
    # ... existing families
    GPT_6 = "gpt-6"
```

### Step 3: Add Tests

Edit `tests/test_package.py`:

```python
def test_get_gpt6_capabilities(self):
    """Test GPT-6 capabilities."""
    caps = get_model_capabilities("gpt-6")
    assert caps is not None
    assert caps.supports_streaming is True
```

### Step 4: Run Tests and Commit

```bash
pytest tests/ -v
git add .
git commit -m "Add support for GPT-6 model"
git push
```

## Adding a New Provider

### Step 1: Create Provider File

Create `src/inflow_unified_ai/providers/new_provider.py`:

```python
"""
New Provider implementation.
"""

from typing import AsyncIterator
from ..providers.base import LLMProvider, ProviderError
from ..models.requests import CompletionRequest
from ..models.responses import CompletionResponse, CompletionChunk


class NewProvider(LLMProvider):
    """Provider for New AI Service."""
    
    def __init__(
        self,
        api_key: str,
        endpoint: str = None,
        **kwargs
    ):
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.newprovider.com"
    
    async def agenerate(
        self,
        request: CompletionRequest
    ) -> CompletionResponse:
        """Generate a completion asynchronously."""
        # Implement API call here
        pass
    
    async def astream(
        self,
        request: CompletionRequest
    ) -> AsyncIterator[CompletionChunk]:
        """Stream a completion asynchronously."""
        # Implement streaming API call here
        pass
```

### Step 2: Register Provider

Edit `src/inflow_unified_ai/providers/__init__.py`:

```python
from .new_provider import NewProvider

__all__ = [
    # ... existing exports
    "NewProvider",
]
```

### Step 3: Add to Factory

Edit `src/inflow_unified_ai/providers/factory.py`:

```python
from .new_provider import NewProvider

# In ModelFactory class
PROVIDERS = {
    # ... existing providers
    "new_provider": NewProvider,
    "new": NewProvider,  # Alias
}
```

### Step 4: Add Tests

Create `tests/test_new_provider.py`:

```python
def test_new_provider_import(self):
    """Test NewProvider can be imported."""
    from inflow_unified_ai.providers import NewProvider
    assert NewProvider is not None
```

## Code Style Guidelines

### Python Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Use docstrings for all public classes and functions
- Maximum line length: 88 characters (ruff default)

### Import Order

```python
# Standard library
import os
import asyncio
from typing import Optional, List

# Third-party
import structlog
from pydantic import BaseModel

# Local
from inflow_unified_ai.models import Message
from inflow_unified_ai.providers import LLMProvider
```

### Docstring Format

```python
def function_name(param1: str, param2: int = 10) -> str:
    """Short description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 10.
    
    Returns:
        Description of return value.
    
    Raises:
        ProviderError: If something goes wrong.
    """
    pass
```

## Commit Message Guidelines

Follow conventional commits:

```
feat: add support for new model
fix: handle streaming timeout error
docs: update API reference
test: add tests for new provider
chore: update dependencies
refactor: simplify retry logic
```

## Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/new-feature
   ```

2. Make changes and commit:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

3. Run tests and linting:
   ```bash
   pytest tests/ -v
   ruff check src/
   ruff format src/
   ```

4. Push and create PR:
   ```bash
   git push origin feature/new-feature
   ```

5. Request review and merge

## Troubleshooting

### Import Errors

If you get import errors, make sure:
1. Virtual environment is activated
2. Package is installed in editable mode: `pip install -e .`

### API Errors

Check:
1. Environment variables are set correctly
2. API keys are valid
3. Endpoint URLs are correct

### Test Failures

1. Run specific failing test with verbose output:
   ```bash
   pytest tests/test_package.py::test_name -v -s
   ```

2. Check for missing dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
