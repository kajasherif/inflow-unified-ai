# GitHub Copilot Instructions

## Project Overview

This is **inflow-unified-ai** - a unified Python abstraction layer for multiple AI providers. The package provides a consistent interface for Azure OpenAI, OpenAI, Anthropic Claude, and Google Gemini models.

### Package Information
- **Name**: `inflow-unified-ai`
- **Version**: `0.1.0`
- **Python**: `3.10+`
- **Build System**: `hatchling`
- **Package Location**: `src/inflow_unified_ai/`

### Key Components

```
src/inflow_unified_ai/
├── __init__.py          # Public API exports
├── client.py            # AIClient - main entry point
├── models/
│   ├── messages.py      # BaseMessage, TextMessage, ImageMessage
│   ├── ai_model.py      # AIModel, ModelFamily, ModelCapabilities
│   └── registry.py      # ModelRegistry with pre-configured models
└── providers/
    ├── base.py          # BaseProvider abstract class
    ├── azure_openai.py  # AzureOpenAIProvider
    ├── openai_provider.py # OpenAIProvider
    ├── anthropic.py     # AnthropicProvider
    └── google.py        # GoogleProvider
```

---

## Development Patterns

### Package Import Structure
All public APIs are exported from the root `__init__.py`:

```python
from inflow_unified_ai import (
    AIClient,
    TextMessage,
    ImageMessage,
    AIModel,
    ModelFamily,
    ModelCapabilities,
    ModelRegistry,
    AzureOpenAIProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider
)
```

### Async-First Design
All provider methods are async:

```python
async def generate(self, messages: List[BaseMessage], **kwargs) -> str:
    """Generate a response from the AI model."""
    pass

async def stream(self, messages: List[BaseMessage], **kwargs) -> AsyncGenerator[str, None]:
    """Stream responses from the AI model."""
    pass
```

### Type Safety
Use Pydantic models and dataclasses with full type hints:

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelCapabilities:
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_function_calling: bool = False
```

### Environment Variables
Never hardcode API keys. Always use environment variables:

```python
import os

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
```

---

## Workflows & Commands

### Development Setup
```bash
# Clone and setup
git clone https://github.com/kajasherif/inflow-unified-ai.git
cd inflow-unified-ai
pip install -e ".[dev]"
```

### Running Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=inflow_unified_ai

# Specific test
pytest tests/test_package.py::test_aiclient_import -v
```

### Linting & Formatting
```bash
# Check lint errors
ruff check src/

# Auto-fix lint errors
ruff check src/ --fix

# Format code
ruff format src/

# Type checking
mypy src/inflow_unified_ai --ignore-missing-imports
```

### Building Package
```bash
# Clean previous builds
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue

# Build
python -m build

# Validate
twine check dist/*
```

### Publishing
```bash
# TestPyPI (automatic via CI/CD on push to main)
git push origin main

# PyPI (automatic via CI/CD on version tag)
git tag v0.2.0
git push origin v0.2.0
```

---

## Agent Modes

### @python-package-dev - Package Development Agent

**Activation**: When working on package source code, adding features, or fixing bugs.

**Behavior**:
- Focus on `src/inflow_unified_ai/` directory
- Follow async-first patterns for all provider methods
- Ensure all public APIs are exported in `__init__.py`
- Use type hints on all functions and methods
- Follow the existing code style and patterns

**Key Files**:
- `src/inflow_unified_ai/client.py` - Main AIClient class
- `src/inflow_unified_ai/providers/*.py` - Provider implementations
- `src/inflow_unified_ai/models/*.py` - Data models

**Conventions**:
```python
# All provider methods must be async
async def generate(self, messages, **kwargs) -> str:
    ...

# Use dataclasses for simple data structures
@dataclass
class ModelCapabilities:
    supports_vision: bool = False

# Use Pydantic for validation-heavy models
class TextMessage(BaseMessage):
    content: str
    role: str = "user"
```

---

### @test-writer - Test Writing Agent

**Activation**: When writing or updating tests.

**Behavior**:
- Write tests in `tests/` directory
- Use pytest with pytest-asyncio for async tests
- Mock external dependencies (API calls)
- Follow Arrange-Act-Assert pattern
- Include docstrings explaining what each test validates

**Test Patterns**:
```python
# Basic test structure
def test_feature_name():
    """Test that feature does X correctly."""
    # Arrange
    input_data = create_input()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected

# Async test
@pytest.mark.asyncio
async def test_async_feature():
    """Test async function."""
    result = await async_function()
    assert result is not None

# Mock external calls
from unittest.mock import MagicMock, patch

def test_with_mock():
    """Test with mocked dependency."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "mocked"
    client = AIClient(provider=mock_provider)
    # ...

# Parametrized tests
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
])
def test_multiple_cases(input, expected):
    assert input.upper() == expected
```

**Coverage Targets**:
- Core (AIClient): 90%+
- Models: 85%+
- Providers: 80%+

---

### @docs-writer - Documentation Agent

**Activation**: When writing or updating documentation.

**Behavior**:
- Write documentation in `docs/` directory
- Use clear, concise language
- Include code examples
- Follow markdown best practices
- Keep API reference in sync with code

**Documentation Structure**:
```
docs/
├── README.md              # Index
├── 01-GETTING-STARTED.md  # Installation & quickstart
├── 02-PACKAGE-ARCHITECTURE.md
├── 03-DEVELOPMENT-GUIDE.md
├── 04-CI-CD-PIPELINE.md
├── 05-TESTING-GUIDE.md
├── 06-PUBLISHING-GUIDE.md
└── 07-API-REFERENCE.md
```

**Style Guide**:
- Use tables for parameter documentation
- Include both basic and advanced examples
- Add "See Also" sections for related topics
- Use admonitions for warnings/notes

---

## Chat Modes

### #plan - Planning Mode

**Activation**: When planning changes before implementation.

**Behavior**:
- Analyze requirements thoroughly
- Identify affected files and components
- Create step-by-step implementation plan
- List potential risks and considerations
- **DO NOT** generate code - only plan

**Output Format**:
```markdown
## Change Plan: [Feature Name]

### Affected Files
- `file1.py` - Description of changes
- `file2.py` - Description of changes

### Implementation Steps
1. Step one
2. Step two
3. Step three

### Testing Requirements
- Test case 1
- Test case 2

### Risks & Considerations
- Risk 1 and mitigation
- Risk 2 and mitigation
```

---

### #review - Code Review Mode

**Activation**: When reviewing code changes.

**Behavior**:
- Check for type safety and proper typing
- Verify async patterns are correct
- Ensure tests exist for new code
- Check for security issues (hardcoded keys, etc.)
- Verify documentation is updated
- Check for breaking changes

**Review Checklist**:
```markdown
## Code Review

### Type Safety
- [ ] All functions have type hints
- [ ] Return types are specified
- [ ] No `Any` types without justification

### Async Patterns
- [ ] Provider methods are async
- [ ] Proper await usage
- [ ] No blocking calls in async code

### Testing
- [ ] Unit tests added/updated
- [ ] Edge cases covered
- [ ] Mocks used for external calls

### Documentation
- [ ] Docstrings present
- [ ] API reference updated
- [ ] CHANGELOG updated (if applicable)

### Security
- [ ] No hardcoded credentials
- [ ] Environment variables used
- [ ] No sensitive data in logs
```

---

## Prompts / Tasks

### /add-provider - Add New AI Provider

**Purpose**: Add a new AI provider to the package.

**Steps**:
1. Create provider file in `src/inflow_unified_ai/providers/`
2. Inherit from `BaseProvider`
3. Implement required async methods: `generate()`, `stream()`
4. Add provider export to `providers/__init__.py`
5. Add provider export to main `__init__.py`
6. Add unit tests in `tests/test_providers.py`
7. Update documentation

**Template**:
```python
# src/inflow_unified_ai/providers/new_provider.py
"""NewProvider implementation."""
from typing import List, AsyncGenerator
from .base import BaseProvider
from ..models.messages import BaseMessage


class NewProvider(BaseProvider):
    """Provider for New AI Service."""

    def __init__(
        self,
        api_key: str,
        model: str = "default-model",
        **kwargs
    ):
        self.api_key = api_key
        self.model = model
        self._client = None

    async def generate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> str:
        """Generate a response."""
        # Implementation here
        pass

    async def stream(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream responses."""
        # Implementation here
        yield ""
```

---

### /add-model - Add New Model to Registry

**Purpose**: Add a new AI model to the ModelRegistry.

**Steps**:
1. Add model family to `ModelFamily` enum (if new family)
2. Create `AIModel` instance with capabilities
3. Register in `ModelRegistry._models`
4. Update tests to include new model
5. Update documentation

**Template**:
```python
# In src/inflow_unified_ai/models/registry.py

# 1. Add to ModelFamily enum (if new)
class ModelFamily(Enum):
    NEW_MODEL = "new-model"

# 2. Add to registry
_models = {
    "new-model-id": AIModel(
        model_id="new-model-id",
        provider="provider-name",
        family=ModelFamily.NEW_MODEL,
        capabilities=ModelCapabilities(
            supports_vision=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
        ),
        context_window=128000,
        max_output_tokens=4096,
    ),
}
```

---

### /release - Release Checklist

**Purpose**: Prepare and execute a new release.

**Pre-Release Checklist**:
```markdown
## Release v{VERSION}

### Pre-Release
- [ ] All tests passing: `pytest tests/ -v`
- [ ] Lint clean: `ruff check src/`
- [ ] Type check clean: `mypy src/inflow_unified_ai`
- [ ] Version updated in `pyproject.toml`
- [ ] CHANGELOG updated (if exists)
- [ ] Documentation updated

### Release
- [ ] Commit: `git commit -am "Bump version to {VERSION}"`
- [ ] Push to main: `git push origin main`
- [ ] Verify TestPyPI build succeeds
- [ ] Test install from TestPyPI
- [ ] Create tag: `git tag v{VERSION}`
- [ ] Push tag: `git push origin v{VERSION}`
- [ ] Verify PyPI build succeeds
- [ ] Test install from PyPI

### Post-Release
- [ ] Create GitHub Release with notes
- [ ] Announce release (if applicable)
- [ ] Update version to next dev version
```

**Commands**:
```bash
# Version bump and release
git add pyproject.toml
git commit -m "Bump version to 0.2.0"
git push origin main

# Wait for TestPyPI, then:
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ inflow-unified-ai==0.2.0

# If successful:
git tag v0.2.0
git push origin v0.2.0
```

---

## CI/CD Pipeline

### Workflow File
`.github/workflows/ci-cd.yml`

### Jobs
| Job | Trigger | Description |
|-----|---------|-------------|
| `lint` | All pushes/PRs | Ruff lint + MyPy type check |
| `test` | All pushes/PRs | pytest on Python 3.10, 3.11, 3.12 |
| `build` | After lint+test | Build wheel and sdist |
| `publish-testpypi` | Push to main | Publish to TestPyPI |
| `publish-pypi` | Version tag (v*) | Publish to PyPI |
| `release` | After PyPI | Create GitHub Release |

### Trusted Publishing Setup
- TestPyPI: Environment `testpypi`
- PyPI: Environment `pypi`
- Workflow: `ci-cd.yml`

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package configuration, dependencies, version |
| `src/inflow_unified_ai/__init__.py` | Public API exports |
| `src/inflow_unified_ai/client.py` | AIClient main class |
| `src/inflow_unified_ai/models/registry.py` | Model registry with all models |
| `tests/test_package.py` | Unit tests |
| `.github/workflows/ci-cd.yml` | CI/CD pipeline |

---

## Links

| Resource | URL |
|----------|-----|
| GitHub | https://github.com/kajasherif/inflow-unified-ai |
| TestPyPI | https://test.pypi.org/project/inflow-unified-ai/ |
| PyPI | https://pypi.org/project/inflow-unified-ai/ |
| Actions | https://github.com/kajasherif/inflow-unified-ai/actions |
| Documentation | `docs/` folder |

---

## Quick Reference

### Common Tasks

| Task | Command |
|------|---------|
| Run tests | `pytest tests/ -v` |
| Fix lint | `ruff check src/ --fix` |
| Format | `ruff format src/` |
| Build | `python -m build` |
| Install dev | `pip install -e ".[dev]"` |

### File Naming Conventions
- Providers: `{provider_name}.py` (e.g., `azure_openai.py`)
- Models: `{model_type}.py` (e.g., `messages.py`)
- Tests: `test_{module}.py` (e.g., `test_package.py`)

### Code Style
- Use `ruff` for linting and formatting
- Maximum line length: 88 characters
- Use double quotes for strings
- Sort imports with isort (via ruff)
