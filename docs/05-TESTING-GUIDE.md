# Testing Guide

## Overview

This guide covers how to run tests, write new tests, and understand the testing infrastructure for the inflow-unified-ai package.

## Test Framework

- **pytest**: Main test runner
- **pytest-asyncio**: For testing async functions
- **pytest-cov**: For code coverage reports

## Running Tests

### Basic Test Run

```bash
# Run all tests
pytest tests/ -v

# Run with verbose output
pytest tests/ -v --tb=short

# Run specific test file
pytest tests/test_package.py -v

# Run specific test function
pytest tests/test_package.py::test_aiclient_import -v
```

### With Coverage

```bash
# Run with coverage report
pytest tests/ -v --cov=inflow_unified_ai

# Generate HTML coverage report
pytest tests/ -v --cov=inflow_unified_ai --cov-report=html

# View coverage in browser
start htmlcov/index.html  # Windows
open htmlcov/index.html   # macOS
```

### Running Tests with Different Python Versions

```bash
# Using py launcher (Windows)
py -3.10 -m pytest tests/ -v
py -3.11 -m pytest tests/ -v
py -3.12 -m pytest tests/ -v

# Using specific Python path
C:\Python310\python.exe -m pytest tests/ -v
```

## Test Structure

### Directory Layout

```
tests/
├── __init__.py           # Makes tests a package
├── conftest.py           # Shared fixtures (if needed)
├── test_package.py       # Main package tests
├── test_models.py        # Model-specific tests
├── test_providers.py     # Provider-specific tests
└── test_integration.py   # Integration tests (needs API keys)
```

### Current Tests

The `test_package.py` file contains 16 tests covering:

| Test Name | Description |
|-----------|-------------|
| `test_aiclient_import` | Verifies AIClient can be imported |
| `test_base_message_import` | Verifies BaseMessage can be imported |
| `test_text_message_import` | Verifies TextMessage can be imported |
| `test_image_message_import` | Verifies ImageMessage can be imported |
| `test_aimodel_import` | Verifies AIModel can be imported |
| `test_model_family_import` | Verifies ModelFamily enum can be imported |
| `test_model_capabilities_import` | Verifies ModelCapabilities can be imported |
| `test_model_registry_import` | Verifies ModelRegistry can be imported |
| `test_create_text_message` | Tests TextMessage creation |
| `test_create_image_message` | Tests ImageMessage creation |
| `test_model_capabilities_defaults` | Tests ModelCapabilities defaults |
| `test_model_capabilities_custom` | Tests custom ModelCapabilities |
| `test_aimodel_creation` | Tests AIModel instantiation |
| `test_model_registry_get_model` | Tests ModelRegistry.get_model() |
| `test_model_registry_list_models` | Tests ModelRegistry.list_models() |
| `test_aiclient_initialization` | Tests AIClient init with mock |

## Writing Tests

### Basic Test Structure

```python
"""Test module docstring."""
import pytest
from inflow_unified_ai import AIClient, TextMessage

def test_feature_works():
    """Test that feature works correctly."""
    # Arrange
    input_data = "test"
    
    # Act
    result = some_function(input_data)
    
    # Assert
    assert result == expected_value

class TestFeatureGroup:
    """Group related tests."""
    
    def test_scenario_one(self):
        """Test first scenario."""
        assert True
    
    def test_scenario_two(self):
        """Test second scenario."""
        assert True
```

### Testing Async Functions

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test an async function."""
    result = await some_async_function()
    assert result is not None
```

### Using Fixtures

```python
# In conftest.py or test file
@pytest.fixture
def sample_client():
    """Create a sample client for testing."""
    from unittest.mock import MagicMock
    mock_provider = MagicMock()
    client = AIClient(provider=mock_provider)
    return client

# In test function
def test_with_fixture(sample_client):
    """Test using the fixture."""
    assert sample_client is not None
```

### Mocking External Dependencies

```python
from unittest.mock import MagicMock, patch

def test_with_mock():
    """Test with mocked dependencies."""
    # Mock a class
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "mocked response"
    
    client = AIClient(provider=mock_provider)
    result = client.generate("test")
    
    assert result == "mocked response"
    mock_provider.generate.assert_called_once_with("test")

@patch('inflow_unified_ai.providers.azure_openai.AzureOpenAI')
def test_with_patch(mock_azure):
    """Test with patched module."""
    mock_azure.return_value.chat.completions.create.return_value = "response"
    # Test code here
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("input_val,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("test", "TEST"),
])
def test_uppercase(input_val, expected):
    """Test multiple inputs."""
    assert input_val.upper() == expected
```

## Test Categories

### Unit Tests (No API Required)

These tests mock external dependencies:

```python
def test_message_creation():
    """Unit test - no external calls."""
    msg = TextMessage(content="Hello")
    assert msg.content == "Hello"
```

### Integration Tests (Requires API Keys)

Mark these to skip when API keys aren't available:

```python
import os
import pytest

@pytest.mark.skipif(
    not os.getenv("AZURE_OPENAI_API_KEY"),
    reason="Azure OpenAI API key not set"
)
async def test_real_api_call():
    """Integration test - needs real API."""
    client = AIClient(
        provider=AzureOpenAIProvider(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    )
    response = await client.generate([TextMessage(content="Hi")])
    assert response is not None
```

## pytest.ini Configuration

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts = -v --tb=short
markers =
    integration: marks tests as integration tests (deselect with '-m "not integration"')
    slow: marks tests as slow (deselect with '-m "not slow"')
```

## Coverage Goals

| Component | Target Coverage |
|-----------|-----------------|
| Core (AIClient) | 90%+ |
| Models | 85%+ |
| Providers | 80%+ |
| Utils | 75%+ |

## Test Best Practices

### Do's

✅ Test one thing per test function
✅ Use descriptive test names
✅ Include docstrings explaining what's tested
✅ Mock external dependencies
✅ Test edge cases and error conditions
✅ Keep tests fast (< 1 second each)
✅ Use fixtures for common setup

### Don'ts

❌ Don't test implementation details
❌ Don't make tests dependent on each other
❌ Don't commit API keys in test files
❌ Don't skip tests without a reason
❌ Don't test third-party code

## Common Test Patterns

### Testing Exceptions

```python
import pytest

def test_raises_error():
    """Test that function raises expected error."""
    with pytest.raises(ValueError) as exc_info:
        function_that_raises(invalid_input)
    
    assert "expected message" in str(exc_info.value)
```

### Testing Multiple Assertions

```python
def test_complex_object():
    """Test object with multiple properties."""
    obj = create_complex_object()
    
    assert obj.name == "expected"
    assert obj.value > 0
    assert obj.items == ["a", "b", "c"]
```

### Testing with Environment Variables

```python
import os
from unittest.mock import patch

def test_with_env_var():
    """Test behavior with environment variable."""
    with patch.dict(os.environ, {"API_KEY": "test-key"}):
        result = function_using_env_var()
        assert result is not None
```

## Continuous Integration

Tests run automatically on:
- Every push to any branch
- Every pull request to main
- Python versions: 3.10, 3.11, 3.12

Check test status at: https://github.com/kajasherif/inflow-unified-ai/actions
