# inflow-unified-ai Documentation

Welcome to the comprehensive documentation for the inflow-unified-ai package - a unified abstraction layer for multiple AI providers.

## Quick Links

| Document | Description |
|----------|-------------|
| [Getting Started](./01-GETTING-STARTED.md) | Installation, basic usage, and quick examples |
| [Package Architecture](./02-PACKAGE-ARCHITECTURE.md) | Directory structure, components, and data flow |
| [Development Guide](./03-DEVELOPMENT-GUIDE.md) | Setup, adding providers/models, code style |
| [CI/CD Pipeline](./04-CI-CD-PIPELINE.md) | GitHub Actions, workflows, and deployment |
| [Testing Guide](./05-TESTING-GUIDE.md) | Running tests, writing tests, coverage |
| [Publishing Guide](./06-PUBLISHING-GUIDE.md) | TestPyPI, PyPI, version management |
| [API Reference](./07-API-REFERENCE.md) | Complete API documentation |

## What is inflow-unified-ai?

inflow-unified-ai is a Python library that provides a unified interface for interacting with multiple AI providers:

- **Azure OpenAI** - GPT-4o, GPT-4.1, o1, o3, o4-mini
- **OpenAI** - Direct OpenAI API access
- **Anthropic** - Claude 3, Claude 3.5, Claude 4
- **Google** - Gemini Pro, Gemini 2, Gemini 2.5

## Key Features

✅ **Unified API** - Same interface for all providers
✅ **Model Registry** - Pre-configured models with capabilities
✅ **Vision Support** - Image analysis across providers
✅ **Streaming** - Real-time response streaming
✅ **Async-First** - Built for async/await patterns
✅ **Type Safety** - Full type hints and validation

## Quick Start

```bash
pip install inflow-unified-ai
```

```python
from inflow_unified_ai import AIClient, AzureOpenAIProvider, TextMessage

provider = AzureOpenAIProvider(
    api_key="your-key",
    endpoint="your-endpoint",
    deployment="gpt-4o"
)

client = AIClient(provider=provider)
response = await client.generate([
    TextMessage(role="user", content="Hello!")
])
```

## Package Status

| Item | Status |
|------|--------|
| Version | 0.1.0 |
| Python | 3.10+ |
| TestPyPI | ✅ Published |
| PyPI | 🔜 Coming soon |
| CI/CD | ✅ Active |
| Tests | 16 passing |

## Links

- **TestPyPI**: https://test.pypi.org/project/inflow-unified-ai/
- **GitHub**: https://github.com/kajasherif/inflow-unified-ai
- **GitHub Actions**: https://github.com/kajasherif/inflow-unified-ai/actions

## Contributing

1. Clone the repository
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest tests/ -v`
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
