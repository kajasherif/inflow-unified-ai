"""
inflow-unified-ai: Unified AI Abstraction Layer for iNextLabs products.

This library provides a consistent API to interact with multiple AI providers
including Azure OpenAI, Anthropic, Google Gemini, and vLLM.
"""

from inflow_unified_ai.client import AIClient
from inflow_unified_ai.models.requests import (
    CompletionRequest,
    StructuredRequest,
    Message,
    MessageRole,
    ResponseFormat,
    ReasoningEffort,
)
from inflow_unified_ai.models.responses import CompletionResponse, CompletionChunk, Usage
from inflow_unified_ai.models.capabilities import (
    ModelCapabilities,
    ModelFamily,
    get_model_capabilities,
    is_reasoning_model,
)
from inflow_unified_ai.providers.base import LLMProvider, ProviderError
from inflow_unified_ai.providers.factory import ModelFactory, register_provider
from inflow_unified_ai.resilience.retry import RetryConfig, with_retry

__version__ = "0.1.0"

__all__ = [
    # Main client
    "AIClient",
    # Request models
    "CompletionRequest",
    "StructuredRequest",
    "Message",
    "MessageRole",
    "ResponseFormat",
    "ReasoningEffort",
    # Response models
    "CompletionResponse",
    "CompletionChunk",
    "Usage",
    # Capabilities
    "ModelCapabilities",
    "ModelFamily",
    "get_model_capabilities",
    "is_reasoning_model",
    # Provider base
    "LLMProvider",
    "ProviderError",
    "ModelFactory",
    "register_provider",
    # Resilience
    "RetryConfig",
    "with_retry",
]
