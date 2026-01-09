"""Models package for inflow-unified-ai."""

from inflow_unified_ai.models.requests import (
    CompletionRequest,
    StructuredRequest,
    Message,
    MessageRole,
    ContentPart,
    ReasoningEffort,
    ResponseFormat,
    ToolDefinition,
)
from inflow_unified_ai.models.responses import (
    CompletionResponse,
    CompletionChunk,
    Usage,
    Choice,
    ToolCall,
)
from inflow_unified_ai.models.capabilities import (
    ModelCapabilities,
    ModelFamily,
    MODEL_CAPABILITIES_REGISTRY,
    get_model_capabilities,
    is_reasoning_model,
)

__all__ = [
    # Requests
    "CompletionRequest",
    "StructuredRequest",
    "Message",
    "MessageRole",
    "ContentPart",
    "ReasoningEffort",
    "ResponseFormat",
    "ToolDefinition",
    # Responses
    "CompletionResponse",
    "CompletionChunk",
    "Usage",
    "Choice",
    "ToolCall",
    # Capabilities
    "ModelCapabilities",
    "ModelFamily",
    "MODEL_CAPABILITIES_REGISTRY",
    "get_model_capabilities",
    "is_reasoning_model",
]
