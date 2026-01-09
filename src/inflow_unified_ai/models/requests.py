"""
Request models for the Unified AI Abstraction Layer.

These models provide a standardized way to define requests across
different AI providers while handling provider-specific parameter
mappings internally.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    """Supported message roles across providers."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    DEVELOPER = "developer"  # For reasoning models (equivalent to system)
    TOOL = "tool"
    FUNCTION = "function"


class ContentPart(BaseModel):
    """Multi-modal content part (text or image)."""

    type: Literal["text", "image_url"] = "text"
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class Message(BaseModel):
    """
    A single message in a conversation.

    Supports both simple text content and multi-modal content (text + images).
    """

    role: Union[MessageRole, str]
    content: Union[str, List[ContentPart], None] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    @field_validator("role", mode="before")
    @classmethod
    def normalize_role(cls, v: Any) -> Union[MessageRole, str]:
        """Normalize role to MessageRole enum if possible."""
        if isinstance(v, MessageRole):
            return v
        if isinstance(v, str):
            try:
                return MessageRole(v.lower())
            except ValueError:
                return v
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        result: Dict[str, Any] = {
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role
        }

        if self.content is not None:
            if isinstance(self.content, str):
                result["content"] = self.content
            else:
                result["content"] = [part.model_dump(exclude_none=True) for part in self.content]

        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        return result


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for reasoning models (o-series, GPT-5+)."""

    NONE = "none"  # Only supported by gpt-5.1
    MINIMAL = "minimal"  # Supported by GPT-5 series (except gpt-5-codex)
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"  # Only supported by gpt-5.1-codex-max


class ResponseFormat(BaseModel):
    """Response format configuration."""

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    """Tool/function definition for function calling."""

    type: Literal["function"] = "function"
    function: Dict[str, Any]


class CompletionRequest(BaseModel):
    """
    Standardized completion request that works across all providers.

    The library automatically handles parameter mapping based on model
    capabilities. For example:
    - Reasoning models (o-series, GPT-5) will ignore temperature and use
      max_completion_tokens instead of max_tokens
    - Standard models use max_tokens and support temperature
    """

    # Required fields
    model: str = Field(..., description="Model identifier (e.g., 'gpt-4o', 'o3-mini', 'gpt-5')")
    messages: List[Message] = Field(..., description="List of messages in the conversation")

    # Token limits - library auto-selects the right one based on model
    max_tokens: Optional[int] = Field(
        None, description="Max tokens for standard models. Ignored for reasoning models."
    )
    max_completion_tokens: Optional[int] = Field(
        None, description="Max completion tokens for reasoning models. Takes precedence if set."
    )

    # Sampling parameters - ignored for reasoning models
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0-2). Not supported by reasoning models.",
    )
    top_p: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Nucleus sampling. Not supported by reasoning models."
    )

    # Penalty parameters - ignored for reasoning models
    presence_penalty: Optional[float] = Field(
        None, ge=-2.0, le=2.0, description="Presence penalty. Not supported by reasoning models."
    )
    frequency_penalty: Optional[float] = Field(
        None, ge=-2.0, le=2.0, description="Frequency penalty. Not supported by reasoning models."
    )

    # Reasoning-specific parameters
    reasoning_effort: Optional[Union[ReasoningEffort, str]] = Field(
        None, description="Reasoning effort level for reasoning models (low/medium/high)."
    )

    # Response configuration
    response_format: Optional[ResponseFormat] = None
    stop: Optional[Union[str, List[str]]] = None

    # Streaming
    stream: bool = Field(False, description="Whether to stream the response")

    # Tools/Functions
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None

    # Other options
    seed: Optional[int] = None
    user: Optional[str] = None
    n: int = Field(1, ge=1, description="Number of completions to generate")

    # Azure-specific
    data_sources: Optional[List[Dict[str, Any]]] = Field(
        None, description="Azure OpenAI data sources for RAG scenarios"
    )

    # Metadata
    metadata: Optional[Dict[str, str]] = None
    store: Optional[bool] = None  # For stored completions

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def normalize_reasoning_effort(cls, v: Any) -> Optional[Union[ReasoningEffort, str]]:
        """Normalize reasoning effort to enum if possible."""
        if v is None:
            return None
        if isinstance(v, ReasoningEffort):
            return v
        if isinstance(v, str):
            try:
                return ReasoningEffort(v.lower())
            except ValueError:
                return v
        return v

    def get_effective_max_tokens(self, is_reasoning_model: bool) -> Optional[int]:
        """
        Get the effective max tokens value based on model type.

        For reasoning models, returns max_completion_tokens.
        For standard models, returns max_tokens.
        """
        if is_reasoning_model:
            return self.max_completion_tokens or self.max_tokens
        return self.max_tokens


class StructuredRequest(CompletionRequest):
    """
    Request for structured output generation (JSON schema-based).

    Extends CompletionRequest with schema definition for structured outputs.
    """

    output_schema: Dict[str, Any] = Field(
        ..., description="JSON schema for the expected output structure"
    )
    strict: bool = Field(True, description="Whether to enforce strict schema validation")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Auto-set response format for structured output
        if self.response_format is None:
            self.response_format = ResponseFormat(
                type="json_schema",
                json_schema={
                    "name": "structured_output",
                    "strict": self.strict,
                    "schema": self.output_schema,
                },
            )
