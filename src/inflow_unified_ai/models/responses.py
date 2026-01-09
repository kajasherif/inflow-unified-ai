"""
Response models for the Unified AI Abstraction Layer.

These models provide a standardized response format regardless of
which AI provider is being used.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = Field(0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(0, description="Number of tokens in the completion")
    total_tokens: int = Field(0, description="Total tokens used")

    # Detailed breakdown (when available)
    reasoning_tokens: Optional[int] = Field(
        None, description="Tokens used for reasoning (reasoning models only)"
    )
    cached_tokens: Optional[int] = Field(None, description="Tokens served from cache")

    # Additional details from provider
    completion_tokens_details: Optional[Dict[str, Any]] = None
    prompt_tokens_details: Optional[Dict[str, Any]] = None

    @classmethod
    def from_openai(cls, usage_dict: Optional[Dict[str, Any]]) -> "Usage":
        """Create Usage from OpenAI-style usage dict."""
        if not usage_dict:
            return cls()

        reasoning_tokens = None
        cached_tokens = None
        completion_details = usage_dict.get("completion_tokens_details")
        prompt_details = usage_dict.get("prompt_tokens_details")

        if completion_details:
            reasoning_tokens = completion_details.get("reasoning_tokens")
        if prompt_details:
            cached_tokens = prompt_details.get("cached_tokens")

        return cls(
            prompt_tokens=usage_dict.get("prompt_tokens", 0),
            completion_tokens=usage_dict.get("completion_tokens", 0),
            total_tokens=usage_dict.get("total_tokens", 0),
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            completion_tokens_details=completion_details,
            prompt_tokens_details=prompt_details,
        )


class ToolCall(BaseModel):
    """A tool/function call from the model."""

    id: str
    type: str = "function"
    function: Dict[str, Any]


class Choice(BaseModel):
    """A single completion choice."""

    index: int = 0
    message: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class CompletionResponse(BaseModel):
    """
    Standardized response from any AI provider.

    This provides a unified interface regardless of whether you're
    using Azure OpenAI, Anthropic, Gemini, or vLLM.
    """

    # Core response data
    id: str = Field(..., description="Unique response identifier")
    content: str = Field("", description="Generated text content")

    # Model information
    model: str = Field("", description="Model that generated the response")
    provider: str = Field("", description="Provider name (e.g., 'azure_openai')")

    # Token usage
    usage: Usage = Field(default_factory=Usage)

    # Completion details
    finish_reason: Optional[str] = Field(
        None, description="Why generation stopped (stop, length, tool_calls, etc.)"
    )

    # Tool calls (if any)
    tool_calls: Optional[List[ToolCall]] = None

    # Full choices (for n > 1)
    choices: List[Choice] = Field(default_factory=list)

    # Metadata
    created: Optional[datetime] = None
    system_fingerprint: Optional[str] = None

    # Raw response (for debugging)
    raw_response: Optional[Dict[str, Any]] = Field(
        None, description="Original provider response (for debugging)"
    )

    # Reasoning summary (for reasoning models)
    reasoning_summary: Optional[str] = None

    @classmethod
    def from_openai(cls, response: Any, provider: str = "azure_openai") -> "CompletionResponse":
        """Create CompletionResponse from OpenAI-style response."""
        # Handle both dict and object responses
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        elif isinstance(response, dict):
            data = response
        else:
            data = dict(response)

        # Extract content from first choice
        content = ""
        finish_reason = None
        tool_calls = None
        choices = []

        raw_choices = data.get("choices", [])
        for i, choice in enumerate(raw_choices):
            message = choice.get("message", {})

            if i == 0:
                content = message.get("content", "") or ""
                finish_reason = choice.get("finish_reason")

                # Extract tool calls
                raw_tool_calls = message.get("tool_calls")
                if raw_tool_calls:
                    tool_calls = [
                        ToolCall(
                            id=tc.get("id", ""),
                            type=tc.get("type", "function"),
                            function=tc.get("function", {}),
                        )
                        for tc in raw_tool_calls
                    ]

            choices.append(
                Choice(
                    index=choice.get("index", i),
                    message=message,
                    finish_reason=choice.get("finish_reason"),
                    logprobs=choice.get("logprobs"),
                )
            )

        # Parse created timestamp
        created = None
        if "created" in data:
            try:
                created = datetime.fromtimestamp(data["created"])
            except (TypeError, ValueError):
                pass

        return cls(
            id=data.get("id", ""),
            content=content,
            model=data.get("model", ""),
            provider=provider,
            usage=Usage.from_openai(data.get("usage")),
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            choices=choices,
            created=created,
            system_fingerprint=data.get("system_fingerprint"),
            raw_response=data,
        )


class CompletionChunk(BaseModel):
    """
    A single chunk from a streaming completion.

    Used when stream=True in the request.
    """

    id: str = Field(..., description="Chunk identifier")
    content: str = Field("", description="Content delta in this chunk")

    # Model info
    model: str = Field("", description="Model generating the stream")
    provider: str = Field("", description="Provider name")

    # Stream state
    finish_reason: Optional[str] = None
    is_final: bool = Field(False, description="Whether this is the last chunk")

    # Tool call deltas (for streaming tool calls)
    tool_calls: Optional[List[Dict[str, Any]]] = None

    # Usage (only in final chunk with stream_options.include_usage)
    usage: Optional[Usage] = None

    @classmethod
    def from_openai_chunk(cls, chunk: Any, provider: str = "azure_openai") -> "CompletionChunk":
        """Create CompletionChunk from OpenAI-style stream chunk."""
        if hasattr(chunk, "model_dump"):
            data = chunk.model_dump()
        elif isinstance(chunk, dict):
            data = chunk
        else:
            data = dict(chunk)

        content = ""
        finish_reason = None
        tool_calls = None

        choices = data.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "") or ""
            finish_reason = choices[0].get("finish_reason")
            tool_calls = delta.get("tool_calls")

        usage = None
        if "usage" in data and data["usage"]:
            usage = Usage.from_openai(data["usage"])

        return cls(
            id=data.get("id", ""),
            content=content,
            model=data.get("model", ""),
            provider=provider,
            finish_reason=finish_reason,
            is_final=finish_reason is not None,
            tool_calls=tool_calls,
            usage=usage,
        )
