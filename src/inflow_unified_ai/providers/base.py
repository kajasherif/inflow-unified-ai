"""
Base provider interface for the Unified AI Abstraction Layer.

All AI provider adapters must implement this abstract base class
to ensure consistent behavior across different providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Iterator, Optional, Any

from inflow_unified_ai.models.requests import CompletionRequest
from inflow_unified_ai.models.responses import CompletionResponse, CompletionChunk
from inflow_unified_ai.models.capabilities import ModelCapabilities, get_model_capabilities


class LLMProvider(ABC):
    """
    Abstract base class for LLM provider adapters.

    Each provider (Azure OpenAI, Anthropic, Gemini, vLLM) implements
    this interface to provide a consistent API for the AIClient.
    """

    # Provider identification
    provider_name: str = "base"

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the provider.

        Args:
            api_key: API key for authentication
            endpoint: API endpoint URL
            api_version: API version (for Azure)
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self._config = kwargs

    @abstractmethod
    def generate(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion synchronously.

        Args:
            request: The completion request

        Returns:
            CompletionResponse with the generated content

        Raises:
            ProviderError: If the API call fails
        """
        pass

    @abstractmethod
    async def agenerate(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion asynchronously.

        Args:
            request: The completion request

        Returns:
            CompletionResponse with the generated content

        Raises:
            ProviderError: If the API call fails
        """
        pass

    @abstractmethod
    def stream(self, request: CompletionRequest) -> Iterator[CompletionChunk]:
        """
        Stream a completion synchronously.

        Args:
            request: The completion request (stream=True)

        Yields:
            CompletionChunk for each streamed part

        Raises:
            ProviderError: If the API call fails
        """
        pass

    @abstractmethod
    async def astream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        """
        Stream a completion asynchronously.

        Args:
            request: The completion request (stream=True)

        Yields:
            CompletionChunk for each streamed part

        Raises:
            ProviderError: If the API call fails
        """
        pass

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """
        Get capabilities for a specific model.

        Args:
            model: Model identifier

        Returns:
            ModelCapabilities describing what the model supports
        """
        return get_model_capabilities(model)

    def prepare_request_params(self, request: CompletionRequest) -> Dict[str, Any]:
        """
        Prepare provider-specific request parameters.

        This method transforms the standardized CompletionRequest into
        provider-specific parameters, handling model-specific differences
        like reasoning model parameter restrictions.

        Args:
            request: The standardized completion request

        Returns:
            Dictionary of provider-specific parameters
        """
        capabilities = self.get_model_capabilities(request.model)
        params: Dict[str, Any] = {"model": request.model}

        # Messages
        params["messages"] = self._prepare_messages(request, capabilities)

        # Token limits - choose the right parameter based on model
        if capabilities.use_max_completion_tokens:
            # Reasoning models use max_completion_tokens
            max_tokens = request.max_completion_tokens or request.max_tokens
            if max_tokens is not None:
                params["max_completion_tokens"] = max_tokens
        else:
            # Standard models use max_tokens
            if request.max_tokens is not None:
                params["max_tokens"] = request.max_tokens

        # Sampling parameters - only for non-reasoning models
        if capabilities.supports_temperature and request.temperature is not None:
            params["temperature"] = request.temperature

        if capabilities.supports_top_p and request.top_p is not None:
            params["top_p"] = request.top_p

        # Penalty parameters - only for non-reasoning models
        if capabilities.supports_presence_penalty and request.presence_penalty is not None:
            params["presence_penalty"] = request.presence_penalty

        if capabilities.supports_frequency_penalty and request.frequency_penalty is not None:
            params["frequency_penalty"] = request.frequency_penalty

        # Reasoning effort - only for reasoning models
        if capabilities.supports_reasoning_effort and request.reasoning_effort is not None:
            effort = request.reasoning_effort
            if hasattr(effort, "value"):
                effort = effort.value
            params["reasoning_effort"] = effort
        elif capabilities.default_reasoning_effort and capabilities.supports_reasoning_effort:
            # Apply default reasoning effort if model has one
            params["reasoning_effort"] = capabilities.default_reasoning_effort

        # Streaming
        params["stream"] = request.stream

        # Response format
        if request.response_format is not None:
            params["response_format"] = request.response_format.model_dump(exclude_none=True)

        # Stop sequences
        if request.stop is not None:
            params["stop"] = request.stop

        # Tools
        if request.tools and capabilities.supports_tools:
            params["tools"] = [t.model_dump() for t in request.tools]

            if request.tool_choice is not None:
                params["tool_choice"] = request.tool_choice

            if (
                request.parallel_tool_calls is not None
                and capabilities.supports_parallel_tool_calls
            ):
                params["parallel_tool_calls"] = request.parallel_tool_calls

        # Seed for reproducibility
        if request.seed is not None:
            params["seed"] = request.seed

        # User identifier
        if request.user is not None:
            params["user"] = request.user

        # Multiple completions
        if request.n > 1:
            params["n"] = request.n

        # Azure-specific: data sources
        if request.data_sources is not None:
            params["data_sources"] = request.data_sources

        # Stored completions
        if request.store is not None:
            params["store"] = request.store

        if request.metadata is not None:
            params["metadata"] = request.metadata

        return params

    def _prepare_messages(
        self, request: CompletionRequest, capabilities: ModelCapabilities
    ) -> list:
        """
        Prepare messages for the API call.

        Handles system/developer message conversion for reasoning models.

        Args:
            request: The completion request
            capabilities: Model capabilities

        Returns:
            List of message dictionaries
        """
        messages = []

        for msg in request.messages:
            msg_dict = msg.to_dict()

            # Convert system to developer for reasoning models that require it
            if (
                capabilities.convert_system_to_developer
                and msg_dict.get("role") == "system"
                and capabilities.supports_developer_message
            ):
                msg_dict["role"] = "developer"

            messages.append(msg_dict)

        return messages

    def validate_request(self, request: CompletionRequest) -> list[str]:
        """
        Validate a request against model capabilities.

        Args:
            request: The request to validate

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []
        capabilities = self.get_model_capabilities(request.model)

        # Warn about unsupported parameters
        if not capabilities.supports_temperature and request.temperature is not None:
            warnings.append(
                f"Model '{request.model}' does not support temperature parameter. "
                "It will be ignored."
            )

        if not capabilities.supports_streaming and request.stream:
            warnings.append(
                f"Model '{request.model}' does not support streaming. "
                "Request will be processed synchronously."
            )

        if not capabilities.supports_reasoning_effort and request.reasoning_effort is not None:
            warnings.append(
                f"Model '{request.model}' does not support reasoning_effort parameter. "
                "It will be ignored."
            )

        # Validate reasoning effort values
        if capabilities.supports_reasoning_effort and request.reasoning_effort is not None:
            effort = request.reasoning_effort
            if hasattr(effort, "value"):
                effort = effort.value
            if effort not in capabilities.supported_reasoning_efforts:
                warnings.append(
                    f"Model '{request.model}' does not support reasoning_effort='{effort}'. "
                    f"Supported values: {capabilities.supported_reasoning_efforts}"
                )

        return warnings


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.response_data = response_data


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""

    pass


class AuthenticationError(ProviderError):
    """Raised when authentication fails."""

    pass


class InvalidRequestError(ProviderError):
    """Raised when the request is invalid."""

    pass


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is not found."""

    pass
