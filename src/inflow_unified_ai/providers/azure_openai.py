"""
Azure OpenAI provider adapter for the Unified AI Abstraction Layer.

This adapter handles all Azure OpenAI models including:
- Standard models: GPT-4o, GPT-4.1, GPT-4 Turbo, GPT-3.5-Turbo
- Reasoning models: o1, o3, o4-mini, GPT-5 series

Key parameter handling:
- Reasoning models use max_completion_tokens (not max_tokens)
- Reasoning models don't support temperature, top_p, penalties
- System messages are converted to developer messages for some reasoning models
"""

from __future__ import annotations

import os
from typing import AsyncIterator, Dict, Iterator, Optional, Any

import structlog

from inflow_unified_ai.models.requests import CompletionRequest
from inflow_unified_ai.models.responses import CompletionResponse, CompletionChunk
from inflow_unified_ai.models.capabilities import ModelCapabilities, get_model_capabilities
from inflow_unified_ai.providers.base import (
    LLMProvider,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
)

# Lazy import openai to avoid import errors if not installed
_openai_client = None
_async_openai_client = None

logger = structlog.get_logger(__name__)


def _get_openai():
    """Lazy import of OpenAI module."""
    try:
        import openai

        return openai
    except ImportError:
        raise ImportError(
            "openai package is required for Azure OpenAI provider. "
            "Install it with: pip install openai>=1.0"
        )


class AzureOpenAIProvider(LLMProvider):
    """
    Azure OpenAI provider adapter.

    Supports all Azure OpenAI models with intelligent parameter mapping
    based on model capabilities. Handles the differences between standard
    chat models and reasoning models automatically.

    Usage:
        provider = AzureOpenAIProvider(
            api_key="your-api-key",
            endpoint="https://your-resource.openai.azure.com",
            api_version="2024-10-21"
        )

        response = provider.generate(CompletionRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="Hello!")]
        ))
    """

    provider_name = "azure_openai"

    # Default API version - updated for latest model support
    DEFAULT_API_VERSION = "2024-10-21"

    # API version for v1 endpoint (for GPT-5 series)
    V1_API_BASE_PATH = "/openai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
        use_v1_endpoint: bool = False,
        timeout: float = 600.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Azure OpenAI provider.

        Args:
            api_key: Azure OpenAI API key. Falls back to AZURE_OPENAI_API_KEY env var.
            endpoint: Azure OpenAI endpoint URL. Falls back to AZURE_OPENAI_ENDPOINT env var.
            api_version: API version to use. Defaults to 2024-10-21.
            deployment_name: Default deployment name (can be overridden per request).
            use_v1_endpoint: Use the v1 endpoint format (for GPT-5 series).
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
            **kwargs: Additional configuration options.
        """
        # Resolve configuration from environment if not provided
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or os.getenv(
            "AZURE_OPENAI_API_VERSION", self.DEFAULT_API_VERSION
        )
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.use_v1_endpoint = use_v1_endpoint
        self.timeout = timeout
        self.max_retries = max_retries

        super().__init__(
            api_key=self.api_key, endpoint=self.endpoint, api_version=self.api_version, **kwargs
        )

        # Initialize clients lazily
        self._client: Optional[Any] = None
        self._async_client: Optional[Any] = None

        logger.info(
            "Azure OpenAI provider initialized",
            endpoint=self.endpoint,
            api_version=self.api_version,
            use_v1_endpoint=self.use_v1_endpoint,
        )

    def _get_client(self) -> Any:
        """Get or create the synchronous OpenAI client."""
        if self._client is None:
            openai = _get_openai()

            if not self.api_key:
                raise AuthenticationError(
                    "Azure OpenAI API key is required. "
                    "Set AZURE_OPENAI_API_KEY environment variable or pass api_key parameter.",
                    provider=self.provider_name,
                )

            if not self.endpoint:
                raise InvalidRequestError(
                    "Azure OpenAI endpoint is required. "
                    "Set AZURE_OPENAI_ENDPOINT environment variable or pass endpoint parameter.",
                    provider=self.provider_name,
                )

            if self.use_v1_endpoint:
                # Use OpenAI client with Azure v1 endpoint
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=f"{self.endpoint.rstrip('/')}{self.V1_API_BASE_PATH}",
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
            else:
                # Use AzureOpenAI client
                self._client = openai.AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.endpoint,
                    api_version=self.api_version,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )

        return self._client

    def _get_async_client(self) -> Any:
        """Get or create the asynchronous OpenAI client."""
        if self._async_client is None:
            openai = _get_openai()

            if not self.api_key:
                raise AuthenticationError(
                    "Azure OpenAI API key is required.", provider=self.provider_name
                )

            if not self.endpoint:
                raise InvalidRequestError(
                    "Azure OpenAI endpoint is required.", provider=self.provider_name
                )

            if self.use_v1_endpoint:
                self._async_client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=f"{self.endpoint.rstrip('/')}{self.V1_API_BASE_PATH}",
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
            else:
                self._async_client = openai.AsyncAzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.endpoint,
                    api_version=self.api_version,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )

        return self._async_client

    def _should_use_v1_endpoint(self, model: str) -> bool:
        """
        Determine if the v1 endpoint should be used for a model.

        GPT-5 series models use the v1 endpoint format.

        Args:
            model: Model identifier

        Returns:
            True if v1 endpoint should be used
        """
        v1_prefixes = [
            "gpt-5",
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-pro",
            "gpt-5-codex",
        ]
        model_lower = model.lower()
        return any(model_lower.startswith(prefix) for prefix in v1_prefixes)

    def _resolve_model(self, request: CompletionRequest) -> str:
        """
        Resolve the model/deployment name to use.

        For Azure OpenAI, the model field should contain the deployment name.

        Args:
            request: The completion request

        Returns:
            Deployment name to use
        """
        return request.model or self.deployment_name or ""

    def prepare_request_params(self, request: CompletionRequest) -> Dict[str, Any]:
        """
        Prepare Azure OpenAI-specific request parameters.

        This method handles all the model-specific parameter requirements:
        - Standard models: max_tokens, temperature, penalties
        - Reasoning models: max_completion_tokens, reasoning_effort

        Args:
            request: The standardized completion request

        Returns:
            Dictionary of Azure OpenAI API parameters
        """
        capabilities = self.get_model_capabilities(request.model)
        params: Dict[str, Any] = {}

        # Model/deployment
        params["model"] = self._resolve_model(request)

        # Messages with proper role handling
        params["messages"] = self._prepare_messages(request, capabilities)

        # =================================================================
        # TOKEN LIMITS
        # =================================================================
        # Reasoning models MUST use max_completion_tokens
        # Standard models can use either, prefer max_tokens for compatibility

        if capabilities.use_max_completion_tokens:
            # Reasoning model - use max_completion_tokens
            max_tokens = request.max_completion_tokens or request.max_tokens
            if max_tokens is not None:
                params["max_completion_tokens"] = max_tokens
            # Do NOT include max_tokens for reasoning models
        else:
            # Standard model - use max_tokens
            if request.max_tokens is not None:
                params["max_tokens"] = request.max_tokens
            # max_completion_tokens also works for newer standard models
            elif request.max_completion_tokens is not None:
                params["max_completion_tokens"] = request.max_completion_tokens

        # =================================================================
        # SAMPLING PARAMETERS (Standard models only)
        # =================================================================
        # Reasoning models do NOT support these parameters

        if capabilities.supports_temperature:
            if request.temperature is not None:
                params["temperature"] = request.temperature
        # If model doesn't support temperature, simply don't include it

        if capabilities.supports_top_p:
            if request.top_p is not None:
                params["top_p"] = request.top_p

        # =================================================================
        # PENALTY PARAMETERS (Standard models only)
        # =================================================================

        if capabilities.supports_presence_penalty:
            if request.presence_penalty is not None:
                params["presence_penalty"] = request.presence_penalty

        if capabilities.supports_frequency_penalty:
            if request.frequency_penalty is not None:
                params["frequency_penalty"] = request.frequency_penalty

        # =================================================================
        # REASONING PARAMETERS (Reasoning models only)
        # =================================================================

        if capabilities.supports_reasoning_effort:
            if request.reasoning_effort is not None:
                effort = request.reasoning_effort
                if hasattr(effort, "value"):
                    effort = effort.value
                params["reasoning_effort"] = effort
            elif capabilities.default_reasoning_effort:
                # Apply model's default reasoning effort
                params["reasoning_effort"] = capabilities.default_reasoning_effort

        # =================================================================
        # STREAMING
        # =================================================================

        if request.stream:
            if capabilities.supports_streaming:
                params["stream"] = True
                # Include usage in stream for newer API versions
                params["stream_options"] = {"include_usage": True}
            else:
                # Model doesn't support streaming, log warning and proceed sync
                logger.warning(
                    "Model does not support streaming, proceeding synchronously",
                    model=request.model,
                )
                params["stream"] = False
        else:
            params["stream"] = False

        # =================================================================
        # RESPONSE FORMAT
        # =================================================================

        if request.response_format is not None:
            params["response_format"] = request.response_format.model_dump(exclude_none=True)

        # =================================================================
        # STOP SEQUENCES
        # =================================================================

        if request.stop is not None:
            params["stop"] = request.stop

        # =================================================================
        # TOOLS / FUNCTION CALLING
        # =================================================================

        if request.tools and capabilities.supports_tools:
            params["tools"] = [t.model_dump() for t in request.tools]

            if request.tool_choice is not None:
                params["tool_choice"] = request.tool_choice

            # Parallel tool calls - only for models that support it
            # Note: Reasoning models generally don't support parallel tool calls
            if request.parallel_tool_calls is not None:
                if capabilities.supports_parallel_tool_calls:
                    params["parallel_tool_calls"] = request.parallel_tool_calls
                else:
                    logger.debug(
                        "Model does not support parallel_tool_calls, ignoring", model=request.model
                    )

        # =================================================================
        # OTHER PARAMETERS
        # =================================================================

        if request.seed is not None:
            params["seed"] = request.seed

        if request.user is not None:
            params["user"] = request.user

        if request.n > 1:
            params["n"] = request.n

        # Azure-specific: On Your Data
        if request.data_sources is not None:
            params["data_sources"] = request.data_sources

        # Stored completions (for distillation/training)
        if request.store is not None:
            params["store"] = request.store

        if request.metadata is not None:
            params["metadata"] = request.metadata

        return params

    def _prepare_messages(
        self, request: CompletionRequest, capabilities: ModelCapabilities
    ) -> list:
        """
        Prepare messages for Azure OpenAI API.

        Handles:
        - System to developer message conversion for reasoning models
        - Multi-modal content (text + images)

        Args:
            request: The completion request
            capabilities: Model capabilities

        Returns:
            List of message dictionaries for the API
        """
        messages = []

        for msg in request.messages:
            msg_dict = msg.to_dict()

            # Convert system messages to developer for certain reasoning models
            # This is required for o1, o3, o4-mini series
            if capabilities.convert_system_to_developer and msg_dict.get("role") == "system":
                if capabilities.supports_developer_message:
                    msg_dict["role"] = "developer"
                    logger.debug(
                        "Converted system message to developer message", model=request.model
                    )
                elif not capabilities.supports_system_message:
                    # Model doesn't support system messages at all
                    # Convert to user message with context prefix
                    msg_dict["role"] = "user"
                    if isinstance(msg_dict.get("content"), str):
                        msg_dict["content"] = f"[System Instructions]: {msg_dict['content']}"
                    logger.debug(
                        "Converted system message to user message (no system support)",
                        model=request.model,
                    )

            messages.append(msg_dict)

        return messages

    def _handle_error(self, error: Exception) -> None:
        """
        Handle and re-raise OpenAI errors with proper types.

        Args:
            error: The original exception

        Raises:
            Appropriate ProviderError subclass
        """
        openai = _get_openai()

        error_message = str(error)
        status_code = getattr(error, "status_code", None)

        if isinstance(error, openai.RateLimitError):
            raise RateLimitError(
                f"Rate limit exceeded: {error_message}",
                provider=self.provider_name,
                status_code=status_code,
            ) from error

        if isinstance(error, openai.AuthenticationError):
            raise AuthenticationError(
                f"Authentication failed: {error_message}",
                provider=self.provider_name,
                status_code=status_code,
            ) from error

        if isinstance(error, openai.BadRequestError):
            raise InvalidRequestError(
                f"Invalid request: {error_message}",
                provider=self.provider_name,
                status_code=status_code,
            ) from error

        if isinstance(error, openai.NotFoundError):
            raise ModelNotFoundError(
                f"Model or deployment not found: {error_message}",
                provider=self.provider_name,
                status_code=status_code,
            ) from error

        # Generic error
        raise ProviderError(
            f"Azure OpenAI error: {error_message}",
            provider=self.provider_name,
            status_code=status_code,
        ) from error

    def generate(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion synchronously.

        Args:
            request: The completion request

        Returns:
            CompletionResponse with the generated content
        """
        # Log warnings for unsupported parameters
        warnings = self.validate_request(request)
        for warning in warnings:
            logger.warning(warning)

        # Prepare parameters
        params = self.prepare_request_params(request)

        # Don't stream for generate()
        params["stream"] = False
        if "stream_options" in params:
            del params["stream_options"]

        logger.debug(
            "Generating completion",
            model=params.get("model"),
            is_reasoning=get_model_capabilities(request.model).is_reasoning_model,
        )

        try:
            client = self._get_client()
            response = client.chat.completions.create(**params)

            return CompletionResponse.from_openai(response, provider=self.provider_name)

        except Exception as e:
            logger.error("Generation failed", error=str(e), model=params.get("model"))
            self._handle_error(e)

    async def agenerate(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion asynchronously.

        Args:
            request: The completion request

        Returns:
            CompletionResponse with the generated content
        """
        warnings = self.validate_request(request)
        for warning in warnings:
            logger.warning(warning)

        params = self.prepare_request_params(request)
        params["stream"] = False
        if "stream_options" in params:
            del params["stream_options"]

        logger.debug(
            "Generating completion (async)",
            model=params.get("model"),
        )

        try:
            client = self._get_async_client()
            response = await client.chat.completions.create(**params)

            return CompletionResponse.from_openai(response, provider=self.provider_name)

        except Exception as e:
            logger.error("Async generation failed", error=str(e))
            self._handle_error(e)

    def stream(self, request: CompletionRequest) -> Iterator[CompletionChunk]:
        """
        Stream a completion synchronously.

        Args:
            request: The completion request

        Yields:
            CompletionChunk for each streamed part
        """
        capabilities = self.get_model_capabilities(request.model)

        if not capabilities.supports_streaming:
            logger.warning(
                "Model does not support streaming, falling back to generate()", model=request.model
            )
            response = self.generate(request)
            yield CompletionChunk(
                id=response.id,
                content=response.content,
                model=response.model,
                provider=self.provider_name,
                finish_reason=response.finish_reason,
                is_final=True,
                usage=response.usage,
            )
            return

        params = self.prepare_request_params(request)
        params["stream"] = True

        try:
            client = self._get_client()
            stream = client.chat.completions.create(**params)

            for chunk in stream:
                yield CompletionChunk.from_openai_chunk(chunk, provider=self.provider_name)

        except Exception as e:
            logger.error("Streaming failed", error=str(e))
            self._handle_error(e)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        """
        Stream a completion asynchronously.

        Args:
            request: The completion request

        Yields:
            CompletionChunk for each streamed part
        """
        capabilities = self.get_model_capabilities(request.model)

        if not capabilities.supports_streaming:
            logger.warning(
                "Model does not support streaming, falling back to agenerate()", model=request.model
            )
            response = await self.agenerate(request)
            yield CompletionChunk(
                id=response.id,
                content=response.content,
                model=response.model,
                provider=self.provider_name,
                finish_reason=response.finish_reason,
                is_final=True,
                usage=response.usage,
            )
            return

        params = self.prepare_request_params(request)
        params["stream"] = True

        try:
            client = self._get_async_client()
            stream = await client.chat.completions.create(**params)

            async for chunk in stream:
                yield CompletionChunk.from_openai_chunk(chunk, provider=self.provider_name)

        except Exception as e:
            logger.error("Async streaming failed", error=str(e))
            self._handle_error(e)
