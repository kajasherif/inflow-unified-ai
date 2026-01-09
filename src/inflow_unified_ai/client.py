"""
AI Client facade for the Unified AI Abstraction Layer.

The AIClient is the main entry point for consumers of the library.
It provides a simple, unified interface for interacting with any
supported AI provider.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union
import structlog

from inflow_unified_ai.models.requests import CompletionRequest, Message, ReasoningEffort
from inflow_unified_ai.models.responses import CompletionResponse, CompletionChunk
from inflow_unified_ai.models.capabilities import get_model_capabilities, is_reasoning_model
from inflow_unified_ai.providers.base import LLMProvider
from inflow_unified_ai.providers.factory import ModelFactory, detect_provider_for_model
from inflow_unified_ai.resilience.retry import RetryConfig, with_retry

logger = structlog.get_logger(__name__)


class AIClient:
    """
    Unified AI Client for interacting with multiple AI providers.

    This is the main entry point for the library. It provides a simple
    interface for generating completions with automatic provider selection,
    request preparation, and response handling.

    Features:
    - Automatic provider detection based on model name
    - Smart parameter mapping based on model capabilities
    - Built-in retry logic with exponential backoff
    - Streaming support
    - Async support

    Usage:
        # Initialize with explicit provider
        client = AIClient(
            provider="azure_openai",
            api_key="your-api-key",
            endpoint="https://your-resource.openai.azure.com"
        )

        # Or with auto-detection
        client = AIClient()  # Uses environment variables

        # Generate a completion
        response = client.generate(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        print(response.content)

        # Streaming
        for chunk in client.stream(model="gpt-4o", messages=[...]):
            print(chunk.content, end="")
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
        **provider_config: Any,
    ) -> None:
        """
        Initialize the AI Client.

        Args:
            provider: Provider name (e.g., "azure_openai", "anthropic").
                     If not specified, will auto-detect from model name.
            api_key: API key for the provider
            endpoint: API endpoint (for Azure)
            api_version: API version (for Azure)
            retry_config: Custom retry configuration
            **provider_config: Additional provider-specific configuration
        """
        self._factory = ModelFactory()
        self._default_provider = provider
        self._retry_config = retry_config or RetryConfig()

        # Store provider configuration
        self._provider_config: Dict[str, Any] = {
            "api_key": api_key,
            "endpoint": endpoint,
            "api_version": api_version,
            **provider_config,
        }

        # Remove None values
        self._provider_config = {k: v for k, v in self._provider_config.items() if v is not None}

        # Cache for provider instances
        self._provider_cache: Dict[str, LLMProvider] = {}

        logger.info(
            "AIClient initialized",
            default_provider=self._default_provider,
            has_api_key=bool(api_key or os.getenv("AZURE_OPENAI_API_KEY")),
        )

    def _get_provider(self, model: str) -> LLMProvider:
        """
        Get the appropriate provider for a model.

        Args:
            model: Model identifier

        Returns:
            Configured LLMProvider instance
        """
        # Determine provider
        provider_name = self._default_provider
        if not provider_name:
            provider_name = detect_provider_for_model(model)
            if not provider_name:
                # Default to Azure OpenAI
                provider_name = "azure_openai"

        # Check cache
        if provider_name in self._provider_cache:
            return self._provider_cache[provider_name]

        # Create provider
        provider = self._factory.get_provider(provider_name, **self._provider_config)
        self._provider_cache[provider_name] = provider

        return provider

    def _prepare_request(
        self,
        model: str,
        messages: Union[List[Dict[str, Any]], List[Message]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[Union[str, ReasoningEffort]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionRequest:
        """
        Prepare a CompletionRequest from input parameters.

        Args:
            model: Model identifier
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens for response
            max_completion_tokens: Maximum completion tokens (for reasoning models)
            reasoning_effort: Reasoning effort level (for reasoning models)
            stream: Whether to stream the response
            **kwargs: Additional request parameters

        Returns:
            Prepared CompletionRequest
        """
        # Convert dict messages to Message objects
        prepared_messages = []
        for msg in messages:
            if isinstance(msg, Message):
                prepared_messages.append(msg)
            else:
                prepared_messages.append(Message(**msg))

        # Build request
        request = CompletionRequest(
            model=model,
            messages=prepared_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            stream=stream,
            **kwargs,
        )

        return request

    def generate(
        self,
        model: str,
        messages: Union[List[Dict[str, Any]], List[Message]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[Union[str, ReasoningEffort]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Generate a completion synchronously.

        This is the primary method for getting AI responses. It automatically
        handles provider selection and parameter mapping based on the model.

        Args:
            model: Model identifier (e.g., "gpt-4o", "o3-mini", "gpt-5")
            messages: List of messages in the conversation
            temperature: Sampling temperature (0-2). Ignored for reasoning models.
            max_tokens: Max tokens for response. For reasoning models, use
                       max_completion_tokens instead.
            max_completion_tokens: Max completion tokens (required for reasoning models)
            reasoning_effort: Reasoning effort level (low/medium/high) for
                            reasoning models (o-series, GPT-5)
            **kwargs: Additional request parameters

        Returns:
            CompletionResponse with the generated content

        Example:
            response = client.generate(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is Python?"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            print(response.content)
        """
        request = self._prepare_request(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            stream=False,
            **kwargs,
        )

        provider = self._get_provider(model)

        # Apply retry logic
        @with_retry(self._retry_config)
        def _call() -> CompletionResponse:
            return provider.generate(request)

        return _call()

    async def agenerate(
        self,
        model: str,
        messages: Union[List[Dict[str, Any]], List[Message]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[Union[str, ReasoningEffort]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Generate a completion asynchronously.

        Same as generate() but for async contexts.

        Args:
            model: Model identifier
            messages: List of messages
            temperature: Sampling temperature (ignored for reasoning models)
            max_tokens: Max tokens for response
            max_completion_tokens: Max completion tokens (for reasoning models)
            reasoning_effort: Reasoning effort level (for reasoning models)
            **kwargs: Additional request parameters

        Returns:
            CompletionResponse with the generated content

        Example:
            response = await client.agenerate(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello!"}]
            )
        """
        request = self._prepare_request(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            stream=False,
            **kwargs,
        )

        provider = self._get_provider(model)

        # Apply retry logic
        @with_retry(self._retry_config)
        async def _call() -> CompletionResponse:
            return await provider.agenerate(request)

        return await _call()

    def stream(
        self,
        model: str,
        messages: Union[List[Dict[str, Any]], List[Message]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[Union[str, ReasoningEffort]] = None,
        **kwargs: Any,
    ) -> Iterator[CompletionChunk]:
        """
        Stream a completion synchronously.

        Yields chunks as they are generated, allowing for real-time
        display of AI responses.

        Note: Some reasoning models (o1, o3-pro) don't support streaming.
        In that case, a single chunk with the full response is returned.

        Args:
            model: Model identifier
            messages: List of messages
            temperature: Sampling temperature (ignored for reasoning models)
            max_tokens: Max tokens for response
            max_completion_tokens: Max completion tokens (for reasoning models)
            reasoning_effort: Reasoning effort level (for reasoning models)
            **kwargs: Additional request parameters

        Yields:
            CompletionChunk for each part of the response

        Example:
            for chunk in client.stream(model="gpt-4o", messages=[...]):
                print(chunk.content, end="", flush=True)
        """
        request = self._prepare_request(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            stream=True,
            **kwargs,
        )

        provider = self._get_provider(model)

        yield from provider.stream(request)

    async def astream(
        self,
        model: str,
        messages: Union[List[Dict[str, Any]], List[Message]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[Union[str, ReasoningEffort]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        """
        Stream a completion asynchronously.

        Same as stream() but for async contexts.

        Args:
            model: Model identifier
            messages: List of messages
            temperature: Sampling temperature (ignored for reasoning models)
            max_tokens: Max tokens for response
            max_completion_tokens: Max completion tokens (for reasoning models)
            reasoning_effort: Reasoning effort level (for reasoning models)
            **kwargs: Additional request parameters

        Yields:
            CompletionChunk for each part of the response

        Example:
            async for chunk in client.astream(model="gpt-4o", messages=[...]):
                print(chunk.content, end="", flush=True)
        """
        request = self._prepare_request(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            stream=True,
            **kwargs,
        )

        provider = self._get_provider(model)

        async for chunk in provider.astream(request):
            yield chunk

    def chat(
        self, model: str, user_message: str, system_message: Optional[str] = None, **kwargs: Any
    ) -> str:
        """
        Simplified chat interface for quick interactions.

        This is a convenience method for simple question-answer scenarios.

        Args:
            model: Model identifier
            user_message: The user's message
            system_message: Optional system prompt
            **kwargs: Additional request parameters

        Returns:
            The AI's response as a string

        Example:
            answer = client.chat(
                model="gpt-4o",
                user_message="What is 2+2?",
                system_message="You are a helpful math tutor."
            )
        """
        messages: List[Dict[str, Any]] = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": user_message})

        response = self.generate(model=model, messages=messages, **kwargs)
        return response.content

    async def achat(
        self, model: str, user_message: str, system_message: Optional[str] = None, **kwargs: Any
    ) -> str:
        """
        Async version of chat().

        Args:
            model: Model identifier
            user_message: The user's message
            system_message: Optional system prompt
            **kwargs: Additional request parameters

        Returns:
            The AI's response as a string
        """
        messages: List[Dict[str, Any]] = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": user_message})

        response = await self.agenerate(model=model, messages=messages, **kwargs)
        return response.content

    def get_model_capabilities(self, model: str):
        """
        Get capabilities for a specific model.

        Useful for checking what parameters a model supports before
        making a request.

        Args:
            model: Model identifier

        Returns:
            ModelCapabilities object describing the model's features
        """
        return get_model_capabilities(model)

    def is_reasoning_model(self, model: str) -> bool:
        """
        Check if a model is a reasoning model.

        Reasoning models (o-series, GPT-5) have different parameter
        requirements than standard chat models.

        Args:
            model: Model identifier

        Returns:
            True if the model is a reasoning model
        """
        return is_reasoning_model(model)

    def list_providers(self) -> List[str]:
        """
        List available providers.

        Returns:
            List of provider names
        """
        return self._factory.list_providers()
