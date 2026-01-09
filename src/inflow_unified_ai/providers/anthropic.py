"""
Anthropic Claude Provider (Stub Implementation)

This module provides a placeholder for Anthropic Claude integration.
Full implementation would be added when Anthropic support is needed.
"""

from typing import AsyncIterator, Optional, Any
from ..providers.base import LLMProvider, ProviderError
from ..models.requests import CompletionRequest
from ..models.responses import CompletionResponse, CompletionChunk


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider implementation stub.

    To use this provider, install the anthropic package:
        pip install anthropic

    And provide your API key:
        provider = AnthropicProvider(api_key="your-api-key")
    """

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs: Any
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            base_url: Custom API base URL (optional)
            **kwargs: Additional configuration
        """
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _ensure_client(self):
        """Lazy initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ProviderError(
                    "Anthropic package not installed. Install with: pip install anthropic"
                )

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion using Anthropic Claude.

        Args:
            request: The completion request

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: Stub implementation - not yet available
        """
        raise ProviderError(
            "Anthropic provider is a stub implementation. Full implementation coming soon."
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        """
        Stream a completion using Anthropic Claude.

        Args:
            request: The completion request

        Yields:
            CompletionChunk objects as they arrive

        Raises:
            ProviderError: Stub implementation - not yet available
        """
        raise ProviderError(
            "Anthropic provider is a stub implementation. Full implementation coming soon."
        )
        # Make this an async generator
        yield  # type: ignore

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "anthropic"

    def get_supported_models(self) -> list[str]:
        """
        Return list of supported Anthropic models.

        Returns:
            List of model identifiers
        """
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]
