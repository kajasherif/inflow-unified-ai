"""
Google Gemini Provider (Stub Implementation)

This module provides a placeholder for Google Gemini integration.
Full implementation would be added when Gemini support is needed.
"""

from typing import AsyncIterator, Optional, Any
from ..providers.base import LLMProvider, ProviderError
from ..models.requests import CompletionRequest
from ..models.responses import CompletionResponse, CompletionChunk


class GeminiProvider(LLMProvider):
    """
    Google Gemini provider implementation stub.

    To use this provider, install the google-generativeai package:
        pip install google-generativeai

    And provide your API key:
        provider = GeminiProvider(api_key="your-api-key")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        **kwargs: Any,
    ):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key (for direct API access)
            project_id: GCP project ID (for Vertex AI)
            location: GCP region for Vertex AI
            **kwargs: Additional configuration
        """
        self.api_key = api_key
        self.project_id = project_id
        self.location = location
        self._client = None

    def _ensure_client(self):
        """Lazy initialize the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                raise ProviderError(
                    "Google Generative AI package not installed. "
                    "Install with: pip install google-generativeai"
                )

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion using Google Gemini.

        Args:
            request: The completion request

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: Stub implementation - not yet available
        """
        raise ProviderError(
            "Gemini provider is a stub implementation. Full implementation coming soon."
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        """
        Stream a completion using Google Gemini.

        Args:
            request: The completion request

        Yields:
            CompletionChunk objects as they arrive

        Raises:
            ProviderError: Stub implementation - not yet available
        """
        raise ProviderError(
            "Gemini provider is a stub implementation. Full implementation coming soon."
        )
        # Make this an async generator
        yield  # type: ignore

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "gemini"

    def get_supported_models(self) -> list[str]:
        """
        Return list of supported Gemini models.

        Returns:
            List of model identifiers
        """
        return [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.0-pro",
        ]
