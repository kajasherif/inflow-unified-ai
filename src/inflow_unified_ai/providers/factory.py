"""
Provider factory for the Unified AI Abstraction Layer.

The factory pattern allows easy registration and lookup of provider
implementations. It also handles provider instantiation with proper
configuration.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, Callable
import structlog

from inflow_unified_ai.providers.base import LLMProvider, ProviderError

logger = structlog.get_logger(__name__)

# Global provider registry
_PROVIDER_REGISTRY: Dict[str, Type[LLMProvider]] = {}
_PROVIDER_INSTANCES: Dict[str, LLMProvider] = {}


def register_provider(
    name: str, provider_class: Optional[Type[LLMProvider]] = None
) -> Callable[[Type[LLMProvider]], Type[LLMProvider]]:
    """
    Register a provider class with the factory.

    Can be used as a decorator or called directly:

        @register_provider("my_provider")
        class MyProvider(LLMProvider):
            pass

        # Or directly:
        register_provider("my_provider", MyProvider)

    Args:
        name: Unique name for the provider
        provider_class: The provider class to register (optional for decorator use)

    Returns:
        Decorator function or the registered class
    """

    def decorator(cls: Type[LLMProvider]) -> Type[LLMProvider]:
        _PROVIDER_REGISTRY[name.lower()] = cls
        logger.debug(f"Registered provider: {name}")
        return cls

    if provider_class is not None:
        return decorator(provider_class)

    return decorator


class ModelFactory:
    """
    Factory for creating and managing LLM provider instances.

    The factory provides:
    - Provider registration and lookup
    - Instance caching (optional)
    - Configuration management

    Usage:
        factory = ModelFactory()

        # Get a provider instance
        provider = factory.get_provider("azure_openai", api_key="...", endpoint="...")

        # Generate a completion
        response = provider.generate(request)
    """

    def __init__(self, cache_instances: bool = True) -> None:
        """
        Initialize the factory.

        Args:
            cache_instances: Whether to cache provider instances for reuse
        """
        self.cache_instances = cache_instances
        self._instances: Dict[str, LLMProvider] = {}

        # Auto-register built-in providers
        self._register_builtin_providers()

    def _register_builtin_providers(self) -> None:
        """Register the built-in provider implementations."""
        # Lazy imports to avoid circular dependencies
        try:
            from inflow_unified_ai.providers.azure_openai import AzureOpenAIProvider

            register_provider("azure_openai", AzureOpenAIProvider)
            register_provider("azure", AzureOpenAIProvider)  # Alias
        except ImportError:
            logger.debug("Azure OpenAI provider not available")

        try:
            from inflow_unified_ai.providers.anthropic import AnthropicProvider

            register_provider("anthropic", AnthropicProvider)
            register_provider("claude", AnthropicProvider)  # Alias
        except ImportError:
            logger.debug("Anthropic provider not available")

        try:
            from inflow_unified_ai.providers.gemini import GeminiProvider

            register_provider("gemini", GeminiProvider)
            register_provider("google", GeminiProvider)  # Alias
        except ImportError:
            logger.debug("Gemini provider not available")

        try:
            from inflow_unified_ai.providers.vllm import VLLMProvider

            register_provider("vllm", VLLMProvider)
        except ImportError:
            logger.debug("vLLM provider not available")

    def get_provider(self, provider_name: str, **config: Any) -> LLMProvider:
        """
        Get a provider instance by name.

        If caching is enabled and an instance exists with the same
        configuration, it will be returned. Otherwise, a new instance
        is created.

        Args:
            provider_name: Name of the provider (e.g., "azure_openai")
            **config: Provider-specific configuration

        Returns:
            Configured LLMProvider instance

        Raises:
            ProviderError: If the provider is not registered
        """
        name_lower = provider_name.lower()

        # Check cache first
        cache_key = self._get_cache_key(name_lower, config)
        if self.cache_instances and cache_key in self._instances:
            logger.debug(f"Returning cached provider: {name_lower}")
            return self._instances[cache_key]

        # Look up provider class
        if name_lower not in _PROVIDER_REGISTRY:
            available = list(_PROVIDER_REGISTRY.keys())
            raise ProviderError(
                f"Unknown provider: '{provider_name}'. Available providers: {available}",
                provider=provider_name,
            )

        provider_class = _PROVIDER_REGISTRY[name_lower]

        # Create instance
        try:
            instance = provider_class(**config)
            logger.info(f"Created provider instance: {name_lower}")
        except Exception as e:
            raise ProviderError(
                f"Failed to create provider '{provider_name}': {e}", provider=provider_name
            ) from e

        # Cache if enabled
        if self.cache_instances:
            self._instances[cache_key] = instance

        return instance

    def _get_cache_key(self, provider_name: str, config: Dict[str, Any]) -> str:
        """Generate a cache key for a provider configuration."""
        # Use relevant config values for cache key
        key_parts = [provider_name]

        # Include key config in cache key (but not sensitive values)
        for k in sorted(config.keys()):
            if k not in ("api_key",):  # Exclude sensitive values
                key_parts.append(f"{k}={config[k]}")

        return "|".join(key_parts)

    def clear_cache(self) -> None:
        """Clear all cached provider instances."""
        self._instances.clear()
        logger.debug("Cleared provider instance cache")

    @staticmethod
    def list_providers() -> list[str]:
        """
        List all registered provider names.

        Returns:
            List of provider names
        """
        return list(_PROVIDER_REGISTRY.keys())

    @staticmethod
    def is_provider_registered(name: str) -> bool:
        """
        Check if a provider is registered.

        Args:
            name: Provider name to check

        Returns:
            True if registered
        """
        return name.lower() in _PROVIDER_REGISTRY


# Model to provider mapping for auto-detection
MODEL_PROVIDER_MAP: Dict[str, str] = {
    # Azure OpenAI models
    "gpt-3.5": "azure_openai",
    "gpt-35": "azure_openai",
    "gpt-4": "azure_openai",
    "gpt-4o": "azure_openai",
    "gpt-4.1": "azure_openai",
    "gpt-4.5": "azure_openai",
    "gpt-5": "azure_openai",
    "o1": "azure_openai",
    "o3": "azure_openai",
    "o4": "azure_openai",
    "codex": "azure_openai",
    # Anthropic models
    "claude": "anthropic",
    # Google models
    "gemini": "gemini",
    # Local models
    "llama": "vllm",
    "mistral": "vllm",
    "mixtral": "vllm",
}


def detect_provider_for_model(model: str) -> Optional[str]:
    """
    Attempt to detect the provider for a given model name.

    Args:
        model: Model identifier

    Returns:
        Provider name if detected, None otherwise
    """
    model_lower = model.lower()

    for prefix, provider in MODEL_PROVIDER_MAP.items():
        if model_lower.startswith(prefix):
            return provider

    return None
