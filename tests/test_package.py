"""
Unit tests for inflow-unified-ai package.
"""

import pytest
from inflow_unified_ai import (
    AIClient,
    Message,
    MessageRole,
    __version__,
)
from inflow_unified_ai.models import (
    get_model_capabilities,
    is_reasoning_model,
    ModelFamily,
)
from inflow_unified_ai.providers import AzureOpenAIProvider


class TestPackageImports:
    """Test that all package imports work correctly."""

    def test_version_exists(self):
        """Test that version is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert __version__ == "0.1.0"

    def test_aiclient_import(self):
        """Test AIClient can be imported."""
        assert AIClient is not None

    def test_message_types(self):
        """Test Message and MessageRole imports."""
        assert Message is not None
        assert MessageRole is not None
        assert MessageRole.USER is not None
        assert MessageRole.ASSISTANT is not None
        assert MessageRole.SYSTEM is not None

    def test_provider_import(self):
        """Test provider imports."""
        assert AzureOpenAIProvider is not None


class TestMessageCreation:
    """Test Message creation and validation."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = Message(role=MessageRole.SYSTEM, content="You are helpful")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are helpful"

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message(role=MessageRole.ASSISTANT, content="Hi there!")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there!"


class TestModelCapabilities:
    """Test model capabilities detection."""

    def test_get_gpt4o_capabilities(self):
        """Test GPT-4o capabilities."""
        caps = get_model_capabilities("gpt-4o")
        assert caps is not None
        assert caps.supports_streaming is True
        assert caps.supports_temperature is True

    def test_get_o1_capabilities(self):
        """Test o1 reasoning model capabilities."""
        caps = get_model_capabilities("o1")
        assert caps is not None
        assert caps.supports_streaming is False
        assert caps.supports_temperature is False

    def test_get_o3_capabilities(self):
        """Test o3 reasoning model capabilities."""
        caps = get_model_capabilities("o3")
        assert caps is not None
        assert caps.supports_streaming is True
        assert caps.supports_temperature is False

    def test_is_reasoning_model(self):
        """Test reasoning model detection."""
        assert is_reasoning_model("o1") is True
        assert is_reasoning_model("o3") is True
        assert is_reasoning_model("o3-mini") is True
        assert is_reasoning_model("gpt-4o") is False
        assert is_reasoning_model("gpt-4.1") is False

    def test_gpt5_is_reasoning(self):
        """Test GPT-5 series are reasoning models."""
        assert is_reasoning_model("gpt-5") is True
        assert is_reasoning_model("gpt-5-mini") is True
        assert is_reasoning_model("gpt-5.1-chat") is True

    def test_unknown_model_defaults(self):
        """Test unknown model returns default capabilities."""
        caps = get_model_capabilities("unknown-model-xyz")
        assert caps is not None  # Should return defaults


class TestModelFamily:
    """Test ModelFamily enum."""

    def test_model_families_exist(self):
        """Test ModelFamily enum values."""
        assert ModelFamily.GPT_4O is not None
        assert ModelFamily.GPT_41 is not None
        assert ModelFamily.GPT_5 is not None
        assert ModelFamily.O1 is not None
        assert ModelFamily.O3 is not None


class TestAIClientInitialization:
    """Test AIClient initialization."""

    def test_client_init_with_provider(self):
        """Test client initialization with provider string."""
        client = AIClient(
            provider="azure_openai",
            api_key="test-key",
            endpoint="https://test.openai.azure.com"
        )
        assert client is not None

    def test_client_init_azure_alias(self):
        """Test client initialization with azure alias."""
        client = AIClient(
            provider="azure",
            api_key="test-key",
            endpoint="https://test.openai.azure.com"
        )
        assert client is not None
