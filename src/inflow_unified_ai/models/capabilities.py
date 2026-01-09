"""
Model capabilities registry for the Unified AI Abstraction Layer.

This module defines what parameters each model supports, enabling
intelligent parameter mapping and validation. Based on the latest
Azure OpenAI documentation (as of January 2026).

Key differences between model families:
- Standard models (GPT-4o, GPT-4.1, etc.): Support temperature, max_tokens
- Reasoning models (o-series, GPT-5): Use max_completion_tokens, reasoning_effort
- Reasoning models do NOT support: temperature, top_p, penalties, logprobs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
import re


class ModelFamily(str, Enum):
    """Model family classification."""

    # Standard chat models
    GPT_35_TURBO = "gpt-35-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_41 = "gpt-4.1"
    GPT_41_MINI = "gpt-4.1-mini"
    GPT_41_NANO = "gpt-4.1-nano"
    GPT_45_PREVIEW = "gpt-4.5-preview"

    # O-series reasoning models
    O1_MINI = "o1-mini"
    O1 = "o1"
    O1_PREVIEW = "o1-preview"
    O3_MINI = "o3-mini"
    O3 = "o3"
    O3_PRO = "o3-pro"
    O4_MINI = "o4-mini"
    CODEX_MINI = "codex-mini"

    # GPT-5 series reasoning models
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5 = "gpt-5"
    GPT_5_PRO = "gpt-5-pro"
    GPT_5_CODEX = "gpt-5-codex"
    GPT_51 = "gpt-5.1"
    GPT_51_CHAT = "gpt-5.1-chat"
    GPT_51_CODEX = "gpt-5.1-codex"
    GPT_51_CODEX_MINI = "gpt-5.1-codex-mini"
    GPT_51_CODEX_MAX = "gpt-5.1-codex-max"
    GPT_52 = "gpt-5.2"
    GPT_52_CHAT = "gpt-5.2-chat"

    # Special/Router models
    MODEL_ROUTER = "model-router"

    # Other providers
    CLAUDE = "claude"
    GEMINI = "gemini"
    VLLM = "vllm"

    # Unknown/custom
    UNKNOWN = "unknown"


@dataclass
class ModelCapabilities:
    """
    Defines the capabilities and parameter support for a model.

    This is used to intelligently map standardized requests to
    provider-specific API calls.
    """

    family: ModelFamily

    # Token parameter support
    supports_max_tokens: bool = True
    supports_max_completion_tokens: bool = False
    use_max_completion_tokens: bool = False  # If True, use max_completion_tokens instead

    # Sampling parameter support
    supports_temperature: bool = True
    supports_top_p: bool = True

    # Penalty parameter support
    supports_presence_penalty: bool = True
    supports_frequency_penalty: bool = True

    # Reasoning support
    is_reasoning_model: bool = False
    supports_reasoning_effort: bool = False
    supported_reasoning_efforts: Set[str] = field(default_factory=lambda: {"low", "medium", "high"})
    default_reasoning_effort: Optional[str] = None

    # Message role support
    supports_system_message: bool = True
    supports_developer_message: bool = False
    convert_system_to_developer: bool = False  # For o-series models

    # Feature support
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_parallel_tool_calls: bool = True
    supports_structured_output: bool = True
    supports_vision: bool = False
    supports_audio: bool = False

    # Logprobs support
    supports_logprobs: bool = True

    # Context window
    max_input_tokens: int = 128000
    max_output_tokens: int = 4096

    # API requirements
    requires_api_version: Optional[str] = None

    # Additional notes
    notes: str = ""

    def get_unsupported_params(self) -> List[str]:
        """Get list of unsupported parameter names."""
        unsupported = []

        if not self.supports_temperature:
            unsupported.append("temperature")
        if not self.supports_top_p:
            unsupported.append("top_p")
        if not self.supports_presence_penalty:
            unsupported.append("presence_penalty")
        if not self.supports_frequency_penalty:
            unsupported.append("frequency_penalty")
        if not self.supports_logprobs:
            unsupported.extend(["logprobs", "top_logprobs", "logit_bias"])
        if not self.supports_max_tokens and not self.supports_max_completion_tokens:
            unsupported.append("max_tokens")

        return unsupported


# =============================================================================
# MODEL CAPABILITIES REGISTRY
# =============================================================================

# Standard GPT models (support temperature, max_tokens)
_STANDARD_GPT_CAPABILITIES = ModelCapabilities(
    family=ModelFamily.GPT_4O,
    supports_max_tokens=True,
    supports_max_completion_tokens=True,
    use_max_completion_tokens=False,
    supports_temperature=True,
    supports_top_p=True,
    supports_presence_penalty=True,
    supports_frequency_penalty=True,
    is_reasoning_model=False,
    supports_reasoning_effort=False,
    supports_streaming=True,
    supports_tools=True,
    supports_parallel_tool_calls=True,
    supports_structured_output=True,
    supports_vision=True,
    supports_logprobs=True,
    max_input_tokens=128000,
    max_output_tokens=16384,
)

# O-series reasoning models (no temperature, use max_completion_tokens)
_O_SERIES_CAPABILITIES = ModelCapabilities(
    family=ModelFamily.O1,
    supports_max_tokens=False,
    supports_max_completion_tokens=True,
    use_max_completion_tokens=True,
    supports_temperature=False,
    supports_top_p=False,
    supports_presence_penalty=False,
    supports_frequency_penalty=False,
    is_reasoning_model=True,
    supports_reasoning_effort=True,
    supported_reasoning_efforts={"low", "medium", "high"},
    supports_system_message=True,
    supports_developer_message=True,
    convert_system_to_developer=True,
    supports_streaming=True,  # Varies by model
    supports_tools=True,
    supports_parallel_tool_calls=False,
    supports_structured_output=True,
    supports_vision=True,
    supports_logprobs=False,
    max_input_tokens=200000,
    max_output_tokens=100000,
)

# GPT-5 series reasoning models
_GPT5_CAPABILITIES = ModelCapabilities(
    family=ModelFamily.GPT_5,
    supports_max_tokens=False,
    supports_max_completion_tokens=True,
    use_max_completion_tokens=True,
    supports_temperature=False,
    supports_top_p=False,
    supports_presence_penalty=False,
    supports_frequency_penalty=False,
    is_reasoning_model=True,
    supports_reasoning_effort=True,
    supported_reasoning_efforts={"minimal", "low", "medium", "high"},
    supports_system_message=True,
    supports_developer_message=True,
    convert_system_to_developer=False,
    supports_streaming=True,
    supports_tools=True,
    supports_parallel_tool_calls=True,
    supports_structured_output=True,
    supports_vision=True,
    supports_logprobs=False,
    max_input_tokens=272000,
    max_output_tokens=128000,
)


# Complete registry mapping model identifiers to capabilities
MODEL_CAPABILITIES_REGISTRY: Dict[str, ModelCapabilities] = {
    # =========================================================================
    # GPT-3.5 Series
    # =========================================================================
    "gpt-35-turbo": ModelCapabilities(
        family=ModelFamily.GPT_35_TURBO,
        supports_max_tokens=True,
        supports_max_completion_tokens=False,
        supports_temperature=True,
        supports_vision=False,
        max_input_tokens=16385,
        max_output_tokens=4096,
    ),
    "gpt-35-turbo-16k": ModelCapabilities(
        family=ModelFamily.GPT_35_TURBO,
        supports_max_tokens=True,
        supports_max_completion_tokens=False,
        supports_temperature=True,
        supports_vision=False,
        max_input_tokens=16385,
        max_output_tokens=4096,
    ),
    # =========================================================================
    # GPT-4 Series
    # =========================================================================
    "gpt-4": ModelCapabilities(
        family=ModelFamily.GPT_4,
        supports_max_tokens=True,
        supports_temperature=True,
        supports_parallel_tool_calls=False,
        supports_vision=False,
        max_input_tokens=8192,
        max_output_tokens=4096,
    ),
    "gpt-4-32k": ModelCapabilities(
        family=ModelFamily.GPT_4,
        supports_max_tokens=True,
        supports_temperature=True,
        supports_parallel_tool_calls=False,
        supports_vision=False,
        max_input_tokens=32768,
        max_output_tokens=4096,
    ),
    "gpt-4-turbo": ModelCapabilities(
        family=ModelFamily.GPT_4_TURBO,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        max_input_tokens=128000,
        max_output_tokens=4096,
    ),
    "gpt-4-turbo-2024-04-09": ModelCapabilities(
        family=ModelFamily.GPT_4_TURBO,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        max_input_tokens=128000,
        max_output_tokens=4096,
    ),
    # =========================================================================
    # GPT-4o Series
    # =========================================================================
    "gpt-4o": ModelCapabilities(
        family=ModelFamily.GPT_4O,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        supports_audio=True,
        max_input_tokens=128000,
        max_output_tokens=16384,
    ),
    "gpt-4o-2024-05-13": ModelCapabilities(
        family=ModelFamily.GPT_4O,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        max_input_tokens=128000,
        max_output_tokens=4096,
    ),
    "gpt-4o-2024-08-06": ModelCapabilities(
        family=ModelFamily.GPT_4O,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        max_input_tokens=128000,
        max_output_tokens=16384,
    ),
    "gpt-4o-2024-11-20": ModelCapabilities(
        family=ModelFamily.GPT_4O,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        max_input_tokens=128000,
        max_output_tokens=16384,
    ),
    "gpt-4o-mini": ModelCapabilities(
        family=ModelFamily.GPT_4O_MINI,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        max_input_tokens=128000,
        max_output_tokens=16384,
    ),
    "gpt-4o-mini-2024-07-18": ModelCapabilities(
        family=ModelFamily.GPT_4O_MINI,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        max_input_tokens=128000,
        max_output_tokens=16384,
    ),
    # =========================================================================
    # GPT-4.1 Series (Latest non-reasoning models)
    # =========================================================================
    "gpt-4.1": ModelCapabilities(
        family=ModelFamily.GPT_41,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        max_input_tokens=128000,
        max_output_tokens=32768,
    ),
    "gpt-4.1-2025-04-14": ModelCapabilities(
        family=ModelFamily.GPT_41,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        max_input_tokens=128000,
        max_output_tokens=32768,
    ),
    "gpt-4.1-mini": ModelCapabilities(
        family=ModelFamily.GPT_41_MINI,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        max_input_tokens=128000,
        max_output_tokens=32768,
    ),
    "gpt-4.1-mini-2025-04-14": ModelCapabilities(
        family=ModelFamily.GPT_41_MINI,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        max_input_tokens=128000,
        max_output_tokens=32768,
    ),
    "gpt-4.1-nano": ModelCapabilities(
        family=ModelFamily.GPT_41_NANO,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        supports_parallel_tool_calls=False,
        max_input_tokens=128000,
        max_output_tokens=32768,
    ),
    "gpt-4.1-nano-2025-04-14": ModelCapabilities(
        family=ModelFamily.GPT_41_NANO,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        supports_parallel_tool_calls=False,
        max_input_tokens=128000,
        max_output_tokens=32768,
    ),
    # =========================================================================
    # GPT-4.5 Preview
    # =========================================================================
    "gpt-4.5-preview": ModelCapabilities(
        family=ModelFamily.GPT_45_PREVIEW,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_vision=True,
        max_input_tokens=128000,
        max_output_tokens=16384,
        notes="Preview model, may be retired",
    ),
    # =========================================================================
    # O1 Series (Reasoning Models)
    # =========================================================================
    "o1-mini": ModelCapabilities(
        family=ModelFamily.O1_MINI,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=False,  # o1-mini doesn't support reasoning_effort
        supports_system_message=False,
        supports_developer_message=False,
        supports_streaming=False,
        supports_tools=False,
        supports_parallel_tool_calls=False,
        supports_vision=False,
        supports_logprobs=False,
        max_input_tokens=128000,
        max_output_tokens=65536,
    ),
    "o1-mini-2024-09-12": ModelCapabilities(
        family=ModelFamily.O1_MINI,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=False,
        supports_system_message=False,
        supports_developer_message=False,
        supports_streaming=False,
        supports_tools=False,
        supports_parallel_tool_calls=False,
        supports_vision=False,
        supports_logprobs=False,
        max_input_tokens=128000,
        max_output_tokens=65536,
    ),
    "o1": ModelCapabilities(
        family=ModelFamily.O1,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        convert_system_to_developer=True,
        supports_streaming=False,
        supports_tools=True,
        supports_parallel_tool_calls=False,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=200000,
        max_output_tokens=100000,
    ),
    "o1-2024-12-17": ModelCapabilities(
        family=ModelFamily.O1,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        convert_system_to_developer=True,
        supports_streaming=False,
        supports_tools=True,
        supports_parallel_tool_calls=False,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=200000,
        max_output_tokens=100000,
    ),
    "o1-preview": ModelCapabilities(
        family=ModelFamily.O1_PREVIEW,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=False,
        supports_system_message=False,
        supports_developer_message=False,
        supports_streaming=False,
        supports_tools=False,
        supports_parallel_tool_calls=False,
        supports_vision=False,
        supports_logprobs=False,
        max_input_tokens=128000,
        max_output_tokens=32768,
        notes="Preview model, will be retired",
    ),
    # =========================================================================
    # O3 Series (Advanced Reasoning Models)
    # =========================================================================
    "o3-mini": ModelCapabilities(
        family=ModelFamily.O3_MINI,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        convert_system_to_developer=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=False,
        supports_vision=False,
        supports_logprobs=False,
        max_input_tokens=200000,
        max_output_tokens=100000,
    ),
    "o3-mini-2025-01-31": ModelCapabilities(
        family=ModelFamily.O3_MINI,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        convert_system_to_developer=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=False,
        supports_vision=False,
        supports_logprobs=False,
        max_input_tokens=200000,
        max_output_tokens=100000,
    ),
    "o3": ModelCapabilities(
        family=ModelFamily.O3,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        convert_system_to_developer=True,
        supports_streaming=True,  # Limited access
        supports_tools=True,
        supports_parallel_tool_calls=False,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=200000,
        max_output_tokens=100000,
    ),
    "o3-2025-04-16": ModelCapabilities(
        family=ModelFamily.O3,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        convert_system_to_developer=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=False,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=200000,
        max_output_tokens=100000,
    ),
    "o3-pro": ModelCapabilities(
        family=ModelFamily.O3_PRO,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        convert_system_to_developer=True,
        supports_streaming=False,  # o3-pro doesn't support streaming
        supports_tools=True,
        supports_parallel_tool_calls=False,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=200000,
        max_output_tokens=100000,
        notes="Use background mode to avoid timeouts",
    ),
    "o3-pro-2025-06-10": ModelCapabilities(
        family=ModelFamily.O3_PRO,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        convert_system_to_developer=True,
        supports_streaming=False,
        supports_tools=True,
        supports_parallel_tool_calls=False,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=200000,
        max_output_tokens=100000,
    ),
    # =========================================================================
    # O4 Series
    # =========================================================================
    "o4-mini": ModelCapabilities(
        family=ModelFamily.O4_MINI,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        convert_system_to_developer=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=False,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=200000,
        max_output_tokens=100000,
    ),
    "o4-mini-2025-04-16": ModelCapabilities(
        family=ModelFamily.O4_MINI,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        convert_system_to_developer=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=False,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=200000,
        max_output_tokens=100000,
    ),
    # =========================================================================
    # Codex-mini (Fine-tuned o4-mini)
    # =========================================================================
    "codex-mini": ModelCapabilities(
        family=ModelFamily.CODEX_MINI,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        convert_system_to_developer=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=False,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=200000,
        max_output_tokens=100000,
    ),
    "codex-mini-2025-05-16": ModelCapabilities(
        family=ModelFamily.CODEX_MINI,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        convert_system_to_developer=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=False,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=200000,
        max_output_tokens=100000,
    ),
    # =========================================================================
    # GPT-5 Series (Next-gen Reasoning Models)
    # =========================================================================
    "gpt-5-nano": ModelCapabilities(
        family=ModelFamily.GPT_5_NANO,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"minimal", "low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5-nano-2025-08-07": ModelCapabilities(
        family=ModelFamily.GPT_5_NANO,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"minimal", "low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5-mini": ModelCapabilities(
        family=ModelFamily.GPT_5_MINI,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"minimal", "low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5-mini-2025-08-07": ModelCapabilities(
        family=ModelFamily.GPT_5_MINI,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"minimal", "low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5": ModelCapabilities(
        family=ModelFamily.GPT_5,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"minimal", "low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5-2025-08-07": ModelCapabilities(
        family=ModelFamily.GPT_5,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"minimal", "low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5-pro": ModelCapabilities(
        family=ModelFamily.GPT_5_PRO,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"high"},  # Only supports high
        default_reasoning_effort="high",
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=False,  # No streaming for pro
        supports_tools=True,
        supports_parallel_tool_calls=False,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5-pro-2025-10-06": ModelCapabilities(
        family=ModelFamily.GPT_5_PRO,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"high"},
        default_reasoning_effort="high",
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=False,
        supports_tools=True,
        supports_parallel_tool_calls=False,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5-codex": ModelCapabilities(
        family=ModelFamily.GPT_5_CODEX,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"low", "medium", "high"},  # No minimal for codex
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5-codex-2025-09-01": ModelCapabilities(
        family=ModelFamily.GPT_5_CODEX,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    # =========================================================================
    # GPT-5.1 Series
    # =========================================================================
    "gpt-5.1": ModelCapabilities(
        family=ModelFamily.GPT_51,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"none", "minimal", "low", "medium", "high"},
        default_reasoning_effort="none",  # Important: defaults to none!
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
        notes="reasoning_effort defaults to 'none' - must explicitly set for reasoning",
    ),
    "gpt-5.1-2025-11-13": ModelCapabilities(
        family=ModelFamily.GPT_51,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"none", "minimal", "low", "medium", "high"},
        default_reasoning_effort="none",
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5.1-chat": ModelCapabilities(
        family=ModelFamily.GPT_51_CHAT,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,  # Reasoning model - no temperature!
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"minimal", "low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=111616,
        max_output_tokens=16384,
        notes="Built-in reasoning - remove temperature parameter when migrating from gpt-5-chat",
    ),
    "gpt-5.1-chat-2025-11-13": ModelCapabilities(
        family=ModelFamily.GPT_51_CHAT,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"minimal", "low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=111616,
        max_output_tokens=16384,
    ),
    "gpt-5.1-codex": ModelCapabilities(
        family=ModelFamily.GPT_51_CODEX,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"none", "minimal", "low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5.1-codex-2025-11-13": ModelCapabilities(
        family=ModelFamily.GPT_51_CODEX,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"none", "minimal", "low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5.1-codex-mini": ModelCapabilities(
        family=ModelFamily.GPT_51_CODEX_MINI,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"none", "minimal", "low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5.1-codex-mini-2025-11-13": ModelCapabilities(
        family=ModelFamily.GPT_51_CODEX_MINI,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"none", "minimal", "low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5.1-codex-max": ModelCapabilities(
        family=ModelFamily.GPT_51_CODEX_MAX,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        },  # Supports xhigh!
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
        notes="Supports xhigh reasoning_effort; 'none' is NOT supported",
    ),
    # =========================================================================
    # GPT-5.2 Series
    # =========================================================================
    "gpt-5.2": ModelCapabilities(
        family=ModelFamily.GPT_52,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"none", "minimal", "low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5.2-chat": ModelCapabilities(
        family=ModelFamily.GPT_52_CHAT,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        use_max_completion_tokens=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_presence_penalty=False,
        supports_frequency_penalty=False,
        is_reasoning_model=True,
        supports_reasoning_effort=True,
        supported_reasoning_efforts={"none", "minimal", "low", "medium", "high"},
        supports_system_message=True,
        supports_developer_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        supports_logprobs=False,
        max_input_tokens=272000,
        max_output_tokens=128000,
        notes="Chat-optimized variant of GPT-5.2",
    ),
    # =========================================================================
    # Model Router (Azure-specific smart routing)
    # =========================================================================
    "model-router": ModelCapabilities(
        family=ModelFamily.MODEL_ROUTER,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        supports_top_p=True,
        supports_presence_penalty=True,
        supports_frequency_penalty=True,
        is_reasoning_model=False,
        supports_system_message=True,
        supports_streaming=True,
        supports_tools=True,
        supports_parallel_tool_calls=True,
        supports_vision=True,
        max_input_tokens=128000,
        max_output_tokens=16384,
        notes="Azure smart routing deployment - routes to optimal model",
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_model_capabilities(model: str) -> ModelCapabilities:
    """
    Get capabilities for a model identifier.

    Attempts exact match first, then tries to match by prefix/pattern.
    Falls back to default capabilities for unknown models.

    Args:
        model: Model identifier (e.g., "gpt-4o", "o3-mini", "gpt-5.1")

    Returns:
        ModelCapabilities for the model
    """
    # Exact match
    if model in MODEL_CAPABILITIES_REGISTRY:
        return MODEL_CAPABILITIES_REGISTRY[model]

    # Normalize model name (lowercase, handle Azure deployment names)
    normalized = model.lower().strip()

    if normalized in MODEL_CAPABILITIES_REGISTRY:
        return MODEL_CAPABILITIES_REGISTRY[normalized]

    # Try prefix matching for versioned models
    for registered_model in MODEL_CAPABILITIES_REGISTRY:
        if normalized.startswith(registered_model) or registered_model.startswith(normalized):
            return MODEL_CAPABILITIES_REGISTRY[registered_model]

    # Pattern-based matching for model families
    patterns = [
        (r"^gpt-5\.2-chat", "gpt-5.2-chat"),
        (r"^gpt-5\.2", "gpt-5.2"),
        (r"^model-router", "model-router"),
        (r"^gpt-5\.1-codex-max", "gpt-5.1-codex-max"),
        (r"^gpt-5\.1-codex-mini", "gpt-5.1-codex-mini"),
        (r"^gpt-5\.1-codex", "gpt-5.1-codex"),
        (r"^gpt-5\.1-chat", "gpt-5.1-chat"),
        (r"^gpt-5\.1", "gpt-5.1"),
        (r"^gpt-5-pro", "gpt-5-pro"),
        (r"^gpt-5-codex", "gpt-5-codex"),
        (r"^gpt-5-mini", "gpt-5-mini"),
        (r"^gpt-5-nano", "gpt-5-nano"),
        (r"^gpt-5", "gpt-5"),
        (r"^o4-mini", "o4-mini"),
        (r"^o3-pro", "o3-pro"),
        (r"^o3-mini", "o3-mini"),
        (r"^o3", "o3"),
        (r"^o1-mini", "o1-mini"),
        (r"^o1-preview", "o1-preview"),
        (r"^o1", "o1"),
        (r"^codex-mini", "codex-mini"),
        (r"^gpt-4\.5", "gpt-4.5-preview"),
        (r"^gpt-4\.1-nano", "gpt-4.1-nano"),
        (r"^gpt-4\.1-mini", "gpt-4.1-mini"),
        (r"^gpt-4\.1", "gpt-4.1"),
        (r"^gpt-4o-mini", "gpt-4o-mini"),
        (r"^gpt-4o", "gpt-4o"),
        (r"^gpt-4-turbo", "gpt-4-turbo"),
        (r"^gpt-4-32k", "gpt-4-32k"),
        (r"^gpt-4", "gpt-4"),
        (r"^gpt-35-turbo-16k", "gpt-35-turbo-16k"),
        (r"^gpt-35-turbo", "gpt-35-turbo"),
        (r"^gpt-3\.5-turbo", "gpt-35-turbo"),
    ]

    for pattern, base_model in patterns:
        if re.match(pattern, normalized):
            if base_model in MODEL_CAPABILITIES_REGISTRY:
                return MODEL_CAPABILITIES_REGISTRY[base_model]

    # Default to GPT-4o-like capabilities for unknown models
    return ModelCapabilities(
        family=ModelFamily.UNKNOWN,
        supports_max_tokens=True,
        supports_max_completion_tokens=True,
        supports_temperature=True,
        max_input_tokens=128000,
        max_output_tokens=4096,
        notes=f"Unknown model '{model}' - using default capabilities",
    )


def is_reasoning_model(model: str) -> bool:
    """
    Check if a model is a reasoning model (o-series or GPT-5+).

    Reasoning models have different parameter requirements:
    - Use max_completion_tokens instead of max_tokens
    - Don't support temperature, top_p, penalties
    - May support reasoning_effort parameter

    Args:
        model: Model identifier

    Returns:
        True if the model is a reasoning model
    """
    capabilities = get_model_capabilities(model)
    return capabilities.is_reasoning_model
