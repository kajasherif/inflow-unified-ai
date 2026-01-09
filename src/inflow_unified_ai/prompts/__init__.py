"""Prompts module for inflow-unified-ai."""

from inflow_unified_ai.prompts.manager import (
    PromptTemplate,
    PromptManager,
    register_prompt,
    get_prompt,
    get_system_prompt,
    SYSTEM_PROMPTS,
)

__all__ = [
    "PromptTemplate",
    "PromptManager",
    "register_prompt",
    "get_prompt",
    "get_system_prompt",
    "SYSTEM_PROMPTS",
]
