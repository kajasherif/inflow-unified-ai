"""
Prompt management for the Unified AI Abstraction Layer.

Provides utilities for working with prompt templates and
normalizing prompts across different providers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import re


class PromptTemplate:
    """
    Simple prompt template with variable substitution.

    Supports both ${variable} and {variable} syntax.

    Example:
        template = PromptTemplate(
            "You are a ${role}. Answer questions about ${topic}."
        )
        prompt = template.format(role="helpful assistant", topic="Python")
    """

    def __init__(self, template: str) -> None:
        """
        Initialize the template.

        Args:
            template: Template string with variables
        """
        self.template = template
        self._variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        """Extract variable names from the template."""
        # Match both ${var} and {var} patterns
        pattern = r"\$\{(\w+)\}|\{(\w+)\}"
        matches = re.findall(pattern, self.template)
        # Flatten and filter empty strings
        return list(set(v for pair in matches for v in pair if v))

    @property
    def variables(self) -> List[str]:
        """Get list of variable names in the template."""
        return self._variables

    def format(self, **kwargs: Any) -> str:
        """
        Format the template with provided variables.

        Args:
            **kwargs: Variable values

        Returns:
            Formatted string

        Raises:
            KeyError: If a required variable is missing
        """
        result = self.template

        # Replace ${var} syntax
        for key, value in kwargs.items():
            result = result.replace(f"${{{key}}}", str(value))
            result = result.replace(f"{{{key}}}", str(value))

        return result

    def safe_format(self, **kwargs: Any) -> str:
        """
        Format the template, leaving missing variables unchanged.

        Args:
            **kwargs: Variable values

        Returns:
            Formatted string with unmatched variables intact
        """
        result = self.template

        for key, value in kwargs.items():
            result = result.replace(f"${{{key}}}", str(value))
            result = result.replace(f"{{{key}}}", str(value))

        return result


class PromptManager:
    """
    Manager for prompt templates.

    Provides centralized storage and retrieval of prompt templates.

    Example:
        manager = PromptManager()
        manager.register("greeting", "Hello, ${name}! How can I help you today?")
        prompt = manager.get("greeting", name="Alice")
    """

    def __init__(self) -> None:
        """Initialize the prompt manager."""
        self._templates: Dict[str, PromptTemplate] = {}

    def register(self, name: str, template: str) -> None:
        """
        Register a prompt template.

        Args:
            name: Template name
            template: Template string
        """
        self._templates[name] = PromptTemplate(template)

    def get(self, name: str, **kwargs: Any) -> str:
        """
        Get and format a registered template.

        Args:
            name: Template name
            **kwargs: Variable values

        Returns:
            Formatted prompt string

        Raises:
            KeyError: If template not found
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found")

        return self._templates[name].format(**kwargs)

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a template object by name.

        Args:
            name: Template name

        Returns:
            PromptTemplate or None if not found
        """
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        """Get list of registered template names."""
        return list(self._templates.keys())

    def clear(self) -> None:
        """Remove all registered templates."""
        self._templates.clear()


# Default prompt manager instance
_default_manager = PromptManager()


def register_prompt(name: str, template: str) -> None:
    """Register a prompt with the default manager."""
    _default_manager.register(name, template)


def get_prompt(name: str, **kwargs: Any) -> str:
    """Get a prompt from the default manager."""
    return _default_manager.get(name, **kwargs)


# Pre-built prompts for common scenarios
SYSTEM_PROMPTS = {
    "assistant": "You are a helpful AI assistant.",
    "coder": "You are an expert programmer. Write clean, efficient, and well-documented code.",
    "analyst": "You are a data analyst. Provide clear, accurate analysis based on the data provided.",
    "writer": "You are a skilled writer. Create engaging, well-structured content.",
    "teacher": "You are a patient and knowledgeable teacher. Explain concepts clearly and thoroughly.",
}


def get_system_prompt(role: str) -> str:
    """
    Get a pre-built system prompt for a role.

    Args:
        role: Role name (assistant, coder, analyst, writer, teacher)

    Returns:
        System prompt string
    """
    return SYSTEM_PROMPTS.get(role, SYSTEM_PROMPTS["assistant"])
