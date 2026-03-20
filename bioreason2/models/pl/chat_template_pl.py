"""
Chat templates for Protein LLM models.

This module provides chat templates for formatting conversations with protein sequences and GO graph context using natural language boundaries.
"""

from pathlib import Path


def _load_template(template_name: str) -> str:
    """Load a chat template from file."""
    template_dir = Path(__file__).parent
    template_path = template_dir / template_name
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


# Chat template for Protein LLM models
CHAT_TEMPLATE = _load_template("qwen3_4b_chat_template.jinja2")


def get_chat_template(model_name: str) -> str:
    """
    Get the chat template for a given model name.

    Args:
        model_name: The name of the model to get the chat template for.

    Returns:
        The chat template for the given model name.
    """
    # Legacy support kept for model_name but we return the same template for all models
    return CHAT_TEMPLATE
