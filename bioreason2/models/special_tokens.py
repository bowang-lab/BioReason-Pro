"""
Special tokens for BioReason2 models.

This module defines only the pad tokens that are added to tokenizer vocabulary.
Start/end boundaries are handled as natural language in chat templates.
"""

from typing import List

# Only pad tokens are needed - these get replaced with meaningful embeddings
PROTEIN_PAD_TOKEN = "<|protein_pad|>"
GO_GRAPH_PAD_TOKEN = "<|go_graph_pad|>"

# All special tokens added to tokenizer vocabulary
ALL_SPECIAL_TOKENS = [PROTEIN_PAD_TOKEN, GO_GRAPH_PAD_TOKEN]

# Token mappings for easy access
SPECIAL_TOKENS = {
    "protein_pad": PROTEIN_PAD_TOKEN,
    "go_graph_pad": GO_GRAPH_PAD_TOKEN,
}


def get_all_special_tokens() -> List[str]:
    """Get all special tokens added to tokenizer vocabulary."""
    return ALL_SPECIAL_TOKENS.copy()


def get_token(token_name: str) -> str:
    """
    Get a specific special token by name.

    Args:
        token_name: Name of the token ('protein_pad' or 'go_graph_pad')

    Returns:
        The special token string

    Raises:
        KeyError: If token_name is not found
    """
    if token_name not in SPECIAL_TOKENS:
        available = list(SPECIAL_TOKENS.keys())
        raise KeyError(f"Token '{token_name}' not found. Available tokens: {available}")

    return SPECIAL_TOKENS[token_name]
