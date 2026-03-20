"""
Protein LLM processing utilities.

This module provides processing classes for handling protein and text inputs for ProteinLLMModel.
"""

from typing import List, Optional, Union, Dict, Any

import torch

from transformers.processing_utils import (
    CommonKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from bioreason2.models.special_tokens import get_token


class PLProteinKwargs(CommonKwargs):
    """Keyword arguments specific to protein processing"""

    max_length_text: Optional[int]
    max_length_protein: Optional[int]


class PLProcessorKwargs(ProcessingKwargs, total=False):
    """Processing keyword arguments for the PL processor"""

    protein_kwargs: PLProteinKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class PLProcessor(ProcessorMixin):
    r"""
    Constructs a PL processor which wraps a ESM3 protein processor and a text tokenizer into a single processor.
    This processor handles both text and protein sequence processing to prepare inputs for the ProteinLLMModel.

    Args:
        tokenizer (PreTrainedTokenizerBase, *optional*):
            The text tokenizer used for processing text inputs.
        protein_tokenizer (PreTrainedTokenizerBase, *optional*):
            The protein tokenizer used for processing protein sequences.
        chat_template (`str`, *optional*):
            A Jinja template for chat formatting. If None, will use the tokenizer's template.
    """

    attributes = ["tokenizer"]
    valid_kwargs = ["model", "chat_template"]
    tokenizer_class = (
        "Qwen2Tokenizer",
        "Qwen2TokenizerFast",
        "GPT2TokenizerFast",
    )

    def __init__(self, tokenizer=None, chat_template=None, **kwargs):
        """
        Initialize the processor with text and protein tokenizers.

        Args:
            tokenizer: Text tokenizer (usually from a language model)
            chat_template: Template for formatting chat conversations
            **kwargs: Additional arguments
        """
        self.tokenizer = tokenizer

        self.protein_token = (
            get_token("protein_pad") if not hasattr(self.tokenizer, "protein_token") else self.tokenizer.protein_token
        )
        self.go_token = (
            get_token("go_graph_pad") if not hasattr(self.tokenizer, "go_token") else self.tokenizer.go_token
        )

        # Get chat template from tokenizer if not provided
        if chat_template is None and hasattr(self.tokenizer, "chat_template"):
            chat_template = self.tokenizer.chat_template
        super().__init__(tokenizer, chat_template=chat_template)

        # The GRPO trainer might expect this to be set
        if not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_protein_sequences(
        self,
        batch_protein_sequences: List[List[str]],
        max_length: int = 2048,
        return_tensors: str = "pt",
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Tokenize a batch of protein sequences.

        Args:
            batch_protein_sequences: List of lists of protein sequences per batch item
            max_length: Maximum allowed length for protein sequences
            return_tensors: Return format for tensors ("pt" for PyTorch)
            device: Device to place tensors on

        Returns:
            Dict containing:
                - protein_sequences: The protein sequence strings
                - batch_idx_map: Mapping of which sequences belong to which batch item
        """
        # Create a mapping to track which sequences belong to which batch item
        batch_idx_map = []
        all_sequences = []

        # Flatten all sequences with batch tracking
        for batch_idx, protein_sequences in enumerate(batch_protein_sequences):
            for seq in protein_sequences:
                all_sequences.append(seq)
                batch_idx_map.append(batch_idx)

        # If no sequences in the entire batch, return empty dict
        if not all_sequences:
            return {"protein_sequences": None, "batch_idx_map": []}

        # For ESM3, we don't tokenize here - we pass the raw sequences
        # The ProteinLLMModel will handle ESM3 encoding internally
        return {"protein_sequences": all_sequences, "batch_idx_map": batch_idx_map}

    def __call__(
        self,
        batch_protein_sequences: Optional[List[List[str]]] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        max_length_text: int = 4096,
        max_length_protein: int = 2048,
        return_tensors: str = "pt",
        device: str = "cuda",
        batch_go_aspects: Optional[List[str]] = None,
        num_go_tokens: int = 200,
        **kwargs: Unpack[PLProcessorKwargs],
    ) -> BatchFeature:
        """
        Process text and protein sequences for model input.

        Args:
            batch_protein_sequences: List of lists of protein sequences per batch item
            text: Input text or list of texts
            max_length_text: Maximum length for text sequences
            max_length_protein: Maximum length for protein sequences
            return_tensors: Return format for tensors
            device: Device to place tensors on
            **kwargs: Additional processor keyword arguments

        Returns:
            BatchFeature with tokenized inputs for the model
        """
        output_kwargs = self._merge_kwargs(
            PLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Ensure text is a list
        if not isinstance(text, list):
            text = [text]

        protein_inputs = {}
        if batch_protein_sequences is not None:
            # Process protein sequences
            protein_processing_result = self.tokenize_protein_sequences(
                batch_protein_sequences,
                max_length=max_length_protein,
                return_tensors=return_tensors,
                device=device,
            )

            # Replace protein tokens in text if needed
            index = 0
            for i in range(len(text)):
                while self.protein_token in text[i]:
                    # For proteins, we estimate token count (since ESM3 handles tokenization)
                    # This is a placeholder - in practice, each residue typically becomes one token
                    if protein_processing_result["protein_sequences"]:
                        seq_length = len(protein_processing_result["protein_sequences"][index])
                        num_protein_tokens = min(seq_length, max_length_protein)
                        num_protein_tokens += 2  # For special tokens of ESM
                    else:
                        num_protein_tokens = 1

                    text[i] = text[i].replace(self.protein_token, "<|placeholder|>" * num_protein_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.protein_token)

            # Add batch info to the output
            protein_inputs = {
                "protein_sequences": protein_processing_result["protein_sequences"],
                "batch_idx_map": protein_processing_result["batch_idx_map"],
                "batch_go_aspects": batch_go_aspects,
            }
        else:
            # Even if no protein sequences, we may still have GO aspects
            protein_inputs = {
                "protein_sequences": None,
                "batch_idx_map": [],
                "batch_go_aspects": batch_go_aspects,
            }

        # Replace GO tokens with the correct number of pad tokens (independent of protein sequences)
        for i in range(len(text)):
            while self.go_token in text[i]:
                # Each GO aspect produces exactly 200 reduced embeddings
                text[i] = text[i].replace(self.go_token, "<|go_placeholder|>" * num_go_tokens, 1)
            text[i] = text[i].replace("<|go_placeholder|>", self.go_token)

        # Tokenize text
        text_kwargs = output_kwargs.get("text_kwargs", {})

        if "padding" in text_kwargs:
            del text_kwargs["padding"]

        text_inputs = self.tokenizer(
            text,
            max_length=max_length_text + num_go_tokens + max_length_protein + 2,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
            **text_kwargs,
        )

        # The BatchFeature should have all required fields for the model's forward pass
        return BatchFeature(data={**text_inputs, **protein_inputs})

    def batch_decode(self, *args, **kwargs) -> List[str]:
        """
        This method forwards all its arguments to the tokenizer's batch_decode.

        Returns:
            List of decoded strings
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs) -> str:
        """
        This method forwards all its arguments to the tokenizer's decode.

        Returns:
            Decoded string
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_protein_to_text(
        self,
        generated_outputs: torch.Tensor,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        """
        Post-process the model output to decode the text.

        Args:
            generated_outputs: The token IDs generated by the model
            skip_special_tokens: Whether to skip special tokens in the output
            **kwargs: Additional arguments for the decoder

        Returns:
            List of decoded strings
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        """
        Get the input names expected by the model.

        Returns:
            List of input names
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        protein_input_names = ["protein_sequences", "batch_idx_map", "batch_go_aspects"]

        return list(dict.fromkeys(tokenizer_input_names + protein_input_names))
