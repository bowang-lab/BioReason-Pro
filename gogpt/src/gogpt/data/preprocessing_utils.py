"""
Reusable preprocessing utilities for GO-GPT data preparation.
"""

import random
import torch
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
import json


def preprocess_single_example(
    example: Dict[str, Any],
    protein_tokenizer: Any,
    go_tokenizer: Any,
    organism_mapper: Any,
    embed_model_type: str = "esm2",
    max_go_terms: int = 300,
    max_protein_length: int = 1024,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Preprocess a single example for GO-GPT model.

    Args:
        example: Dictionary containing 'sequence', 'go_terms', 'aspect', 'organism'
        protein_tokenizer: Tokenizer for protein sequences (ESM2 via transformers)
        go_tokenizer: Custom GO term tokenizer
        organism_mapper: Organism ID mapper
        embed_model_type: Type of protein model (only "esm2" supported)
        max_go_terms: Maximum number of GO terms to keep
        max_protein_length: Maximum protein sequence length
        seed: Random seed for GO term sampling

    Returns:
        Preprocessed example with tokenized inputs
    """
    if seed is not None:
        random.seed(seed)

    sequence = example['sequence']
    go_terms = example['go_terms']
    aspect = example['aspect']
    organism = example['organism']

    if len(go_terms) > max_go_terms:
        go_terms = random.sample(go_terms, max_go_terms)

    inputs = protein_tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=max_protein_length,
        padding=False,
        add_special_tokens=False
    )
    protein_tokens = inputs["input_ids"][0]
    protein_mask = inputs["attention_mask"][0]

    if len(protein_tokens) > max_protein_length:
        protein_tokens = protein_tokens[:max_protein_length]
        protein_mask = protein_mask[:max_protein_length]

    go_tokens = go_tokenizer.encode(go_terms, aspect)
    go_tokens = torch.tensor(go_tokens, dtype=torch.long)

    go_input_tokens = go_tokens[:-1] if len(go_tokens) > 1 else go_tokens
    go_targets = go_tokens[1:] if len(go_tokens) > 1 else go_tokens

    organism_id = organism_mapper.map_organism(organism)

    return {
        'protein_tokens': protein_tokens,
        'protein_mask': protein_mask,
        'go_tokens': go_tokens,
        'go_input_tokens': go_input_tokens,
        'go_targets': go_targets,
        'organism_id': organism_id,
        'protein_id': example.get('protein_id', 'unknown'),
        'aspect': aspect,
        'organism': organism,
        'go_terms': go_terms
    }


def load_preprocessing_artifacts(artifacts_dir: Path) -> Tuple[Any, Any, Dict]:
    """
    Load preprocessing artifacts (tokenizer, organism mapper, tokenizer info).

    Args:
        artifacts_dir: Directory containing the artifacts

    Returns:
        Tuple of (go_tokenizer, organism_mapper, tokenizer_info)
    """
    tokenizer_path = artifacts_dir / "go_tokenizer.pkl"
    with open(tokenizer_path, 'rb') as f:
        go_tokenizer = pickle.load(f)

    mapper_path = artifacts_dir / "organism_mapper.pkl"
    with open(mapper_path, 'rb') as f:
        organism_mapper = pickle.load(f)

    info_path = artifacts_dir / "tokenizer_info.json"
    with open(info_path, 'r') as f:
        tokenizer_info = json.load(f)

    return go_tokenizer, organism_mapper, tokenizer_info


def get_protein_tokenizer(embed_model_path: str, embed_model_type: str = "esm2"):
    """
    Get the appropriate protein tokenizer based on model path and type.

    Args:
        embed_model_path: Path to the protein model (ESM2 via transformers)
        embed_model_type: Type of model (only "esm2" supported)

    Returns:
        Tuple of (protein_tokenizer, embed_model_type)
    """
    from transformers import AutoTokenizer
    protein_tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
    return protein_tokenizer, "esm2"


def determine_max_protein_length(embed_model_path: str, embed_model_type: str = "esm2") -> int:
    """
    Determine the maximum protein length based on the model.

    Args:
        embed_model_path: Path to the protein model
        embed_model_type: Type of model (only "esm2" supported)

    Returns:
        Maximum protein sequence length
    """
    return 1024
