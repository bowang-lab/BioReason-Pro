import pickle
import os
import json
from pathlib import Path

from gogpt.utils.organism_mapper import OrganismMapper
from gogpt.data.tokenizer import GOTermTokenizer
from gogpt.data.preprocessing_utils import (
    preprocess_single_example,
    load_preprocessing_artifacts,
    get_protein_tokenizer,
    determine_max_protein_length
)

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from collections import Counter

from datasets import load_dataset, load_from_disk
from typing import Optional, Tuple
import random
from tqdm import tqdm


class PreprocessedProteinGODataset(Dataset):
    """Dataset that loads preprocessed data directly without tokenization."""

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        print(f"Loaded preprocessed dataset with {len(self.dataset)} examples")

        if len(self.dataset) > 0:
            aspects = self.dataset['aspect']
            aspect_counts = Counter(aspects)
            print(f"Aspect distribution: {dict(aspect_counts)}")

            sample_size = min(1000, len(self.dataset))
            sample = self.dataset.select(range(sample_size))
            organisms = sample['organism']
            org_counts = Counter(organisms)
            print(f"Organism distribution (top 5 from first {sample_size}): {dict(list(org_counts.most_common(5)))}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        if "go_input_tokens" in example:
            go_tokens_to_return = torch.tensor(example["go_input_tokens"])
            targets = torch.tensor(example["go_targets"])
            go_mask_len = len(example["go_input_tokens"])
        else:
            go_tokens_to_return = torch.tensor(example["go_tokens"])
            targets = torch.tensor(example["go_tokens"])
            go_mask_len = len(example["go_tokens"])

        result = {
            "protein_tokens": torch.tensor(example["protein_tokens"]),
            "protein_mask": torch.tensor(example["protein_mask"]),
            "go_tokens": go_tokens_to_return,
            "targets": targets,
            "go_mask": torch.ones(go_mask_len, dtype=torch.bool),
            "organism_id": torch.tensor(example["organism_id"], dtype=torch.long),
            "protein_id": example["protein_id"]
        }

        if "go_terms_list" in example:
            result["go_terms_list"] = example["go_terms_list"]

        return result


class ProteinGODataset(Dataset):
    def __init__(self, sequences, go_terms, aspects, tokenizer, protein_tokenizer, organisms, protein_ids):
        self.sequences = sequences
        self.go_terms = go_terms
        self.aspects = aspects
        self.tokenizer = tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.max_length = 1024
        self.organisms = organisms
        self.protein_ids = protein_ids

        print()
        print(f"Dataset initialized with {len(sequences)} annotations")
        if aspects and len(aspects) > 0:
            aspect_counts = Counter(aspects)
            print(f"Aspect distribution: {dict(aspect_counts)}")
        if organisms and len(organisms) > 0:
            org_counts = Counter(organisms)
            print(f"Organism distribution (top 5): {dict(list(org_counts.items())[:5])}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        go_terms = self.go_terms[idx]
        aspect = self.aspects[idx]
        organism = self.organisms[idx]

        if len(go_terms) > 300:
            import random
            go_terms = random.sample(go_terms, 300)

        tok_out = self.protein_tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        go_tokens = self.tokenizer.encode(go_terms, aspect=aspect)

        input_tokens = go_tokens[:-1]
        shifted_targets = [0] * len(input_tokens)

        for i in range(len(input_tokens) - 1):
            shifted_targets[i] = go_tokens[i + 1]

        shifted_targets[-1] = go_tokens[-1]

        assert len(input_tokens) == len(shifted_targets), f"Length mismatch: {len(input_tokens)} vs {len(shifted_targets)}"

        return {
            "protein_tokens": tok_out["input_ids"].squeeze(0),
            "protein_mask": tok_out["attention_mask"].squeeze(0),
            "go_tokens": torch.tensor(input_tokens),
            "targets": torch.tensor(shifted_targets),
            "go_mask": torch.ones(len(input_tokens), dtype=torch.bool),
            "organism_id": torch.tensor(organism, dtype=torch.long),
            "protein_id": self.protein_ids[idx]
        }


def collate_batch(batch):
    protein_tokens, protein_masks, go_tokens, go_masks, targets, organism_ids, protein_ids = zip(
        *(
            (
                item["protein_tokens"],
                item["protein_mask"],
                item["go_tokens"],
                item["go_mask"],
                item["targets"],
                item["organism_id"],
                item["protein_id"]
            )
            for item in batch
        )
    )

    result = {
        "protein_tokens": pad_sequence(
            protein_tokens, batch_first=True, padding_value=1
        ),
        "protein_mask": pad_sequence(
            protein_masks, batch_first=True, padding_value=False
        ),
        "go_tokens": pad_sequence(go_tokens, batch_first=True, padding_value=0),
        "go_mask": pad_sequence(go_masks, batch_first=True, padding_value=False),
        "targets": pad_sequence(targets, batch_first=True, padding_value=0),
        "organism_id": torch.stack(organism_ids),
        "protein_ids": protein_ids,
    }

    if "go_terms_list" in batch[0]:
        result["go_terms_list"] = [item["go_terms_list"] for item in batch]

    return result


def load_preprocessed_data(
    preprocessed_path: str,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 8,
):
    """Load preprocessed data from disk."""
    preprocessed_path = Path(preprocessed_path)

    print(f"Loading preprocessed data from: {preprocessed_path}")

    dataset_path = preprocessed_path / "dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found at {dataset_path}")

    dataset = load_from_disk(str(dataset_path))
    train_data = dataset["train"]
    val_data = dataset["validation"]

    print(f"Loaded preprocessed data: {len(train_data)} train, {len(val_data)} val samples")

    artifacts_dir = preprocessed_path / "artifacts"

    tokenizer_path = artifacts_dir / "go_tokenizer.pkl"
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Loaded GO tokenizer with vocab size: {len(tokenizer.token_to_id)}")

    mapper_path = artifacts_dir / "organism_mapper.pkl"
    with open(mapper_path, 'rb') as f:
        organism_mapper = pickle.load(f)
    print(f"Loaded organism mapper with {organism_mapper.get_vocab_size()} organisms")

    info_path = artifacts_dir / "tokenizer_info.json"
    with open(info_path, 'r') as f:
        tokenizer_info = json.load(f)

    print("\nCreating dataset objects...")
    train_dataset = PreprocessedProteinGODataset(train_data)
    val_dataset = PreprocessedProteinGODataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    print(f"\nData loading complete!")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    return tokenizer, train_loader, val_loader, tokenizer_info


def load_eval_data(
    hf_dataset_name: str,
    hf_dataset_config: str,
    split: str,
    preprocessed_artifacts_path: str,
    embed_model_path: str,
    batch_size: int = 256,
    max_go_terms: int = 300,
    max_protein_length: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 8,
    seed: int = 42
) -> Tuple[DataLoader, dict, object]:
    """
    Load and preprocess any split from any HuggingFace dataset for evaluation.

    Args:
        hf_dataset_name: HuggingFace dataset name (e.g., "wanglab/cafa5")
        hf_dataset_config: Dataset configuration
        split: Split to load (e.g., "validation", "test", "test_superset")
        preprocessed_artifacts_path: Path to directory containing tokenizer and organism mapper
        embed_model_path: Path to protein language model
        batch_size: Batch size for DataLoader
        max_go_terms: Maximum number of GO terms per protein
        max_protein_length: Maximum protein sequence length (auto-detected if None)
        num_workers: Number of workers for DataLoader
        pin_memory: Whether to pin memory
        persistent_workers: Whether to use persistent workers
        prefetch_factor: Prefetch factor for DataLoader
        seed: Random seed for GO term sampling

    Returns:
        Tuple of (dataloader, tokenizer_info, go_tokenizer)
    """
    print(f"Loading {split} split from {hf_dataset_name}/{hf_dataset_config}")

    artifacts_dir = Path(preprocessed_artifacts_path)
    go_tokenizer, organism_mapper, tokenizer_info = load_preprocessing_artifacts(artifacts_dir)
    print(f"Loaded preprocessing artifacts from {artifacts_dir}")

    protein_tokenizer, embed_model_type = get_protein_tokenizer(embed_model_path)
    print(f"Loaded protein tokenizer for {embed_model_path} (type: {embed_model_type})")

    if max_protein_length is None:
        max_protein_length = determine_max_protein_length(embed_model_path, embed_model_type)
    print(f"Using max protein length: {max_protein_length}")

    dataset = load_dataset(hf_dataset_name, name=hf_dataset_config, trust_remote_code=True)

    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")

    data = dataset[split]
    print(f"Loaded {len(data)} examples from {split} split")

    processed_examples = []

    for example in tqdm(data, desc=f"Processing {split} examples"):
        protein_id = example.get('EntryID', example.get('protein_id', 'unknown'))
        sequence = example.get('Sequence', example.get('sequence', ''))
        organism = example.get('Organism', example.get('organism', 'unknown'))

        for aspect_key in ['go_mf', 'go_bp', 'go_cc']:
            if aspect_key in example and example[aspect_key]:
                aspect = aspect_key.split('_')[1].upper()

                formatted_example = {
                    'protein_id': protein_id,
                    'sequence': sequence,
                    'go_terms': example[aspect_key],
                    'aspect': aspect,
                    'organism': organism
                }

                try:
                    processed = preprocess_single_example(
                        formatted_example,
                        protein_tokenizer,
                        go_tokenizer,
                        organism_mapper,
                        embed_model_type,
                        max_go_terms,
                        max_protein_length,
                        seed
                    )
                    processed_examples.append(processed)
                except Exception as e:
                    print(f"Warning: Failed to process example {protein_id} for aspect {aspect}: {e}")
                    continue

    print(f"Processed {len(processed_examples)} examples")

    from datasets import Dataset as HFDataset
    hf_dataset = HFDataset.from_list(processed_examples)
    eval_dataset = PreprocessedProteinGODataset(hf_dataset)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    print(f"\nData loading complete!")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches: {len(eval_loader)}")

    return eval_loader, tokenizer_info, go_tokenizer
