#!/usr/bin/env python3
"""
Data preprocessing script for GO-GPT.
Creates a vocabulary with GO terms filtered by frequency from training data.
Processes all protein-aspect pairs in the HuggingFace dataset splits.
"""

import os
import sys

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*Deprecated call to `pkg_resources.declare_namespace.*")

import json
import pickle
from pathlib import Path
from collections import Counter
import random

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm

@hydra.main(version_base="1.3", config_path="../configs", config_name="preprocess/base")
def prepare_data(cfg: DictConfig) -> None:
    """
    Preprocess data with GO vocabulary filtered by frequency from training data.
    Uses HuggingFace dataset splits.
    """
    # Get the original working directory (before Hydra changes it)
    original_cwd = hydra.utils.get_original_cwd()

    # Add src to path for imports
    sys.path.append(os.path.join(original_cwd, "src"))

    from gogpt.data.tokenizer import GOTermTokenizer
    from gogpt.utils.organism_mapper import OrganismMapper

    # Print config
    print("="*60)
    print("GO-GPT Data Preprocessing Configuration:")
    print("(No protein-aspect pair filtering)")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60)

    # Set seed for reproducibility
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Vocabulary selection configuration per aspect
    vocab_config = {
        'MF': cfg.vocab_limits.mf,
        'BP': cfg.vocab_limits.bp,
        'CC': cfg.vocab_limits.cc
    }
    print(f"Vocabulary selection per aspect: {OmegaConf.to_yaml(vocab_config)}")

    # ESM2 is the only supported model type
    embed_model_type = "esm2"
    max_length = 1024

    print(f"Model type: {embed_model_type}")
    print(f"Max sequence length: {max_length}")

    # Load dataset
    print(f"\nLoading dataset: {cfg.hf_dataset_name}")
    dataset = load_dataset(cfg.hf_dataset_name, name=cfg.hf_dataset_config)
    train_data = dataset["train"]
    val_data = dataset["validation"]
    print(f"✓ Loaded: {len(train_data)} train, {len(val_data)} val samples")

    # STEP 1: Count GO term frequencies per aspect in TRAINING DATA ONLY
    print("\nCounting GO term frequencies per aspect in training data...")
    aspect_go_counters = {
        'MF': Counter(),
        'BP': Counter(),
        'CC': Counter()
    }

    for example in tqdm(train_data, desc="Counting GO terms from training data"):
        # Count MF terms
        if example['go_mf']:
            aspect_go_counters['MF'].update(example['go_mf'])

        # Count BP terms
        if example['go_bp']:
            aspect_go_counters['BP'].update(example['go_bp'])

        # Count CC terms
        if example['go_cc']:
            aspect_go_counters['CC'].update(example['go_cc'])

    # Print frequency statistics
    for aspect, counter in aspect_go_counters.items():
        print(f"\n{aspect} aspect:")
        print(f"  Total unique GO terms: {len(counter)}")
        print(f"  Total annotations: {sum(counter.values())}")
        if counter:
            most_common = counter.most_common(3)
            print(f"  Top 3: {most_common}")

    # STEP 2: Select GO terms per aspect using configured method
    print(f"\nSelecting GO terms per aspect using configured methods...")
    limited_vocab_per_aspect = {}
    vocab_stats = {}  # Store statistics for cross-reporting

    for aspect, config in vocab_config.items():
        counter = aspect_go_counters[aspect]
        method = config.method
        value = config.value

        if not counter:
            limited_vocab_per_aspect[aspect] = set()
            vocab_stats[aspect] = {"selected": 0, "method": method, "value": value}
            print(f"  {aspect}: No GO terms found")
            continue

        if method == "top_k":
            # Select top K most frequent terms
            available_terms = len(counter)
            k = min(value, available_terms)  # Don't request more than available

            selected_terms = [term for term, count in counter.most_common(k)]
            limited_vocab_per_aspect[aspect] = set(selected_terms)

            # Get minimum frequency (frequency of the k-th term)
            min_freq = counter.most_common(k)[-1][1] if k > 0 else 0

            vocab_stats[aspect] = {
                "method": method,
                "requested": value,
                "selected": len(selected_terms),
                "min_freq": min_freq,
                "available_terms": available_terms
            }

            print(f"  {aspect}: Selected top {len(selected_terms)} GO terms (requested {value})")
            print(f"    → Min frequency: {min_freq} proteins")
            if value > available_terms:
                print(f"    → Warning: Requested {value} but only {available_terms} available")

        elif method == "min_freq":
            # Select all terms with frequency >= threshold
            selected_terms = [term for term, count in counter.items() if count >= value]
            limited_vocab_per_aspect[aspect] = set(selected_terms)

            # Get frequency range
            if selected_terms:
                selected_counts = [counter[term] for term in selected_terms]
                min_freq = min(selected_counts)  # Should be >= value
                max_freq = max(selected_counts)
            else:
                min_freq = max_freq = 0

            vocab_stats[aspect] = {
                "method": method,
                "threshold": value,
                "selected": len(selected_terms),
                "freq_range": (min_freq, max_freq),
                "total_terms": len(counter)
            }

            print(f"  {aspect}: Selected {len(selected_terms)} GO terms (min freq ≥{value})")
            if selected_terms:
                print(f"    → Frequency range: {min_freq}-{max_freq} proteins")
            else:
                print(f"    → Warning: No terms meet frequency threshold of {value}")

        else:
            raise ValueError(f"Unknown vocabulary selection method: {method}")

    # Combine all pruned vocabulary terms
    all_pruned_go_terms = set()
    for aspect_terms in limited_vocab_per_aspect.values():
        all_pruned_go_terms.update(aspect_terms)

    print(f"\nTotal pruned vocabulary size: {len(all_pruned_go_terms)} GO terms")

    # Create GO tokenizer with pruned vocabulary
    go_tokenizer = GOTermTokenizer(all_pruned_go_terms)
    print(f"GO tokenizer created with vocab size: {len(go_tokenizer.token_to_id)}")

    # Load protein tokenizer (ESM2 via transformers)
    print(f"\nLoading protein tokenizer: {cfg.embed_model_path}")
    protein_tokenizer = AutoTokenizer.from_pretrained(cfg.embed_model_path)
    print("ESM2 tokenizer loaded")

    # OPTIMIZED: Use HuggingFace multiprocessing for vocabulary filtering
    print("\nApplying vocabulary filtering with multiprocessing...")

    def filter_and_prepare_training_data(example):
        """Fast vocabulary filtering for training data using multiprocessing"""
        results = []

        # Check each aspect
        aspects = [('MF', 'go_mf'), ('BP', 'go_bp'), ('CC', 'go_cc')]

        for aspect_name, aspect_key in aspects:
            # Check if this protein-aspect pair should be included
            has_annotations = example[aspect_key] is not None and len(example[aspect_key]) > 0
            has_valid_sequence = example['sequence'] is not None and len(example['sequence'].strip()) > 0

            if has_annotations and has_valid_sequence:
                original_go_terms = example[aspect_key]

                # Filter GO terms to limited vocabulary
                filtered_go_terms = [
                    term for term in original_go_terms
                    if term in limited_vocab_per_aspect[aspect_name]
                ]

                # Only include if still has GO terms after filtering
                if filtered_go_terms:
                    results.append({
                        'protein_id': example['protein_id'],
                        'sequence': example['sequence'],
                        'go_terms': filtered_go_terms,  # Filtered subset → used for tokenization
                        'aspect': aspect_name,
                        'organism': example['organism'] if example['organism'] else "<UNKNOWN>",
                        'original_count': len(original_go_terms),
                        'filtered_count': len(filtered_go_terms)
                    })

        return {'examples': results}

    def filter_and_prepare_validation_data(example):
        """Fast preparation for validation data (filters sequences but keeps raw GO terms for metrics)"""
        results = []

        # Check each aspect
        aspects = [('MF', 'go_mf'), ('BP', 'go_bp'), ('CC', 'go_cc')]

        for aspect_name, aspect_key in aspects:
            # Check if this protein-aspect pair should be included
            has_annotations = example[aspect_key] is not None and len(example[aspect_key]) > 0
            has_valid_sequence = example['sequence'] is not None and len(example['sequence'].strip()) > 0

            if has_annotations and has_valid_sequence:
                # Get all GO terms (for ground truth metrics)
                all_go_terms = example[aspect_key]

                # Filter GO terms to vocabulary (for clean tokenization without PAD in middle)
                filtered_go_terms = [
                    term for term in all_go_terms
                    if term in limited_vocab_per_aspect[aspect_name]
                ]

                # Only include if still has GO terms after filtering
                if filtered_go_terms:
                    results.append({
                        'protein_id': example['protein_id'],
                        'sequence': example['sequence'],
                        # NOTE: Naming is confusing but necessary for backward compatibility:
                        'go_terms': filtered_go_terms,      # Filtered subset → used for tokenization (no PAD in middle)
                        'go_terms_list': all_go_terms,      # Complete raw set → used for validation metrics (unbiased)
                        'aspect': aspect_name,
                        'organism': example['organism'] if example['organism'] else "<UNKNOWN>",
                        'original_count': len(all_go_terms),
                        'filtered_count': len(filtered_go_terms)
                    })

        return {'examples': results}

    # Apply filtering with multiprocessing
    print("  Filtering training data...")
    train_filtered = train_data.map(
        filter_and_prepare_training_data,
        num_proc=cfg.num_proc,
        desc="Filtering training vocabulary",
        load_from_cache_file=False  # Force recomputation for testing
    )

    print("  Filtering validation data...")
    val_filtered = val_data.map(
        filter_and_prepare_validation_data,
        num_proc=cfg.num_proc,
        desc="Preparing validation data",
        load_from_cache_file=False  # Force recomputation for testing
    )

    # Flatten results and collect statistics
    train_examples = []
    val_examples = []

    # Statistics tracking
    stats = {
        'MF': {'train_examples': 0, 'val_examples': 0, 'train_removed': 0, 'train_total': 0},
        'BP': {'train_examples': 0, 'val_examples': 0, 'train_removed': 0, 'train_total': 0},
        'CC': {'train_examples': 0, 'val_examples': 0, 'train_removed': 0, 'train_total': 0}
    }

    # Collect examples by aspect to maintain same ordering as original approach
    train_examples_by_aspect = {'MF': [], 'BP': [], 'CC': []}
    val_examples_by_aspect = {'MF': [], 'BP': [], 'CC': []}

    # Process training results and group by aspect
    for filtered_result in train_filtered:
        for example in filtered_result['examples']:
            aspect = example['aspect']
            train_examples_by_aspect[aspect].append(example)
            stats[aspect]['train_examples'] += 1
            stats[aspect]['train_total'] += example['original_count']
            stats[aspect]['train_removed'] += example['original_count'] - example['filtered_count']

    # Process validation results and group by aspect
    for filtered_result in val_filtered:
        for example in filtered_result['examples']:
            aspect = example['aspect']
            val_examples_by_aspect[aspect].append(example)
            stats[aspect]['val_examples'] += 1

    # Flatten in aspect order (MF, BP, CC) to match original approach
    train_examples = []
    val_examples = []
    for aspect in ['MF', 'BP', 'CC']:
        train_examples.extend(train_examples_by_aspect[aspect])
        val_examples.extend(val_examples_by_aspect[aspect])

    # Print statistics
    print("\nVocabulary filtering results:")
    for aspect in ['MF', 'BP', 'CC']:
        s = stats[aspect]
        print(f"  {aspect}: {s['train_examples']} train, {s['val_examples']} val examples")
        if s['train_total'] > 0:
            print(f"    {aspect} training: {s['train_examples']} examples, "
                  f"removed {s['train_removed']}/{s['train_total']} GO terms")

    # Create organism mapper with top-N filtering
    print("\nCreating organism mapper...")
    all_organisms = [ex['organism'] for ex in train_examples + val_examples]

    # Configure top N organisms - adjust this value as needed
    top_n_organisms = cfg.get('top_n_organisms', None)  # None means use all organisms (original behavior)

    organism_mapper = OrganismMapper(all_organisms, top_n_organisms=top_n_organisms)
    print(f"Organism mapper created with vocab size: {organism_mapper.get_vocab_size()}")

    # Print mapping statistics
    stats = organism_mapper.get_mapping_stats()
    if stats['top_n_organisms'] is not None:
        coverage_pct = (stats['top_n_organisms'] / stats['total_original_organisms']) * 100
        print(f"  → Using top {stats['top_n_organisms']} organisms ({coverage_pct:.1f}% of unique organisms)")
        print(f"  → {stats['total_original_organisms'] - stats['top_n_organisms']} organisms mapped to <UNKNOWN>")
    else:
        print(f"  → Using all {stats['total_original_organisms']} unique organisms")

    # Tokenize and preprocess all examples
    def preprocess_example(example):
        """Preprocess a single example"""
        sequence = example['sequence']
        go_terms = example['go_terms']  # Already filtered to vocabulary (training and validation)
        aspect = example['aspect']
        organism = example['organism']

        # Limit GO terms if needed (using truncation to preserve original order)
        if len(go_terms) > cfg.max_go_terms:
            go_terms = go_terms[:cfg.max_go_terms]

        # Tokenize protein sequence using ESM2 tokenizer
        tok_out = protein_tokenizer(
            sequence,
            padding=False,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        protein_tokens = tok_out["input_ids"]
        protein_mask = tok_out["attention_mask"]

        # Encode GO terms (already filtered, so no PAD in middle of sequence)
        go_tokens = go_tokenizer.encode(go_terms, aspect=aspect)

        # Create input and target sequences
        input_tokens = go_tokens[:-1]
        shifted_targets = [0] * len(input_tokens)

        for i in range(len(input_tokens) - 1):
            shifted_targets[i] = go_tokens[i + 1]
        shifted_targets[-1] = go_tokens[-1]

        # Map organism to ID
        organism_id = organism_mapper.map_organism(organism)

        result = {
            'protein_id': example['protein_id'],
            'protein_tokens': protein_tokens,
            'protein_mask': protein_mask,
            'go_input_tokens': input_tokens,
            'go_targets': shifted_targets,
            'aspect': aspect,
            'organism_id': organism_id,
            'organism_name': organism,
            'num_go_terms': len(go_terms),
            'protein_length': len(protein_tokens)
        }

        # Pass through raw GO terms list if it exists (validation data only)
        # This field contains ALL GO terms (unfiltered) for unbiased metric computation
        if 'go_terms_list' in example:
            result['go_terms_list'] = example['go_terms_list']

        return result

    # Process all examples using multiprocessing
    print("\nTokenizing and preprocessing examples...")

    # Create HuggingFace datasets first, then apply preprocessing with multiprocessing
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    # Apply tokenization and preprocessing with multiprocessing
    train_dataset = train_dataset.map(
        preprocess_example,
        num_proc=cfg.num_proc,
        desc="Processing training examples",
        load_from_cache_file=False
    )

    val_dataset = val_dataset.map(
        preprocess_example,
        num_proc=cfg.num_proc,
        desc="Processing validation examples",
        load_from_cache_file=False
    )

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    # Create output directory
    output_dir = Path(cfg.output_dir) / cfg.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save dataset
    print(f"\nSaving preprocessed dataset to {output_dir}")
    dataset_dict.save_to_disk(str(output_dir / "dataset"))

    # Save artifacts
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Save GO tokenizer
    tokenizer_path = artifacts_dir / "go_tokenizer.pkl"
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(go_tokenizer, f)
    print(f"Saved GO tokenizer to {tokenizer_path}")

    # Save organism mapper
    mapper_path = artifacts_dir / "organism_mapper.pkl"
    with open(mapper_path, 'wb') as f:
        pickle.dump(organism_mapper, f)
    print(f"Saved organism mapper to {mapper_path}")

    # === Save JSON versions for HuggingFace compatibility ===

    # Save GO tokenizer as JSON
    go_tokenizer_json = {
        "token_to_id": go_tokenizer.token_to_id,
        "id_to_token": {str(k): v for k, v in go_tokenizer.id_to_token.items()},
    }
    go_tokenizer_json_path = artifacts_dir / "go_tokenizer.json"
    with open(go_tokenizer_json_path, 'w') as f:
        json.dump(go_tokenizer_json, f, indent=2)
    print(f"Saved GO tokenizer (JSON) to {go_tokenizer_json_path}")

    # Save organism mapper as JSON
    organism_mapper_json = {
        "organism_to_idx": organism_mapper.organism_to_idx,
        "idx_to_organism": {str(k): v for k, v in organism_mapper.idx_to_organism.items()},
        "vocab_size": organism_mapper.vocab_size,
    }
    organism_mapper_json_path = artifacts_dir / "organism_mapper.json"
    with open(organism_mapper_json_path, 'w') as f:
        json.dump(organism_mapper_json, f, indent=2)
    print(f"Saved organism mapper (JSON) to {organism_mapper_json_path}")

    # Save organism list (human-readable)
    organism_list_path = artifacts_dir / "organism_list.txt"
    with open(organism_list_path, 'w') as f:
        for org in sorted(organism_mapper.organism_to_idx.keys()):
            f.write(f"{org}\n")
    print(f"Saved organism list to {organism_list_path}")

    # Save tokenizer info
    tokenizer_info = {
        'vocab_size': len(go_tokenizer.token_to_id),
        'pad_token_id': go_tokenizer.pad_token_id,
        'mf_start_token_id': go_tokenizer.mf_start_token_id,
        'mf_end_token_id': go_tokenizer.mf_end_token_id,
        'bp_start_token_id': go_tokenizer.bp_start_token_id,
        'bp_end_token_id': go_tokenizer.bp_end_token_id,
        'cc_start_token_id': go_tokenizer.cc_start_token_id,
        'cc_end_token_id': go_tokenizer.cc_end_token_id,
        'organism_vocab_size': organism_mapper.get_vocab_size(),
        'embed_model_type': embed_model_type,
        'max_protein_length': max_length,
        'vocab_config': {k: OmegaConf.to_container(v) for k, v in vocab_config.items()},  # Convert DictConfigs to dicts
        'vocab_stats': vocab_stats,    # Save detailed statistics
    }

    info_path = artifacts_dir / "tokenizer_info.json"
    with open(info_path, 'w') as f:
        json.dump(tokenizer_info, f, indent=2)
    print(f"Saved tokenizer info to {info_path}")

    # Save vocabulary per aspect for analysis
    vocab_analysis = {
        aspect: {
            'selected_terms': sorted(list(terms)),
            'count': len(terms)
        }
        for aspect, terms in limited_vocab_per_aspect.items()
    }

    vocab_path = artifacts_dir / "limited_vocabulary.json"
    with open(vocab_path, 'w') as f:
        json.dump(vocab_analysis, f, indent=2)
    print(f"Saved vocabulary analysis to {vocab_path}")

    # Save preprocessing config
    config_path = output_dir / "preprocess_config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)
    print(f"Saved preprocessing config to {config_path}")

    # Print final statistics
    print("\n" + "="*60)
    print("GO-GPT Data Preprocessing Complete!")
    print("="*60)
    print(f"Dataset saved to: {output_dir}")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    # Detailed vocabulary statistics with cross-reporting
    print(f"\nDetailed Vocabulary Statistics:")
    print(f"  Total GO terms in pruned vocab: {len(all_pruned_go_terms)}")
    print()

    for aspect, stats in vocab_stats.items():
        print(f"  {aspect} Aspect:")
        if stats["method"] == "top_k":
            print(f"    Method: Top-K selection")
            print(f"    Requested: {stats['requested']} terms")
            print(f"    Selected: {stats['selected']} terms")
            print(f"    Min frequency (cross-stat): {stats['min_freq']} proteins")
            print(f"    Available terms: {stats['available_terms']}")
        elif stats["method"] == "min_freq":
            print(f"    Method: Minimum frequency selection")
            print(f"    Threshold: ≥{stats['threshold']} proteins")
            print(f"    Selected (cross-stat): {stats['selected']} terms")
            if stats['selected'] > 0:
                print(f"    Frequency range: {stats['freq_range'][0]}-{stats['freq_range'][1]} proteins")
            print(f"    Total available terms: {stats['total_terms']}")
        print()

    print("\nComputing final statistics...")

    # Aspect distribution (using column access for efficiency)
    train_aspects = Counter(train_dataset['aspect'])
    val_aspects = Counter(val_dataset['aspect'])
    print(f"\nTraining aspect distribution: {dict(train_aspects)}")
    print(f"Validation aspect distribution: {dict(val_aspects)}")

    # Protein length statistics (using column access for efficiency)
    train_lengths = train_dataset['protein_length']
    val_lengths = val_dataset['protein_length']
    print(f"\nProtein length stats (train): min={min(train_lengths)}, max={max(train_lengths)}, avg={sum(train_lengths)/len(train_lengths):.1f}")
    print(f"Protein length stats (val): min={min(val_lengths)}, max={max(val_lengths)}, avg={sum(val_lengths)/len(val_lengths):.1f}")

    # GO terms statistics (using column access for efficiency)
    train_go_counts = train_dataset['num_go_terms']
    val_go_counts = val_dataset['num_go_terms']
    print(f"\nGO terms per protein (train): min={min(train_go_counts)}, max={max(train_go_counts)}, avg={sum(train_go_counts)/len(train_go_counts):.1f}")
    print(f"GO terms per protein (val): min={min(val_go_counts)}, max={max(val_go_counts)}, avg={sum(val_go_counts)/len(val_go_counts):.1f}")

if __name__ == "__main__":
    prepare_data()
