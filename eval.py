#!/usr/bin/env python3
"""
CAFA-5 Evaluation Script

Features:
- Modular function design for maintainability
- Individual JSON file output per protein_id + go_aspect combination
- Robust error handling and logging
- Progress tracking and resumable execution
- Multi-GPU safe concurrent execution
- Professional argument parsing with grouped options

Usage:
    python eval.py --ckpt_dir /path/to/checkpoint --evals_path /path/to/results [options]
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List
import torch
from tqdm import tqdm
import traceback

from bioreason2.models.protein_vllm import ProteinLLMModel
from bioreason2.dataset.cafa5.load import load_cafa5_dataset
from bioreason2.utils import str2bool

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STOP_TOKENS = ["<|im_end|>"]
ERROR_LOG_FILE = "evaluation_errors.json"

# GO Aspect mapping for cleaner filenames
GO_ASPECT_CODES = {"molecular_function": "MF", "cellular_component": "CC", "biological_process": "BP"}

def get_go_aspect_code(go_aspect: str) -> str:
    """Convert GO aspect to short code for cleaner filenames."""
    return GO_ASPECT_CODES.get(go_aspect, go_aspect)


def _get_ground_truth(sample: Dict[str, Any]) -> str:
    """Extracts the ground truth assistant reasoning and answer from the sample."""
    prompt_data = sample.get("prompt")
    if isinstance(prompt_data, list):
        for message in prompt_data:
            if message.get("role") == "assistant":
                reasoning = message.get("reasoning_content", "")
                answer = ""
                content = message.get("content", [])
                if isinstance(content, list) and content:
                    answer = content[0].get("text", "")
                return f"{reasoning}\n\n{answer}" if reasoning and answer else reasoning or answer
    return sample.get("answer", "")


def initialize_model(args) -> ProteinLLMModel:
    """Initialize and return the ProteinLLMModel."""
    print(f"📥 Loading ProteinLLMModel from checkpoint: {args.ckpt_dir}...")
    model = ProteinLLMModel(
        ckpt_dir=args.ckpt_dir,
        protein_model_name=args.protein_model_name,
        protein_embedding_layer=args.protein_embedding_layer,
        go_obo_path=args.go_obo_path,
        precomputed_embeddings_path=args.precomputed_embeddings_path,
        max_length_protein=args.max_length_protein,
        max_length_text=args.max_model_len,
        max_model_len=args.max_model_len,
        unified_go_encoder=args.unified_go_encoder,
        go_hidden_dim=args.go_hidden_dim,
        go_num_gat_layers=args.go_num_gat_layers,
        go_num_heads=args.go_num_heads,
        go_num_reduced_embeddings=args.go_num_reduced_embeddings,
        go_embedding_dim=args.go_embedding_dim,
        text_model_finetune=False,
        protein_model_finetune=False,
        go_model_finetune=False,
    )
    print("Model initialized successfully.")
    return model


def load_dataset(args):
    """Load and prepare the CAFA-5 validation dataset."""
    print("\n📥 Loading and preparing CAFA-5 validation dataset...")
    _, val_ds, _ = load_cafa5_dataset(
        dataset=args.cafa5_dataset,
        dataset_name=args.cafa5_dataset_name,
        cache_dir=args.dataset_cache_dir,
        dataset_subset=args.cafa5_dataset_subset,
        max_length=args.max_length_protein,
        seed=args.seed,
        val_split_ratio=args.val_split_ratio,
        return_as_chat_template=True,
        split_go_aspects=args.split_go_aspects,
        structure_dir=args.structure_dir,
        include_go_defs=args.include_go_defs,
        interpro_dataset_name=args.interpro_dataset_name,
        include_protein_function_summary=args.include_protein_function_summary,
        interpro_in_prompt=args.interpro_in_prompt,
        predict_interpro=args.predict_interpro,
        ppi_in_prompt=args.ppi_in_prompt,
        reasoning_dataset_name=args.reasoning_dataset_name,
        go_gpt_predictions_column=args.go_gpt_predictions_column,
        min_go_mf_freq=args.min_go_mf_freq,
        min_go_bp_freq=args.min_go_bp_freq,
        min_go_cc_freq=args.min_go_cc_freq,
        apply_go_filtering_to_val_test=args.apply_go_filtering_to_val_test,
        add_uniprot_summary=args.add_uniprot_summary,
        debug=args.debug,
    )
    val_ds = val_ds.shuffle(seed=args.seed)

    if not val_ds or len(val_ds) == 0:
        raise ValueError("Validation dataset is empty or failed to load.")

    n_samples = len(val_ds) if args.max_samples <= 0 else min(args.max_samples, len(val_ds))

    # Handle chunking for multi-GPU processing
    if args.num_chunks > 1:
        chunk_size = n_samples // args.num_chunks
        start_idx = args.chunk_id * chunk_size

        if args.chunk_id == args.num_chunks - 1:
            # Last chunk gets any remaining samples
            end_idx = n_samples
        else:
            end_idx = start_idx + chunk_size

        print(f"Processing chunk {args.chunk_id + 1}/{args.num_chunks}: samples {start_idx} to {end_idx-1}")
        samples = val_ds.select(range(start_idx, end_idx))
    else:
        print("📊 Processing full dataset (no chunking)")
        samples = val_ds.select(range(n_samples))

    print(f"Loaded {len(samples)} samples for evaluation.")
    return samples


def filter_unprocessed_samples(samples, evals_path: str) -> List[Dict[str, Any]]:
    """Filter out already processed samples and return only unprocessed ones.
    
    Simplified logic: If ANY file exists for a (protein_id, go_aspect) combination,
    skip it entirely. Don't worry about whether all k iterations are complete.
    """
    os.makedirs(evals_path, exist_ok=True)
    processed_ids = set()

    if os.path.exists(evals_path):
        existing_files = os.listdir(evals_path)
        for filename in existing_files:
            if filename.endswith(".json"):
                # Parse filename: {protein_id}_{go_aspect_code}_k{i:02d}.json
                parts = filename.split("_")
                if len(parts) >= 2:
                    processed_unique_id = f"{parts[0]}_{parts[1]}"
                    processed_ids.add(processed_unique_id)
        
        print(f"🔄 Found {len(processed_ids)} samples with at least one result file.")

    # Filter out already processed samples
    print("🔍 Filtering out already processed samples...")
    unprocessed_samples = []
    for sample in samples:
        protein_id = sample.get("protein_id")
        go_aspect = sample.get("go_aspect")
        go_aspect_code = get_go_aspect_code(go_aspect)
        sample_unique_id = f"{protein_id}_{go_aspect_code}"
        
        if sample_unique_id not in processed_ids:
            unprocessed_samples.append(sample)
    
    print(f"📊 Total samples: {len(samples)}")
    print(f"Already processed: {len(samples) - len(unprocessed_samples)}")
    print(f"Remaining to process: {len(unprocessed_samples)}")

    return unprocessed_samples


def save_result(result_record: Dict[str, Any], protein_id: str, go_aspect: str, evals_path: str, k_idx: int = 0) -> None:
    """Save individual result to its own JSON file using short GO aspect codes."""
    go_aspect_code = get_go_aspect_code(go_aspect)
    result_filename = f"{protein_id}_{go_aspect_code}_k{k_idx:02d}.json"
    result_filepath = os.path.join(evals_path, result_filename)

    with open(result_filepath, "w") as f:
        json.dump(result_record, f, indent=4)


def log_error(error_type: str, protein_id: str, go_aspect: str, go_bp: str, go_mf: str, go_cc: str, go_bp_leaf: str, go_mf_leaf: str, go_cc_leaf: str, error_msg: str = "") -> None:
    """Log errors to a centralized JSON file."""
    error_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "error_type": error_type,
        "protein_id": protein_id,
        "go_aspect": go_aspect,
        "go_bp": go_bp,
        "go_mf": go_mf,
        "go_cc": go_cc,
        "go_bp_leaf": go_bp_leaf,
        "go_mf_leaf": go_mf_leaf,
        "go_cc_leaf": go_cc_leaf,
        "error_message": error_msg if error_msg else ("Out of Memory" if error_type == "oom" else "Unknown error"),
    }

    # Load existing errors or create new list
    errors = []
    if os.path.exists(ERROR_LOG_FILE):
        try:
            with open(ERROR_LOG_FILE, "r") as f:
                errors = json.load(f)
        except (json.JSONDecodeError, Exception):
            errors = []

    # Append new error
    errors.append(error_record)

    # Save back to file
    with open(ERROR_LOG_FILE, "w") as f:
        json.dump(errors, f, indent=4)


def process_single_sample(
    model: ProteinLLMModel, sample: Dict[str, Any], protein_id: str, go_aspect: str, go_bp: str, go_mf: str, go_cc: str, go_bp_leaf: str, go_mf_leaf: str, go_cc_leaf: str, args
) -> Dict[str, Any]:
    """Process a single sample and return the result."""
    conversation_data = sample.get("prompt")
    if conversation_data is None:
        print(f"No prompt data for protein {protein_id}, skipping...")
        return None

    # Extract only system and user messages for generation
    # Filter out assistant messages to create proper generation prompt
    user_conversation = []
    for message in conversation_data:
        if message.get("role") in ["system", "user"]:
            user_conversation.append(message)
        elif message.get("role") == "assistant":
            # Stop at first assistant message - we only want the input
            break

    final_prompt_string = model.text_tokenizer.apply_chat_template(
        user_conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,  # Avoid empty thinking injection
    )

    sequence = sample.get("sequence")
    if sequence is None:
        print(f"No sequence data for protein {protein_id}, skipping...")
        return None

    processed_inputs = model.processor(
        text=[final_prompt_string],
        batch_protein_sequences=[[sequence]],
        batch_go_aspects=[go_aspect],
        max_length_text=model.max_length_text,
        max_length_protein=model.max_length_protein,
        return_tensors="pt",
    )

    input_ids = processed_inputs.get("input_ids").to(DEVICE)
    attention_mask = processed_inputs.get("attention_mask").to(DEVICE)
    structure_coords = processed_inputs.get("structure_coords")

    # Run Inference
    with torch.inference_mode():
        generated_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            protein_sequences=[sequence],
            batch_idx_map=[0],
            go_aspects=[go_aspect],
            structure_coords=structure_coords,
            # Pass generation parameters from args
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            stop=STOP_TOKENS,
        )

    response_text = generated_outputs[0] if generated_outputs else "Error: Empty response"

    result_record = {
        "protein_id": protein_id,
        "go_aspect": go_aspect,
        "ground_truth": _get_ground_truth(sample),
        "generated_response": response_text,
        "success": True,
        "protein_sequence": sequence,
        "input_prompt": final_prompt_string,
        "sequence_length": len(sequence) if sequence else 0,
        "go_bp": go_bp,
        "go_mf": go_mf,
        "go_cc": go_cc,
        "go_bp_leaf": go_bp_leaf,
        "go_mf_leaf": go_mf_leaf,
        "go_cc_leaf": go_cc_leaf,
    }

    return result_record


def print_final_statistics(newly_processed: int, total_time: float, evals_path: str) -> None:
    """Print final evaluation statistics."""
    total_files = len([f for f in os.listdir(evals_path) if f.endswith(".json")])

    print("\nEvaluation complete.")
    print(f"⏱️  Processed {newly_processed} new samples in {total_time:.2f}s")
    if newly_processed > 0:
        print(f"📈 Processing rate: {newly_processed/total_time:.2f} samples/s")
    print(f"💾 Total result files: {total_files} in directory: {evals_path}")
    print("Individual JSON files saved for each protein_id + aspect combination")


def run_local_inference(args):
    """
    Main function to orchestrate data loading, model inference, and result saving.
    """
    print("--- Starting Local CAFA-5 Inference ---")

    try:
        # Initialize model
        model = initialize_model(args)

        # Load dataset
        samples = load_dataset(args)

        # Filter out already processed samples
        unprocessed_samples = filter_unprocessed_samples(samples, args.evals_path)

        # Main inference loop - only process unprocessed samples
        print(f"\nStarting inference loop with pass@{args.pass_at_k}...")
        t_start = time.time()
        successfully_processed = 0

        for sample in tqdm(unprocessed_samples, desc="Processing Samples", total=len(unprocessed_samples), unit="sample"):
            protein_id = sample.get("protein_id")
            go_aspect = sample.get("go_aspect", "all")
            go_bp = sample.get("go_bp", "")
            go_mf = sample.get("go_mf", "")
            go_cc = sample.get("go_cc", "")
            go_bp_leaf = sample.get("go_bp_leaf", "")
            go_mf_leaf = sample.get("go_mf_leaf", "")
            go_cc_leaf = sample.get("go_cc_leaf", "")

            # Generate k samples for pass@k
            sample_has_success = False
            for k_idx in range(args.pass_at_k):
                try:
                    result_record = process_single_sample(model, sample, protein_id, go_aspect, go_bp, go_mf, go_cc, go_bp_leaf, go_mf_leaf, go_cc_leaf, args)
                    if result_record is not None:
                        save_result(result_record, protein_id, go_aspect, args.evals_path, k_idx=k_idx)
                        if not sample_has_success:
                            successfully_processed += 1
                            sample_has_success = True

                except torch.cuda.OutOfMemoryError:
                    print(f"CUDA Out of Memory on sample ID: {protein_id}, k={k_idx}. Skipping this k iteration.")
                    log_error("oom", protein_id, go_aspect, go_bp, go_mf, go_cc, go_bp_leaf, go_mf_leaf, go_cc_leaf)
                    torch.cuda.empty_cache()
                    continue

                except Exception as e:
                    print(f"Unexpected error on sample ID {protein_id}, k={k_idx}: {e}")
                    log_error("other", protein_id, go_aspect, go_bp, go_mf, go_cc, go_bp_leaf, go_mf_leaf, go_cc_leaf, str(e))
                    traceback.print_exc()
                    continue

        # Print final statistics
        t_end = time.time()
        dt = t_end - t_start
        print_final_statistics(successfully_processed, dt, args.evals_path)

    except Exception as e:
        print(f"Critical Error: {e}")
        traceback.print_exc()
        return


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup and return the argument parser."""
    parser = argparse.ArgumentParser(description="Local CAFA inference with ProteinLLMModel")

    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--ckpt_dir", type=str, required=True, help="Path to the ProteinLLMModel checkpoint directory."
    )
    model_group.add_argument(
        "--protein_model_name", type=str, default="esm3_sm_open_v1", help="Name of the protein encoder model."
    )
    model_group.add_argument(
        "--protein_embedding_layer",
        type=int,
        default=-1,
        help="ESM3 layer to extract embeddings from. Use -1 for final output (default), 0-N for specific transformer layers. Only works with ESM3 models."
    )
    model_group.add_argument("--go_obo_path", type=str, required=True, help="Path to GO ontology .obo file.")
    model_group.add_argument(
        "--precomputed_embeddings_path",
        type=str,
        required=True,
        help="Path to directory with precomputed GO embeddings.",
    )
    model_group.add_argument(
        "--unified_go_encoder",
        type=str2bool,
        default=False,
        help="If True, use unified GOGraphEncoderUnified; if False, use original GOGraphEncoder.",
    )
    model_group.add_argument("--max_model_len", type=int, default=32768, help="Maximum length of the model.")
    model_group.add_argument(
        "--go_hidden_dim", type=int, default=512, help="Hidden dimension for GO GAT layers (must match training)."
    )
    model_group.add_argument(
        "--go_num_gat_layers", type=int, default=3, help="Number of GAT layers in GO encoder (must match training)."
    )
    model_group.add_argument(
        "--go_num_heads", type=int, default=8, help="Number of attention heads in GO GAT (must match training)."
    )
    model_group.add_argument(
        "--go_num_reduced_embeddings",
        type=int,
        default=200,
        help="Number of reduced embeddings per GO namespace (must match training).",
    )
    model_group.add_argument(
        "--go_embedding_dim", type=int, default=2560, help="GO embedding dimension (must match training)."
    )

    # Dataset options
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument("--cafa5_dataset", type=str, default="wanglab/cafa5")
    dataset_group.add_argument("--cafa5_dataset_name", type=str, default="cafa5_reasoning")
    dataset_group.add_argument("--cafa5_dataset_subset", type=str, default=None)
    dataset_group.add_argument("--dataset_cache_dir", type=str, default=None)
    dataset_group.add_argument(
        "--structure_dir", type=str, default=None
    )
    dataset_group.add_argument("--include_go_defs", type=str2bool, default=False)
    dataset_group.add_argument("--interpro_dataset_name", type=str, default="interpro_metadata")
    dataset_group.add_argument("--split_go_aspects", type=str2bool, default=True)
    dataset_group.add_argument("--interpro_in_prompt", type=str2bool, default=True)
    dataset_group.add_argument("--predict_interpro", type=str2bool, default=False)
    dataset_group.add_argument("--ppi_in_prompt", type=str2bool, default=True)
    dataset_group.add_argument("--include_protein_function_summary", type=str2bool, default=True)
    dataset_group.add_argument("--val_split_ratio", type=float, default=0.1)
    dataset_group.add_argument("--seed", type=int, default=23)
    dataset_group.add_argument("--debug", type=str2bool, default=False)
    dataset_group.add_argument(
        "--max_length_protein", type=int, default=2048, help="Maximum length of protein sequences."
    )
    dataset_group.add_argument("--enable_thinking", type=str2bool, default=True)
    dataset_group.add_argument(
        "--reasoning_dataset_name",
        type=str,
        default=None,
        help="Config name for reasoning traces dataset (e.g., 'experiment_data_reasoning'). If provided, uses reasoning data instead of generating assistant reasoning. Requires split_go_aspects=False since reasoning contains comprehensive analysis for all GO aspects together.",
    )
    dataset_group.add_argument(
        "--go_gpt_predictions_column",
        type=str,
        default="go_pred",
        help="Column name for GO-GPT predictions (must match training).",
    )
    dataset_group.add_argument(
        "--min_go_mf_freq",
        type=int,
        default=50,
        help="Minimum frequency for molecular function GO terms to include in dataset (must match training).",
    )
    dataset_group.add_argument(
        "--min_go_bp_freq",
        type=int,
        default=100,
        help="Minimum frequency for biological process GO terms to include in dataset (must match training).",
    )
    dataset_group.add_argument(
        "--min_go_cc_freq",
        type=int,
        default=50,
        help="Minimum frequency for cellular component GO terms to include in dataset (must match training).",
    )
    dataset_group.add_argument(
        "--apply_go_filtering_to_val_test",
        type=str2bool,
        default=False,
        help="Whether to apply GO frequency filtering to validation/test sets (must match training).",
    )
    dataset_group.add_argument("--add_uniprot_summary", type=str2bool, default=False)

    # Evaluation controls
    eval_group = parser.add_argument_group("Evaluation Configuration")
    eval_group.add_argument("--max_samples", type=int, default=-1, help="Max samples to process (-1 for all).")
    eval_group.add_argument("--max_new_tokens", type=int, default=1024)
    eval_group.add_argument("--temperature", type=float, default=0.1)
    eval_group.add_argument("--top_p", type=float, default=0.9)
    eval_group.add_argument("--repetition_penalty", type=float, default=1.0)
    eval_group.add_argument(
        "--pass_at_k", 
        type=int, 
        default=1, 
        help="Number of inference attempts per sample for pass@k evaluation (default: 1). Use temperature > 0 for diversity."
    )

    # Data chunking (optional)
    chunk_group = parser.add_argument_group("Data Chunking (Optional)")
    chunk_group.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help="Total number of chunks for distributed processing. Default: 1 (no chunking).",
    )
    chunk_group.add_argument(
        "--chunk_id", type=int, default=0, help="ID of this chunk (0-indexed). Only used when num_chunks > 1."
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--evals_path", type=str, required=True, help="Directory path to save individual evaluation results."
    )

    return parser


if __name__ == "__main__":
    parser = setup_argument_parser()
    args = parser.parse_args()
    run_local_inference(args)
