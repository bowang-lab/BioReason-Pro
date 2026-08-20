#!/usr/bin/env python3
"""
CAFA Evaluation Script (batched inference)

Features:
- Batched inference: one vLLM call per batch
- Individual JSON file output per protein_id + go_aspect combination
- Resumable: a protein that already has a result file is skipped
- Multi-GPU safe concurrent execution via --num_chunks / --chunk_id
- Per-sample retry on OOM

Usage:
    python eval.py --ckpt_dir /path/to/checkpoint --evals_path /path/to/results [options]
"""

import argparse
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from bioreason2.models.protein_vllm import ProteinLLMModel
from bioreason2.dataset.cafa5.load import load_cafa5_dataset
from bioreason2.utils import str2bool

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STOP_TOKENS = ["<|im_end|>"]

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


def text_token_budget(args) -> int:
    """Tokenizer cap for the text part, so prompt + protein + GO fits max_model_len.

    The processor truncates at max_length_text + 200 GO tokens + max_length_protein + 2;
    letting that exceed the context window makes vLLM reject the request outright.
    """
    return max(1, args.max_model_len - 200 - args.max_length_protein - 2)


def initialize_model(args) -> ProteinLLMModel:
    """Initialize and return the ProteinLLMModel."""
    print(f"Loading ProteinLLMModel from checkpoint: {args.ckpt_dir}...")
    model = ProteinLLMModel(
        ckpt_dir=args.ckpt_dir,
        protein_model_name=args.protein_model_name,
        protein_embedding_layer=args.protein_embedding_layer,
        go_obo_path=args.go_obo_path,
        precomputed_embeddings_path=args.precomputed_embeddings_path,
        max_length_protein=args.max_length_protein,
        max_length_text=text_token_budget(args),
        max_model_len=args.max_model_len,
        unified_go_encoder=args.unified_go_encoder,
        go_hidden_dim=args.go_hidden_dim,
        go_num_gat_layers=args.go_num_gat_layers,
        go_num_heads=args.go_num_heads,
        go_num_reduced_embeddings=args.go_num_reduced_embeddings,
        go_embedding_dim=args.go_embedding_dim,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        text_model_finetune=False,
        protein_model_finetune=False,
        go_model_finetune=False,
    )
    print("Model initialized successfully.")
    return model


def load_dataset(args):
    """Load and prepare the evaluation dataset."""
    print(f"\nLoading dataset (split: {args.cafa5_dataset_split})...")
    train_ds, val_ds, test_ds = load_cafa5_dataset(
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
        # Default True: the prompt never depends on the example's own labels, and
        # matches predict.py. Set both False to reproduce the published benchmark,
        # whose prompts did derive these from the labels.
        force_uniprot_summary=args.force_uniprot_summary,
        ask_all_go_aspects=args.ask_all_go_aspects,
        debug=args.debug,
    )

    dataset = {"train": train_ds, "val": val_ds, "test": test_ds}[args.cafa5_dataset_split]
    if not dataset or len(dataset) == 0:
        raise ValueError(f"Dataset split '{args.cafa5_dataset_split}' is empty or failed to load.")

    dataset = dataset.shuffle(seed=args.seed)
    n_samples = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))

    # Handle chunking for multi-GPU processing
    if args.num_chunks > 1:
        chunk_size = n_samples // args.num_chunks
        start_idx = args.chunk_id * chunk_size
        # Last chunk gets any remaining samples
        end_idx = n_samples if args.chunk_id == args.num_chunks - 1 else start_idx + chunk_size
        print(f"Processing chunk {args.chunk_id + 1}/{args.num_chunks}: samples {start_idx} to {end_idx - 1}")
        samples = dataset.select(range(start_idx, end_idx))
    else:
        print("Processing full dataset (no chunking)")
        samples = dataset.select(range(n_samples))

    print(f"Loaded {len(samples)} samples for evaluation.")
    return samples


def filter_unprocessed_samples(samples, evals_path: str):
    """Drop samples that already have a result file, returning a Dataset.

    Reads protein_id/go_aspect columnwise; row-by-row iteration would decode the
    full sequence and prompt just to read two strings.
    """
    os.makedirs(evals_path, exist_ok=True)
    processed_ids = set()

    for filename in os.listdir(evals_path):
        if filename.endswith(".json"):
            # Filename is {protein_id}_{go_aspect_code}_k{i:02d}.json
            parts = filename.split("_")
            if len(parts) >= 2:
                processed_ids.add(f"{parts[0]}_{parts[1]}")
    print(f"Found {len(processed_ids)} samples with at least one result file.")

    protein_ids = samples["protein_id"]
    go_aspects = samples["go_aspect"]
    keep = [
        i
        for i, (pid, aspect) in enumerate(zip(protein_ids, go_aspects))
        if f"{pid}_{get_go_aspect_code(aspect)}" not in processed_ids
    ]

    print(f"Total samples: {len(samples)}")
    print(f"Already processed: {len(samples) - len(keep)}")
    print(f"Remaining to process: {len(keep)}")
    return samples.select(keep)


def batch_slice(dataset, start: int, end: int) -> List[Dict[str, Any]]:
    """Rows [start, end) as a list of dicts, via a single Arrow slice."""
    columns = dataset[start:end]
    n = end - start
    return [{key: values[i] for key, values in columns.items()} for i in range(n)]


def save_result(result_record: Dict[str, Any], protein_id: str, go_aspect: str, evals_path: str, k_idx: int = 0) -> None:
    """Save individual result to its own JSON file using short GO aspect codes."""
    go_aspect_code = get_go_aspect_code(go_aspect)
    result_filename = f"{protein_id}_{go_aspect_code}_k{k_idx:02d}.json"
    with open(os.path.join(evals_path, result_filename), "w") as f:
        json.dump(result_record, f, indent=4)


def log_error(error_log_path: str, error_type: str, protein_id: str, go_aspect: str, error_msg: str = "") -> None:
    """Append an error record to the run's error log."""
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "error_type": error_type,
        "protein_id": protein_id,
        "go_aspect": go_aspect,
        "error_message": error_msg or ("Out of Memory" if error_type == "oom" else "Unknown error"),
    }

    errors = []
    if os.path.exists(error_log_path):
        try:
            with open(error_log_path) as f:
                errors = json.load(f)
        except Exception:
            errors = []
    errors.append(record)

    with open(error_log_path, "w") as f:
        json.dump(errors, f, indent=4)


def _build_prompt_string(model, sample: Dict[str, Any], args) -> Optional[str]:
    """Chat-template string for one sample, or None if it cannot be built."""
    conversation_data = sample.get("prompt")
    if conversation_data is None:
        return None

    # Keep system and user messages; stop at the first assistant turn
    user_conversation = []
    for message in conversation_data:
        if message.get("role") in ["system", "user"]:
            user_conversation.append(message)
        elif message.get("role") == "assistant":
            break

    return model.text_tokenizer.apply_chat_template(
        user_conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )


def prepare_batch_inputs(model: ProteinLLMModel, batch_samples: List[Dict[str, Any]], args) -> Optional[Dict[str, Any]]:
    """Tokenize a batch. Returns None when no sample in the batch is usable."""
    prompts, sequences, go_aspects, keep = [], [], [], []
    for sample in batch_samples:
        prompt = _build_prompt_string(model, sample, args)
        sequence = sample.get("sequence")
        if prompt is None or sequence is None:
            continue
        prompts.append(prompt)
        sequences.append(sequence)
        go_aspects.append(sample.get("go_aspect", "all"))
        keep.append(sample)

    if not prompts:
        return None

    # Left-pad so content sits at the end of every row. generate() strips padding
    # before handing embeddings to vLLM either way; this matches predict.py.
    original_padding_side = model.text_tokenizer.padding_side
    model.text_tokenizer.padding_side = "left"
    try:
        processed_inputs = model.processor(
            # Copy: the processor expands <|protein_pad|> in place.
            text=list(prompts),
            batch_protein_sequences=[[s] for s in sequences],
            batch_go_aspects=go_aspects,
            max_length_text=model.max_length_text,
            max_length_protein=model.max_length_protein,
            return_tensors="pt",
        )
    finally:
        model.text_tokenizer.padding_side = original_padding_side

    return {
        "input_ids": processed_inputs.get("input_ids").to(DEVICE),
        "attention_mask": processed_inputs.get("attention_mask").to(DEVICE),
        "structure_coords": processed_inputs.get("structure_coords"),
        "sequences": sequences,
        "go_aspects": go_aspects,
        "prompts": prompts,
        "samples": keep,
    }


def process_batch(model: ProteinLLMModel, batch_samples: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """Run one batched generation and build a result record per sample."""
    batch_inputs = prepare_batch_inputs(model, batch_samples, args)
    if batch_inputs is None:
        return []

    with torch.inference_mode():
        generated_outputs = model.generate(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            protein_sequences=batch_inputs["sequences"],
            batch_idx_map=list(range(len(batch_inputs["sequences"]))),
            go_aspects=batch_inputs["go_aspects"],
            structure_coords=batch_inputs["structure_coords"],
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            stop=STOP_TOKENS,
        )

    results = []
    for i, sample in enumerate(batch_inputs["samples"]):
        text = generated_outputs[i] if i < len(generated_outputs) else ""
        sequence = batch_inputs["sequences"][i]
        results.append({
            "protein_id": sample.get("protein_id"),
            "go_aspect": batch_inputs["go_aspects"][i],
            "ground_truth": _get_ground_truth(sample),
            "generated_response": text,
            # Always True: marking an empty generation as failed makes
            # cafa_evals.py drop the protein from ground truth too, inflating Fmax.
            "success": True,
            "protein_sequence": sequence,
            "input_prompt": batch_inputs["prompts"][i],
            "sequence_length": len(sequence) if sequence else 0,
            "go_bp": sample.get("go_bp", ""),
            "go_mf": sample.get("go_mf", ""),
            "go_cc": sample.get("go_cc", ""),
            "go_bp_leaf": sample.get("go_bp_leaf", ""),
            "go_mf_leaf": sample.get("go_mf_leaf", ""),
            "go_cc_leaf": sample.get("go_cc_leaf", ""),
        })
    return results


def print_final_statistics(newly_processed: int, empty_responses: int, total_time: float, evals_path: str) -> None:
    """Print final evaluation statistics."""
    total_files = len([f for f in os.listdir(evals_path) if f.endswith(".json")])

    print("\nEvaluation complete.")
    print(f"Processed {newly_processed} new samples in {total_time:.2f}s")
    if newly_processed > 0:
        print(f"Processing rate: {newly_processed / total_time:.2f} samples/s")
    if empty_responses:
        print(f"WARNING: {empty_responses} samples generated an empty response; "
              f"they are scored as a miss, not dropped.")
    print(f"Total result files: {total_files} in directory: {evals_path}")


def run_inference(args):
    """Orchestrate data loading, batched model inference, and result saving."""
    print("--- Starting batched CAFA inference ---")
    print(f"Batch size: {args.batch_size}")

    # Not .json: everything scanning evals_path for *.json treats those as results.
    error_log_path = os.path.join(args.evals_path, f"evaluation_errors_chunk{args.chunk_id:03d}.log")

    try:
        model = initialize_model(args)
        samples = load_dataset(args)
        unprocessed = filter_unprocessed_samples(samples, args.evals_path)

        n = len(unprocessed)
        if n == 0:
            print("All samples already processed. Nothing to do.")
            return

        bs = max(1, args.batch_size)
        num_batches = (n + bs - 1) // bs
        print(f"\nStarting inference: {n} samples, batch_size={bs}, {num_batches} batches, pass@{args.pass_at_k}")

        t_start = time.time()
        successfully_processed = 0
        empty_responses = 0

        for batch_idx in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
            batch_samples = batch_slice(unprocessed, batch_idx * bs, min((batch_idx + 1) * bs, n))

            for k_idx in range(args.pass_at_k):
                try:
                    records = process_batch(model, batch_samples, args)

                except torch.cuda.OutOfMemoryError:
                    print(f"\nCUDA OOM on batch {batch_idx} (k={k_idx}); retrying its samples individually.")
                    torch.cuda.empty_cache()
                    records = []
                    for sample in batch_samples:
                        try:
                            records.extend(process_batch(model, [sample], args))
                        except Exception as exc:
                            log_error(error_log_path, "oom", sample.get("protein_id", "unknown"),
                                      sample.get("go_aspect", "all"), str(exc))
                            torch.cuda.empty_cache()

                except Exception as exc:
                    ids = ",".join(str(s.get("protein_id")) for s in batch_samples)
                    print(f"\nError on batch {batch_idx} (k={k_idx}) [{ids}]: {exc}")
                    traceback.print_exc()
                    for sample in batch_samples:
                        log_error(error_log_path, "batch_error", sample.get("protein_id", "unknown"),
                                  sample.get("go_aspect", "all"), str(exc))
                    continue

                for record in records:
                    save_result(record, record["protein_id"], record["go_aspect"], args.evals_path, k_idx=k_idx)
                    if not record["generated_response"].strip():
                        empty_responses += 1
                    if k_idx == 0:
                        successfully_processed += 1

        print_final_statistics(successfully_processed, empty_responses, time.time() - t_start, args.evals_path)

    except Exception as exc:
        # Must not swallow this: exiting 0 makes a dead shard look COMPLETED.
        print(f"Critical Error: {exc}")
        traceback.print_exc()
        raise SystemExit(1)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup and return the argument parser."""
    parser = argparse.ArgumentParser(description="Batched CAFA inference with ProteinLLMModel")

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
    model_group.add_argument("--max_model_len", type=int, default=8192,
                             help="Maximum context length for vLLM (prompt + generation).")
    model_group.add_argument(
        "--gpu_memory_utilization", type=float, default=0.5,
        help="Fraction of GPU memory vLLM may use. ESM3, the GO encoder and the "
             "batch's prompt embeddings are allocated outside it, so leave headroom.",
    )
    model_group.add_argument(
        "--max_num_seqs", type=int, default=256,
        help="Upper bound on sequences vLLM runs concurrently.",
    )
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
    dataset_group.add_argument(
        "--cafa5_dataset_split", type=str, default="val", choices=["train", "val", "test"],
        help="Which split to evaluate. A dataset with only a `test` split is returned "
             "as the validation split too, so the default works for it.",
    )
    dataset_group.add_argument("--dataset_cache_dir", type=str, default=None)
    dataset_group.add_argument(
        "--structure_dir", type=str, default=None
    )
    dataset_group.add_argument("--include_go_defs", type=str2bool, default=False)
    dataset_group.add_argument(
        "--interpro_dataset_name",
        type=str,
        default=None,
        help="Name of InterPro metadata dataset config, resolved against --cafa5_dataset. "
             "Not needed on the reasoning path, which reads the dataset's own "
             "`interpro_formatted` column.",
    )
    dataset_group.add_argument("--split_go_aspects", type=str2bool, default=True)
    dataset_group.add_argument(
        "--ask_all_go_aspects", type=str2bool, default=True,
        help="Ask for all three GO aspects instead of deriving them from the "
             "example's labels. False reproduces the published benchmark.",
    )
    dataset_group.add_argument(
        "--force_uniprot_summary", type=str2bool, default=True,
        help="Always request a UniProt summary instead of conditioning on whether "
             "the protein has a known function. False reproduces the benchmark.",
    )
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
    eval_group.add_argument(
        "--batch_size", type=int, default=16,
        help="Samples per vLLM generation call, and so the number of sequences "
             "decoded concurrently. Raise for throughput; set 1 for one at a time.",
    )
    eval_group.add_argument("--max_new_tokens", type=int, default=3000,
                            help="Must leave room for the model to finish reasoning and emit "
                                 "its GO summary; truncating produces empty predictions.")
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
    run_inference(args)
