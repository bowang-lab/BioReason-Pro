#!/usr/bin/env python3
"""
BioReason-Pro: Protein Function Prediction Pipeline

Single-file inference tool that runs InterPro, GO-GPT, and BioReason-Pro
to predict protein function from sequence.

Usage:
    python predict.py --input proteins.tsv --output results.tsv --model_type rl
    python predict.py --input proteins.tsv --output results.tsv --model_type sft --resume

Input TSV columns (tab-separated, with header):
    protein_id    organism    sequence

Output TSV columns (tab-separated, with header):
    protein_id    organism    sequence    sequence_length    interpro    gogpt    generated_response

Supported organisms are listed in organism_list.txt.
"""

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve repo root and add to path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gogpt" / "src"))

from interpro_api import run_interproscan_online, format_interpro_output
from gogpt_api import load_predictor, predict_go_terms, format_go_output
from bioreason2.models.protein_vllm import ProteinLLMModel
from bioreason2.dataset.prompts.cafa5 import (
    CAFA5_REASONING_TEMPLATE,
    CAFA5_REASONING_TEMPLATE_WITH_CONTEXT,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_REPO = {
    "sft": "wanglab/bioreason-pro-sft",
    "rl": "wanglab/bioreason-pro-rl",
}
GO_OBO_PATH = str(REPO_ROOT / "bioreason2" / "dataset" / "go-basic.obo")
ORGANISM_LIST_PATH = str(REPO_ROOT / "organism_list.txt")
STOP_TOKENS = ["<|im_end|>"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model architecture (must match training config)
MODEL_ARCH = dict(
    protein_model_name="esm3_sm_open_v1",
    protein_embedding_layer=37,
    unified_go_encoder=True,
    go_hidden_dim=512,
    go_num_gat_layers=3,
    go_num_heads=8,
    go_num_reduced_embeddings=200,
    go_embedding_dim=2560,
)

# Generation defaults. Kept in step with eval.py.
GEN_DEFAULTS = dict(
    max_new_tokens=3000,
    temperature=0.0,
    top_p=0.95,
    repetition_penalty=1.0,
    batch_size=16,
    max_length_protein=2000,
    max_model_len=8192,
    # vLLM's share only; ESM3, the GO encoder and prompt embeddings sit outside it.
    gpu_memory_utilization=0.5,
    max_num_seqs=256,
)

TSV_COLUMNS = [
    "protein_id", "organism", "sequence", "sequence_length",
    "interpro", "gogpt", "generated_response",
]

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ===================================================================
# Input Validation
# ===================================================================

def clean_sequence(seq: str) -> str:
    """Clean a protein sequence: remove whitespace, newlines, tabs, and non-AA characters."""
    seq = seq.strip()
    seq = re.sub(r"[\s\n\t\r]+", "", seq)
    seq = seq.upper()
    cleaned = "".join(c for c in seq if c in VALID_AA)
    if len(cleaned) != len(seq):
        removed = len(seq) - len(cleaned)
        print(f"  Warning: Removed {removed} non-amino-acid characters from sequence")
    return cleaned


def load_organism_list() -> set:
    """Load the set of supported organisms from organism_list.txt."""
    if not os.path.exists(ORGANISM_LIST_PATH):
        return set()
    with open(ORGANISM_LIST_PATH) as f:
        return {line.strip() for line in f if line.strip()}


def validate_organism(organism: str, valid_organisms: set) -> str:
    """Validate and clean organism string. Warns if not in supported list."""
    organism = organism.strip()
    if valid_organisms and organism not in valid_organisms:
        print(f"  Warning: Organism '{organism}' not in organism_list.txt. "
              f"GO-GPT predictions may be less accurate.")
    return organism


# ===================================================================
# I/O Helpers
# ===================================================================

def read_input_tsv(path: str) -> List[Dict[str, str]]:
    """Read input TSV. Requires columns: protein_id, organism, sequence."""
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise ValueError("Input TSV is empty")
    for col in ("protein_id", "organism", "sequence"):
        if col not in rows[0]:
            raise ValueError(f"Input TSV missing required column: '{col}'")
    return rows


def get_completed_ids(output_path: str) -> set:
    """Scan output TSV for already-completed protein IDs."""
    if not os.path.exists(output_path):
        return set()
    with open(output_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["protein_id"] for row in reader if row.get("generated_response")}


def load_checkpoint(path: str) -> Dict[str, str]:
    """Load a stage checkpoint JSON."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(data: Dict[str, str], path: str):
    """Save a stage checkpoint JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def append_results_tsv(output_path: str, results: List[Dict[str, str]]):
    """Append results to output TSV, creating header if needed."""
    write_header = not os.path.exists(output_path)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow({col: r.get(col, "") for col in TSV_COLUMNS})


# ===================================================================
# Stage 1: InterPro (online API, multi-threaded)
# ===================================================================

def _interpro_single(protein_id: str, sequence: str) -> Tuple[str, str]:
    """Run InterPro online for a single protein. Returns (protein_id, formatted_str)."""
    try:
        result_df = run_interproscan_online(sequence)
        if result_df.empty:
            return protein_id, ""
        formatted = format_interpro_output(result_df, {})
        return protein_id, formatted
    except Exception as e:
        print(f"  InterPro failed for {protein_id}: {e}")
        return protein_id, ""


def run_interpro_stage(
    proteins: List[Dict[str, str]],
    checkpoint_path: str,
    resume: bool,
) -> Dict[str, str]:
    """Run InterPro on all proteins using online API with thread parallelism."""
    print(f"\n[1/3] Running InterPro annotations (online API)...")

    results = load_checkpoint(checkpoint_path) if resume else {}
    todo = [p for p in proteins if p["protein_id"] not in results]

    if not todo:
        print(f"  All {len(proteins)} proteins already have InterPro results.")
        return results

    workers = min(os.cpu_count() or 4, len(todo))
    print(f"  {len(todo)} proteins to process, {workers} threads")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_interpro_single, p["protein_id"], p["sequence"]): p["protein_id"]
            for p in todo
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="  InterPro"):
            pid, formatted = future.result()
            results[pid] = formatted

    save_checkpoint(results, checkpoint_path)
    with_hits = sum(1 for v in results.values() if v)
    print(f"  Done. {with_hits}/{len(results)} proteins have InterPro annotations.")
    return results


# ===================================================================
# Stage 2: GO-GPT (GPU, load -> run -> unload)
# ===================================================================

def run_gogpt_stage(
    proteins: List[Dict[str, str]],
    checkpoint_path: str,
    resume: bool,
) -> Dict[str, str]:
    """Run GO-GPT predictions. Loads model on GPU, runs, then frees GPU memory."""
    print(f"\n[2/3] Running GO-GPT predictions...")

    results = load_checkpoint(checkpoint_path) if resume else {}
    todo = [p for p in proteins if p["protein_id"] not in results]

    if not todo:
        print(f"  All {len(proteins)} proteins already have GO-GPT results.")
        return results

    print(f"  Loading GO-GPT model...")
    predictor = load_predictor(cache_dir=None)
    print(f"  Model loaded. Processing {len(todo)} proteins...")

    for p in tqdm(todo, desc="  GO-GPT"):
        pid = p["protein_id"]
        try:
            predictions = predict_go_terms(predictor, p["sequence"], p["organism"])
            results[pid] = format_go_output(predictions)
        except Exception as e:
            print(f"  GO-GPT failed for {pid}: {e}")
            results[pid] = ""

    # Free GPU memory
    del predictor
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Done. GPU memory released.")

    save_checkpoint(results, checkpoint_path)
    return results


# ===================================================================
# Stage 3: BioReason-Pro (GPU, load from HuggingFace -> batch inference)
# ===================================================================

def _download_checkpoint(model_type: str) -> str:
    """Download checkpoint from HuggingFace and return local path."""
    from huggingface_hub import snapshot_download
    repo_id = HF_REPO[model_type]
    print(f"  Downloading {repo_id} from HuggingFace...")
    local_dir = snapshot_download(repo_id=repo_id)
    print(f"  Checkpoint ready at: {local_dir}")
    return local_dir


def _build_prompt(organism: str, interpro: str, gogpt: str) -> Dict[str, str]:
    """Build prompt dict for the model."""
    go_aspects_suffix = " and focus more on its Molecular Function, Biological Process, Cellular Component."
    uniprot_summary = " Summarize in UniProt format."

    if interpro or gogpt:
        system = CAFA5_REASONING_TEMPLATE_WITH_CONTEXT["system_prompt"]
        user = CAFA5_REASONING_TEMPLATE_WITH_CONTEXT["user_prompt"].format(
            organism=organism,
            interpro_data=interpro if interpro else "None",
            go_speculations=gogpt if gogpt else "None",
        )
    else:
        system = CAFA5_REASONING_TEMPLATE["system_prompt"]
        user = CAFA5_REASONING_TEMPLATE["user_prompt"].format(organism=organism)

    user = user.rstrip(".") + go_aspects_suffix + uniprot_summary
    return {"system": system, "user": user}


def _build_chat_messages(organism: str, interpro: str, gogpt: str) -> List[Dict]:
    """Build chat message list for the model."""
    prompt = _build_prompt(organism, interpro, gogpt)
    return [
        {
            "role": "user",
            "content": [
                {"type": "protein", "text": None},
                {"type": "go_graph", "text": None},
                {
                    "type": "text",
                    "text": f"{prompt['system'].strip()}\n\n{prompt['user'].strip()}",
                },
            ],
        },
    ]


def run_bioreason_stage(
    proteins: List[Dict[str, str]],
    interpro_results: Dict[str, str],
    gogpt_results: Dict[str, str],
    args,
) -> None:
    """Run BioReason-Pro inference and write results incrementally to output TSV."""
    print(f"\n[3/3] Running BioReason-Pro inference ({args.model_type.upper()} model)...")

    # Check what's already done
    completed = get_completed_ids(args.output) if args.resume else set()
    todo = [p for p in proteins if p["protein_id"] not in completed]

    if not todo:
        print(f"  All {len(proteins)} proteins already have predictions.")
        return

    # Download checkpoint
    ckpt_dir = _download_checkpoint(args.model_type)

    # GO embeddings: go_embedding.pt in the HF checkpoint provides the cached
    # embedding, and go_projection.pt provides the projection weights.
    # No need for the per-GO-term safetensors directory.
    precomputed_path = args.go_embeddings_path  # None unless user overrides

    print(f"  Loading model...")
    model = ProteinLLMModel(
        ckpt_dir=ckpt_dir,
        go_obo_path=GO_OBO_PATH,
        precomputed_embeddings_path=precomputed_path,
        max_length_protein=GEN_DEFAULTS["max_length_protein"],
        max_length_text=GEN_DEFAULTS["max_model_len"] - 200 - GEN_DEFAULTS["max_length_protein"] - 2,
        max_model_len=GEN_DEFAULTS["max_model_len"],
        gpu_memory_utilization=GEN_DEFAULTS["gpu_memory_utilization"],
        max_num_seqs=GEN_DEFAULTS["max_num_seqs"],
        text_model_finetune=False,
        protein_model_finetune=False,
        go_model_finetune=False,
        **MODEL_ARCH,
    )
    print(f"  Model loaded. Processing {len(todo)} proteins...")

    batch_size = args.batch_size
    num_batches = (len(todo) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="  BioReason", unit="batch"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(todo))
        batch = todo[start:end]

        try:
            # Build prompts and inputs
            prompts = []
            sequences = []
            go_aspects = []

            for sample in batch:
                pid = sample["protein_id"]
                messages = _build_chat_messages(
                    sample["organism"],
                    interpro_results.get(pid, ""),
                    gogpt_results.get(pid, ""),
                )
                prompt_string = model.text_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                prompts.append(prompt_string)
                sequences.append(sample["sequence"])
                go_aspects.append("all")

            original_padding_side = model.text_tokenizer.padding_side
            model.text_tokenizer.padding_side = "left"

            processed = model.processor(
                text=prompts,
                batch_protein_sequences=[[seq] for seq in sequences],
                batch_go_aspects=go_aspects,
                max_length_text=model.max_length_text,
                max_length_protein=model.max_length_protein,
                return_tensors="pt",
            )

            model.text_tokenizer.padding_side = original_padding_side

            input_ids = processed.get("input_ids").to(DEVICE)
            attention_mask = processed.get("attention_mask").to(DEVICE)

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    protein_sequences=sequences,
                    batch_idx_map=list(range(len(batch))),
                    go_aspects=go_aspects,
                    structure_coords=processed.get("structure_coords"),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    stop=STOP_TOKENS,
                )

            # Collect results
            batch_results = []
            for i, sample in enumerate(batch):
                response = outputs[i] if i < len(outputs) else ""
                batch_results.append({
                    "protein_id": sample["protein_id"],
                    "organism": sample["organism"],
                    "sequence": sample["sequence"],
                    "sequence_length": str(len(sample["sequence"])),
                    "interpro": interpro_results.get(sample["protein_id"], ""),
                    "gogpt": gogpt_results.get(sample["protein_id"], ""),
                    "generated_response": response,
                })

            append_results_tsv(args.output, batch_results)

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM on batch {batch_idx}. Skipping.")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n  Error on batch {batch_idx}: {e}")
            traceback.print_exc()

    total = len(get_completed_ids(args.output))
    print(f"  Done. {total}/{len(proteins)} proteins have predictions.")


# ===================================================================
# Pipeline Orchestrator
# ===================================================================

def run_pipeline(args):
    """Run the full prediction pipeline."""
    print(f"BioReason-Pro Prediction Pipeline")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Model:  {args.model_type.upper()}")

    # Read and validate input
    proteins = read_input_tsv(args.input)
    valid_organisms = load_organism_list()

    print(f"  Loaded {len(proteins)} proteins. Validating...")
    for p in proteins:
        p["sequence"] = clean_sequence(p["sequence"])
        p["organism"] = validate_organism(p["organism"], valid_organisms)
        p["sequence_length"] = str(len(p["sequence"]))
        if not p["sequence"]:
            raise ValueError(f"Protein '{p['protein_id']}' has an empty sequence after cleaning")

    print(f"  Validation complete.")

    # Checkpoint paths (timestamped to avoid collisions across runs)
    stem = Path(args.output).stem
    out_dir = str(Path(args.output).parent) or "."
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    interpro_ckpt = os.path.join(out_dir, f"{stem}_interpro_{timestamp}.json")
    gogpt_ckpt = os.path.join(out_dir, f"{stem}_gogpt_{timestamp}.json")

    # Stage 1: InterPro
    interpro_results = run_interpro_stage(proteins, interpro_ckpt, args.resume)

    # Stage 2: GO-GPT
    gogpt_results = run_gogpt_stage(proteins, gogpt_ckpt, args.resume)

    # Stage 3: BioReason-Pro
    run_bioreason_stage(proteins, interpro_results, gogpt_results, args)

    print(f"\nPipeline complete. Results at: {args.output}")


# ===================================================================
# CLI
# ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="BioReason-Pro: Predict protein function from sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, help="Input TSV file (protein_id, organism, sequence)")
    parser.add_argument("--output", required=True, help="Output TSV file")
    parser.add_argument("--model_type", choices=["sft", "rl"], default="rl", help="Model type (default: rl)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints / skip completed proteins")
    parser.add_argument("--batch_size", type=int, default=GEN_DEFAULTS["batch_size"])
    parser.add_argument("--max_new_tokens", type=int, default=GEN_DEFAULTS["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=GEN_DEFAULTS["temperature"])
    parser.add_argument("--top_p", type=float, default=GEN_DEFAULTS["top_p"])
    parser.add_argument("--repetition_penalty", type=float, default=GEN_DEFAULTS["repetition_penalty"])
    parser.add_argument("--go_embeddings_path", type=str, default=None,
                        help="Path to GO embeddings dir (optional, only if go_embedding.pt missing from checkpoint)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
