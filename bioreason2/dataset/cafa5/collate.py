import os
from typing import Dict, List

import torch
import numpy as np

from bioreason2.models.pl.processing_pl import PLProcessor
from bioreason2.protein_modules import ESMProteinModule

from esm.sdk.api import ESMProtein

# BioPython imports for structure parsing
from Bio.PDB import MMCIFParser, is_aa


def _coords_from_cif(cif_path, chain_id="A", atom_order=["N", "CA", "C"]):
    """Load protein structure coordinates from a CIF file.

    Args:
        cif_path (str): Path to the CIF file.
        chain_id (str): Chain ID to extract. Defaults to "A".
        atom_order (list): List of atom names to extract in order. Defaults to ["N", "CA", "C"].

    Returns:
        numpy.ndarray: Coordinate array with shape [L, len(atom_order), 3] where L is the number of residues.
    """
    # Parse with BioPython
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)
    model = structure[0]

    if chain_id not in model:
        raise ValueError(f"Chain {chain_id} not found in {cif_path}")

    chain = model[chain_id]
    coords = []

    # Collect residue info
    for residue in chain:
        if not is_aa(residue, standard=True):
            continue

        atom_coords = []
        for atom_name in atom_order:
            try:
                coord = residue[atom_name].get_coord()
            except KeyError:
                coord = np.full(3, np.nan)
            atom_coords.append(coord)
        coords.append(atom_coords)

    coords = np.array(coords)  # shape: [L, len(atom_order), 3]

    return coords


def _coords_from_pdb(pdb_path, chain_id="A"):
    """Load protein structure coordinates from a PDB file.

    Args:
        pdb_path (str): Path to the PDB file.
        chain_id (str): Chain ID to extract. Defaults to "A".

    Returns:
        numpy.ndarray: Coordinate array from the ESMProtein object.
    """
    return ESMProtein.from_pdb(pdb_path, chain_id=chain_id).coordinates


def _truncate_after_assistant_start(text: str) -> str:
    """
    Keep everything up to and including the first '<|im_start|>assistant\n',
    drop any assistant answer that follows. Used for inference mode.
    """
    marker = "<|im_end|>\n<|im_start|>assistant\n"
    idx = text.find(marker)
    if idx != -1:
        return text[: idx + len(marker)]
    return text


def qwen_protein_collate_fn(
    examples: List[Dict],
    processor: PLProcessor,
    max_length_text: int,
    max_length_protein: int,
    return_answer_in_batch: bool = False,
    inference_mode: bool = False,
) -> Dict:
    """Custom collate function for Qwen Protein models.

    Creates a batch with proper labels for supervised fine-tuning where only
    the assistant responses contribute to the loss calculation. Handles both
    text and protein structure data.

    Args:
        examples (List[Dict]): List of example dictionaries containing text,
            protein sequences, structure paths, and answers.
        processor (PLProcessor): Protein language model processor for tokenization.
        max_length_text (int): Maximum length for text sequences.
        max_length_protein (int): Maximum length for protein sequences.
        return_answer_in_batch (bool): Whether to include answers in the batch.
        inference_mode (bool): If True, truncate inputs after the assistant start
            marker to prepare for generation. Defaults to False.

    Returns:
        Dict: Batch dictionary containing:
            - input_ids: Tokenized input sequences
            - attention_mask: Attention masks for the sequences
            - structure_coords: Protein structure coordinates as tensor
            - labels: Labels for supervised fine-tuning (only assistant responses)
            - answer: Ground truth answers (if return_answer_in_batch=True)
            - prompt: Text prompts (if inference_mode=True)
    """
    protein_module = ESMProteinModule()
    prompts_text = protein_module.prepare_prompt(processing_class=processor, inputs=examples)
    batch_protein_sequences = [example["protein_sequences"] for example in examples]
    batch_protein_structures = [example["structure_path"] for example in examples]
    batch_go_aspects = [example.get("go_aspect") for example in examples]

    batch = processor(
        text=prompts_text,
        batch_protein_sequences=batch_protein_sequences,
        batch_go_aspects=batch_go_aspects,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        max_length_text=max_length_text,
        max_length_protein=max_length_protein,
    )

    # -------------------------------------------------------------------
    # Process protein structure coordinates (dynamic padding to batch max)
    # -------------------------------------------------------------------
    structure_coord_list = []
    for struct_path in batch_protein_structures:
        if struct_path is not None and os.path.exists(struct_path):
            try:
                if struct_path.endswith(".cif"):
                    coords = _coords_from_cif(struct_path)
                elif struct_path.endswith(".pdb"):
                    coords = _coords_from_pdb(struct_path)
                else:
                    raise ValueError(f"Unsupported structure format: {struct_path}")
            except Exception:
                # On error, fall back to empty coordinates
                coords = np.full((0, 3, 3), np.nan)
        else:
            coords = np.full((0, 3, 3), np.nan)

        # Truncate if number of residues exceeds max_length_protein
        if coords.shape[0] > max_length_protein:
            coords = coords[:max_length_protein]

        structure_coord_list.append(coords)

    # Dynamic padding: pad to batch max length (more memory efficient)
    lengths = [arr.shape[0] for arr in structure_coord_list]
    max_len = max(lengths) if lengths else 0
    batch_size = len(structure_coord_list)

    # Build padded tensor
    if max_len == 0:
        padded = torch.zeros((batch_size, 0, 3, 3), dtype=torch.float32)
    else:
        padded = torch.zeros((batch_size, max_len, 3, 3), dtype=torch.float32)
        for i, arr in enumerate(structure_coord_list):
            L = arr.shape[0]
            if L > 0:
                padded[i, :L] = torch.from_numpy(arr).to(torch.float32)

    batch["structure_coords"] = padded.to(batch["input_ids"].device)

    # -------------------------------------------------------------------
    # Create labels for supervised fine-tuning
    # -------------------------------------------------------------------

    # Create labels tensor filled with -100 (ignored in loss calculation)
    labels = torch.full_like(batch["input_ids"], -100)

    # Get token IDs for special markers
    assistant_start_marker = "<|im_start|>assistant\n"
    im_end_marker = "<|im_end|>"

    assistant_start_token_ids = processor.tokenizer.encode(assistant_start_marker, add_special_tokens=False)
    im_end_token_ids = processor.tokenizer.encode(im_end_marker, add_special_tokens=False)

    # Convert token arrays to tensors for faster comparison
    assistant_marker_tensor = torch.tensor(assistant_start_token_ids, device=batch["input_ids"].device)
    im_end_marker_tensor = torch.tensor(im_end_token_ids, device=batch["input_ids"].device)

    # Get dimensions for easier reference
    assistant_marker_len = len(assistant_start_token_ids)
    im_end_marker_len = len(im_end_token_ids)

    # For each sequence in the batch
    for i in range(batch["input_ids"].shape[0]):
        input_ids = batch["input_ids"][i]
        seq_len = input_ids.size(0)

        # Track assistant sections
        assistant_sections = []

        # Find all assistant start markers
        start_positions = []
        for pos in range(seq_len - assistant_marker_len + 1):
            if torch.all(input_ids[pos : pos + assistant_marker_len] == assistant_marker_tensor):
                start_positions.append(pos + assistant_marker_len)  # Store position after marker

        # Find all end markers
        end_positions = []
        for pos in range(seq_len - im_end_marker_len + 1):
            if torch.all(input_ids[pos : pos + im_end_marker_len] == im_end_marker_tensor):
                end_positions.append(pos)  # Store position at start of end marker

        # Match start and end markers to create sections
        for start_pos in start_positions:
            # Find the next end marker after this start position
            valid_ends = [pos for pos in end_positions if pos > start_pos]
            if valid_ends:
                end_pos = min(valid_ends)  # Take the first end marker after start
                # Only include content between markers (not the markers themselves)
                if start_pos < end_pos:
                    assistant_sections.append((start_pos, end_pos))
            else:
                # If no end marker, assume the section runs to the end of the sequence
                assistant_sections.append((start_pos, seq_len))

        # Set labels for all identified assistant sections
        for start_pos, end_pos in assistant_sections:
            if start_pos < end_pos and start_pos < seq_len:
                end_pos = min(end_pos, seq_len)  # Safety check
                labels[i, start_pos:end_pos] = input_ids[start_pos:end_pos]

    # Also mask padding tokens
    labels[batch["input_ids"] == processor.tokenizer.pad_token_id] = -100

    # Add labels to batch
    batch["labels"] = labels

    # Add answer to batch
    if return_answer_in_batch:
        batch["answer"] = [example["answer"].strip() for example in examples]

    # -------------------------------------------------------------------
    # Inference mode: truncate after assistant start marker
    # -------------------------------------------------------------------
    if inference_mode:
        device = batch["input_ids"].device
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = processor.tokenizer.eos_token_id

        # Composite marker to match _truncate_after_assistant_start
        composite = "<|im_end|>\n<|im_start|>assistant\n"
        comp_ids = processor.tokenizer.encode(composite, add_special_tokens=False)
        comp_t = torch.tensor(comp_ids, device=device)
        comp_len = len(comp_ids)

        B, L = batch["input_ids"].shape
        keep_lens: List[int] = []

        for i in range(B):
            ids = batch["input_ids"][i]
            keep = L  # default: keep all if marker not found
            # Scan for FIRST occurrence
            for j in range(0, L - comp_len + 1):
                if torch.all(ids[j : j + comp_len] == comp_t):
                    keep = j + comp_len
                    break
            keep_lens.append(keep)

        new_max = max(keep_lens) if keep_lens else 0

        # Allocate new left-padded tensors
        new_input_ids = torch.full((B, new_max), pad_id, dtype=batch["input_ids"].dtype, device=device)
        new_attention = torch.zeros((B, new_max), dtype=batch["attention_mask"].dtype, device=device)
        new_labels = torch.full((B, new_max), -100, dtype=batch["labels"].dtype, device=device)

        for i, k in enumerate(keep_lens):
            if k == 0:
                continue
            # Take the first k tokens (truncate from the RIGHT), then left-pad to new_max
            src_ids = batch["input_ids"][i, :k]
            src_attn = batch["attention_mask"][i, :k]
            src_lbls = batch["labels"][i, :k]

            new_input_ids[i, -k:] = src_ids
            new_attention[i, -k:] = src_attn
            new_labels[i, -k:] = src_lbls

        batch["input_ids"] = new_input_ids
        batch["attention_mask"] = new_attention
        batch["labels"] = new_labels

        # Also truncate the text prompts
        batch["prompt"] = [_truncate_after_assistant_start(p) for p in prompts_text]

    return batch
