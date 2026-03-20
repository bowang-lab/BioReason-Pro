import torch
import gc
from typing import Optional, List, Dict, Any


def generate_single_response(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    protein_sequences: Optional[List[str]] = None,
    structure_coords: Optional[torch.Tensor] = None,
    batch_idx_map: Optional[List[int]] = None,
    go_aspects: Optional[List[str]] = None,
    example_idx: int = 0,
    assistant_marker: str = "<|im_start|>assistant\n",
    **generation_kwargs,
) -> Dict[str, Any]:
    """
    Generate response for a single example from batch, truncating at assistant marker.

    Args:
        model: The ProteinLLM model to use for generation
        tokenizer: The tokenizer for encoding/decoding
        input_ids: Batch of input token IDs [batch_size, seq_len]
        attention_mask: Attention mask for the inputs [batch_size, seq_len]
        labels: Label tensor for ground truth extraction [batch_size, seq_len]
        protein_sequences: List of protein sequence strings
        structure_coords: Tensor of protein structure coordinates
        batch_idx_map: Mapping of protein sequences to batch indices
        go_aspects: List of GO aspects for each example in the batch
        example_idx: Which example from the batch to generate for
        assistant_marker: The marker that indicates start of assistant response
        **generation_kwargs: Additional arguments for model.generate()

    Returns:
        Dictionary with keys:
        - user_input: The decoded user input prompt
        - generation: Complete generated sequence
        - ground_truth: The target response (if labels provided)
        - success: Boolean indicating if generation succeeded
    """
    result = {"user_input": "", "generation": "", "ground_truth": "", "success": False}

    try:
        # Encode assistant marker
        assistant_marker_tokens = tokenizer.encode(assistant_marker, add_special_tokens=False)
        marker_tensor = torch.tensor(assistant_marker_tokens, device=input_ids.device)
        marker_len = len(assistant_marker_tokens)

        # Find non-padding tokens for this example
        non_pad = (input_ids[example_idx] != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        start_idx = non_pad[0].item() if len(non_pad) > 0 else 0

        # Find assistant marker position
        assistant_pos = None
        for pos in range(start_idx, input_ids.size(1) - marker_len + 1):
            if torch.all(input_ids[example_idx, pos : pos + marker_len] == marker_tensor):
                assistant_pos = pos
                break

        if assistant_pos is None:
            return result

        # Prepare generation input (up to and including assistant marker)
        gen_input_ids = input_ids[example_idx : example_idx + 1, start_idx : assistant_pos + marker_len]
        gen_attention_mask = attention_mask[example_idx : example_idx + 1, start_idx : assistant_pos + marker_len]

        # Extract protein sequences for this example
        example_protein_sequences = None
        example_structure_coords = None
        example_batch_map = None
        example_go_aspects = None

        if protein_sequences is not None and batch_idx_map is not None:
            # Find protein sequences belonging to this example
            example_indices = [i for i, idx in enumerate(batch_idx_map) if idx == example_idx]

            if len(example_indices) > 0:
                example_protein_sequences = [protein_sequences[i] for i in example_indices]

                # If structures are present, extract the corresponding coordinates
                if structure_coords is not None:
                    example_structure_coords = structure_coords[example_indices]

                # Map all sequences to batch index 0 for single-item generation
                example_batch_map = [0] * len(example_indices)

        # Extract GO aspect for this example
        if go_aspects is not None and example_idx < len(go_aspects):
            example_go_aspects = [go_aspects[example_idx]]

        # Generate response
        with torch.no_grad():
            generated = model.generate(
                input_ids=gen_input_ids,
                attention_mask=gen_attention_mask,
                protein_sequences=example_protein_sequences,
                structure_coords=example_structure_coords,
                batch_idx_map=example_batch_map,
                go_aspects=example_go_aspects,
                **generation_kwargs,
            )

        # Decode results
        user_input = tokenizer.decode(gen_input_ids[0], skip_special_tokens=False).strip()
        generation = tokenizer.decode(generated[0], skip_special_tokens=False).strip()

        # Extract ground truth if labels are provided
        ground_truth = ""
        if labels is not None:
            # Find all positions where we have valid labels (not -100)
            valid_label_pos = (labels[example_idx] != -100).nonzero(as_tuple=True)[0]

            if len(valid_label_pos) > 0:
                # Check if valid labels start after assistant marker
                if valid_label_pos[0] >= assistant_pos + marker_len:
                    ground_truth = tokenizer.decode(
                        input_ids[example_idx, valid_label_pos],
                        skip_special_tokens=False,
                    ).strip()

        # Log result
        result.update(
            {
                "user_input": user_input,
                "generation": generation,
                "ground_truth": ground_truth,
                "success": True,
            }
        )

        return result

    except Exception as e:
        print(f"Error generating for example {example_idx}: {str(e)}")
        return result
