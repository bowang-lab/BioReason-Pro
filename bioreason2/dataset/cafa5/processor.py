# protein_training_generator.py

import os
from pathlib import Path
from goatools.obo_parser import GODag
from goatools.godag.reldepth import get_go2reldepth
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from bioreason2.dataset.utils import (
    ASPECT_ORDER,
    ASPECT_TO_COLUMN,
    ASPECT_FULL_NAMES,
    ASPECT_SPECIAL_TOKENS,
    ASPECT_SUMMARY_TOKENS,
    ASPECT_ROOT_TERMS,
    FUNCTION_SUMMARY_TOKENS,
    INTERPRO_SUMMARY_TOKENS,
)
from bioreason2.dataset.prompts.cafa5 import (
    CAFA5_PROMPT_TEMPLATE_WITH_INTERPRO,
    CAFA5_PROMPT_TEMPLATE_GO_ONLY,
    CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_WITH_INTERPRO,
    CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_GO_ONLY,
    CAFA5_PROMPT_TEMPLATE_WITH_INTERPRO_NO_FUNC,
    CAFA5_PROMPT_TEMPLATE_GO_ONLY_NO_FUNC,
    CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_WITH_INTERPRO_NO_FUNC,
    CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_GO_ONLY_NO_FUNC,
    CAFA5_PROMPT_TEMPLATE_INTERPRO_IN_PROMPT,
    CAFA5_PROMPT_TEMPLATE_INTERPRO_IN_PROMPT_NO_FUNC,
    CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_INTERPRO_IN_PROMPT,
    CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_INTERPRO_IN_PROMPT_NO_FUNC,
    CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_PPI_IN_PROMPT,
    CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_PPI_IN_PROMPT_NO_FUNC,
    CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_INTERPRO_PPI_IN_PROMPT,
    CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_INTERPRO_PPI_IN_PROMPT_NO_FUNC,
)
import json

# ────────────────── GO DAG & caches (load **once**) ──────────────────
BASE_DIR = Path(__file__).resolve().parent
OBO = (BASE_DIR.parent / "go-basic.obo").as_posix()
_GODAG = GODag(OBO, optional_attrs=["def", "relationship"])
_GO_DEPTH: Dict[str, int] = get_go2reldepth(_GODAG.values(), relationship_set={"part_of"})

# term → (name, definition)
_GO_INFO: Dict[str, Tuple[str, str]] = {}
for go_id, term in _GODAG.items():
    name = term.name.replace("_", " ")
    definition = term.defn.split('"')[1] if '"' in term.defn else term.defn
    _GO_INFO[go_id] = (name, definition)

# parent → children (is_a + part_of)
_CHILDREN: Dict[str, set[str]] = defaultdict(set)
for parent in _GODAG.values():
    for child in parent.children:
        _CHILDREN[parent.id].add(child.id)
for term in _GODAG.values():
    if hasattr(term, "relationship") and "part_of" in term.relationship:
        for parent in term.relationship["part_of"]:
            _CHILDREN[parent.id].add(term.id)


# ────────────────── helpers using caches ──────────────────
def _generate_aspect_traversal(go_terms: List[str], aspect: str, include_go_defs: bool = True) -> str:
    if not go_terms:
        return ""

    go_set = set(go_terms)

    # group by depth (skip invalid)
    depth_bins: Dict[int, List[str]] = defaultdict(list)
    for gid in go_terms:
        d = _GO_DEPTH.get(gid, -1)
        if d >= 0:
            depth_bins[d].append(gid)

    lines = [ASPECT_SPECIAL_TOKENS[aspect]["start"]]
    for depth in sorted(depth_bins):
        lines.append(f"Depth {depth}:")
        for gid in sorted(depth_bins[depth]):
            name, definition = _GO_INFO.get(gid, ("Unknown", "No definition"))
            if gid in ASPECT_ROOT_TERMS.values():
                name = name.replace("_", " ")

            # Toggle definition based on parameter
            if include_go_defs:
                lines.append(f"- {gid}: {name} - {definition}")
            else:
                lines.append(f"- {gid}: {name}")

            selected = [c for c in _CHILDREN.get(gid, []) if c in go_set]
            if selected:
                lines.append("  Selected children:")
                for cid in sorted(selected):
                    cname, _ = _GO_INFO.get(cid, ("Unknown", ""))
                    lines.append(f"    - {cid}: {cname}")
            else:
                lines.append("  No selected children")
    lines.append(ASPECT_SPECIAL_TOKENS[aspect]["end"])
    return "\n".join(lines).strip()


# Cache for InterPro metadata lookups
_INTERPRO_CACHE = {}


def _process_interpro_data(row: pd.Series, interpro_metadata: pd.DataFrame = None) -> tuple[str, list]:
    """Process InterPro data for a protein and return summary and ID list.

    Args:
        row: Protein data row
        interpro_metadata: DataFrame with InterPro metadata

    Returns:
        Tuple of (interpro_summary_string, interpro_ids_list)
    """
    interpro_sum = ""
    interpro_ids = []

    if interpro_metadata is not None and row.get("interpro_ids") is not None and len(row["interpro_ids"]) > 0:

        # Build cache if needed (once per process)
        global _INTERPRO_CACHE
        if not _INTERPRO_CACHE and interpro_metadata is not None:
            for _, meta_row in interpro_metadata.iterrows():
                _INTERPRO_CACHE[meta_row["interpro_id"]] = {
                    "entry_name": meta_row["entry_name"],
                    "type": meta_row["type"],
                }

        interpro_parts = [INTERPRO_SUMMARY_TOKENS["start"]]
        interpro_ids = row["interpro_ids"]

        # Parse location data
        locations = {}
        if row.get("interpro_location") and pd.notna(row.get("interpro_location")):
            try:
                locations = json.loads(row["interpro_location"])
            except json.JSONDecodeError:
                locations = {}

        # Process each InterPro ID in order
        for ipro_id in interpro_ids:
            # Get metadata from cache (O(1) lookup)
            meta = _INTERPRO_CACHE.get(ipro_id)
            if meta:
                name = meta["entry_name"]
                type_str = meta["type"]
            else:
                name = "Unknown"
                type_str = "Unknown"

            # Get location if available
            loc_str = ""
            if ipro_id in locations:
                start, end = locations[ipro_id]
                loc_str = f" [{start}-{end}]"

            interpro_parts.append(f"- {ipro_id}: {name} ({type_str}){loc_str}")

        interpro_parts.append(f"{INTERPRO_SUMMARY_TOKENS['end']}\n")
        interpro_sum = "\n".join(interpro_parts)

    return interpro_sum, interpro_ids


# ────────────────── example construction ──────────────────
def _build_instruction(
    row: pd.Series,
    tpl: Dict[str, str],
    interpro_metadata: pd.DataFrame = None,
    interpro_in_prompt: bool = False,
) -> Dict[str, str]:
    present = [
        a
        for a in ASPECT_ORDER
        if isinstance(row[ASPECT_TO_COLUMN[a]], (list, np.ndarray)) and len(row[ASPECT_TO_COLUMN[a]]) > 0
    ]

    # Build GO aspects string
    if len(present) == 1:
        go_aspects = ASPECT_FULL_NAMES[present[0]]
    elif len(present) == 2:
        go_aspects = f"{ASPECT_FULL_NAMES[present[0]]} and {ASPECT_FULL_NAMES[present[1]]}"
    else:
        go_aspects = ", ".join(ASPECT_FULL_NAMES[a] for a in present[:-1]) + f", and {ASPECT_FULL_NAMES[present[-1]]}"

    # Handle InterPro data in user prompt if requested
    user_prompt = tpl["user_prompt"]
    if interpro_in_prompt and interpro_metadata is not None:
        interpro_sum, _ = _process_interpro_data(row, interpro_metadata)
        if interpro_sum:
            # Remove the InterPro summary tokens from the formatted InterPro data
            # since we want the raw data for the user prompt
            interpro_data_clean = (
                interpro_sum.replace(f"{INTERPRO_SUMMARY_TOKENS['start']}\n", "")
                .replace(f"{INTERPRO_SUMMARY_TOKENS['end']}\n", "")
                .strip()
            )
            user_prompt = user_prompt.format(
                go_aspects=go_aspects,
                organism=row["organism"],
                interpro_data=interpro_data_clean,
            )
        else:
            # No InterPro data, use empty string
            user_prompt = user_prompt.format(go_aspects=go_aspects, organism=row["organism"], interpro_data="")
    else:
        user_prompt = user_prompt.format(go_aspects=go_aspects, organism=row["organism"])

    return {
        "system": tpl["system_prompt"],
        "user": user_prompt,
    }


def _build_instruction_single_aspect(
    row: pd.Series,
    aspect: str,
    tpl: Dict[str, str],
    interpro_metadata: pd.DataFrame = None,
    interpro_in_prompt: bool = False,
    ppi_in_prompt: bool = False,
) -> Dict[str, str]:
    """Build instruction for a single GO aspect."""
    go_aspects = ASPECT_FULL_NAMES[aspect]

    # Handle InterPro data in user prompt if requested
    user_prompt = tpl["user_prompt"]
    interpro_data_clean = ""
    if interpro_in_prompt and interpro_metadata is not None:
        interpro_sum, _ = _process_interpro_data(row, interpro_metadata)
        if interpro_sum:
            # Remove the InterPro summary tokens from the formatted InterPro data
            # since we want the raw data for the user prompt
            interpro_data_clean = (
                interpro_sum.replace(f"{INTERPRO_SUMMARY_TOKENS['start']}\n", "")
                .replace(f"{INTERPRO_SUMMARY_TOKENS['end']}\n", "")
                .strip()
            )

    # Handle PPI data from cached column if requested
    ppi_data_clean = ""
    if ppi_in_prompt:
        ppi_data_clean = row.get("ppi_formatted", "")
        # Handle case where ppi_formatted might be None
        if ppi_data_clean is None:
            ppi_data_clean = ""

    # Format user prompt based on available data
    if interpro_in_prompt and ppi_in_prompt:
        user_prompt = user_prompt.format(
            go_aspects=go_aspects,
            organism=row["organism"],
            interpro_data=interpro_data_clean,
            ppi_data=ppi_data_clean,
        )
    elif interpro_in_prompt:
        user_prompt = user_prompt.format(
            go_aspects=go_aspects,
            organism=row["organism"],
            interpro_data=interpro_data_clean,
        )
    elif ppi_in_prompt:
        user_prompt = user_prompt.format(go_aspects=go_aspects, organism=row["organism"], ppi_data=ppi_data_clean)
    else:
        user_prompt = user_prompt.format(go_aspects=go_aspects, organism=row["organism"])

    return {
        "system": tpl["system_prompt"],
        "user": user_prompt,
    }


def _build_response(
    row: pd.Series,
    interpro_metadata: pd.DataFrame = None,
    include_go_defs: bool = True,
    interpro_in_prompt: bool = False,
    predict_interpro: bool = False,
) -> tuple[str, str]:
    reasoning_parts, collected = [], defaultdict(list)

    # InterPro summary
    interpro_sum, interpro_ids = _process_interpro_data(row, interpro_metadata)

    # Add InterPro to reasoning only if not already in user prompt
    if interpro_sum and predict_interpro and not interpro_in_prompt:
        reasoning_parts.append(interpro_sum)

    # Then GO terms
    for aspect in ASPECT_ORDER:
        terms = row[ASPECT_TO_COLUMN[aspect]]
        if isinstance(terms, np.ndarray):
            terms = terms.tolist()
        if terms:
            traversal = _generate_aspect_traversal(terms, aspect, include_go_defs)
            reasoning_parts.append(traversal)
            collected[aspect] = sorted(terms)

    reasoning = "\n".join(reasoning_parts) if reasoning_parts else ""

    # GO-summary
    summary = [ASPECT_SUMMARY_TOKENS["start"]]
    for aspect in ASPECT_ORDER:
        if aspect in collected:
            ordered = sorted(collected[aspect], key=lambda t: (_GO_DEPTH.get(t, -1), t))
            summary.append(f"{aspect}: {', '.join(ordered)}")
    summary.append(ASPECT_SUMMARY_TOKENS["end"])

    # function summary
    func_sum = ""
    func = row.get("protein_function")
    if func and str(func).strip():
        func_sum = f"\n{FUNCTION_SUMMARY_TOKENS['start']}\n{str(func).strip()}\n{FUNCTION_SUMMARY_TOKENS['end']}"

    # Combine all parts for the answer - InterPro, then GO, then function
    answer_parts = []
    if interpro_ids and predict_interpro and not interpro_in_prompt:
        answer_parts.append(
            f"{INTERPRO_SUMMARY_TOKENS['start']}\n{', '.join(interpro_ids)}\n{INTERPRO_SUMMARY_TOKENS['end']}\n"
        )
    answer_parts.append("\n".join(summary))  # GO summary
    if func_sum:
        answer_parts.append(func_sum)

    answer = "\n".join(answer_parts)
    return reasoning, answer


def _build_response_single_aspect(
    row: pd.Series,
    aspect: str,
    interpro_metadata: pd.DataFrame = None,
    include_go_defs: bool = True,
    interpro_in_prompt: bool = False,
    predict_interpro: bool = False,
) -> tuple[str, str]:
    """Build response for a single GO aspect."""
    reasoning_parts, collected = [], defaultdict(list)

    # InterPro summary (same for all aspects)
    interpro_sum, interpro_ids = _process_interpro_data(row, interpro_metadata)

    # Add InterPro to reasoning only if not already in user prompt
    if interpro_sum and predict_interpro and not interpro_in_prompt:
        reasoning_parts.append(interpro_sum)

    # Process only the specified GO aspect
    terms = row[ASPECT_TO_COLUMN[aspect]]
    if isinstance(terms, np.ndarray):
        terms = terms.tolist()
    if terms:
        traversal = _generate_aspect_traversal(terms, aspect, include_go_defs)
        reasoning_parts.append(traversal)
        collected[aspect] = sorted(terms)

    reasoning = "\n".join(reasoning_parts) if reasoning_parts else ""

    # GO-summary for single aspect
    summary = [ASPECT_SUMMARY_TOKENS["start"]]
    if aspect in collected:
        ordered = sorted(collected[aspect], key=lambda t: (_GO_DEPTH.get(t, -1), t))
        summary.append(f"{aspect}: {', '.join(ordered)}")
    summary.append(ASPECT_SUMMARY_TOKENS["end"])

    # function summary
    func_sum = ""
    func = row.get("protein_function")
    if func and str(func).strip():
        func_sum = f"\n{FUNCTION_SUMMARY_TOKENS['start']}\n{str(func).strip()}\n{FUNCTION_SUMMARY_TOKENS['end']}"

    # Combine all parts for the answer - InterPro, then GO, then function
    answer_parts = []
    if interpro_ids and predict_interpro and not interpro_in_prompt:
        answer_parts.append(
            f"{INTERPRO_SUMMARY_TOKENS['start']}\n{', '.join(interpro_ids)}\n{INTERPRO_SUMMARY_TOKENS['end']}\n"
        )
    answer_parts.append("\n".join(summary))  # GO summary
    if func_sum:
        answer_parts.append(func_sum)

    answer = "\n".join(answer_parts)
    return reasoning, answer


def get_appropriate_template(
    protein_data: pd.Series,
    interpro_metadata: pd.DataFrame = None,
    interpro_in_prompt: bool = False,
    predict_interpro: bool = False,
) -> Dict[str, str]:
    """Select the appropriate prompt template based on protein data and metadata availability."""

    # Check if InterPro data is available
    interpro_ids = protein_data.get("interpro_ids")
    has_interpro = (
        interpro_metadata is not None  # Add this check
        and interpro_ids is not None
        and isinstance(interpro_ids, (list, np.ndarray))
        and len(interpro_ids) > 0
    )

    # Check if protein function is available
    protein_func = protein_data.get("protein_function")
    has_protein_function = (
        protein_func is not None
        and not pd.isna(protein_func)  # Handle pandas NaN
        and isinstance(protein_func, str)
        and protein_func.strip() != ""
    )

    # Select template based on available data
    if interpro_in_prompt and has_interpro:
        # Use templates with InterPro data in the user prompt
        if has_protein_function:
            return CAFA5_PROMPT_TEMPLATE_INTERPRO_IN_PROMPT
        else:
            return CAFA5_PROMPT_TEMPLATE_INTERPRO_IN_PROMPT_NO_FUNC
    else:
        # Use traditional templates
        if predict_interpro and has_interpro:
            # Use InterPro prediction templates
            if has_protein_function:
                return CAFA5_PROMPT_TEMPLATE_WITH_INTERPRO
            else:
                return CAFA5_PROMPT_TEMPLATE_WITH_INTERPRO_NO_FUNC
        else:
            if has_protein_function:
                return CAFA5_PROMPT_TEMPLATE_GO_ONLY
            else:
                return CAFA5_PROMPT_TEMPLATE_GO_ONLY_NO_FUNC


def get_appropriate_template_single_aspect(
    protein_data: pd.Series,
    interpro_metadata: pd.DataFrame = None,
    interpro_in_prompt: bool = False,
    ppi_in_prompt: bool = False,
    predict_interpro: bool = False,
) -> Dict[str, str]:
    """Select the appropriate prompt template for single aspect based on protein data and metadata availability."""

    # Check if InterPro data is available
    interpro_ids = protein_data.get("interpro_ids")
    has_interpro = (
        interpro_metadata is not None
        and interpro_ids is not None
        and isinstance(interpro_ids, (list, np.ndarray))
        and len(interpro_ids) > 0
    )

    # Check if PPI data is available
    ppi_formatted = protein_data.get("ppi_formatted", "")
    has_ppi = ppi_formatted is not None and ppi_formatted.strip() != ""

    # Check if protein function is available
    protein_func = protein_data.get("protein_function")
    has_protein_function = (
        protein_func is not None
        and not pd.isna(protein_func)  # Handle pandas NaN
        and isinstance(protein_func, str)
        and protein_func.strip() != ""
    )

    # Select template based on available data - prioritize prompt inclusion
    if interpro_in_prompt and ppi_in_prompt and has_interpro and has_ppi:
        # Both InterPro and PPI in user prompt
        if has_protein_function:
            return CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_INTERPRO_PPI_IN_PROMPT
        else:
            return CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_INTERPRO_PPI_IN_PROMPT_NO_FUNC
    elif interpro_in_prompt and has_interpro:
        # Only InterPro in user prompt
        if has_protein_function:
            return CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_INTERPRO_IN_PROMPT
        else:
            return CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_INTERPRO_IN_PROMPT_NO_FUNC
    elif ppi_in_prompt and has_ppi:
        # Only PPI in user prompt
        if has_protein_function:
            return CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_PPI_IN_PROMPT
        else:
            return CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_PPI_IN_PROMPT_NO_FUNC
    else:
        # Use traditional templates
        if predict_interpro and has_interpro:
            # Use InterPro prediction templates
            if has_protein_function:
                return CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_WITH_INTERPRO
            else:
                return CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_WITH_INTERPRO_NO_FUNC
        else:
            # Use GO-only templates
            if has_protein_function:
                return CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_GO_ONLY
            else:
                return CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_GO_ONLY_NO_FUNC


# =========== Main Function ===========
def generate_cafa5_example(
    protein_data: pd.Series,
    prompt_template: Dict[str, str] = None,
    godag: Optional[GODag] = None,
    go2reldepth: Optional[Dict[str, int]] = None,
    include_go_defs: bool = True,
    interpro_metadata: pd.DataFrame = None,
    interpro_in_prompt: bool = False,
    predict_interpro: bool = False,
) -> Dict[str, str]:
    """
    Generate complete training example for a single protein including instruction and response.

    Args:
        protein_data: Series containing protein data row from CAFA5 dataset
        prompt_template: Default dictionary with templates (can be either WITH_INTERPRO or GO_ONLY)
        interpro_metadata: DataFrame with interpro_id, entry_name, and type columns (required)
        include_go_defs: Whether to include GO term definitions in the output
        godag: GO DAG object (optional, uses pre-loaded by default)
        go2reldepth: GO term depth mapping (optional, uses pre-loaded by default)
        interpro_in_prompt: Whether to include InterPro data in user prompt instead of generation

    Returns:
        Dictionary containing the training example with 'instruction' and 'response' keys
    """
    # Use module-level godag and go2reldepth if not provided
    if godag is None:
        godag = globals()["_GODAG"]
    if go2reldepth is None:
        go2reldepth = globals()["_GO_DEPTH"]

    # Auto-select template based on SAME conditions that control data inclusion
    if prompt_template is None:
        prompt_template = get_appropriate_template(protein_data, interpro_metadata, interpro_in_prompt, predict_interpro)

    # Build training example components
    instruction = _build_instruction(protein_data, prompt_template, interpro_metadata, interpro_in_prompt)
    reasoning, answer = _build_response(protein_data, interpro_metadata, include_go_defs, interpro_in_prompt, predict_interpro)

    # Return as dictionary
    return {
        "system": instruction["system"],
        "user": instruction["user"],
        "assistant_reasoning": reasoning,
        "assistant_answer": answer,
    }


def generate_cafa5_example_single_aspect(
    protein_data: pd.Series,
    aspect: str,
    prompt_template: Dict[str, str] = None,
    godag: Optional[GODag] = None,
    go2reldepth: Optional[Dict[str, int]] = None,
    include_go_defs: bool = True,
    interpro_metadata: pd.DataFrame = None,
    interpro_in_prompt: bool = False,
    ppi_in_prompt: bool = False,
    predict_interpro: bool = False,
) -> Dict[str, str]:
    """
    Generate training example for a single GO aspect of a protein.

    Args:
        protein_data: Series containing protein data row from CAFA5 dataset
        aspect: GO aspect to generate example for ("MF", "BP", or "CC")
        prompt_template: Dictionary with templates (can be either WITH_INTERPRO or GO_ONLY)
        interpro_metadata: DataFrame with interpro_id, entry_name, and type columns (required)
        include_go_defs: Whether to include GO term definitions in the output
        godag: GO DAG object (optional, uses pre-loaded by default)
        go2reldepth: GO term depth mapping (optional, uses pre-loaded by default)
        interpro_in_prompt: Whether to include InterPro data in user prompt instead of generation

    Returns:
        Dictionary containing the training example with 'instruction' and 'response' keys
    """
    # Use module-level godag and go2reldepth if not provided
    if godag is None:
        godag = globals()["_GODAG"]
    if go2reldepth is None:
        go2reldepth = globals()["_GO_DEPTH"]

    # Auto-select template based on data availability
    if prompt_template is None:
        prompt_template = get_appropriate_template_single_aspect(
            protein_data, interpro_metadata, interpro_in_prompt, ppi_in_prompt, predict_interpro
        )

    # Build training example components
    instruction = _build_instruction_single_aspect(
        protein_data,
        aspect,
        prompt_template,
        interpro_metadata,
        interpro_in_prompt,
        ppi_in_prompt,
    )
    reasoning, answer = _build_response_single_aspect(
        protein_data, aspect, interpro_metadata, include_go_defs, interpro_in_prompt, predict_interpro
    )

    # Return as dictionary
    return {
        "system": instruction["system"],
        "user": instruction["user"],
        "assistant_reasoning": reasoning,
        "assistant_answer": answer,
    }


def generate_cafa5_examples_split_aspects(
    protein_data: pd.Series,
    prompt_template: Dict[str, str] = None,
    godag: Optional[GODag] = None,
    go2reldepth: Optional[Dict[str, int]] = None,
    include_go_defs: bool = True,
    interpro_metadata: pd.DataFrame = None,
    interpro_in_prompt: bool = False,
    ppi_in_prompt: bool = False,
    predict_interpro: bool = False,
) -> List[Dict[str, str]]:
    """
    Generate separate training examples for each GO aspect of a protein.

    Args:
        protein_data: Series containing protein data row from CAFA5 dataset
        prompt_template: Dictionary with templates (can be either WITH_INTERPRO or GO_ONLY)
        interpro_metadata: DataFrame with interpro_id, entry_name, and type columns (required)
        include_go_defs: Whether to include GO term definitions in the output
        godag: GO DAG object (optional, uses pre-loaded by default)
        go2reldepth: GO term depth mapping (optional, uses pre-loaded by default)
        interpro_in_prompt: Whether to include InterPro data in user prompt instead of generation

    Returns:
        List of dictionaries, each containing a training example for one GO aspect
    """
    examples = []

    # Check which aspects have data
    for aspect in ASPECT_ORDER:
        terms = protein_data[ASPECT_TO_COLUMN[aspect]]
        if isinstance(terms, (list, np.ndarray)) and len(terms) > 0:
            example = generate_cafa5_example_single_aspect(
                protein_data=protein_data,
                aspect=aspect,
                prompt_template=prompt_template,
                godag=godag,
                go2reldepth=go2reldepth,
                include_go_defs=include_go_defs,
                interpro_metadata=interpro_metadata,
                interpro_in_prompt=interpro_in_prompt,
                ppi_in_prompt=ppi_in_prompt,
                predict_interpro=predict_interpro,
            )
            # Add metadata to identify the aspect
            example["go_aspect"] = aspect
            example["protein_id"] = protein_data.get("protein_id", "unknown")
            examples.append(example)

    return examples


def _format_cafa5_for_protein_llm_wrapper(example, interpro_metadata, include_go_defs):
    """Helper function for dataset mapping to format CAFA5 examples."""
    example["prompt"] = generate_cafa5_example(
        example,
        prompt_template=None,  # Automatically determined based on example
        interpro_metadata=interpro_metadata,
        include_go_defs=include_go_defs,
    )
    return example


if __name__ == "__main__":
    print("Resolved OBO path:", OBO)
    print("Exists?", Path(OBO).exists())
