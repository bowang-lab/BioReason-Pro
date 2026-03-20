import ast
import os
import traceback
from collections import defaultdict

from datasets import load_dataset, disable_caching

from bioreason2.dataset.cafa5.processor import (
    generate_cafa5_example,
    generate_cafa5_examples_split_aspects,
)
from bioreason2.dataset.prompts.cafa5 import (
    CAFA5_REASONING_TEMPLATE,
    CAFA5_REASONING_TEMPLATE_WITH_CONTEXT,
    CAFA5_REASONING_TEMPLATE_WITH_CONTEXT_PPI,
    CAFA5_REASONING_TEMPLATE_WITH_CONTEXT_PPI_UNIPROT,
    CAFA5_REASONING_TEMPLATE_SWISSPROT,
)
from bioreason2.dataset.cafa5.format import format_cafa5_for_protein_llm
from bioreason2.dataset.utils import truncate_protein, ASPECT_ORDER, ASPECT_TO_COLUMN

# disable_caching()  # Since multiple people might use the same cache directory


def _count_go_term_frequencies(dataset):
    """Count the frequency of each GO term across the entire dataset.

    Args:
        dataset: HuggingFace dataset with GO terms in go_mf, go_bp, go_cc columns

    Returns:
        dict: Mapping from aspect to dict of {go_term: frequency}
    """
    print("Counting GO term frequencies across dataset...")

    # Initialize frequency counters for each aspect
    go_frequencies = {aspect: defaultdict(int) for aspect in ASPECT_ORDER}

    for i, sample in enumerate(dataset):
        for aspect in ASPECT_ORDER:
            column_name = ASPECT_TO_COLUMN[aspect]
            go_terms = sample.get(column_name, [])

            # Handle different data types (list, None, etc.)
            if go_terms is not None and len(go_terms) > 0:
                if isinstance(go_terms, list):
                    for term in go_terms:
                        if term and isinstance(term, str) and term.startswith("GO:"):
                            go_frequencies[aspect][term] += 1

    # Print statistics
    print("GO term frequency statistics:")
    for aspect in ASPECT_ORDER:
        frequencies = go_frequencies[aspect]
        if frequencies:
            total_terms = len(frequencies)
            total_occurrences = sum(frequencies.values())
            print(
                f"  {aspect}: {total_terms} unique terms, {total_occurrences} total occurrences"
            )
        else:
            print(f"  {aspect}: No terms found")

    return go_frequencies


def _filter_go_terms_by_frequency(
    dataset, go_frequencies, min_frequencies, num_proc=None
):
    """Filter GO terms based on minimum frequency thresholds.

    Args:
        dataset: HuggingFace dataset to filter
        go_frequencies: Dict from _count_go_term_frequencies
        min_frequencies: Dict with keys 'MF', 'BP', 'CC' and minimum frequency values
        num_proc: Number of processes for parallel processing

    Returns:
        Filtered dataset with infrequent GO terms removed
    """
    print("Filtering GO terms based on frequency thresholds...")

    # Create sets of terms that meet the minimum frequency for each aspect
    valid_terms = {}
    for aspect in ASPECT_ORDER:
        min_freq = min_frequencies[aspect]
        valid_terms[aspect] = {
            term for term, freq in go_frequencies[aspect].items() if freq >= min_freq
        }

        original_count = len(go_frequencies[aspect])
        filtered_count = len(valid_terms[aspect])
        print(
            f"  {aspect}: {original_count} -> {filtered_count} terms (min_freq={min_freq})"
        )

    def _filter_sample_go_terms(sample):
        """Filter GO terms in a single sample."""
        filtered_sample = sample.copy()

        for aspect in ASPECT_ORDER:
            column_name = ASPECT_TO_COLUMN[aspect]
            go_terms = sample.get(column_name, [])

            if go_terms is not None and len(go_terms) > 0:
                if isinstance(go_terms, list):
                    # Filter terms to only include those that meet frequency threshold
                    filtered_terms = [
                        term
                        for term in go_terms
                        if term
                        and isinstance(term, str)
                        and term in valid_terms[aspect]
                    ]
                    filtered_sample[column_name] = filtered_terms
                else:
                    # Keep non-list values as is
                    filtered_sample[column_name] = go_terms
            else:
                # Keep None/empty values as is
                filtered_sample[column_name] = go_terms

        return filtered_sample

    # Apply filtering to the dataset
    filtered_dataset = dataset.map(
        _filter_sample_go_terms,
        num_proc=num_proc,
        desc="Filtering GO terms by frequency",
    )

    print("GO term filtering completed")
    return filtered_dataset


def _add_structure_prefix(example, structure_dir):
    """Helper function to add structure directory prefix to structure paths."""
    if example["structure_path"] is not None:
        example["structure_path"] = os.path.join(
            structure_dir, example["structure_path"]
        )
    return example


def _add_ground_truth_go_terms(example):
    """Helper function to add ground_truth_go_terms field from go_ids."""
    go_ids = example.get("go_ids", [])
    if go_ids is None:
        go_ids = []
    if isinstance(go_ids, str):
        go_ids = ast.literal_eval(go_ids) if go_ids else []
    example["ground_truth_go_terms"] = " ".join(go_ids)
    return example


def _add_uniprot_summary(example):
    """Helper function to add UniProt summary to final answer and return it."""
    final_answer_lines = example.get("final_answer", "").split("\n")
    protein_function = example.get("protein_function")
    if protein_function is None:
        protein_function = ""
    final_answer = final_answer_lines[0] + "\n- UniProt Summary: " + protein_function.strip() + "\n" + "\n".join(final_answer_lines[1:])
    return final_answer


def _format_reasoning_prompt(example, go_gpt_predictions_column=None, interpro_in_prompt=False, ppi_in_prompt=False, include_ground_truth_in_final_answer=False, add_uniprot_summary=False, is_swissprot=False, ask_all_go_aspects=False):
    """Format reasoning data into prompt structure.
    
    Args:
        example: Reasoning dataset example with reasoning and final_answer fields
        go_gpt_predictions_column: Optional column name containing pre-computed GO-GPT predictions (e.g., 'go_pred')
        interpro_in_prompt: Whether to include InterPro data in user prompt
        ppi_in_prompt: Whether to include PPI data in user prompt
        include_ground_truth_in_final_answer: Whether to append ground truth GO terms to final answer
        add_uniprot_summary: Whether to add UniProt summary to final answer and use UniProt template
        is_swissprot: Whether to use dynamic SwissProt template that mentions only available data
        
    Returns:
        Example with prompt field added
    """
    organism = example.get("organism", "Unknown")
    
    # Prepare InterPro data
    interpro_data = ""
    if interpro_in_prompt and example.get("interpro_formatted"):
        interpro_data = example["interpro_formatted"]
    
    # Prepare PPI data
    ppi_data = ""
    if ppi_in_prompt and example.get("ppi_formatted"):
        ppi_data = example["ppi_formatted"]
    
    # Prepare GO predictions from pre-computed column
    go_speculations = ""
    if go_gpt_predictions_column and example.get(go_gpt_predictions_column):
        go_speculations = example[go_gpt_predictions_column]
    
    # Build GO aspects suffix based on available data or ask for all 3 if flag is set
    if ask_all_go_aspects:
        go_aspects_suffix = " and focus more on its Molecular Function, Biological Process, Cellular Component."
    else:
        go_aspects = []
        if example.get("go_mf"):
            go_aspects.append("Molecular Function")
        if example.get("go_cc"):
            go_aspects.append("Cellular Component")
        if example.get("go_bp"):
            go_aspects.append("Biological Process")
        go_aspects_suffix = f" and focus more on its {', '.join(go_aspects)}." if go_aspects else "."
    
    # Prepare UniProt summary suffix and assistant answer
    protein_function = example.get("protein_function", "")
    uniprot_summary = " Summarize in UniProt format."
    assistant_answer = _add_uniprot_summary(example) if add_uniprot_summary else (example["final_answer"] if "final_answer" in example else "")
    
    # Handle SwissProt template with dynamic system prompt
    if is_swissprot:
        # Build GO terms text based on available data
        go_terms_parts = []
        if example.get("go_mf"):
            go_terms_parts.append("molecular function")
        if example.get("go_bp"):
            go_terms_parts.append("biological process")
        if example.get("go_cc"):
            go_terms_parts.append("cellular component")
        
        go_terms_text = f", Gene Ontology (GO) terms regarding {', '.join(go_terms_parts)}" if go_terms_parts else ""
        
        # Build PPI text based on available data
        ppi_text = ", protein-protein interactions (PPI)" if example.get("ppi_formatted") else ""
        
        # Format system prompt with dynamic content
        system_prompt = CAFA5_REASONING_TEMPLATE_SWISSPROT["system_prompt"].format(
            go_terms_text=go_terms_text,
            ppi_text=ppi_text
        )
        
        prompt_dict = {
            "system": system_prompt,
            "user": CAFA5_REASONING_TEMPLATE_SWISSPROT["user_prompt"].format(organism=organism),
            "assistant_reasoning": example["reasoning"] if "reasoning" in example else "",
            "assistant_answer": assistant_answer,
        }
        
        return {
            **example,
            "prompt": prompt_dict,
        }
    
    # Choose template based on available context
    if ppi_in_prompt and (interpro_data or go_speculations):
        # Use PPI context template when PPI is requested and we have some context
        if add_uniprot_summary:
            prompt_dict = {
                "system": CAFA5_REASONING_TEMPLATE_WITH_CONTEXT_PPI_UNIPROT["system_prompt"],
                "user": CAFA5_REASONING_TEMPLATE_WITH_CONTEXT_PPI_UNIPROT["user_prompt"].format(
                    organism=organism,
                    interpro_data=interpro_data if interpro_data else "None",
                    ppi_data=ppi_data if ppi_data else "None",
                    go_speculations=go_speculations if go_speculations else "None",
                    go_aspects_suffix=go_aspects_suffix,
                    uniprot_summary=uniprot_summary
                ),
                "assistant_reasoning": example["reasoning"] if "reasoning" in example else "",
                "assistant_answer": assistant_answer,
            }
        else:
            prompt_dict = {
                "system": CAFA5_REASONING_TEMPLATE_WITH_CONTEXT_PPI["system_prompt"],
                "user": CAFA5_REASONING_TEMPLATE_WITH_CONTEXT_PPI["user_prompt"].format(
                    organism=organism,
                    interpro_data=interpro_data if interpro_data else "None",
                    ppi_data=ppi_data if ppi_data else "None",
                    go_speculations=go_speculations if go_speculations else "None",
                    go_aspects_suffix=go_aspects_suffix
                ),
                "assistant_reasoning": example["reasoning"] if "reasoning" in example else "",
                "assistant_answer": assistant_answer,
            }
    elif interpro_data or go_speculations:
        # Use context template without PPI
        prompt_dict = {
            "system": CAFA5_REASONING_TEMPLATE_WITH_CONTEXT["system_prompt"],
            "user": CAFA5_REASONING_TEMPLATE_WITH_CONTEXT["user_prompt"].format(
                organism=organism,
                interpro_data=interpro_data if interpro_data else "",
                go_speculations=go_speculations if go_speculations else ""
            ),
            "assistant_reasoning": example["reasoning"] if "reasoning" in example else "",
            "assistant_answer": assistant_answer,
        }
    else:
        # Use standard template (no additional context)
        prompt_dict = {
            "system": CAFA5_REASONING_TEMPLATE["system_prompt"],
            "user": CAFA5_REASONING_TEMPLATE["user_prompt"].format(organism=organism),
            "assistant_reasoning": example["reasoning"] if "reasoning" in example else "",
            "assistant_answer": assistant_answer,
        }
    
    # Optionally append ground truth GO terms to final answer
    if include_ground_truth_in_final_answer:
        ground_truth_parts = []
        
        # Process each aspect
        for aspect in ["MF", "BP", "CC"]:
            column_name = ASPECT_TO_COLUMN[aspect]
            go_terms = example.get(column_name, [])
            
            # Handle both string and list inputs
            if isinstance(go_terms, str):
                go_terms = ast.literal_eval(go_terms) if go_terms else []
            elif go_terms is None:
                go_terms = []
            
            # Format terms if present
            if go_terms and len(go_terms) > 0:
                ground_truth_parts.append(f"{aspect}: {', '.join(go_terms)}")
            else:
                ground_truth_parts.append(f"{aspect}: None")
        
        # Append to assistant answer if we have any ground truth
        if ground_truth_parts:
            confident_section = "\n\n- Confident GO Terms:\n" + "\n".join(ground_truth_parts)
            prompt_dict["assistant_answer"] = prompt_dict["assistant_answer"] + confident_section
    
    # Set go_aspect: when ask_all_go_aspects=True use "all", otherwise keep existing or set "all"
    go_aspect = example.get("go_aspect")
    if go_aspect is None:
        go_aspect = "all"  # Default to "all" when not split by aspect

    return {
        **example,
        "prompt": prompt_dict,
        "go_aspect": go_aspect,
    }


def _generate_and_flatten_split_examples(
    batch,
    interpro_metadata,
    include_go_defs,
    interpro_in_prompt,
    ppi_in_prompt,
    predict_interpro,
):
    """Single-pass function to generate and flatten split GO aspect examples.

    This function combines the generation and flattening operations into a single pass,
    eliminating the need for intermediate data structures and reducing memory usage.

    Args:
        batch: Dictionary containing batch data
        interpro_metadata: InterPro metadata for protein domains
        include_go_defs: Whether to include GO term definitions
        interpro_in_prompt: Whether to include InterPro data in user prompt
        ppi_in_prompt: Whether to include PPI data in user prompt
        predict_interpro: Whether to ask model to predict InterPro terms

    Returns:
        Dictionary with flattened examples where each protein is expanded into multiple GO aspect examples
    """
    # Initialize the output batch structure
    new_batch = {key: [] for key in batch.keys()}
    new_batch["prompt"] = []
    new_batch["go_aspect"] = []

    # Get batch size from any key (they should all have the same length)
    batch_size = len(batch[next(iter(batch))])

    # Process each example in the batch
    for i in range(batch_size):
        # Extract single example from batch
        example = {key: values[i] for key, values in batch.items()}

        # Generate split examples for this protein
        split_examples = generate_cafa5_examples_split_aspects(
            example,
            prompt_template=None,
            interpro_metadata=interpro_metadata,
            include_go_defs=include_go_defs,
            interpro_in_prompt=interpro_in_prompt,
            ppi_in_prompt=ppi_in_prompt,
            predict_interpro=predict_interpro,
        )

        # Directly flatten the split examples into the output batch
        for split_example in split_examples:
            new_batch["prompt"].append(split_example)
            new_batch["go_aspect"].append(split_example.get("go_aspect", ""))

            # Copy all other fields from the original example
            for key, value in example.items():
                if key not in ["prompt", "go_aspect"] and key in new_batch:
                    new_batch[key].append(value)
            
            # Ensure ground_truth_go_terms is included
            if "ground_truth_go_terms" not in example:
                new_batch["ground_truth_go_terms"].append("")
            else:
                new_batch["ground_truth_go_terms"].append(example["ground_truth_go_terms"])

    return new_batch


def _process_dataset_split(
    dataset,
    interpro_metadata,
    include_go_defs,
    split_go_aspects,
    max_assistant_reasoning_length,
    max_length,
    return_as_chat_template,
    structure_dir,
    include_protein_function_summary,
    num_proc,
    interpro_in_prompt,
    ppi_in_prompt=False,
    predict_interpro=False,
    debug=False,
    reasoning_dataset_name=None,
    go_gpt_predictions_column=None,
    include_ground_truth_in_final_answer=False,
    add_uniprot_summary=False,
    is_swissprot=False,
    ask_all_go_aspects=False,
):
    """Process a single dataset split with all transformations."""
    # For testing, limit to 50 datapoints
    if debug and len(dataset) > 50:
        print(f"Debug mode: limiting split from {len(dataset)} to 50 samples")
        dataset = dataset.select(range(50))

    # Add ground truth GO terms
    dataset = dataset.map(
        _add_ground_truth_go_terms,
        num_proc=num_proc,
        desc="Adding ground truth GO terms",
    )
    
    # Remove protein function summaries if requested
    if not include_protein_function_summary:
        dataset = dataset.map(
            lambda x: {"protein_function": None, **x},
            num_proc=num_proc,
            desc="Removing protein function summaries",
        )

    # Format for Protein-LLM
    if split_go_aspects:
        # Single-pass generation and flattening of split GO aspect examples
        dataset = dataset.map(
            _generate_and_flatten_split_examples,
            batched=True,
            num_proc=num_proc,
            desc="Generating and flattening split GO aspect examples",
            fn_kwargs={
                "interpro_metadata": interpro_metadata,
                "include_go_defs": include_go_defs,
                "interpro_in_prompt": interpro_in_prompt,
                "ppi_in_prompt": ppi_in_prompt,
                "predict_interpro": predict_interpro,
            },
        )
    else:
        # Original logic - single example per protein
        if reasoning_dataset_name:
            dataset = dataset.map(
                _format_reasoning_prompt,
                num_proc=num_proc,
                desc="Formatting reasoning prompts",
                fn_kwargs={
                    "go_gpt_predictions_column": go_gpt_predictions_column,
                    "interpro_in_prompt": interpro_in_prompt,
                    "ppi_in_prompt": ppi_in_prompt,
                    "include_ground_truth_in_final_answer": include_ground_truth_in_final_answer,
                    "add_uniprot_summary": add_uniprot_summary,
                    "is_swissprot": is_swissprot,
                    "ask_all_go_aspects": ask_all_go_aspects,
                },
            )
        else:
            # Standard generation
            dataset = dataset.map(
                lambda example: {
                    **example,
                    "prompt": generate_cafa5_example(
                        example,
                        prompt_template=None,
                        interpro_metadata=interpro_metadata,
                        include_go_defs=include_go_defs,
                        interpro_in_prompt=interpro_in_prompt,
                        predict_interpro=predict_interpro,
                    ),
                },
                num_proc=num_proc,
                desc="Formatting CAFA5 examples",
            )

    # Filter by assistant reasoning length
    # dataset = dataset.filter(
    #     lambda x: len(x["prompt"]["assistant_reasoning"].split())
    #     < max_assistant_reasoning_length,
    #     num_proc=num_proc,
    #     desc="Filtering by assistant reasoning length",
    # )

    # Truncate protein sequences
    dataset = dataset.map(
        lambda x: truncate_protein(x, max_length),
        num_proc=num_proc,
        desc="Truncating sequences",
    )

    # Set chat template
    if return_as_chat_template:
        dataset = dataset.map(
            lambda x: format_cafa5_for_protein_llm(x),
            num_proc=num_proc,
            desc="Setting chat template",
        )

    # Set structure paths
    if structure_dir is not None:
        dataset = dataset.map(
            lambda x: _add_structure_prefix(x, structure_dir),
            num_proc=num_proc,
            desc="Adding structure paths",
        )

    return dataset


def load_cafa5_dataset(
    dataset: str = "wanglab/cafa5",
    dataset_name: str = "experiment_data",
    dataset_subset: str = None,
    max_length: int = 2048,
    val_split_ratio: float = 0.1,
    seed: int = 23,
    cache_dir: str = "cache_dir",
    structure_dir: str = None,
    num_proc: int = None,
    return_as_chat_template: bool = False,
    debug: bool = False,
    include_go_defs: bool = True,
    interpro_dataset_name: str = "interpro_metadata",
    max_assistant_reasoning_length: int = 10000,
    split_go_aspects: bool = False,
    include_protein_function_summary: bool = True,
    interpro_in_prompt: bool = False,
    ppi_in_prompt: bool = False,
    predict_interpro: bool = False,
    reasoning_dataset_name: str = None,
    min_go_mf_freq: int = 1,
    min_go_bp_freq: int = 1,
    min_go_cc_freq: int = 1,
    apply_go_filtering_to_val_test: bool = False,
    go_gpt_predictions_column: str = None,
    include_ground_truth_in_final_answer: bool = True,
    add_uniprot_summary: bool = False,
    is_swissprot: bool = False,
    ask_all_go_aspects: bool = False,
):
    """
    Load CAFA5 dataset, format it into the Protein-LLM format, and split into train/val sets.

    Args:
        dataset: Base dataset name to load
        dataset_subset: Subset of the dataset to load
        max_length: Maximum length for protein sequences
        val_split_ratio: Ratio of training data to use for validation
        seed: Random seed for reproducible splits
        cache_dir: Directory to cache the dataset
        structure_dir: Directory containing protein structure files
        num_proc: Number of CPU cores to use (None = auto-detect)
        return_as_chat_template: Whether to return the dataset as a chat template
        debug: Whether to limit the dataset to 100 datapoints for testing
        include_go_defs: Whether to include GO term definitions in the output
        interpro_dataset_name: Name of the InterPro dataset to load
        max_assistant_reasoning_length: Maximum length for assistant reasoning
        split_go_aspects: Whether to create separate examples for each GO aspect
        include_protein_function_summary: Whether to include protein function summaries in the data
        interpro_in_prompt: Whether to include InterPro data in user prompt instead of generation
        predict_interpro: Whether to ask model to predict InterPro terms
        reasoning_dataset_name: Optional config name for reasoning traces dataset (e.g., "experiment_data_reasoning")
        min_go_mf_freq: Minimum frequency for Molecular Function GO terms to be included (default=1)
        min_go_bp_freq: Minimum frequency for Biological Process GO terms to be included (default=1)
        min_go_cc_freq: Minimum frequency for Cellular Component GO terms to be included (default=1)
        apply_go_filtering_to_val_test: Whether to apply GO frequency filtering to validation/test sets (default=False).
                                         For pre-split datasets: Controls whether val/test are filtered (False=train only, True=all splits).
                                         For non-pre-split datasets: Must be True if any min_go_*_freq > 1, otherwise raises ValueError.
        go_gpt_predictions_column: Optional column name containing pre-computed GO-GPT predictions (e.g., "go_pred").
                                   If the dataset has this column, predictions will be included in reasoning prompts.
                                   Only works with reasoning_dataset_name.
        include_ground_truth_in_final_answer: When using reasoning data, if True, appends ground truth GO terms
                                              to the final answer in a "Confident GO Terms:" section (default=True).
                                              Only works with reasoning_dataset_name.
        add_uniprot_summary: When using reasoning data, if True, adds UniProt summary to final answer and uses
                            UniProt template with appropriate suffix (default=False).
                            Only works with reasoning_dataset_name.
        is_swissprot: When using reasoning data, if True, uses dynamic SwissProt template that mentions only
                     available data (GO terms and PPI) in the system prompt (default=False).
                     Only works with reasoning_dataset_name.
        ask_all_go_aspects: When using reasoning data, if True, always asks for all 3 GO aspects
                           (Molecular Function, Biological Process, Cellular Component) regardless
                           of what ground truth data is available (default=False). Useful for evaluation.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) where test_dataset is the original test split
    """
    try:
        # Auto-detect number of processes if not specified
        if num_proc is None:
            num_proc = os.cpu_count() - 2

        print(f"Using {num_proc} CPU cores for parallel processing")

        # Load reasoning dataset as primary dataset if provided
        if reasoning_dataset_name:
            # Reasoning data works with non-split GO aspects (comprehensive reasoning per protein)
            if split_go_aspects:
                raise ValueError(
                    "reasoning_dataset_name requires split_go_aspects=False since reasoning data contains comprehensive analysis for all GO aspects together. "
                    "Set split_go_aspects=False when using reasoning data."
                )

            print(f"Using reasoning dataset '{reasoning_dataset_name}' as primary dataset...")
            try:
                datasets = load_dataset(dataset, name=reasoning_dataset_name, cache_dir=cache_dir)
                # Report on available splits
                available_splits = list(datasets.keys())
                total_examples = sum(len(datasets[split]) for split in available_splits)
                print(f"✓ Loaded reasoning dataset with {total_examples} examples across splits: {available_splits}")
            except Exception as e:
                print(f"Error loading reasoning dataset: {e}")
                raise e
        else:
            # Load regular dataset
            if dataset_subset:
                datasets = load_dataset(
                    dataset,
                    name=dataset_name,
                    dataset_subset=dataset_subset,
                    cache_dir=cache_dir,
                )
            else:
                datasets = load_dataset(dataset, name=dataset_name, cache_dir=cache_dir)

        # Check if this is a pre-split dataset (has validation split)
        if "validation" in datasets or "test" in datasets:
            print("Detected pre-split dataset - using existing splits")
            using_presplit = True
        else:
            print("Using single training set - will split during preprocessing")
            using_presplit = False
            full_train_dataset = datasets["train"]

            # For testing, limit to 50 datapoints
            if debug:
                print("Limiting to 50 datapoints for testing...")
                full_train_dataset = full_train_dataset.select(
                    range(min(50, len(full_train_dataset)))
                )

        # Load InterPro metadata only if specified (needed for both paths)
        interpro_metadata = None
        if interpro_dataset_name is not None:
            print(f"Loading InterPro metadata with name='{interpro_dataset_name}'...")
            ds_interpro = load_dataset(
                dataset, name=interpro_dataset_name, cache_dir=cache_dir
            )
            interpro_metadata = ds_interpro["metadata"].to_pandas()

            # Validate interpro_metadata
            required_columns = ["interpro_id", "entry_name", "type"]
            missing_columns = [
                col for col in required_columns if col not in interpro_metadata.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"interpro_metadata is missing required columns: {missing_columns}"
                )

            print(f"Loaded InterPro metadata with {len(interpro_metadata)} entries")
        else:
            print("No InterPro metadata requested, proceeding with GO terms only")

        # Apply GO term frequency filtering if any threshold > 1
        min_frequencies = {
            "MF": min_go_mf_freq,
            "BP": min_go_bp_freq,
            "CC": min_go_cc_freq,
        }
        apply_filtering = any(freq > 1 for freq in min_frequencies.values())

        if apply_filtering:
            print(
                f"Applying GO term frequency filtering with thresholds: {min_frequencies}"
            )

            if using_presplit:
                # For pre-split datasets, we need to count frequencies across all splits
                # Use first available split to determine frequency thresholds
                available_splits = list(datasets.keys())
                reference_split = available_splits[0]  # Use first available split
                print(f"Counting GO term frequencies in '{reference_split}' split...")
                go_frequencies = _count_go_term_frequencies(datasets[reference_split])

                # Apply filtering to each available split
                for split_name in available_splits:
                    if split_name == "train" or apply_go_filtering_to_val_test:
                        print(f"Filtering {split_name} split...")
                        datasets[split_name] = _filter_go_terms_by_frequency(
                            datasets[split_name], go_frequencies, min_frequencies, num_proc
                        )

                if not apply_go_filtering_to_val_test:
                    print(
                        "Skipping validation/test filtering (apply_go_filtering_to_val_test=False)"
                    )
                    print(
                        "Validation and test sets will retain all GO terms for comprehensive evaluation"
                    )
            else:
                # For single dataset, check if selective filtering is requested
                if not apply_go_filtering_to_val_test and apply_filtering:
                    raise ValueError(
                        "apply_go_filtering_to_val_test=False is not supported for non-pre-split datasets. "
                        "Selective filtering (train-only) requires pre-split datasets. "
                        "Either use a pre-split dataset, set apply_go_filtering_to_val_test=True, "
                        "or disable filtering entirely by setting all min_go_*_freq=1."
                    )

                # For single dataset, apply filtering to entire dataset before splitting
                print("Counting GO term frequencies in full training dataset...")
                go_frequencies = _count_go_term_frequencies(full_train_dataset)

                print("Filtering full training dataset...")
                full_train_dataset = _filter_go_terms_by_frequency(
                    full_train_dataset, go_frequencies, min_frequencies, num_proc
                )
        else:
            print("No GO term frequency filtering applied (all thresholds = 1)")

        # Process based on whether we have pre-split data or not
        if using_presplit:
            # Process pre-split datasets
            print("Processing pre-split datasets...")
            available_splits = list(datasets.keys())
            print(f"Available splits: {available_splits}")

            # Helper to process a split if it exists, otherwise return None
            def process_split_if_exists(split_name):
                if split_name in datasets:
                    return _process_dataset_split(
                        datasets[split_name],
                        interpro_metadata=interpro_metadata,
                        include_go_defs=include_go_defs,
                        split_go_aspects=split_go_aspects,
                        max_assistant_reasoning_length=max_assistant_reasoning_length,
                        max_length=max_length,
                        return_as_chat_template=return_as_chat_template,
                        structure_dir=structure_dir,
                        include_protein_function_summary=include_protein_function_summary,
                        num_proc=num_proc,
                        interpro_in_prompt=interpro_in_prompt,
                        ppi_in_prompt=ppi_in_prompt,
                        predict_interpro=predict_interpro,
                        debug=debug,
                        reasoning_dataset_name=reasoning_dataset_name,
                        go_gpt_predictions_column=go_gpt_predictions_column,
                        include_ground_truth_in_final_answer=include_ground_truth_in_final_answer,
                        add_uniprot_summary=add_uniprot_summary,
                        is_swissprot=is_swissprot,
                        ask_all_go_aspects=ask_all_go_aspects,
                    )
                return None

            # Process each available split
            train_dataset = process_split_if_exists("train")
            val_dataset = process_split_if_exists("validation")
            test_dataset = process_split_if_exists("test")

            # Fall back to available splits for missing ones
            # Priority: use test for evaluation if val is missing, use any available for train
            first_available = train_dataset or val_dataset or test_dataset
            if train_dataset is None:
                train_dataset = first_available
                print("  - Note: No 'train' split, using fallback")
            if val_dataset is None:
                val_dataset = test_dataset or train_dataset
                print("  - Note: No 'validation' split, using fallback")
            if test_dataset is None:
                test_dataset = val_dataset or train_dataset
                print("  - Note: No 'test' split, using fallback")

            print("Pre-split dataset processed successfully:")
            print(f"  - Training: {len(train_dataset) if train_dataset else 0} samples")
            print(f"  - Validation: {len(val_dataset) if val_dataset else 0} samples")
            print(f"  - Test: {len(test_dataset) if test_dataset else 0} samples")

        else:
            # Original behavior - process single dataset and split
            print("Processing single dataset for splitting...")

            # Add ground truth GO terms
            full_train_dataset = full_train_dataset.map(
                _add_ground_truth_go_terms,
                num_proc=num_proc,
                desc="Adding ground truth GO terms",
            )

            # Remove protein function summaries if requested
            if not include_protein_function_summary:
                print("Removing protein function summaries from dataset...")
                full_train_dataset = full_train_dataset.map(
                    lambda x: {"protein_function": None, **x},
                    num_proc=num_proc,
                    desc="Removing protein function summaries",
                )

            # Format for Protein-LLM
            if split_go_aspects:
                # Single-pass generation and flattening of split GO aspect examples
                full_train_dataset = full_train_dataset.map(
                    _generate_and_flatten_split_examples,
                    batched=True,
                    num_proc=num_proc,
                    desc="Generating and flattening split GO aspect examples",
                    fn_kwargs={
                        "interpro_metadata": interpro_metadata,
                        "include_go_defs": include_go_defs,
                        "interpro_in_prompt": interpro_in_prompt,
                        "ppi_in_prompt": ppi_in_prompt,
                        "predict_interpro": predict_interpro,
                    },
                )

            else:
                # Original logic - single example per protein
                if reasoning_dataset_name:
                    full_train_dataset = full_train_dataset.map(
                        _format_reasoning_prompt,
                        num_proc=num_proc,
                        desc="Formatting reasoning prompts",
                        fn_kwargs={
                            "go_gpt_predictions_column": go_gpt_predictions_column,
                            "interpro_in_prompt": interpro_in_prompt,
                            "ppi_in_prompt": ppi_in_prompt,
                            "include_ground_truth_in_final_answer": include_ground_truth_in_final_answer,
                            "add_uniprot_summary": add_uniprot_summary,
                            "is_swissprot": is_swissprot,
                            "ask_all_go_aspects": ask_all_go_aspects,
                        },
                    )
                else:
                    # Standard generation
                    full_train_dataset = full_train_dataset.map(
                        lambda example: {
                            **example,
                            "prompt": generate_cafa5_example(
                                example,
                                prompt_template=None,  # automatically determined based on example
                                interpro_metadata=interpro_metadata,
                                include_go_defs=include_go_defs,
                                interpro_in_prompt=interpro_in_prompt,
                                predict_interpro=predict_interpro,
                            ),
                        },
                        num_proc=num_proc,
                        desc="Formatting CAFA5 examples",
                    )

            # Drop rows with null values
            # full_train_dataset = full_train_dataset.filter(
            #     lambda x: len(x["prompt"]["assistant_reasoning"].split())
            #     < max_assistant_reasoning_length,
            #     num_proc=num_proc,
            #     desc="Filtering by assistant reasoning length",
            # )

            # Truncate protein sequences
            full_train_dataset = full_train_dataset.map(
                lambda x: truncate_protein(x, max_length),
                num_proc=num_proc,
                desc="Truncating sequences",
            )

            # Set chat template
            if return_as_chat_template:
                full_train_dataset = full_train_dataset.map(
                    lambda x: format_cafa5_for_protein_llm(x),
                    num_proc=num_proc,
                    desc="Setting chat template",
                )

            # Set structure paths
            if structure_dir is None:
                print(
                    "No structure directory provided, skipping structure path setting."
                )
            else:
                print(f"Setting structure paths using directory: {structure_dir}")
                full_train_dataset = full_train_dataset.map(
                    lambda x: _add_structure_prefix(x, structure_dir)
                )

            # Shuffle the dataset before splitting
            print(f"Shuffling dataset with seed={seed}")
            full_train_dataset = full_train_dataset.shuffle(seed=seed)

            # Calculate split sizes
            total_train_size = len(full_train_dataset)
            val_size = int(total_train_size * val_split_ratio)
            train_size = total_train_size - val_size

            # Create train/val split with seed
            train_val_split = full_train_dataset.train_test_split(
                test_size=val_size, seed=seed
            )
            train_dataset = train_val_split["train"]
            val_dataset = train_val_split["test"]

            # Use the same validation set as test for now (since we only have train from CAFA5)
            test_dataset = val_dataset

            print("CAFA5 Dataset loaded and split successfully:")
            print(f"  - Total original train: {total_train_size} samples")
            print(f"  - Training: {len(train_dataset)} samples ({train_size})")
            print(f"  - Validation: {len(val_dataset)} samples ({val_size})")
            print(f"  - Test: {len(test_dataset)} samples (same as validation)")

        return train_dataset, val_dataset, test_dataset

    except Exception as e:
        print(f"Failed to load CAFA5 dataset: {e}")
        print("Returning empty datasets")
        traceback.print_exc()
        return [], [], []
