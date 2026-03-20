import pandas as pd
import re
import json
import os
import sys
import ast
import shutil
import random
from pathlib import Path
from typing import Set, List, Tuple, Dict
from tqdm import tqdm
from cafaeval.evaluation import cafa_eval
from colorama import init, Fore, Style
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Add BioReason2 to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bioreason2.utils.argparse_utils import str2bool

# GO aspect classification constants
NAMESPACE_TO_ASPECT = {
    'molecular_function': 'MF',
    'biological_process': 'BP', 
    'cellular_component': 'CC'
}



def extract_go_terms(text: str) -> Set[str]:
    """
    Extract all GO terms from text in format GO:XXXXXXX.
    Removes duplicates by returning a set.
    """
    # Find all GO terms in format GO:XXXXXXX (7 digits)
    go_pattern = r"GO:\d{7}"
    go_terms = set(re.findall(go_pattern, text))
    return go_terms


def extract_reasoning_ground_truth(sample: Dict) -> Tuple[Set[str], Set[str]]:
    """
    Extract ground truth GO terms from reasoning data columns.
    
    Args:
        sample: Dictionary containing go_bp, go_mf, go_cc fields (lists of GO terms)
        
    Returns:
        Tuple of (all_gt_terms, present_aspects) where:
        - all_gt_terms: Set of all GO terms from present aspects
        - present_aspects: Set of aspect codes that have non-empty ground truth
    """
    all_gt_terms = set()
    present_aspects = set()
    
    gt_columns = {
        "BP": "go_bp",
        "MF": "go_mf", 
        "CC": "go_cc"
    }
    
    for aspect, column in gt_columns.items():

        gt_data = sample.get(column, [])
        if isinstance(gt_data, str):
            gt_data = ast.literal_eval(gt_data) if gt_data else []
        elif gt_data is None:
            gt_data = []
        
        # gt_data is a list of GO terms
        if gt_data:
            all_gt_terms.update(gt_data)
            present_aspects.add(aspect)
    
    return all_gt_terms, present_aspects


def classify_go_term_by_aspect(go_term: str, go_dag) -> str:
    """
    Classify a GO term into its aspect (MF/BP/CC) using the GO ontology namespace.
    
    Args:
        go_term: GO term ID (e.g., "GO:0008150")
        go_dag: GO DAG object from goatools
        
    Returns:
        Aspect code ("MF", "BP", "CC") or None if not found
    """
    if not go_dag or go_term not in go_dag:
        return None
    
    # Use the namespace attribute for efficient classification
    namespace = go_dag[go_term].namespace
    
    return NAMESPACE_TO_ASPECT.get(namespace, None)


def filter_predictions_by_aspects(predicted_terms: Set[str], present_aspects: Set[str], go_dag) -> Set[str]:
    """
    Filter predicted GO terms to only include those belonging to aspects with ground truth.
    
    Args:
        predicted_terms: Set of predicted GO term IDs
        present_aspects: Set of aspect codes that have ground truth ("MF", "BP", "CC")
        go_dag: GO DAG object for classification
        
    Returns:
        Filtered set of GO terms belonging to present aspects
    """
    if not go_dag:
        # Fallback: return all predictions if no GO DAG available
        return predicted_terms
    
    filtered_terms = set()
    
    for go_term in predicted_terms:
        aspect = classify_go_term_by_aspect(go_term, go_dag)
        if aspect and aspect in present_aspects:
            filtered_terms.add(go_term)
    
    return filtered_terms


def parse_prediction_format(text: str, final_answer_only: bool = False) -> Set[str]:
    """
    Extract GO terms from prediction text.

    Args:
        text: The prediction text to parse
        final_answer_only: If True, split on </think> and only extract GO terms after it.
                          If False, extract from entire text.
    """
    if final_answer_only and "</think>" in text:
        # Split on </think> and take everything after
        text = text.split("</think>")[-1].strip()
    
    return extract_go_terms(text)


def evaluate_single_prediction(predicted_terms: Set[str], gt_terms: Set[str]) -> float:
    """
    Calculate F1 score for a single prediction.
    
    Args:
        predicted_terms: Set of predicted GO terms
        gt_terms: Set of ground truth GO terms
        
    Returns:
        F1 score (0.0 if no terms or division by zero)
    """
    if not predicted_terms or not gt_terms:
        return 0.0
    
    # Calculate precision, recall, and F1
    true_positives = len(predicted_terms & gt_terms)
    
    if true_positives == 0:
        return 0.0
    
    precision = true_positives / len(predicted_terms)
    recall = true_positives / len(gt_terms)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def select_best_from_k_samples(
    k_samples: List[Dict], 
    gt_terms: Set[str],
    final_answer_only: bool = False
) -> Dict:
    """
    Evaluate k samples in parallel and return the one with highest F1 score.
    
    Args:
        k_samples: List of sample dictionaries for the same protein
        gt_terms: Ground truth GO terms
        final_answer_only: If True, extract predictions only after </think> tag
        
    Returns:
        Best sample dictionary (highest F1 score)
    """
    if len(k_samples) == 1:
        return k_samples[0]
    
    def evaluate_sample(sample):
        generated_response = sample.get("generated_response", "")
        predicted_terms = parse_prediction_format(
            generated_response,
            final_answer_only=final_answer_only
        )
        f1_score = evaluate_single_prediction(predicted_terms, gt_terms)
        return f1_score, sample
    
    # Evaluate all k samples in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(evaluate_sample, k_samples))
    
    # Select sample with highest F1 score
    best_score, best_sample = max(results, key=lambda x: x[0])
    
    # Select sample randomly instead of best F1 score
    # best_score, best_sample = random.choice(results)
    
    return best_sample


def load_json_files_from_directory(directory: str) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Load all JSON files from a single chunk directory and group by (protein_id, go_aspect).
    Handles both old format (without _k suffix) and new format (with _k{i:02d} suffix).
    Automatically handles any number of k samples per protein.
    
    Args:
        directory: Directory containing JSON files
        
    Returns:
        Dictionary mapping (protein_id, go_aspect) to list of k sample dictionaries
    """
    grouped_data = defaultdict(list)
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"{Fore.YELLOW}Warning: Directory {directory} does not exist{Style.RESET_ALL}")
        return grouped_data

    # Find all .json files, excluding errors.jsonl
    for json_file in dir_path.glob("*.json"):
        if json_file.name != "errors.jsonl":
            try:
                filename = json_file.name
                base_name = filename[:-5]  # Remove .json
                
                protein_id = None
                go_aspect_code = None
                
                # Check if this is new format with _k suffix: {protein_id}_{go_aspect_code}_k{i:02d}.json
                if "_k" in base_name and base_name.rsplit("_k", 1)[1].isdigit():
                    # New format with _k suffix
                    parts = base_name.rsplit("_k", 1)
                    if len(parts) == 2:
                        protein_aspect = parts[0]  # e.g., "protein123_MF"
                        k_idx = parts[1]  # e.g., "00", "01"
                        
                        # Extract protein_id and go_aspect_code from protein_aspect
                        if "_" in protein_aspect:
                            aspect_parts = protein_aspect.rsplit("_", 1)
                            if len(aspect_parts) == 2:
                                protein_id, go_aspect_code = aspect_parts
                else:
                    # Old format without _k suffix: {protein_id}_{go_aspect_code}.json
                    if "_" in base_name:
                        aspect_parts = base_name.rsplit("_", 1)
                        if len(aspect_parts) == 2:
                            protein_id, go_aspect_code = aspect_parts
                
                # Load the file if we successfully parsed the filename
                if protein_id and go_aspect_code:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                grouped_data[(protein_id, go_aspect_code)].append(item)
                        else:
                            grouped_data[(protein_id, go_aspect_code)].append(data)
                            
            except Exception as e:
                print(f"{Fore.RED}Error loading {json_file}: {e}{Style.RESET_ALL}")

    return grouped_data


def process_json_data(
    base_dir: str, 
    reasoning_mode: bool = False, 
    final_answer_only: bool = False,
    go_dag=None
) -> Tuple[List[Tuple[str, Set[str]]], List[Tuple[str, Set[str]]]]:
    """
    Process JSON files from directory and extract predictions and ground truth.
    Automatically handles pass@k by selecting best sample when multiple k samples exist.
    
    Args:
        base_dir: Directory containing JSON files or chunk directories with JSON files
        reasoning_mode: If True, use go_bp/go_mf/go_cc for ground truth
        final_answer_only: If True, extract predictions only after </think> tag
        go_dag: GO DAG object (unused in current implementation, kept for compatibility)
    Returns: (predictions_list, ground_truth_list)
    """
    print(f"{Fore.CYAN}Loading and processing JSON data from {base_dir}...{Style.RESET_ALL}")
    
    if reasoning_mode:
        print(f"{Fore.YELLOW}Using reasoning evaluation mode with ground truth from go_bp/go_mf/go_cc{Style.RESET_ALL}")
    
    extraction_mode = "entire response" if not final_answer_only else "after </think> tag"
    print(f"{Fore.YELLOW}Extracting predictions from: {extraction_mode}{Style.RESET_ALL}")

    predictions = []
    ground_truth = []
    total_samples = 0
    successful_samples = 0
    aspect_stats = {"MF": 0, "BP": 0, "CC": 0}  # Track aspect presence

    base_path = Path(base_dir)

    # Load JSON files and group by (protein_id, go_aspect)
    grouped_data = load_json_files_from_directory(base_dir)

    # If no JSON files found directly, look for chunk directories
    if not grouped_data:
        chunk_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
        for chunk_dir in chunk_dirs:
            chunk_grouped = load_json_files_from_directory(str(chunk_dir))
            for key, samples in chunk_grouped.items():
                grouped_data[key].extend(samples)

    # Process grouped data
    total_proteins = len(grouped_data)
    
    # Count k samples per protein for reporting
    k_counts = [len(samples) for samples in grouped_data.values()]
    if k_counts:
        avg_k = sum(k_counts) / len(k_counts)
        max_k = max(k_counts)
        min_k = min(k_counts)
        if max_k > 1:
            print(f"{Fore.YELLOW}Pass@k detected: Found {total_proteins} unique proteins with {min_k}-{max_k} samples each (avg: {avg_k:.1f}){Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Selecting best scoring sample per protein{Style.RESET_ALL}")
        else:
            print(f"Found {total_proteins} unique proteins (single sample each)")

    for (protein_id, go_aspect_code), k_samples in tqdm(grouped_data.items(), desc="Processing proteins"):

        total_samples += 1
        
        # Filter successful samples
        successful_k_samples = [s for s in k_samples if s.get("success", False)]
        
        if not successful_k_samples:
            continue
        
        # Extract ground truth from first sample (same for all k)
        sample = successful_k_samples[0]
        
        if reasoning_mode:
            # Extract from go_bp, go_mf, go_cc columns
            gt_terms, present_aspects = extract_reasoning_ground_truth(sample)
            
            # Track aspect statistics
            for aspect in present_aspects:
                aspect_stats[aspect] += 1
        else:
            # Extract from ground_truth text field
            gt_text = sample.get("ground_truth", "")
            gt_terms = extract_go_terms(gt_text)
        
        # Select best sample from k samples if multiple exist
        if len(successful_k_samples) > 1:
            best_sample = select_best_from_k_samples(
                successful_k_samples,
                gt_terms,
                final_answer_only=final_answer_only
            )
        else:
            # Only one sample available
            best_sample = successful_k_samples[0]
        
        successful_samples += 1
        target_id = best_sample.get("protein_id", f"unknown_protein_{successful_samples}")
        
        # Extract predicted GO terms from best sample
        generated_response = best_sample.get("generated_response", "")
        predicted_terms = parse_prediction_format(
            generated_response, 
            final_answer_only=final_answer_only
        )
        
        # Only add if we have data
        if predicted_terms:
            predictions.append((target_id, predicted_terms))
        if gt_terms:
            ground_truth.append((target_id, gt_terms))

    # Count total annotations
    total_predicted_annotations = sum(len(terms) for _, terms in predictions)
    total_gt_annotations = sum(len(terms) for _, terms in ground_truth)

    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}PROCESSING SUMMARY{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"Total samples processed: {total_samples}")
    print(f"Successful samples: {successful_samples}")
    print(f"Failed samples: {total_samples - successful_samples}")
    
    print(f"\n{Fore.CYAN}PREDICTION STATISTICS{Style.RESET_ALL}")
    print(f"Proteins with predictions: {len(predictions)}")
    print(f"Total predicted annotations: {total_predicted_annotations}")
    
    print(f"\n{Fore.CYAN}GROUND TRUTH STATISTICS{Style.RESET_ALL}")
    print(f"Proteins with ground truth: {len(ground_truth)}")
    print(f"Total ground truth annotations: {total_gt_annotations}")
    
    if reasoning_mode:
        print("\nAspect presence in ground truth:")
        for aspect, count in aspect_stats.items():
            print(f"  {aspect}: {count} proteins")
    
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

    return predictions, ground_truth


def create_cafa_prediction_file(predictions: List[Tuple[str, Set[str]]], output_path: str):
    """Create CAFA-format prediction file: target_id, term_id, score"""
    print(f"Creating prediction file: {output_path}")

    with open(output_path, "w") as f:
        for target_id, go_terms in predictions:
            for go_term in go_terms:
                # Assign score=1.0 to all predicted GO terms
                f.write(f"{target_id}\t{go_term}\t1.0\n")


def create_cafa_ground_truth_file(ground_truth: List[Tuple[str, Set[str]]], output_path: str):
    """Create CAFA-format ground truth file: target_id, term_id"""
    print(f"Creating ground truth file: {output_path}")

    with open(output_path, "w") as f:
        for target_id, go_terms in ground_truth:
            for go_term in go_terms:
                f.write(f"{target_id}\t{go_term}\n")


def run_cafa_evaluation(
    ontology_path: str,
    predictions_dir: str,
    ground_truth_path: str,
    ia_file_path: str = None,
    n_cpu: int = 0,
    th_step: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run CAFA evaluation using cafaeval package.
    """
    print("Running CAFA evaluation...")
    print(f"Ontology: {ontology_path}")
    print(f"Predictions: {predictions_dir}")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Information Accretion file: {ia_file_path if ia_file_path else 'Not provided'}")
    print(f"Using {n_cpu if n_cpu > 0 else 'all available'} CPU cores for evaluation")

    # Run evaluation
    if ia_file_path and os.path.exists(ia_file_path):
        results = cafa_eval(
            ontology_path,
            predictions_dir,
            ground_truth_path,
            ia_file_path,
            th_step=th_step,
        )
    else:
        results = cafa_eval(ontology_path, predictions_dir, ground_truth_path, th_step=th_step)

    return results


def extract_metrics_summary(results) -> Dict[str, float]:
    """
    Extract F1 and weighted F1 scores for each subontology and compute overall means.
    """
    evaluation_df, best_scores_dict = results
    metrics = {}

    # Use any of the best score dataframes (they all have the same structure)
    df = best_scores_dict.get("f", best_scores_dict.get("f_w"))
    if df is None:
        print("ERROR: No valid dataframe found in best_scores_dict")
        return metrics

    # Reset index to access namespace column
    df = df.reset_index()

    # Extract metrics for each namespace
    for ns in df["ns"].unique():
        ns_data = df[df["ns"] == ns].iloc[0]  # Get the row for this namespace
        metrics[f"{ns}_f1"] = ns_data["f"]
        metrics[f"{ns}_weighted_f1"] = ns_data["f_w"]

    # Compute overall means
    f1_values = [metrics[f"{ns}_f1"] for ns in df["ns"].unique()]
    fw1_values = [metrics[f"{ns}_weighted_f1"] for ns in df["ns"].unique()]

    metrics["overall_mean_f1"] = sum(f1_values) / len(f1_values)
    metrics["overall_mean_weighted_f1"] = sum(fw1_values) / len(fw1_values)

    # Print summary
    print("\nF1 scores by aspect:")
    for ns in df["ns"].unique():
        print(f"  {ns}: {metrics[f'{ns}_f1']:.4f}")
    print(f"  Overall mean: {metrics['overall_mean_f1']:.4f}")

    print("\nWeighted F1 scores by aspect:")
    for ns in df["ns"].unique():
        print(f"  {ns}: {metrics[f'{ns}_weighted_f1']:.4f}")
    print(f"  Overall mean: {metrics['overall_mean_weighted_f1']:.4f}")

    return metrics


def print_results_summary(metrics: Dict[str, float]):
    """Print formatted results summary."""
    print("\n" + "=" * 60)
    print("CAFA EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    # Print F1 scores
    print("\nF1 SCORES:")
    print("-" * 30)
    for aspect in ["biological_process", "molecular_function", "cellular_component"]:
        if f"{aspect}_f1" in metrics:
            print(f"{aspect:25}: {metrics[f'{aspect}_f1']:.4f}")
    print(f"{'OVERALL AVERAGE':25}: {metrics['overall_mean_f1']:.4f}")

    # Print weighted F1 scores
    print("\nWEIGHTED F1 SCORES:")
    print("-" * 35)
    for aspect in ["biological_process", "molecular_function", "cellular_component"]:
        if f"{aspect}_weighted_f1" in metrics:
            print(f"{aspect:25}: {metrics[f'{aspect}_weighted_f1']:.4f}")
    print(f"{'OVERALL AVERAGE':25}: {metrics['overall_mean_weighted_f1']:.4f}")

    print("=" * 60)


def main():
    """Main pipeline for CAFA GO term evaluation."""

    # Initialize colorama for colored output
    init()

    parser = argparse.ArgumentParser(description="CAFA GO Term Evaluation Pipeline")
    parser.add_argument(
        "--input_dir",
        "-i",
        required=True,
        help="Input directory containing chunk directories with JSON files",
    )
    parser.add_argument(
        "--ontology",
        "-o",
        required=True,
        help="Path to GO ontology file (go-basic.obo)",
    )
    parser.add_argument(
        "--ia_file",
        "-a",
        required=True,
        help="Path to Information Accretion file (IA.txt)",
    )
    parser.add_argument(
        "--output_dir",
        "-d",
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=0,
        help="Number of CPU threads to use (0 = all available, default: 0)",
    )
    parser.add_argument(
        "--reasoning_mode",
        "-r",
        type=str2bool,
        default=False,
        help="Use reasoning mode: ground truth from go_bp/go_mf/go_cc columns instead of text (default: False)",
    )
    parser.add_argument(
        "--final_answer_only",
        "-f",
        type=str2bool,
        default=False,
        help="Extract predictions only after </think> tag. If False, extract from entire response (default: False)",
    )
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    GO_ONTOLOGY_PATH = args.ontology
    IA_FILE_PATH = args.ia_file
    OUTPUT_DIR = args.output_dir
    NUM_THREADS = args.threads
    REASONING_MODE = args.reasoning_mode
    FINAL_ANSWER_ONLY = args.final_answer_only

    print(f"{Fore.CYAN}Starting CAFA GO Term Evaluation Pipeline{Style.RESET_ALL}")
    print("-" * 40)

    # GO DAG not needed (CAFA handles aspect separation)
    go_dag = None

    # Create output directory
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    print(f"Creating new output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=False)

    # Step 1: Process JSON data from chunk directories
    predictions, ground_truth = process_json_data(
        INPUT_DIR, 
        REASONING_MODE, 
        FINAL_ANSWER_ONLY,
        go_dag
    )

    if not predictions:
        print(f"{Fore.RED}ERROR: No predictions found in the data!{Style.RESET_ALL}")
        return

    if not ground_truth:
        print(f"{Fore.RED}ERROR: No ground truth data found!{Style.RESET_ALL}")
        return

    # Step 2: Create CAFA format files
    predictions_dir = os.path.join(OUTPUT_DIR, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    prediction_file = os.path.join(predictions_dir, "llm_predictions.tsv")
    ground_truth_file = os.path.join(OUTPUT_DIR, "ground_truth.tsv")

    create_cafa_prediction_file(predictions, prediction_file)
    create_cafa_ground_truth_file(ground_truth, ground_truth_file)

    # Step 3: Run CAFA evaluation
    try:
        results = run_cafa_evaluation(
            GO_ONTOLOGY_PATH,
            predictions_dir,
            ground_truth_file,
            ia_file_path=IA_FILE_PATH,
            n_cpu=NUM_THREADS,
            th_step=0.99,  # since every predicted GO term score=1.0
        )

        # Step 4: Extract and display metrics
        metrics = extract_metrics_summary(results)
        print_results_summary(metrics)

        # Step 5: Save detailed results
        evaluation_df, best_scores_dict = results

        # Save main evaluation results
        evaluation_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results.tsv"), sep="\t")
        print(f"\nEvaluation results saved to: {os.path.join(OUTPUT_DIR, 'evaluation_results.tsv')}")

        # Save best scores for each metric
        for metric, df in best_scores_dict.items():
            metric_path = os.path.join(OUTPUT_DIR, f"best_{metric}.tsv")
            df.to_csv(metric_path, sep="\t")
        print(f"Best score files saved to: {OUTPUT_DIR}")

    except Exception as e:
        print(f"{Fore.RED}ERROR during evaluation: {e}{Style.RESET_ALL}")
        print("Please check that all input files exist and are in the correct format.")


if __name__ == "__main__":
    main()
