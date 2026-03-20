#!/usr/bin/env python
"""
GO-GPT API: Predict Gene Ontology (GO) terms from a protein sequence.

Usage (CLI):
    python gogpt_api.py --sequence <protein_sequence>
    python gogpt_api.py --sequence <protein_sequence> --organism "Organism name"
    
Usage (Import):
    from gogpt_api import load_predictor, predict_and_format
    
    predictor = load_predictor()
    result = predict_and_format(predictor, sequence, organism)

Example:
    python gogpt_api.py --sequence "MVLSPADKTN..."
    python gogpt_api.py --sequence "MVLSPADKTN..." --organism "Mus musculus"
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add paths for imports
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "gogpt" / "src"))
sys.path.insert(0, str(REPO_ROOT))

from gogpt import GOGPTPredictor
from bioreason2.dataset.cafa5.processor import _GO_INFO


def load_predictor(
    model_name: str = "armansa1/gogpt-test",
    cache_dir: Optional[str] = None
) -> GOGPTPredictor:
    """
    Load GO-GPT predictor from HuggingFace Hub.
    
    Args:
        model_name: HuggingFace model name.
        cache_dir: Cache directory for model weights.
        
    Returns:
        Loaded GOGPTPredictor instance.
    """
    return GOGPTPredictor.from_pretrained(model_name, cache_dir=cache_dir)


def predict_go_terms(
    predictor: GOGPTPredictor,
    sequence: str,
    organism: str = "Homo sapiens"
) -> Dict[str, List[str]]:
    """
    Predict GO terms for a protein sequence.
    
    Args:
        predictor: Loaded GOGPTPredictor.
        sequence: Protein sequence (amino acids).
        organism: Organism name.
        
    Returns:
        Dict with keys "MF", "BP", "CC" containing lists of GO IDs.
    """
    return predictor.predict(sequence=sequence, organism=organism)


def format_go_output(predictions: Dict[str, List[str]]) -> str:
    """
    Format GO predictions as a human-readable string.
    
    Args:
        predictions: Dict with "MF", "BP", "CC" keys.
        
    Returns:
        Formatted string with GO terms and their names.
    """
    aspect_names = {
        "MF": "Molecular Function",
        "BP": "Biological Process",
        "CC": "Cellular Component"
    }
    
    parts = []
    for aspect in ["MF", "BP", "CC"]:
        terms = predictions.get(aspect, [])
        if terms:
            formatted = []
            for go_id in terms:
                name, _ = _GO_INFO.get(go_id, ("Unknown", ""))
                formatted.append(f"{go_id} ({name})")
            parts.append(f"{aspect_names[aspect]} ({aspect}): {', '.join(formatted)}")
        else:
            parts.append(f"{aspect_names[aspect]} ({aspect}): None")
    
    return "\n".join(parts)


def predict_and_format(
    predictor: GOGPTPredictor,
    sequence: str,
    organism: str = "Homo sapiens"
) -> str:
    """
    Predict GO terms and format as human-readable string.
    
    Args:
        predictor: Loaded GOGPTPredictor.
        sequence: Protein sequence (amino acids).
        organism: Organism name.
        
    Returns:
        Formatted string with GO terms and their names.
    """
    predictions = predict_go_terms(predictor, sequence, organism)
    return format_go_output(predictions)


def main():
    parser = argparse.ArgumentParser(
        description="Predict GO terms from a protein sequence using GO-GPT."
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="Protein sequence (amino acids)"
    )
    parser.add_argument(
        "--organism",
        default="Homo sapiens",
        help="Organism name (default: 'Homo sapiens')"
    )
    parser.add_argument(
        "--model",
        default="armansa1/gogpt-test",
        help="HuggingFace model name (default: armansa1/gogpt-test)"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for model weights (default: HuggingFace default)"
    )
    args = parser.parse_args()
    
    # Load model
    predictor = load_predictor(args.model, args.cache_dir)
    
    # Run prediction and format
    output = predict_and_format(predictor, args.sequence, args.organism)
    print(output)


if __name__ == "__main__":
    main()