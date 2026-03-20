"""
This script will provide a comprehensive analysis of the CAFA5 dataset.
"""

from datasets import load_dataset
import numpy as np
from collections import Counter


def print_header(title, width=60):
    """Print a nicely formatted header"""
    print("\n" + "=" * width)
    print(f" {title.center(width-2)} ")
    print("=" * width)


def print_subheader(title, width=40):
    """Print a nicely formatted subheader"""
    print(f"\n{'-'*width}")
    print(f" {title} ")
    print(f"{'-'*width}")


def print_stats_table(stats_dict, title="Statistics"):
    """Print statistics in a nicely formatted table"""
    print(f"\n{title}:")
    max_key_length = max(len(str(k)) for k in stats_dict.keys()) + 2
    for key, value in stats_dict.items():
        if isinstance(value, float):
            print(f"  {str(key).ljust(max_key_length)}: {value:.2f}")
        else:
            print(f"  {str(key).ljust(max_key_length)}: {value}")


def load_cafa5_dataset():
    """Load the CAFA5 dataset"""
    print("🔄 Loading CAFA5 dataset from HuggingFace...")
    return load_dataset("wanglab/cafa5")


def analyze_dataset_structure(dataset):
    """Analyze the basic structure of the dataset"""
    print_header("📊 DATASET STRUCTURE OVERVIEW")

    print(f"Available splits: {list(dataset.keys())}")

    for split_name, split_data in dataset.items():
        print_subheader(f"{split_name.upper()} Split Info")
        print(f"  📦 Number of samples: {len(split_data):,}")
        print(f"  🏷️  Features: {list(split_data.features.keys())}")

        # Show first example (truncated)
        if len(split_data) > 0:
            print(f"\n  📋 Sample entry from {split_name}:")
            example = split_data[0]
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 80:
                    print(f"    {key}: {value[:80]}...")
                elif isinstance(value, list) and len(value) > 3:
                    print(f"    {key}: [{', '.join(map(str, value[:3]))}, ...] ({len(value)} total)")
                else:
                    print(f"    {key}: {value}")


def analyze_organism_distribution(split_data, split_name):
    """Analyze organism distribution in the dataset - optimized version"""
    print_header(f"🧬 ORGANISM DISTRIBUTION - {split_name.upper()}")

    print("🔄 Processing organism data...")

    # More efficient: extract all organisms at once
    organisms = [item["organism"] for item in split_data if item.get("organism")]
    organism_counts = Counter(organisms)

    stats = {
        "Total unique organisms": len(organism_counts),
        "Total samples": len(organisms),
        "Average samples per organism": (len(organisms) / len(organism_counts) if len(organism_counts) > 0 else 0),
    }
    print_stats_table(stats, "🔍 Summary Statistics")

    print_subheader("🏆 Top 15 Most Common Organisms")
    for i, (organism, count) in enumerate(organism_counts.most_common(15), 1):
        percentage = (count / len(organisms)) * 100
        # Truncate long organism names for better formatting
        org_display = organism[:35] + "..." if len(organism) > 35 else organism
        print(f"  {i:2d}. {org_display:<38} {count:>6,} ({percentage:>5.1f}%)")

    return organism_counts


def analyze_protein_function_stats(split_data, split_name):
    """Analyze protein function text statistics"""
    print_header(f"📝 PROTEIN FUNCTION TEXT ANALYSIS - {split_name.upper()}")

    print("🔄 Processing protein function data...")

    # More efficient processing
    functions = [item["protein_function"] for item in split_data if item.get("protein_function")]

    # Vectorized word and character count analysis
    word_counts = [len(func.split()) for func in functions if func]
    char_counts = [len(func) for func in functions if func]

    coverage_stats = {
        "Valid protein function entries": len(word_counts),
        "Missing/empty entries": len(split_data) - len(word_counts),
        "Coverage percentage": (len(word_counts) / len(split_data)) * 100,
    }
    print_stats_table(coverage_stats, "📊 Data Coverage")

    if word_counts:
        word_stats = {
            "Average": np.mean(word_counts),
            "Median": np.median(word_counts),
            "Min": np.min(word_counts),
            "Max": np.max(word_counts),
            "25th percentile": np.percentile(word_counts, 25),
            "75th percentile": np.percentile(word_counts, 75),
            "95th percentile": np.percentile(word_counts, 95),
            "99th percentile": np.percentile(word_counts, 99),
            "Standard deviation": np.std(word_counts),
        }
        print_stats_table(word_stats, "📊 Word Count Statistics")

        char_stats = {
            "Average": np.mean(char_counts),
            "Median": np.median(char_counts),
            "Min": np.min(char_counts),
            "Max": np.max(char_counts),
            "25th percentile": np.percentile(char_counts, 25),
            "75th percentile": np.percentile(char_counts, 75),
            "95th percentile": np.percentile(char_counts, 95),
            "99th percentile": np.percentile(char_counts, 99),
            "Standard deviation": np.std(char_counts),
        }
        print_stats_table(char_stats, "📊 Character Count Statistics")

    return word_counts, char_counts


def analyze_protein_length_stats(split_data, split_name):
    """Analyze protein length statistics"""
    print_header(f"📏 PROTEIN LENGTH ANALYSIS - {split_name.upper()}")

    print("🔄 Processing protein length data...")

    lengths = [item["length"] for item in split_data if item.get("length") is not None]

    coverage_stats = {
        "Valid length entries": len(lengths),
        "Missing length entries": len(split_data) - len(lengths),
        "Coverage percentage": (len(lengths) / len(split_data)) * 100,
    }
    print_stats_table(coverage_stats, "📊 Data Coverage")

    if lengths:
        length_stats = {
            "Average": np.mean(lengths),
            "Median": np.median(lengths),
            "Min": int(np.min(lengths)),
            "Max": int(np.max(lengths)),
            "25th percentile": np.percentile(lengths, 25),
            "75th percentile": np.percentile(lengths, 75),
            "95th percentile": np.percentile(lengths, 95),
            "99th percentile": np.percentile(lengths, 99),
            "Standard deviation": np.std(lengths),
        }
        print_stats_table(length_stats, "📊 Length Statistics (amino acids)")

        # Length distribution
        print_subheader("📈 Length Distribution")
        length_ranges = [
            ("Very short (< 100 aa)", lambda x: x < 100),
            ("Short (100-300 aa)", lambda x: 100 <= x < 300),
            ("Medium (300-600 aa)", lambda x: 300 <= x < 600),
            ("Long (600-1000 aa)", lambda x: 600 <= x < 1000),
            ("Very long (≥ 1000 aa)", lambda x: x >= 1000),
        ]

        for range_name, condition in length_ranges:
            count = sum(1 for length in lengths if condition(length))
            percentage = (count / len(lengths)) * 100
            print(f"  {range_name:<25} {count:>6,} ({percentage:>5.1f}%)")

    return lengths


def analyze_sequence_stats(split_data, split_name):
    """Analyze protein sequence statistics"""
    print_header(f"🧬 PROTEIN SEQUENCE ANALYSIS - {split_name.upper()}")

    print("🔄 Processing protein sequence data...")

    sequences = [item["sequence"] for item in split_data if item.get("sequence")]

    # Character count analysis
    char_counts = [len(seq) for seq in sequences]

    coverage_stats = {
        "Valid sequence entries": len(sequences),
        "Missing sequence entries": len(split_data) - len(sequences),
        "Coverage percentage": (len(sequences) / len(split_data)) * 100,
    }
    print_stats_table(coverage_stats, "📊 Data Coverage")

    if char_counts:
        seq_stats = {
            "Average length": np.mean(char_counts),
            "Median length": np.median(char_counts),
            "Min length": np.min(char_counts),
            "Max length": np.max(char_counts),
            "25th percentile": np.percentile(char_counts, 25),
            "75th percentile": np.percentile(char_counts, 75),
            "95th percentile": np.percentile(char_counts, 95),
            "Standard deviation": np.std(char_counts),
        }
        print_stats_table(seq_stats, "📊 Sequence Length Statistics")

        # Amino acid composition analysis - optimized
        print("🔄 Analyzing amino acid composition...")
        all_sequences = "".join(sequences)
        aa_counts = Counter(all_sequences)
        total_aa = len(all_sequences)

        print_subheader("🧪 Amino Acid Composition (Top 15)")
        for i, (aa, count) in enumerate(aa_counts.most_common(15), 1):
            percentage = (count / total_aa) * 100
            print(f"  {i:2d}. {aa}: {count:>12,} ({percentage:>5.2f}%)")

        # Calculate some interesting sequence properties
        hydrophobic_aas = set(["A", "V", "I", "L", "M", "F", "Y", "W"])
        charged_aas = set(["R", "K", "D", "E"])
        polar_aas = set(["S", "T", "N", "Q", "H"])

        hydrophobic_count = sum(aa_counts.get(aa, 0) for aa in hydrophobic_aas)
        charged_count = sum(aa_counts.get(aa, 0) for aa in charged_aas)
        polar_count = sum(aa_counts.get(aa, 0) for aa in polar_aas)

        composition_stats = {
            "Hydrophobic residues": f"{(hydrophobic_count/total_aa)*100:.1f}%",
            "Charged residues": f"{(charged_count/total_aa)*100:.1f}%",
            "Polar residues": f"{(polar_count/total_aa)*100:.1f}%",
        }
        print_stats_table(composition_stats, "🧪 Amino Acid Properties")

    return char_counts


def analyze_go_terms_stats(split_data, split_name):
    """Analyze GO terms statistics"""
    print_header(f"🎯 GO TERMS ANALYSIS - {split_name.upper()}")

    print("🔄 Processing GO terms data...")

    # More efficient GO terms processing
    go_counts = []
    all_go_terms = []
    aspect_counts = {"BPO": 0, "CCO": 0, "MFO": 0}

    for item in split_data:
        if item.get("go_ids"):
            go_counts.append(len(item["go_ids"]))
            all_go_terms.extend(item["go_ids"])

            if item.get("go_aspect"):
                for aspect_info in item["go_aspect"]:
                    if aspect_info.get("value") in aspect_counts:
                        aspect_counts[aspect_info["value"]] += 1

    coverage_stats = {
        "Proteins with GO annotations": len(go_counts),
        "Proteins without GO annotations": len(split_data) - len(go_counts),
        "Coverage percentage": (len(go_counts) / len(split_data)) * 100,
    }
    print_stats_table(coverage_stats, "📊 Annotation Coverage")

    if go_counts:
        go_stats = {
            "Average GO terms per protein": np.mean(go_counts),
            "Median GO terms per protein": np.median(go_counts),
            "Min GO terms": np.min(go_counts),
            "Max GO terms": np.max(go_counts),
            "25th percentile": np.percentile(go_counts, 25),
            "75th percentile": np.percentile(go_counts, 75),
            "95th percentile": np.percentile(go_counts, 95),
            "Standard deviation": np.std(go_counts),
        }
        print_stats_table(go_stats, "📊 GO Terms per Protein Statistics")

        unique_go_terms = len(set(all_go_terms))
        total_annotations = len(all_go_terms)

        go_overview = {
            "Total unique GO terms": unique_go_terms,
            "Total GO annotations": total_annotations,
            "Average annotations per unique term": (total_annotations / unique_go_terms if unique_go_terms > 0 else 0),
        }
        print_stats_table(go_overview, "📊 GO Terms Overview")

        print_subheader("🎯 GO Aspect Distribution")
        total_aspects = sum(aspect_counts.values())
        for aspect, count in aspect_counts.items():
            aspect_name = {
                "BPO": "Biological Process",
                "CCO": "Cellular Component",
                "MFO": "Molecular Function",
            }[aspect]
            percentage = (count / total_aspects) * 100 if total_aspects > 0 else 0
            print(f"  {aspect} ({aspect_name}): {count:>8,} ({percentage:>5.1f}%)")

        # Most common GO terms
        go_term_counts = Counter(all_go_terms)
        print_subheader("🏆 Top 15 Most Common GO Terms")
        for i, (go_term, count) in enumerate(go_term_counts.most_common(15), 1):
            percentage = (count / total_annotations) * 100
            print(f"  {i:2d}. {go_term}: {count:>8,} ({percentage:>5.2f}%)")

    return go_counts


def comprehensive_analysis():
    """Run comprehensive analysis on the CAFA5 dataset"""
    print_header("🧬 CAFA5 DATASET COMPREHENSIVE ANALYSIS")
    print("🔬 Analyzing the Critical Assessment of protein Function Annotation dataset")
    print("📅 This analysis covers protein structure, function, and annotation statistics")

    # Load dataset
    dataset = load_cafa5_dataset()

    # Basic structure analysis
    analyze_dataset_structure(dataset)

    # Analyze each split
    for split_name, split_data in dataset.items():
        print(f"\n\n{'='*60}")
        print(f"🔍 DETAILED ANALYSIS: {split_name.upper()} SPLIT ({len(split_data):,} samples)")
        print(f"{'='*60}")

        # Run all analyses
        # analyze_organism_distribution(split_data, split_name)
        analyze_protein_function_stats(split_data, split_name)
        # analyze_protein_length_stats(split_data, split_name)
        # analyze_sequence_stats(split_data, split_name)
        # analyze_go_terms_stats(split_data, split_name)

    print_header("✅ ANALYSIS COMPLETE")
    print("🎉 All dataset statistics have been generated successfully!")
    print("📊 Use this information for downstream analysis and model development")


if __name__ == "__main__":
    comprehensive_analysis()
