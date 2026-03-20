#!/usr/bin/env python
"""
InterPro API: Run InterProScan on a protein sequence and print domain results.

Usage (CLI - single sequence):
    python interpro_api.py --sequence <protein_sequence>
    python interpro_api.py --sequence <protein_sequence> --online

Usage (CLI - batch from FASTA):
    python interpro_api.py --fasta <path_to_fasta> --output <output.tsv>

Usage (Import):
    from interpro_api import analyze_sequence, format_interpro_output, load_interpro_metadata
    
    type_cache = load_interpro_metadata(metadata_path)
    result_df = analyze_sequence(sequence)
    formatted = format_interpro_output(result_df, type_cache)

Example:
    python interpro_api.py --sequence "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
    python interpro_api.py --sequence "MVLSPADKTN..." --online
    python interpro_api.py --fasta proteins.fasta --output results.tsv
"""

import argparse
import json
import subprocess
import tempfile
import time
import os
from typing import Optional
import requests
import pandas as pd


TSV_COLS = [
    "protein", "md5", "length", "analysis", "signature_acc", "signature_desc",
    "start", "end", "score", "status", "date",
    "interpro_id", "interpro_desc", "go_terms", "pathway"
]

IPRSCAN_API_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"

# Global cache for IPR type lookup
_IPR_TYPE_CACHE: dict[str, str] = {}


def load_interpro_metadata(metadata_path: str) -> dict[str, str]:
    """
    Load InterPro metadata JSON and return a dict mapping IPR ID -> type.
    
    Args:
        metadata_path: Path to interpro_metadata.json.
        
    Returns:
        Dict mapping IPR accession to type (e.g., "domain", "family").
    """
    global _IPR_TYPE_CACHE
    
    if _IPR_TYPE_CACHE:
        return _IPR_TYPE_CACHE
    
    if not os.path.exists(metadata_path):
        return {}
    
    with open(metadata_path) as f:
        data = json.load(f)
    
    for entry in data:
        acc = entry.get("metadata", {}).get("accession", "")
        ipr_type = entry.get("metadata", {}).get("type", "")
        if acc and ipr_type:
            _IPR_TYPE_CACHE[acc] = ipr_type
    
    return _IPR_TYPE_CACHE


def run_interproscan_online(sequence: str, email: str = "anonymous@example.com") -> pd.DataFrame:
    """
    Run InterProScan via the EBI REST API.
    
    Args:
        sequence: Protein sequence string.
        email: Email for job submission (required by EBI).
        
    Returns:
        DataFrame with domain annotations.
    """
    # Submit job
    submit_url = f"{IPRSCAN_API_URL}/run"
    data = {
        "email": email,
        "sequence": sequence,
        "stype": "p",  # protein
    }
    
    resp = requests.post(submit_url, data=data)
    resp.raise_for_status()
    job_id = resp.text.strip()
    
    # Poll for completion
    status_url = f"{IPRSCAN_API_URL}/status/{job_id}"
    while True:
        resp = requests.get(status_url)
        resp.raise_for_status()
        status = resp.text.strip()
        
        if status == "FINISHED":
            break
        elif status in ("FAILURE", "ERROR"):
            raise RuntimeError(f"Job failed with status: {status}")
        
        time.sleep(5)
    
    # Get TSV results
    result_url = f"{IPRSCAN_API_URL}/result/{job_id}/tsv"
    resp = requests.get(result_url)
    resp.raise_for_status()
    
    # Parse TSV from response text
    lines = resp.text.strip().split("\n")
    if not lines or not lines[0]:
        return pd.DataFrame(columns=["accession", "interpro_id", "entry_name", "start", "end", "n_fragments"])
    
    # Write to temp file for parsing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        f.write(resp.text)
        tsv_path = f.name
    
    result = parse_and_collapse(tsv_path)
    os.unlink(tsv_path)
    return result


def run_interproscan_local(
    sequence: str,
    prefix: str,
    interproscan_path: str,
    cpu: int,
    appl: Optional[str]
) -> str:
    """
    Run InterProScan locally.
    
    Args:
        sequence: Protein sequence string (amino acids).
        prefix: Output file prefix.
        interproscan_path: Path to interproscan.sh.
        cpu: Number of CPU cores.
        appl: Comma-separated list of analyses (None = all).
        
    Returns:
        Path to the TSV output file.
    """
    fasta_path = f"{prefix}.fasta"
    with open(fasta_path, "w") as f:
        f.write(">query\n")
        f.write(sequence + "\n")
    
    tsv_output = f"{prefix}.tsv"
    
    cmd = [
        interproscan_path,
        "-i", fasta_path,
        "-f", "tsv",
        "-o", tsv_output,
        "-cpu", str(cpu),
    ]
    
    if appl:
        cmd.extend(["-appl", appl])
    
    subprocess.run(cmd, check=True)
    
    if not os.path.exists(tsv_output):
        raise FileNotFoundError(f"Could not find InterProScan output: {tsv_output}")
    
    return tsv_output


def run_interproscan_local_fasta(
    fasta_path: str,
    prefix: str,
    interproscan_path: str,
    cpu: int,
    appl: Optional[str]
) -> str:
    """
    Run InterProScan locally on a FASTA file.
    
    Args:
        fasta_path: Path to input FASTA file.
        prefix: Output file prefix.
        interproscan_path: Path to interproscan.sh.
        cpu: Number of CPU cores.
        appl: Comma-separated list of analyses (None = all).
        
    Returns:
        Path to the TSV output file.
    """
    tsv_output = f"{prefix}.tsv"
    
    cmd = [
        interproscan_path,
        "-i", fasta_path,
        "-f", "tsv",
        "-o", tsv_output,
        "-cpu", str(cpu),
    ]
    
    if appl:
        cmd.extend(["-appl", appl])
    
    subprocess.run(cmd, check=True)
    
    if not os.path.exists(tsv_output):
        raise FileNotFoundError(f"Could not find InterProScan output: {tsv_output}")
    
    return tsv_output


def parse_and_collapse(tsv_path: str) -> pd.DataFrame:
    """
    Parse InterProScan TSV and collapse duplicate InterPro IDs.
    
    Args:
        tsv_path: Path to InterProScan TSV output.
        
    Returns:
        DataFrame with collapsed domain annotations.
    """
    df = pd.read_csv(tsv_path, sep="\t", header=None, comment="#")
    
    ncols = df.shape[1]
    cols = TSV_COLS[:ncols] + [f"extra_{i}" for i in range(ncols - len(TSV_COLS))]
    df.columns = cols
    
    df = df[df["interpro_id"].notna() & (df["interpro_id"] != "-")]
    
    if df.empty:
        return pd.DataFrame(columns=["accession", "interpro_id", "entry_name", "start", "end", "n_fragments"])
    
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    
    grouped = df.groupby(
        ["protein", "interpro_id", "interpro_desc"],
        dropna=False
    ).agg(
        start=("start", "min"),
        end=("end", "max"),
        n_fragments=("start", "count")
    ).reset_index()
    
    grouped.rename(columns={
        "protein": "accession",
        "interpro_desc": "entry_name"
    }, inplace=True)
    
    return grouped


def analyze_sequence(
    sequence: str,
    online: bool = False,
    interproscan_dir: Optional[str] = None,
    cpu: int = 8,
    appl: Optional[str] = None,
    email: str = "anonymous@example.com"
) -> pd.DataFrame:
    """
    Analyze a protein sequence using InterProScan.
    
    Args:
        sequence: Protein sequence string.
        online: Use online API instead of local installation.
        interproscan_dir: Path to InterProScan directory (local mode).
        cpu: Number of CPU cores (local mode).
        appl: Comma-separated list of analyses (local mode).
        email: Email for API submission (online mode).
        
    Returns:
        DataFrame with domain annotations.
    """
    if online:
        return run_interproscan_online(sequence, email)
    else:
        interproscan_path = os.path.join(interproscan_dir, "interproscan.sh")
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "interpro")
            tsv_path = run_interproscan_local(sequence, prefix, interproscan_path, cpu, appl)
            return parse_and_collapse(tsv_path)


def analyze_fasta(
    fasta_path: str,
    output_path: str,
    interproscan_dir: str = None,
    cpu: int = 8,
    appl: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze a FASTA file using InterProScan and output a TSV with formatted results.
    
    Args:
        fasta_path: Path to input FASTA file.
        output_path: Path to output TSV file.
        interproscan_dir: Path to InterProScan directory.
        cpu: Number of CPU cores.
        appl: Comma-separated list of analyses (None = all).
        
    Returns:
        DataFrame with columns [entry_name, interpro_formatted].
    """
    interproscan_path = os.path.join(interproscan_dir, "interproscan.sh")
    metadata_path = os.path.join(interproscan_dir, "interpro_metadata.json")
    type_cache = load_interpro_metadata(metadata_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "interpro")
        tsv_path = run_interproscan_local_fasta(fasta_path, prefix, interproscan_path, cpu, appl)
        results_df = parse_and_collapse(tsv_path)
    
    # Group by protein accession and format
    output_rows = []
    for accession, group in results_df.groupby("accession"):
        formatted = format_interpro_output(group, type_cache)
        output_rows.append({"entry_name": accession, "interpro_formatted": formatted})
    
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_path, sep="\t", index=False)
    
    return output_df


def format_interpro_output(df: pd.DataFrame, type_cache: dict[str, str]) -> str:
    """
    Format InterPro results as a human-readable string.
    
    Args:
        df: DataFrame with domain annotations.
        type_cache: Dict mapping IPR ID to type.
        
    Returns:
        Formatted string with one domain per line.
    """
    if df.empty:
        return ""
    
    parts = []
    for _, row in df.iterrows():
        ipr_id = row["interpro_id"]
        ipr_type = type_cache.get(ipr_id, "unknown")
        loc_str = f" [{int(row['start'])}-{int(row['end'])}]"
        parts.append(f"- {ipr_id}: {row['entry_name']} ({ipr_type}){loc_str}")
    
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Run InterProScan on a protein sequence and print domain results."
    )
    parser.add_argument(
        "--sequence",
        help="Protein sequence (amino acids)"
    )
    parser.add_argument(
        "--fasta",
        help="Path to FASTA file (batch mode)"
    )
    parser.add_argument(
        "--output",
        help="Path to output TSV file (batch mode)"
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use EBI online API instead of local installation"
    )
    parser.add_argument(
        "--email",
        default="anonymous@example.com",
        help="Email for API submission (online mode)"
    )
    parser.add_argument(
        "--interproscan-dir",
        default=None,
        help="Path to InterProScan installation directory (local mode)"
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=8,
        help="Number of CPU cores (local mode, default: 8)"
    )
    parser.add_argument(
        "--appl",
        default=None,
        help="Comma-separated analyses to run (local mode, default: all)"
    )
    args = parser.parse_args()
    
    # Validate arguments
    if args.fasta and args.sequence:
        parser.error("Cannot use both --sequence and --fasta")
    if args.fasta and not args.output:
        parser.error("--output is required when using --fasta")
    if not args.fasta and not args.sequence:
        parser.error("Either --sequence or --fasta is required")
    
    if args.fasta:
        # Batch mode
        analyze_fasta(
            args.fasta,
            args.output,
            interproscan_dir=args.interproscan_dir,
            cpu=args.cpu,
            appl=args.appl
        )
        print(f"Results written to {args.output}")
    else:
        # Single sequence mode
        metadata_path = os.path.join(args.interproscan_dir, "interpro_metadata.json")
        type_cache = load_interpro_metadata(metadata_path)
        
        results = analyze_sequence(
            args.sequence,
            online=args.online,
            interproscan_dir=args.interproscan_dir,
            cpu=args.cpu,
            appl=args.appl,
            email=args.email
        )
        formatted = format_interpro_output(results, type_cache)
        
        print(formatted)


if __name__ == "__main__":
    main()
