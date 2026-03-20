#!/usr/bin/env python3
"""
Download CAFA5 protein structures.

This script downloads and extracts CAFA5 protein structure files from Hugging Face,
with proper caching and directory management.
"""

import os
import tarfile
import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from huggingface_hub import snapshot_download
from bioreason2.utils.argparse_utils import str2bool


def _tar_extract_file(tar_file_path, extracted_dir):
    """Helper function to extract a single tar file with flattened structure"""
    try:
        extracted_files = []
        with tarfile.open(tar_file_path, "r:gz") as tar:
            # Get all file members for progress tracking
            members = [member for member in tar.getmembers() if member.isfile()]

            # Extract members but flatten the directory structure
            for member in members:
                # Extract the file content and write it with the flattened name
                file_obj = tar.extractfile(member)
                if file_obj:
                    output_path = os.path.join(extracted_dir, os.path.basename(member.name))
                    with open(output_path, "wb") as out_file:
                        out_file.write(file_obj.read())
                    extracted_files.append(os.path.basename(member.name))

        return f"Successfully extracted {len(extracted_files)} files from {os.path.basename(tar_file_path)}"
    except Exception as e:
        return f"Failed to extract {os.path.basename(tar_file_path)}: {e}"


def _download_and_collect_tar_files(
    repo_id: str,
    cache_dir: str,
    repo_type: str,
    structure_type: str,
    pattern: str,
    subdirs: list = None,
):
    """
    Helper function to download structures and collect tar files.
    
    Args:
        repo_id: Hugging Face repository ID
        cache_dir: Directory to cache the downloaded files
        repo_type: Repository type (dataset, model, etc.)
        structure_type: Type of structure (e.g., "SwissProt", "AlphaFold", "Interlabel")
        pattern: Glob pattern for snapshot_download
        subdirs: Optional list of subdirectories to search for tar files
        
    Returns:
        List of paths to tar.gz files
    """
    print("\n" + "=" * 60)
    print(f"DOWNLOADING {structure_type.upper()} STRUCTURES")
    print("=" * 60)

    data_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=cache_dir,
        repo_type=repo_type,
        allow_patterns=pattern,
        cache_dir=cache_dir,
    )

    # Extract directory name from pattern (e.g., "swissprot_structures/*" -> "swissprot_structures")
    dir_name = pattern.split("/*")[0]
    base_dir = os.path.join(data_dir, dir_name)
    print(f"{structure_type} structures directory: {base_dir}")

    tar_files = []
    
    if subdirs:
        # Search in subdirectories (e.g., AlphaFold shards)
        for subdir in subdirs:
            shard_dir = os.path.join(base_dir, subdir)
            if os.path.isdir(shard_dir):
                for fname in os.listdir(shard_dir):
                    if fname.endswith(".tar.gz"):
                        tar_files.append(os.path.join(shard_dir, fname))
        print(f"Found {len(tar_files)} {structure_type} tar.gz files in {subdirs}")
    else:
        # Search directly in base directory
        if os.path.isdir(base_dir):
            for fname in os.listdir(base_dir):
                if fname.endswith(".tar.gz"):
                    tar_files.append(os.path.join(base_dir, fname))
        print(f"Found {len(tar_files)} {structure_type} tar.gz files")
    
    return tar_files


def _download_structure_files(
    cache_dir: str,
    num_proc: int = None,
    repo_id: str = "wanglab/cafa5",
    repo_type: str = "dataset",
    shard_subdirs: list = None,
    extracted_dir_name: str = "extracted",
    download_swissprot: bool = True,
    download_af: bool = True,
    download_interlabel: bool = True,
    download_temp_holdout: bool = True,
):
    """Download and extract CAFA5 protein structure files from the Hugging Face hub.

    Args:
        cache_dir (str): Directory to cache the downloaded structures.
        num_proc (int): Number of processes to use for parallel extraction.
                       If None, auto-detects based on CPU count.
        repo_id (str): Hugging Face repository ID.
        repo_type (str): Repository type (dataset, model, etc.).
        shard_subdirs (list): List of subdirectories containing tar shards for AlphaFold.
        extracted_dir_name (str): Name of the extracted directory.
        download_swissprot (bool): Whether to download SwissProt structures.
        download_af (bool): Whether to download AlphaFold structures.
        download_interlabel (bool): Whether to download interlabel structures.
        download_temp_holdout (bool): Whether to download temp holdout structures.

    Returns:
        str: Path to the directory containing the extracted structure files.
    """
    if num_proc is None:
        num_proc = max(8, os.cpu_count())
    if shard_subdirs is None:
        shard_subdirs = ["af_shards"]

    print(f"Using {num_proc} CPU cores for parallel processing")
    print(f"Repository: {repo_id}")
    print(f"Download SwissProt: {download_swissprot}")
    print(f"Download AlphaFold: {download_af}")
    print(f"Download Interlabel: {download_interlabel}")
    print(f"Download Temp Holdout: {download_temp_holdout}")

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")

    # Create final extracted directory
    extracted_dir = os.path.join(cache_dir, extracted_dir_name)
    os.makedirs(extracted_dir, exist_ok=True)
    print(f"Final extracted directory: {extracted_dir}")

    all_tar_files = []

    # Process SwissProt structures if requested
    if download_swissprot:
        swissprot_tar_files = _download_and_collect_tar_files(
            repo_id=repo_id,
            cache_dir=cache_dir,
            repo_type=repo_type,
            structure_type="SwissProt",
            pattern="swissprot_structures/*",
        )
        all_tar_files.extend(swissprot_tar_files)

    # Process AlphaFold structures if requested
    if download_af:
        af_tar_files = _download_and_collect_tar_files(
            repo_id=repo_id,
            cache_dir=cache_dir,
            repo_type=repo_type,
            structure_type="AlphaFold",
            pattern="structures_af/*",
            subdirs=shard_subdirs,
        )
        all_tar_files.extend(af_tar_files)

    # Process Interlabel structures if requested
    if download_interlabel:
        interlabel_tar_files = _download_and_collect_tar_files(
            repo_id=repo_id,
            cache_dir=cache_dir,
            repo_type=repo_type,
            structure_type="Interlabel",
            pattern="structures_interlabel/*",
        )
        all_tar_files.extend(interlabel_tar_files)

    # Process Temp Holdout structures if requested
    if download_temp_holdout:
        temp_holdout_tar_files = _download_and_collect_tar_files(
            repo_id=repo_id,
            cache_dir=cache_dir,
            repo_type=repo_type,
            structure_type="TempHoldout",
            pattern="structures_temp_holdout_2022_2025/*",
        )
        all_tar_files.extend(temp_holdout_tar_files)

    total_tar_files = len(all_tar_files)
    print(f"\nTotal tar.gz files to process: {total_tar_files}")

    if total_tar_files == 0:
        print("Warning: No tar.gz files found in specified directories")
        return extracted_dir

    try:
        print(f"Extracting {total_tar_files} tar files in parallel...")

        # Create a partial function with the extracted_dir argument
        extract_func = partial(_tar_extract_file, extracted_dir=extracted_dir)

        with ThreadPoolExecutor(max_workers=min(num_proc, total_tar_files)) as executor:
            results = list(
                tqdm(
                    executor.map(extract_func, all_tar_files),
                    total=total_tar_files,
                    desc="Extracting",
                )
            )

        # Print extraction results
        for result in results:
            print(result)

    except Exception as e:
        print(f"An error occurred during structure tar file extraction: {e}")

    print(f"\nAll structure files extracted to: {extracted_dir}")
    return extracted_dir


def download_structures(
    cache_dir: str,
    structure_dir: str = None,
    download_swissprot: bool = True,
    download_af: bool = True,
    download_interlabel: bool = False,
    download_temp_holdout: bool = False,
    **kwargs,
) -> str:
    """
    Download and extract CAFA5 protein structures.

    Args:
        cache_dir: Directory to cache downloaded files
        structure_dir: Directory for extracted structure files.
                      If None, uses cache_dir/structures as default.
        download_swissprot: Whether to download SwissProt structures.
        download_af: Whether to download AlphaFold structures.
        download_interlabel: Whether to download interlabel structures.
        download_temp_holdout: Whether to download temp holdout structures.
        **kwargs: Additional arguments passed to _download_structure_files

    Returns:
        str: Path to the directory containing extracted structure files
    """
    print("=" * 80)
    print("DOWNLOADING CAFA5 PROTEIN STRUCTURES")
    print("=" * 80)

    # Set default structure directory if not provided
    if structure_dir is None:
        structure_dir = os.path.join(cache_dir, "structures")
    else:
        # Ensure structure_dir is absolute path
        structure_dir = os.path.abspath(structure_dir)

    print(f"Cache directory: {cache_dir}")
    print(f"Structure directory: {structure_dir}")
    print(f"Download SwissProt: {download_swissprot}")
    print(f"Download AlphaFold: {download_af}")
    print(f"Download Interlabel: {download_interlabel}")
    print(f"Download Temp Holdout: {download_temp_holdout}")

    # Check if structures already exist
    if os.path.exists(structure_dir):
        print(f"Using existing structure directory: {structure_dir}")
        return structure_dir
    else:
        print("Downloading and extracting protein structures...")
        # Use the structure directory as the final output directory
        extracted_structure_dir = _download_structure_files(
            cache_dir=cache_dir,
            download_swissprot=download_swissprot,
            download_af=download_af,
            download_interlabel=download_interlabel,
            download_temp_holdout=download_temp_holdout,
            num_proc=kwargs.get("num_proc"),
            repo_id=kwargs.get("repo_id", "wanglab/cafa5"),
            repo_type=kwargs.get("repo_type", "dataset"),
            shard_subdirs=kwargs.get("shard_subdirs", ["af_shards"]),
            extracted_dir_name=os.path.basename(structure_dir),
        )
        print(f"Structures extracted to: {extracted_structure_dir}")
        return extracted_structure_dir


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download CAFA5 protein structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_structures.py --cache-dir /path/to/cache
    # Downloads to /path/to/cache/structures/ (default: SwissProt + AlphaFold)
  python download_structures.py --cache-dir /path/to/cache --structure-dir /path/to/structures
    # Downloads to custom directory
  python download_structures.py --cache-dir /path/to/cache --num-proc 16
    # Use 16 cores for extraction
  python download_structures.py --cache-dir /path/to/cache --download-af false
    # SwissProt only
  python download_structures.py --cache-dir /path/to/cache --download-swissprot false
    # AlphaFold only
  python download_structures.py --cache-dir /path/to/cache --download-interlabel true
    # SwissProt + AlphaFold + Interlabel
        """,
    )

    # Basic directories
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory to cache downloaded files",
    )

    parser.add_argument(
        "--structure-dir",
        type=str,
        default=None,
        help="Directory for extracted structure files. If not provided, uses cache_dir/structures (default)",
    )

    # Processing configuration
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of CPU cores for parallel processing. If not provided, auto-detects based on CPU count",
    )

    # Hugging Face repository configuration
    parser.add_argument(
        "--repo-id",
        type=str,
        default="wanglab/cafa5",
        help="Hugging Face repository ID (default: wanglab/cafa5)",
    )

    parser.add_argument(
        "--repo-type",
        type=str,
        default="dataset",
        help="Repository type (default: dataset)",
    )

    # Directory structure configuration

    parser.add_argument(
        "--shard-subdirs",
        type=str,
        nargs="+",
        default=["af_shards"],
        help="List of subdirectories containing tar shards (default: af_shards)",
    )

    parser.add_argument(
        "--download-swissprot",
        type=str2bool,
        default=True,
        help="Download SwissProt structures (default: True). Use true/false, yes/no, 1/0.",
    )

    parser.add_argument(
        "--download-af",
        type=str2bool,
        default=True,
        help="Download AlphaFold structures (default: True). Use true/false, yes/no, 1/0.",
    )

    parser.add_argument(
        "--download-interlabel",
        type=str2bool,
        default=True,
        help="Download interlabel structures (default: False). Use true/false, yes/no, 1/0.",
    )

    parser.add_argument(
        "--download-temp-holdout",
        type=str2bool,
        default=True,
        help="Download temp holdout structures (default: False). Use true/false, yes/no, 1/0.",
    )

    args = parser.parse_args()

    # Validate cache directory
    if not os.path.exists(args.cache_dir):
        print(f"Creating cache directory: {args.cache_dir}")
        os.makedirs(args.cache_dir, exist_ok=True)

    # Download structures with all hyperparameters
    structure_path = download_structures(
        cache_dir=args.cache_dir,
        structure_dir=args.structure_dir,
        num_proc=args.num_proc,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        shard_subdirs=args.shard_subdirs,
        download_swissprot=args.download_swissprot,
        download_af=args.download_af,
        download_interlabel=args.download_interlabel,
        download_temp_holdout=args.download_temp_holdout,
    )

    print("\n✅ Structure download completed!")
    print(f"📁 Structure directory: {structure_path}")
    print("🔧 Used hyperparameters:")
    print(f"   - Repository: {args.repo_id}")
    print(f"   - CPU cores: {args.num_proc if args.num_proc else 'auto'}")
    print(f"   - Shard subdirs: {args.shard_subdirs}")
    print(f"   - Structure directory: {structure_path}")
    print(f"   - Download SwissProt: {args.download_swissprot}")
    print(f"   - Download AlphaFold: {args.download_af}")
    print(f"   - Download Interlabel: {args.download_interlabel}")
    print(f"   - Download Temp Holdout: {args.download_temp_holdout}")


if __name__ == "__main__":
    main()
