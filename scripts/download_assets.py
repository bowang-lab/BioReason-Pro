#!/usr/bin/env python3
"""
Download the two large assets BioReason-Pro training and evaluation need but that
are not part of the git repository:

  * protein structures   -> wanglab/bioreason-pro-structures   (~34 GB download, ~57 GB on disk)
  * GO term embeddings   -> wanglab/bioreason-pro-go-embeddings (177 MB download, 338 MB on disk)

By default only the ~132k structures the released datasets actually reference are
written to disk; the shards themselves hold ~370k. Pass --all-structures to keep
everything (~150 GB).

Both land as plain directories that you point the training scripts at:

    python scripts/download_assets.py --dest /data/bioreason

    # -> /data/bioreason/structures      (STRUCTURE_DIR)
    # -> /data/bioreason/go_embeddings   (GO_EMBEDDINGS_PATH)

The download is resumable: re-running skips shards that are already extracted.

Structures are optional. ESM3 falls back to sequence-only when a structure is
missing, so training runs without them -- but the released checkpoint was trained
with them, so skipping changes the model. Use --verify to confirm that what you
downloaded actually covers the dataset (see the note on silent fallback below).
"""

import argparse
import gzip
import os
import shutil
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

STRUCTURES_REPO = "wanglab/bioreason-pro-structures"
GO_EMBEDDINGS_REPO = "wanglab/bioreason-pro-go-embeddings"

# Shard folders required to cover every structure_path in the released datasets.
# Verified: these three give 131,838/131,838 (100%). Listed explicitly so that
# adding unrelated shards to the repo later cannot silently inflate the download.
STRUCTURE_SHARD_PREFIXES = ("af_shards/", "af_shards_extra/", "interlabel_shards/")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_shard(tar_path: str, dest_dir: str, keep: set = None) -> int:
    """Extract one shard into dest_dir as a flat directory of .cif files.

    Members inside the shards are gzipped (`AF-XXXX-F1-model_v4.cif.gz`) but the
    `structure_path` column of the datasets refers to the *decompressed* name
    (`AF-XXXX-F1-model_v4.cif`). We therefore gunzip on the way out. Getting this
    wrong is silent: collate.py checks os.path.exists() and falls back to empty
    coordinates without warning, so every structure would be ignored.

    `keep` restricts extraction to the filenames the datasets actually reference.
    The shards hold ~370k structures but only ~132k are referenced, so filtering
    saves roughly 90 GB of disk.
    """
    written = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if name.endswith(".gz"):
                name = name[: -len(".gz")]
            if keep is not None and name not in keep:
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            data = fileobj.read()
            if os.path.basename(member.name).endswith(".gz"):
                data = gzip.decompress(data)
            with open(os.path.join(dest_dir, name), "wb") as out:
                out.write(data)
            written += 1
    return written


def _referenced_structure_names() -> set:
    """Filenames referenced by the released datasets (~132k of the ~370k shipped)."""
    from datasets import load_dataset

    names = set()
    for repo, split in [
        ("wanglab/bioreason-pro-sft-reasoning-data", "train"),
        ("wanglab/bioreason-pro-sft-reasoning-data", "validation"),
        ("wanglab/bioreason-pro-test-data", "test"),
    ]:
        ds = load_dataset(repo, "default", split=split)
        names.update(p for p in ds["structure_path"] if p)
    return names


def _shard_marker(dest_dir: str, shard: str) -> str:
    return os.path.join(dest_dir, ".shards", shard.replace("/", "__") + ".done")


def download_structures(dest_dir: str, num_workers: int, keep_archives: bool,
                        all_structures: bool = False) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(os.path.join(dest_dir, ".shards"), exist_ok=True)

    keep = None
    if not all_structures:
        print("structures: resolving which files the datasets reference …")
        try:
            keep = _referenced_structure_names()
            print(f"  {len(keep)} referenced (shards hold ~370k; "
                  f"filtering saves ~90 GB of disk)")
        except Exception as exc:
            print(f"  could not read datasets ({exc}); extracting everything")
            keep = None

    api = HfApi()
    shards = sorted(
        f for f in api.list_repo_files(STRUCTURES_REPO, repo_type="dataset")
        if f.endswith(".tar.gz") and f.startswith(STRUCTURE_SHARD_PREFIXES)
    )
    todo = [s for s in shards if not os.path.exists(_shard_marker(dest_dir, s))]
    print(f"structures: {len(shards)} shards total, {len(todo)} to fetch")
    if not todo:
        print("  already complete")
        return

    archive_dir = os.path.join(dest_dir, ".archives") if keep_archives else tempfile.mkdtemp()
    os.makedirs(archive_dir, exist_ok=True)

    def one(shard: str) -> tuple:
        path = hf_hub_download(
            STRUCTURES_REPO, shard, repo_type="dataset", cache_dir=archive_dir
        )
        n = _extract_shard(path, dest_dir, keep)
        if not keep_archives:
            try:
                os.remove(path)
            except OSError:
                pass
        open(_shard_marker(dest_dir, shard), "w").close()
        return shard, n

    total = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(one, s): s for s in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="shards"):
            shard, n = fut.result()
            total += n
    print(f"  extracted {total} structure files into {dest_dir}")

    if not keep_archives and archive_dir.startswith(tempfile.gettempdir()):
        shutil.rmtree(archive_dir, ignore_errors=True)


def download_go_embeddings(dest_dir: str) -> None:
    parent = os.path.dirname(os.path.abspath(dest_dir)) or "."
    os.makedirs(parent, exist_ok=True)

    if os.path.isdir(dest_dir) and len(os.listdir(dest_dir)) > 40000:
        print(f"go embeddings: already present in {dest_dir} "
              f"({len(os.listdir(dest_dir))} files)")
        return

    print("go embeddings: downloading …")
    with tempfile.TemporaryDirectory() as tmp:
        path = hf_hub_download(
            GO_EMBEDDINGS_REPO, "go_embeddings.tar.gz", repo_type="dataset", cache_dir=tmp
        )
        print("  extracting …")
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(parent)
    n = len(os.listdir(dest_dir)) if os.path.isdir(dest_dir) else 0
    print(f"  {n} embedding files in {dest_dir}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(structures_dir: str, go_dir: str) -> int:
    """Check that the downloaded assets actually cover the released datasets."""
    problems = 0

    if os.path.isdir(go_dir):
        n = len([f for f in os.listdir(go_dir) if f.endswith(".safetensors")])
        ok = n >= 43000
        print(f"[{'ok ' if ok else 'FAIL'}] go embeddings: {n} .safetensors "
              f"(expected ~43,248) in {go_dir}")
        problems += 0 if ok else 1
    else:
        print(f"[FAIL] go embeddings: {go_dir} does not exist")
        problems += 1

    if not os.path.isdir(structures_dir):
        print(f"[skip] structures: {structures_dir} does not exist "
              f"(training will run sequence-only)")
        return problems

    try:
        from datasets import load_dataset
    except ImportError:
        print("[skip] structures coverage: `datasets` not installed")
        return problems

    for repo, config, split in [
        ("wanglab/bioreason-pro-sft-reasoning-data", "default", "train"),
        ("wanglab/bioreason-pro-test-data", "default", "test"),
    ]:
        ds = load_dataset(repo, config, split=split)
        want = [p for p in ds["structure_path"] if p]
        have = sum(1 for p in want if os.path.exists(os.path.join(structures_dir, p)))
        pct = 100.0 * have / len(want) if want else 0.0
        ok = pct >= 99.0
        print(f"[{'ok ' if ok else 'FAIL'}] structures for {repo}: "
              f"{have}/{len(want)} resolve ({pct:.1f}%)")
        if not ok:
            print("        NOTE: unresolved structures are silently ignored at "
                  "training time (empty coordinates), so this must be ~100%.")
            problems += 1

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download BioReason-Pro structures and GO embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dest", default="data",
                    help="Parent directory; creates <dest>/structures and <dest>/go_embeddings")
    ap.add_argument("--structures-dir", default=None, help="Override structures destination")
    ap.add_argument("--go-embeddings-dir", default=None, help="Override GO embeddings destination")
    ap.add_argument("--skip-structures", action="store_true",
                    help="Only fetch GO embeddings (training then runs sequence-only)")
    ap.add_argument("--skip-go-embeddings", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="Parallel shard downloads")
    ap.add_argument("--keep-archives", action="store_true",
                    help="Keep the downloaded .tar.gz shards instead of deleting them")
    ap.add_argument("--all-structures", action="store_true",
                    help="Extract every structure in the shards (~370k files, ~150 GB) "
                         "instead of only the ~132k the released datasets reference (~57 GB)")
    ap.add_argument("--verify", action="store_true",
                    help="Only verify existing downloads against the released datasets")
    args = ap.parse_args()

    structures_dir = args.structures_dir or os.path.join(args.dest, "structures")
    go_dir = args.go_embeddings_dir or os.path.join(args.dest, "go_embeddings")

    if args.verify:
        return 1 if verify(structures_dir, go_dir) else 0

    if not args.skip_go_embeddings:
        download_go_embeddings(go_dir)
    if not args.skip_structures:
        download_structures(structures_dir, args.num_workers, args.keep_archives,
                            args.all_structures)

    print("\nVerifying …")
    problems = verify(structures_dir, go_dir)
    if problems:
        print(f"\n{problems} problem(s) found.")
        return 1

    print("\nAll good. Point the training script at:")
    print(f"  STRUCTURE_DIR={os.path.abspath(structures_dir)}")
    print(f"  GO_EMBEDDINGS_PATH={os.path.abspath(go_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
