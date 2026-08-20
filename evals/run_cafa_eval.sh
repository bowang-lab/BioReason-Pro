#!/bin/bash
# Score a directory of eval.py outputs with CAFA Fmax / weighted Fmax.
# INPUT_DIR is the --evals_path you passed to eval.py.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input_dir> [output_dir]" >&2
  exit 1
fi

# Resolve paths BEFORE cd, so relative arguments work from the caller's directory.
INPUT_DIR="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUTPUT_DIR="${2:-eval_results}"
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$PWD/$OUTPUT_DIR" ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/evals"

python cafa_evals.py \
    --input_dir "$INPUT_DIR" \
    --ontology "$REPO_ROOT/bioreason2/dataset/go-basic.obo" \
    --ia_file "$REPO_ROOT/data/IA.txt" \
    --output_dir "$OUTPUT_DIR" \
    --reasoning_mode True \
    --final_answer_only False \
    --threads 0
