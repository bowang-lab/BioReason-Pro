#!/bin/bash
# Score a directory of eval.py outputs with CAFA Fmax / weighted Fmax.
# INPUT_DIR is the --evals_path you passed to eval.py.

set -euo pipefail
cd "$(dirname "$0")"

INPUT_DIR="${1:-}"
OUTPUT_DIR="${2:-eval_results}"

if [ -z "$INPUT_DIR" ]; then
  echo "Usage: $0 <input_dir> [output_dir]" >&2
  exit 1
fi

python cafa_evals.py \
    --input_dir "$INPUT_DIR" \
    --ontology "../bioreason2/dataset/go-basic.obo" \
    --ia_file "../data/IA.txt" \
    --output_dir "$OUTPUT_DIR" \
    --reasoning_mode True \
    --final_answer_only False \
    --threads 0
