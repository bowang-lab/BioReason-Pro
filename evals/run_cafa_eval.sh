#!/bin/bash

python cafa_evals.py \
    --input_dir "" \
    --ontology "../data/go-basic.obo" \
    --ia_file "../data/IA.txt" \
    --output_dir "eval_results" \
    --reasoning_mode True \
    --final_answer_only False \
    --threads 0
