#!/bin/bash

# ===================================================================================================
# SLURM Configuration
# Adjust these to match your cluster. Example values shown.
# ===================================================================================================
# #SBATCH --job-name=eval_bioreason
# #SBATCH --partition=your_gpu_partition
# #SBATCH --account=your_account
# #SBATCH --time=24:00:00
# #SBATCH --nodes=1
# #SBATCH --gpus=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=128gb
# #SBATCH --output=eval_bioreason_%j.out
# #SBATCH --error=eval_bioreason_%j.err

set -eo pipefail

# Run from project root
cd "$(dirname "$0")/.."

# ===================================================================================================
# Environment Setup
# Set these to your conda environment and project root.
# ===================================================================================================
# export PATH=/path/to/your/conda/envs/bin:$PATH     # e.g., /home/user/miniconda/envs/bio/bin
# source /path/to/conda/etc/profile.d/conda.sh       # e.g., /home/user/miniconda/etc/profile.d/conda.sh
# conda activate your_env_name
# ROOT_DIR=/path/to/BioReason-Pro                     # e.g., /home/user/BioReason-Pro
# cd $ROOT_DIR

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

EVALS_DIR="./evals"
mkdir -p "$EVALS_DIR"

export TRANSFORMERS_CACHE="$EVALS_DIR/transformers"
mkdir -p "$TRANSFORMERS_CACHE"

# ===================================================================================================
# Model Checkpoint
# Download from Hugging Face and set the path below.
# ===================================================================================================

# SFT checkpoint — download from https://huggingface.co/wanglab/bioreason-pro-sft
# MODEL_PATH="/path/to/bioreason-pro-sft"

# RL checkpoint — download from https://huggingface.co/wanglab/bioreason-pro-rl
MODEL_PATH="/path/to/bioreason-pro-rl"

# ===================================================================================================
# Paths: Set these to your local directories
# ===================================================================================================
PROTEIN_MODEL_NAME="esm3_sm_open_v1"
GO_OBO_PATH=""                      # e.g., /path/to/go-basic.obo
GO_EMBEDDINGS_PATH=""               # e.g., /data/bioreason/go_embeddings
DATASET_CACHE_DIR=""                # e.g., /data/bioreason/data
STRUCTURE_DIR=""                    # e.g., /data/bioreason/structures

EVAL_SCRIPT="eval.py"
EVALS_PATH="$EVALS_DIR/results"

# Chunking parameters (can be overridden via command line)
NUM_CHUNKS=${NUM_CHUNKS:-1}
CHUNK_ID=${CHUNK_ID:-0}

# ===================================================================================================
# Evaluation Parameters
# ===================================================================================================
MAX_SAMPLES=-1
MAX_LENGTH_PROTEIN=2000
MAX_NEW_TOKENS=5000
TEMPERATURE=0
TOP_P=0.95
REPETITION_PENALTY=1.0
PASS_AT_K=1
PROTEIN_EMBEDDING_LAYER=37
UNIFIED_GO_ENCODER=True

# GO Model Architecture (must match training config)
GO_HIDDEN_DIM=512
GO_NUM_GAT_LAYERS=3
GO_NUM_HEADS=8
GO_NUM_REDUCED_EMBEDDINGS=200
GO_EMBEDDING_DIM=2560

# Dataset configuration
DATASET_NAME="interlabel_test_dataset_with_gogpt_memorized_copy"
REASONING_DATASET_NAME="interlabel_test_dataset_with_gogpt_memorized_copy"
SPLIT_GO_ASPECTS=False
INTERPRO_IN_PROMPT=True
PREDICT_INTERPRO=False
PPI_IN_PROMPT=False
GO_GPT_PREDICTIONS_COLUMN="go_pred"
ADD_UNIPROT_SUMMARY=True

# GO Filtering (must match training config)
MIN_GO_MF_FREQ=1
MIN_GO_BP_FREQ=1
MIN_GO_CC_FREQ=1
APPLY_GO_FILTERING_TO_VAL_TEST=False

# ===================================================================================================
# Execute Evaluation
# ===================================================================================================
echo "Starting reasoning evaluation..."
echo "Model checkpoint: $MODEL_PATH"
echo "Protein model: $PROTEIN_MODEL_NAME"
echo "Results will be saved to: $EVALS_PATH"

mkdir -p "$EVALS_PATH"

CHUNK_ARGS=""
if [ "$NUM_CHUNKS" -gt 1 ]; then
    CHUNK_ARGS="--num_chunks $NUM_CHUNKS --chunk_id $CHUNK_ID"
    echo "Processing chunk $((CHUNK_ID + 1))/$NUM_CHUNKS"
fi

python "$EVAL_SCRIPT" \
    --ckpt_dir "$MODEL_PATH" \
    --protein_model_name "$PROTEIN_MODEL_NAME" \
    --protein_embedding_layer "$PROTEIN_EMBEDDING_LAYER" \
    --go_obo_path "$GO_OBO_PATH" \
    --precomputed_embeddings_path "$GO_EMBEDDINGS_PATH" \
    --unified_go_encoder "$UNIFIED_GO_ENCODER" \
    --go_hidden_dim $GO_HIDDEN_DIM \
    --go_num_gat_layers $GO_NUM_GAT_LAYERS \
    --go_num_heads $GO_NUM_HEADS \
    --go_num_reduced_embeddings $GO_NUM_REDUCED_EMBEDDINGS \
    --go_embedding_dim $GO_EMBEDDING_DIM \
    --cafa5_dataset "wanglab/cafa5" \
    --cafa5_dataset_name "$DATASET_NAME" \
    --reasoning_dataset_name "$REASONING_DATASET_NAME" \
    --go_gpt_predictions_column "$GO_GPT_PREDICTIONS_COLUMN" \
    --dataset_cache_dir "$DATASET_CACHE_DIR" \
    --structure_dir "$STRUCTURE_DIR" \
    --split_go_aspects "$SPLIT_GO_ASPECTS" \
    --interpro_in_prompt "$INTERPRO_IN_PROMPT" \
    --predict_interpro "$PREDICT_INTERPRO" \
    --ppi_in_prompt "$PPI_IN_PROMPT" \
    --add_uniprot_summary "$ADD_UNIPROT_SUMMARY" \
    --min_go_mf_freq $MIN_GO_MF_FREQ \
    --min_go_bp_freq $MIN_GO_BP_FREQ \
    --min_go_cc_freq $MIN_GO_CC_FREQ \
    --apply_go_filtering_to_val_test $APPLY_GO_FILTERING_TO_VAL_TEST \
    --seed 23 \
    --debug False \
    --max_samples $MAX_SAMPLES \
    --max_length_protein $MAX_LENGTH_PROTEIN \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --pass_at_k $PASS_AT_K \
    --repetition_penalty $REPETITION_PENALTY \
    --evals_path "$EVALS_PATH" \
    $CHUNK_ARGS

echo "Evaluation finished. Results are in $EVALS_PATH"
