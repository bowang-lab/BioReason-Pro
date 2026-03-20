#!/bin/bash

# Convert a GRPO checkpoint to HuggingFace format.
# The input CHECKPOINT_PATH is the output of GRPO training (e.g., checkpoint-700/).
# The output SAVE_DIR will contain the merged HF model ready for inference.

# Run from project root
cd "$(dirname "$0")/.."

# ===================================================================================================
# Configuration — Set these to match your environment and training run
# ===================================================================================================

# Input: path to the GRPO training checkpoint directory
CHECKPOINT_PATH=""                  # e.g., /data/checkpoints/grpo/checkpoint-700

# Output: where to save the converted HuggingFace model
SAVE_DIR=""                         # e.g., /data/checkpoints/grpo/checkpoint-700-hf

# Model configuration (must match your GRPO training config)
TEXT_MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
PROTEIN_MODEL_NAME="esm3_sm_open_v1"

# Paths — set these to your local directories
CACHE_DIR=""                        # e.g., /data/bioreason/cache
GO_OBO_PATH=""                      # e.g., /path/to/go-basic.obo
GO_EMBEDDINGS_PATH=""               # e.g., /data/bioreason/go_embeddings

# Training hyperparameters (must match your GRPO training config)
MAX_LENGTH_TEXT=10000
MAX_LENGTH_PROTEIN=2000
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Protein and GO settings (must match training config)
PROTEIN_EMBEDDING_LAYER=37
GO_HIDDEN_DIM=512
GO_NUM_GAT_LAYERS=3
GO_NUM_HEADS=8
GO_NUM_REDUCED_EMBEDDINGS=200
GO_EMBEDDING_DIM=2560
UNIFIED_GO_ENCODER=True
PROTEIN_MODEL_FINETUNE=False

# ===================================================================================================
# Run conversion
# ===================================================================================================

python bioreason2/utils/save_grpo_ckpt.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --save_dir "$SAVE_DIR" \
    --text_model_name "$TEXT_MODEL_NAME" \
    --protein_model_name "$PROTEIN_MODEL_NAME" \
    --cache_dir "$CACHE_DIR" \
    --max_length_text $MAX_LENGTH_TEXT \
    --max_length_protein $MAX_LENGTH_PROTEIN \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --protein_embedding_layer $PROTEIN_EMBEDDING_LAYER \
    --go_obo_path "$GO_OBO_PATH" \
    --precomputed_embeddings_path "$GO_EMBEDDINGS_PATH" \
    --go_hidden_dim $GO_HIDDEN_DIM \
    --go_num_gat_layers $GO_NUM_GAT_LAYERS \
    --go_num_heads $GO_NUM_HEADS \
    --go_num_reduced_embeddings $GO_NUM_REDUCED_EMBEDDINGS \
    --go_embedding_dim $GO_EMBEDDING_DIM \
    --unified_go_encoder $UNIFIED_GO_ENCODER \
    --protein_model_finetune $PROTEIN_MODEL_FINETUNE
