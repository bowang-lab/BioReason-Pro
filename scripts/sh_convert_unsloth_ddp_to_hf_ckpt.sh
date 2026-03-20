#!/bin/bash

# Convert an Unsloth+DDP (Lightning) checkpoint to HuggingFace format.
# The input CHECKPOINT_PATH is a .ckpt file from PyTorch Lightning training.
# The output SAVE_DIR will contain the merged HF model ready for inference.
#
# Usage: ./sh_convert_unsloth_ddp_to_hf_ckpt.sh [checkpoint_path] [save_dir]
#   If no arguments provided, uses the defaults configured below.

# Run from project root
cd "$(dirname "$0")/.."

# ===================================================================================================
# Configuration — Set these to match your environment and training run
# ===================================================================================================

# Input: path to the Lightning .ckpt file from SFT training
CHECKPOINT_PATH=""                  # e.g., /data/checkpoints/sft/last.ckpt

# Output: where to save the converted HuggingFace model
SAVE_DIR=""                         # e.g., /data/checkpoints/sft/last-hf.ckpt

# Model configuration (must match your SFT training config)
TEXT_MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
PROTEIN_MODEL_NAME="esm3_sm_open_v1"

# Paths — set these to your local directories
CACHE_DIR=""                        # e.g., /data/bioreason/cache
GO_OBO_PATH=""                      # e.g., /path/to/go-basic.obo
GO_EMBEDDINGS_PATH=""               # e.g., /data/bioreason/go_embeddings

# Training hyperparameters (must match your SFT training config)
MAX_LENGTH_TEXT=10000
MAX_LENGTH_PROTEIN=2000
LORA_RANK=128
LORA_ALPHA=256
LORA_DROPOUT=0.05
PROTEIN_EMBEDDING_LAYER=37
GO_HIDDEN_DIM=512
GO_NUM_GAT_LAYERS=3
GO_NUM_HEADS=8
GO_NUM_REDUCED_EMBEDDINGS=200
GO_EMBEDDING_DIM=2560
UNIFIED_GO_ENCODER=True

# ===================================================================================================
# Validation
# ===================================================================================================
if [ "$#" -ge 1 ]; then
    CHECKPOINT_PATH="$1"
fi

if [ "$#" -ge 2 ]; then
    SAVE_DIR="$2"
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: $0 <checkpoint_path> [save_dir]"
    echo "Example: $0 /path/to/last.ckpt /path/to/output/hf_model"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file does not exist: $CHECKPOINT_PATH"
    exit 1
fi

echo "Converting Unsloth+DDP checkpoint to HuggingFace format..."
echo "Input: $CHECKPOINT_PATH"
echo "Output: $SAVE_DIR"

# ===================================================================================================
# Run conversion
# ===================================================================================================

python bioreason2/utils/save_unsloth_ckpt.py \
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
    --unified_go_encoder $UNIFIED_GO_ENCODER

if [ $? -eq 0 ]; then
    echo "Conversion completed successfully!"
    echo "HuggingFace model saved to: $SAVE_DIR"
    echo ""
    echo "Saved components:"
    echo "  - Text model (merged): $SAVE_DIR/"
    echo "  - Tokenizer: $SAVE_DIR/"
    echo "  - Protein projection: $SAVE_DIR/protein_projection.pt"
    echo "  - GO projection: $SAVE_DIR/go_projection.pt"
    echo "  - GO encoder: $SAVE_DIR/go_encoder.pt"
    echo "  - Protein model: $SAVE_DIR/protein_model/"
else
    echo "Conversion failed!"
    exit 1
fi
