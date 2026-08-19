#!/bin/bash

# ===================================================================================================
# SLURM Configuration
# Adjust these to match your cluster. Example values shown.
# ===================================================================================================
# #SBATCH --job-name=train_protein_qwen_staged
# #SBATCH --partition=your_gpu_partition
# #SBATCH --time=120:00:00
# #SBATCH --gpus=8
# #SBATCH --ntasks-per-node=4
# #SBATCH --nodes=2
# #SBATCH --cpus-per-task=16
# #SBATCH --mem=256gb
# #SBATCH --output=train_protein_qwen_staged_%j_%x.out
# #SBATCH --error=train_protein_qwen_staged_%j_%x.err


# Run from project root
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ===================================================================================================
# Environment Setup
# Set these to your conda environment and project root.
# ===================================================================================================
# export PATH=/path/to/your/conda/envs/bin:$PATH     # e.g., /home/user/miniconda/envs/bio/bin
# ROOT_DIR=/path/to/BioReason-Pro                     # e.g., /home/user/BioReason-Pro
# cd $ROOT_DIR

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

unset SLURM_TRES_PER_TASK
# ===================================================================================================



# ===================================================================================================
# Shared Configuration
# ===================================================================================================
BASE_WANDB_PROJECT="bioreason-pro-finetune"
WANDB_ENTITY="${WANDB_ENTITY:-}"    # your W&B entity, or leave empty to use the default
TEXT_MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
EXPERIMENT_NAME="reasoning-sft"

# --- Cluster topology (must match the SBATCH header above) ---
NUM_NODES=2
BATCH_SIZE=4

# --- Paths: Set these to your local directories ---
BASE_CHECKPOINT_DIR=""              # e.g., /data/checkpoints
DATASET_CACHE_DIR=""                # e.g., /data/bioreason/data
CACHE_DIR=""                        # e.g., /data/bioreason/cache
STRUCTURE_DIR=""                    # e.g., /data/bioreason/structures  (optional, see README)
GO_EMBEDDINGS_PATH=""               # e.g., /data/bioreason/go_embeddings  (build with bioreason2/utils/go_embed.py)
GO_OBO_PATH="$REPO_ROOT/bioreason2/dataset/go-basic.obo"

# --- Dataset Configuration ---
# The SFT corpus is its own HuggingFace repo with a single "default" config.
# HF_DATASET_REPO is passed to --cafa5_dataset and REASONING_DATASET_NAME to
# --reasoning_dataset_name, so these two must be a valid (repo, config) pair.
HF_DATASET_REPO="wanglab/bioreason-pro-sft-reasoning-data"
REASONING_DATASET_NAME="default"
STAGE1_DATASET_NAME="default"
STAGE2_DATASET_NAME="default"
STAGE2_DATASET_WEIGHTS="1"
GO_GPT_PREDICTIONS_COLUMN="go_pred"
INCLUDE_GROUND_TRUTH_IN_FINAL_ANSWER=False
ADD_UNIPROT_SUMMARY=True
IS_SWISSPROT=False

# --- Model Configuration ---
MAX_LENGTH_TEXT=10000
MAX_LENGTH_PROTEIN=2000
LORA_RANK=128
LORA_ALPHA=256
LORA_DROPOUT=0          # the released checkpoint was trained with dropout disabled
ESM_LAYER=37

INTERPRO_IN_PROMPT=True
PREDICT_INTERPRO=False
PPI_IN_PROMPT=True


BASE_COMMAND="srun python train_protein_llm.py \
    --cache_dir $CACHE_DIR \
    ${WANDB_ENTITY:+--wandb_entity $WANDB_ENTITY} \
    --text_model_name ${TEXT_MODEL_NAME} \
    --protein_model_name esm3_sm_open_v1 \
    --strategy ddp_find_unused_parameters_false \
    --use_qlora False \
    --use_unsloth True \
    --num_gpus -1 \
    --batch_size $BATCH_SIZE \
    --num_nodes $NUM_NODES \
    --gradient_accumulation_steps 1 \
    --model_type protein-llm \
    --dataset_type cafa5 \
    --cafa5_dataset $HF_DATASET_REPO \
    --reasoning_dataset_name $REASONING_DATASET_NAME \
    --go_gpt_predictions_column $GO_GPT_PREDICTIONS_COLUMN \
    --include_ground_truth_in_final_answer $INCLUDE_GROUND_TRUTH_IN_FINAL_ANSWER \
    --add_uniprot_summary $ADD_UNIPROT_SUMMARY \
    --is_swissprot $IS_SWISSPROT \
    --dataset_cache_dir $DATASET_CACHE_DIR \
    --cache_dir $CACHE_DIR \
    --structure_dir $STRUCTURE_DIR \
    --val_split_ratio 0.1 \
    --max_length_protein $MAX_LENGTH_PROTEIN \
    --max_length_text $MAX_LENGTH_TEXT \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --protein_model_finetune False \
    --protein_embedding_layer $ESM_LAYER \
    --go_model_finetune True \
    --attn_implementation flash_attention_2 \
    --go_obo_path $GO_OBO_PATH \
    --precomputed_embeddings_path $GO_EMBEDDINGS_PATH \
    --go_hidden_dim 512 \
    --go_num_gat_layers 3 \
    --go_num_heads 8 \
    --go_num_reduced_embeddings 200 \
    --go_embedding_dim 2560 \
    --unified_go_encoder True \
    --return_answer_in_batch False \
    --num_workers 8 \
    --weight_decay 0.01 \
    --seed 23 \
    --save_top_k 1 \
    --include_go_defs False \
    --split_go_aspects False \
    --interpro_in_prompt $INTERPRO_IN_PROMPT \
    --predict_interpro $PREDICT_INTERPRO \
    --ppi_in_prompt $PPI_IN_PROMPT \
    --include_protein_function_summary True \
    --min_go_mf_freq 1 \
    --min_go_bp_freq 1 \
    --min_go_cc_freq 1 \
    --apply_go_filtering_to_val_test False \
    --log_every_n_steps 200 \
    --enable_sample_generation True \
    --verbose_sample_generation False \
    --val_check_interval 0.2 \
    --debug False"
# ===================================================================================================


# ===================================================================================================
# Preflight: fail loudly on unset paths instead of creating stray relative dirs
# ===================================================================================================
for v in BASE_CHECKPOINT_DIR DATASET_CACHE_DIR CACHE_DIR GO_EMBEDDINGS_PATH GO_OBO_PATH; do
  if [ -z "${!v}" ]; then
    echo "Error: $v is not set. Edit the 'Paths' block at the top of this script." >&2
    exit 1
  fi
done
if [ ! -f "$GO_OBO_PATH" ]; then
  echo "Error: GO_OBO_PATH does not exist: $GO_OBO_PATH" >&2
  exit 1
fi
if [ ! -d "$GO_EMBEDDINGS_PATH" ] || [ -z "$(ls -A "$GO_EMBEDDINGS_PATH" 2>/dev/null)" ]; then
  echo "Error: GO_EMBEDDINGS_PATH is empty or missing: $GO_EMBEDDINGS_PATH" >&2
  echo "       Build it first:  python -m bioreason2.utils.go_embed --output_dir $GO_EMBEDDINGS_PATH" >&2
  exit 1
fi
if [ -n "$STRUCTURE_DIR" ] && [ ! -d "$STRUCTURE_DIR" ]; then
  echo "Error: STRUCTURE_DIR is set but does not exist: $STRUCTURE_DIR" >&2
  exit 1
fi
# ===================================================================================================



# ===================================================================================================
# --- Stage 1: Warm-up (Projector + GO Training) ---
#
# OFF by default: the released BioReason-Pro SFT checkpoint was trained with
# Stage 2 only, starting from randomly-initialised projectors. Set RUN_STAGE1=true
# to run the warm-up first and initialise Stage 2 from its projector / GO weights.
# ===================================================================================================
RUN_STAGE1="${RUN_STAGE1:-false}"
STAGE1_ARGS=""

if [ "$RUN_STAGE1" = "true" ]; then
echo "--- Starting Stage 1: Projector Training"

RUN_NAME_S1_DIR="${BASE_WANDB_PROJECT}-$(basename ${TEXT_MODEL_NAME})-stage1"
TIMESTAMP_S1=$(date +%Y%m%d-%H%M%S)
STAGE1_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/${RUN_NAME_S1_DIR}-${TIMESTAMP_S1}"
WANDB_RUN_NAME_S1="stage1-$(basename ${TEXT_MODEL_NAME})-${EXPERIMENT_NAME}"
mkdir -p $STAGE1_CHECKPOINT_DIR

stdbuf -oL -eL $BASE_COMMAND \
  --run_name "${WANDB_RUN_NAME_S1}" \
  --cafa5_dataset_name $STAGE1_DATASET_NAME \
  --training_stage 1 \
  --max_epochs 1 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.1 \
  --text_model_finetune False \
  --checkpoint_dir $STAGE1_CHECKPOINT_DIR \
  --wandb_project "${BASE_WANDB_PROJECT}-stage1"

PROJECTOR_WEIGHTS_PATH="$STAGE1_CHECKPOINT_DIR/projector_weights.pt"
if [ ! -f "$PROJECTOR_WEIGHTS_PATH" ]; then
  echo "Error: Stage 1 failed. Projector weights not found at $PROJECTOR_WEIGHTS_PATH"
  exit 1
fi

GO_PROJECTOR_WEIGHTS_PATH="$STAGE1_CHECKPOINT_DIR/go_projection_weights.pt"
GO_ENCODER_WEIGHTS_PATH="$STAGE1_CHECKPOINT_DIR/go_encoder_weights.pt"
if [ ! -f "$GO_PROJECTOR_WEIGHTS_PATH" ] || [ ! -f "$GO_ENCODER_WEIGHTS_PATH" ]; then
  echo "Error: Stage 1 failed. GO projector or GO encoder weights not found"
  exit 1
fi

echo "--- Stage 1 Complete. Projector weights saved to $PROJECTOR_WEIGHTS_PATH"
echo "--- Stage 1 Complete. GO projector weights saved to $GO_PROJECTOR_WEIGHTS_PATH"
echo "--- Stage 1 Complete. GO encoder weights saved to $GO_ENCODER_WEIGHTS_PATH"

STAGE1_ARGS="--projector_checkpoint_path $PROJECTOR_WEIGHTS_PATH \
    --go_projection_checkpoint_path $GO_PROJECTOR_WEIGHTS_PATH \
    --go_encoder_checkpoint_path $GO_ENCODER_WEIGHTS_PATH"
else
echo "--- Skipping Stage 1 (RUN_STAGE1=false); Stage 2 starts from scratch"
fi
# ===================================================================================================



# ===================================================================================================
# --- Stage 2: Full Model Fine-tuning ---
# ===================================================================================================
echo "--- Starting Stage 2: Full Model Fine-tuning"

RUN_NAME_S2_DIR="${BASE_WANDB_PROJECT}-$(basename ${TEXT_MODEL_NAME})-${EXPERIMENT_NAME}-stage2"
STAGE2_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/${RUN_NAME_S2_DIR}"
WANDB_RUN_NAME_S2="stage2-$(basename ${TEXT_MODEL_NAME})-${EXPERIMENT_NAME}"
mkdir -p $STAGE2_CHECKPOINT_DIR

CKPT_ARG=""
LATEST_CKPT="${STAGE2_CHECKPOINT_DIR}/last.ckpt"

if [ -e "$LATEST_CKPT" ]; then
    echo "   Found existing checkpoint at $LATEST_CKPT, resuming training"
    CKPT_ARG="--ckpt_path $LATEST_CKPT"
else
    echo "   No existing checkpoint found, starting fresh training"
    CKPT_ARG=""
fi

stdbuf -oL -eL $BASE_COMMAND \
    --run_name "${WANDB_RUN_NAME_S2}" \
    --cafa5_dataset_name $STAGE2_DATASET_NAME \
    --training_stage 2 \
    --max_epochs 10 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --text_model_finetune True \
    $STAGE1_ARGS \
    --checkpoint_dir $STAGE2_CHECKPOINT_DIR \
    --every_n_train_steps 10000000000 \
    --wandb_project "${BASE_WANDB_PROJECT}" \
    $CKPT_ARG
# ===================================================================================================
