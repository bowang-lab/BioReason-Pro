import os
import shutil
import torch
from bioreason2.models.protein_llm import ProteinLLMModel, _get_target_modules
from pathlib import Path
import argparse
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from bioreason2.utils.argparse_utils import str2bool


def _setup_lora_for_checkpoint_loading(
    model: ProteinLLMModel,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
):
    """Setup LoRA for checkpoint loading without full training preparation"""
    print(f"🔧 Setting up LoRA for checkpoint loading (rank={lora_rank}, alpha={lora_alpha})")
    
    # Get target modules
    target_modules = _get_target_modules(model)
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Prepare text model for LoRA
    model.text_model = prepare_model_for_kbit_training(model.text_model)
    model.text_model = get_peft_model(model.text_model, lora_config)
    
    print("✅ LoRA setup complete for checkpoint loading")
    return lora_config


class DeepSpeedCheckpointAnalyzer:
    """Analyzes and works with DeepSpeed checkpoint structure"""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.checkpoint = None

    def load_deepspeed_checkpoint(self):
        """Load the DeepSpeed model states checkpoint"""
        checkpoint_dir = Path(self.checkpoint_path)

        # Try different possible checkpoint files
        possible_files = [
            "checkpoint/mp_rank_00_model_states.pt",  # DeepSpeed model states
            "output_dir/pytorch_model.bin",  # Alternative format
            "pytorch_model.bin",  # Direct format
        ]

        checkpoint_file = None
        for filename in possible_files:
            full_path = checkpoint_dir / filename
            if full_path.exists():
                checkpoint_file = full_path
                break

        if checkpoint_file is None:
            raise FileNotFoundError(f"Could not find model checkpoint in {checkpoint_dir}")

        print(f"📥 Loading DeepSpeed checkpoint: {checkpoint_file}")

        # Load with weights_only=False for DeepSpeed compatibility
        try:
            self.checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
            print("✅ Successfully loaded checkpoint")
        except Exception as e:
            print(f"❌ Failed to load with weights_only=False: {e}")
            # Try with weights_only=True as fallback
            self.checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
            print("✅ Successfully loaded checkpoint with weights_only=True")

    def extract_model_state_dict(self):
        """Extract the clean model state dict from DeepSpeed checkpoint"""
        if self.checkpoint is None:
            raise ValueError("Checkpoint not loaded. Call load_deepspeed_checkpoint() first.")

        # Get the module state dict
        if "module" in self.checkpoint:
            state_dict = self.checkpoint["module"]
        else:
            state_dict = self.checkpoint

        print(f"📤 Extracting model state dict with {len(state_dict)} parameters...")
        return state_dict



def save_ckpt(args):
    # Use the unified loading pipeline from analysis.py (same as reason.py)
    print("🔄 Loading model via DeepSpeedCheckpointAnalyzer ...")
    analyzer = DeepSpeedCheckpointAnalyzer(args.checkpoint_path)
    analyzer.load_deepspeed_checkpoint()
    # Extract raw state dict and load with identical logic to reason.py to avoid key mismatches
    state_dict = analyzer.extract_model_state_dict()

    print("🔧 Building base ProteinLLMModel …")

    # Create a custom model using current ProteinLLM architecture
    model = ProteinLLMModel(
        text_model_name=args.text_model_name,
        protein_model_name=args.protein_model_name,
        cache_dir=args.cache_dir,
        max_length_protein=args.max_length_protein,
        max_length_text=args.max_length_text,
        text_model_finetune=True,
        protein_model_finetune=args.protein_model_finetune,
        protein_embedding_layer=args.protein_embedding_layer,
        go_model_finetune=True,
        attn_implementation="flash_attention_2",
        go_obo_path=args.go_obo_path,
        precomputed_embeddings_path=args.precomputed_embeddings_path,
        go_hidden_dim=args.go_hidden_dim,
        go_num_gat_layers=args.go_num_gat_layers,
        go_num_heads=args.go_num_heads,
        go_num_reduced_embeddings=args.go_num_reduced_embeddings,
        go_embedding_dim=args.go_embedding_dim,
        unified_go_encoder=args.unified_go_encoder,
    )

    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"📍 Model moved to {device}")

    # ------------------------------------------------------------------
    # Load pretrained GO components if paths are provided
    # ------------------------------------------------------------------
    if hasattr(args, "go_projection_path") and args.go_projection_path and os.path.exists(args.go_projection_path):
        if hasattr(model, "go_projection") and model.go_projection is not None:
            print(f"🔧 Loading GO projection weights from: {args.go_projection_path}")
            go_projection_state_dict = torch.load(args.go_projection_path, map_location=device)
            model.go_projection.load_state_dict(go_projection_state_dict)
            print("✅ GO projection weights loaded successfully")
        else:
            print("⚠️ GO projection path provided but model has no GO projection component")

    if hasattr(args, "go_encoder_path") and args.go_encoder_path and os.path.exists(args.go_encoder_path):
        if hasattr(model, "go_encoder") and model.go_encoder is not None:
            print(f"🔧 Loading GO encoder weights from: {args.go_encoder_path}")
            # Handle both .pt files and directories
            if args.go_encoder_path.endswith(".pt"):
                go_encoder_state_dict = torch.load(args.go_encoder_path, map_location=device)
                model.go_encoder.load_state_dict(go_encoder_state_dict)
            else:
                # Try to load as a directory with pytorch_model.bin
                go_encoder_file = os.path.join(args.go_encoder_path, "pytorch_model.bin")
                if os.path.exists(go_encoder_file):
                    go_encoder_state_dict = torch.load(go_encoder_file, map_location=device)
                    model.go_encoder.load_state_dict(go_encoder_state_dict)
                else:
                    print(f"⚠️ Could not find GO encoder weights in {args.go_encoder_path}")
            print("✅ GO encoder weights loaded successfully")
        else:
            print("⚠️ GO encoder path provided but model has no GO encoder component")

    # ------------------------------------------------------------------
    # CRITICAL: Check vocabulary size compatibility with checkpoint FIRST
    # ------------------------------------------------------------------
    print("🔧 Checking vocabulary size compatibility...")

    # First, let's check what vocab size the checkpoint expects
    # Handle PEFT-style embedding keys with different suffixes
    checkpoint_vocab_size = None
    for k in state_dict.keys():
        if "embed_tokens" in k and (
            "weight" in k or "original_module.weight" in k or "modules_to_save.default.weight" in k
        ):
            checkpoint_vocab_size = state_dict[k].shape[0]
            print(f"📊 Found embedding key: {k} with vocab size: {checkpoint_vocab_size}")
            break

    if checkpoint_vocab_size:
        current_vocab_size = len(model.text_tokenizer)
        print(f"📊 Checkpoint vocab size: {checkpoint_vocab_size}")
        print(f"📊 Current vocab size: {current_vocab_size}")

        # If vocab sizes don't match, we have a problem with special tokens
        if current_vocab_size != checkpoint_vocab_size:
            print(f"⚠️  Vocab size mismatch! Checkpoint has {checkpoint_vocab_size}, model has {current_vocab_size}")
            print("🔧 Will resize embeddings to match checkpoint after LoRA preparation")

    # ------------------------------------------------------------------
    # Setup LoRA for checkpoint loading so that all LoRA modules exist **before**
    # we load the checkpoint.
    # ------------------------------------------------------------------
    _setup_lora_for_checkpoint_loading(
        model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # ------------------------------------------------------------------
    # CRITICAL: After LoRA prep, resize embeddings to match checkpoint
    # ------------------------------------------------------------------
    if checkpoint_vocab_size:
        print("🔧 Post-LoRA: Ensuring embedding sizes match checkpoint...")

        try:
            # Get current embedding size after LoRA prep
            if hasattr(model.text_model, "base_model") and hasattr(model.text_model.base_model, "model"):
                actual_model = model.text_model.base_model.model
                current_embed_size = actual_model.model.embed_tokens.weight.shape[0]
                current_lm_head_size = actual_model.lm_head.weight.shape[0]
            else:
                current_embed_size = model.text_model.model.embed_tokens.weight.shape[0]
                current_lm_head_size = model.text_model.lm_head.weight.shape[0]

            print(f"📊 Current embed_tokens size: {current_embed_size}")
            print(f"📊 Current lm_head size: {current_lm_head_size}")
            print(f"📊 Target vocab size: {checkpoint_vocab_size}")

            if current_embed_size != checkpoint_vocab_size or current_lm_head_size != checkpoint_vocab_size:
                print(f"🔧 Resizing embeddings to match checkpoint ({checkpoint_vocab_size})")

                # Resize using the correct model reference
                if hasattr(model.text_model, "base_model") and hasattr(model.text_model.base_model, "model"):
                    model.text_model.base_model.model.resize_token_embeddings(checkpoint_vocab_size)
                else:
                    model.text_model.resize_token_embeddings(checkpoint_vocab_size)

                print(f"✅ Successfully resized embeddings to {checkpoint_vocab_size}")
            else:
                print("✅ Embedding sizes already match checkpoint")

        except Exception as e:
            print(f"⚠️  Error checking/resizing embeddings: {e}")
            print("🔧 Will attempt to load checkpoint anyway...")
    else:
        print("⚠️  Could not detect checkpoint vocabulary size")

    # ---- key remapping identical to reason.py ----
    def new_key(k: str) -> str:
        if k.startswith("=model."):  # deepspeed save
            return k[6:]
        if k.startswith("_forward_module."):
            return k[len("_forward_module.") :]
        return k

    magic = {new_key(k): v for k, v in state_dict.items()}

    model_keys = set(model.state_dict().keys())
    print(f"📊 Model has {len(model_keys)} keys")
    print(f"📊 Checkpoint has {len(magic)} keys")

    # Print some sample model keys to understand the structure
    sample_model_keys = [k for k in model_keys if "text_model" in k][:5]
    print("🔍 Sample model keys:")
    for key in sample_model_keys:
        print(f"   • {key}")

    sample_checkpoint_keys = [k for k in magic.keys() if "text_model" in k][:5]
    print("🔍 Sample checkpoint keys:")
    for key in sample_checkpoint_keys:
        print(f"   • {key}")

    remapped = {}
    for k, v in magic.items():
        new_k = k

        # ------------------------------------------------------------------
        # FIXED KEY MAPPING - Handle the actual checkpoint structure
        # ------------------------------------------------------------------

        # 1) Strip the leading "model." that appears in DeepSpeed checkpoints
        if new_k.startswith("model."):
            new_k = new_k[len("model.") :]

        # 2) Handle PEFT-style embedding keys FIRST before general text model mapping
        # PEFT saves embeddings with special suffixes that need to be handled
        if "embed_tokens" in new_k or "lm_head" in new_k:
            # Handle PEFT embedding patterns:
            # text_model.base_model.model.model.embed_tokens.original_module.weight -> text_model_hf.base_model.model.model.embed_tokens.weight
            # text_model.base_model.model.model.embed_tokens.modules_to_save.default.weight -> text_model_hf.base_model.model.model.embed_tokens.weight

            if "original_module.weight" in new_k:
                # Use the original module weights
                new_k = new_k.replace(".original_module.weight", ".weight")
                print(f"🔧 Mapping original_module embedding: {k} -> {new_k}")
            elif "modules_to_save.default.weight" in new_k:
                # Use the trained/modified weights (prefer these over original)
                new_k = new_k.replace(".modules_to_save.default.weight", ".weight")
                print(f"🔧 Mapping modules_to_save embedding: {k} -> {new_k}")

            # Now apply standard text model mapping (only if not already in PEFT format)
            if new_k.startswith("text_model.") and not new_k.startswith("text_model.base_model.model."):
                suffix = new_k[len("text_model.") :]
                new_k = f"text_model.base_model.model.{suffix}"

        # 3) Map text model components - FIXED FOR CORRECT PEFT STRUCTURE!
        # The checkpoint may already have correct PEFT structure: text_model.base_model.model.*
        # Only apply transformation if it doesn't already have the correct structure
        elif new_k.startswith("text_model.") and not new_k.startswith("text_model.base_model.model."):
            suffix = new_k[len("text_model.") :]  # Get everything after "text_model."
            new_k = f"text_model.base_model.model.{suffix}"

        # 4) Map Protein components - remove duplicate "model." prefix
        elif new_k.startswith("protein_model."):
            # Keep as is - this should map to model.protein_model.*
            pass
        elif new_k.startswith("protein_projection."):
            # Keep as is - this should map to model.protein_projection.*
            pass

        # 5) Handle the case where we have both "model.protein_projection.*" and "protein_projection.*"
        # Prefer the version without "model." prefix to avoid duplication
        if k.startswith("model.protein_projection.") and f"protein_projection.{k.split('.')[-1]}" in magic:
            print(f"⚠️  Skipping duplicate key: {k} (using protein_projection version)")
            continue

        # 6) Skip if we already have this key (prefer modules_to_save over original_module)
        if new_k in remapped and "modules_to_save" in k:
            print(f"🔧 Replacing with modules_to_save version: {new_k}")
        elif new_k in remapped and "original_module" in k:
            print(f"⚠️  Skipping original_module (already have modules_to_save): {k}")
            continue

        # Move tensor to the same device as the model
        if isinstance(v, torch.Tensor):
            v = v.to(device)

        remapped[new_k] = v

    magic = remapped
    print(f"🔄 After key mapping: {len(magic)} keys")

    # After remapping keys, filter out 4-bit base layer weights
    filtered_magic = {}
    for k, v in magic.items():
        # Skip 4-bit quantized base layer weights and their metadata
        if '.base_layer.weight' in k and v.shape[-1] == 1:
            print(f"⚠️  Skipping 4-bit quantized weight: {k}")
            continue
        if any(x in k for x in ['.absmax', '.quant_map', '.quant_state', '.nested_absmax', '.nested_quant_map']):
            print(f"⚠️  Skipping quantization metadata: {k}")
            continue
        filtered_magic[k] = v

    magic = filtered_magic
    print(f"🔄 After filtering quantized weights: {len(magic)} keys")

    # Load the state dict
    result = model.load_state_dict(magic, strict=False)

    print(
        f"📥 load_state_dict completed → missing {len(result.missing_keys)} | unexpected {len(result.unexpected_keys)}"
    )
    if result.missing_keys:
        print("⚠️  Sample missing keys:", result.missing_keys[:5])
    if result.unexpected_keys:
        print("⚠️  Sample unexpected keys:", result.unexpected_keys[:5])

    # Check if LoRA was loaded and merge it BEFORE saving
    if hasattr(model.text_model, "peft_config"):
        print("🔗 Merging LoRA adapters...")
        model.text_model = model.text_model.merge_and_unload()
        print("✅ LoRA adapters merged into base model")
    else:
        print("⚠️  No LoRA adapters found to merge")

    # Safety check: only remove if it's clearly a checkpoint/model save directory
    if os.path.exists(args.save_dir):
        # Check if directory looks like a model save directory (contains model files)
        is_model_dir = any(
            f in os.listdir(args.save_dir)
            for f in [
                "pytorch_model.bin",
                "model.safetensors",
                "config.json",
                "tokenizer.json",
            ]
            if os.path.isfile(os.path.join(args.save_dir, f))
        )

        # Extra safety: don't remove if directory looks like a source code directory
        unsafe_patterns = [
            ".git",
            "src",
            "bioreason2",
            "__pycache__",
            "train_",
            "test_",
        ]
        is_unsafe = any(pattern in args.save_dir.lower() for pattern in unsafe_patterns)

        if is_model_dir and not is_unsafe:
            print(f"🗑️ Removing existing model directory: {args.save_dir}")
            shutil.rmtree(args.save_dir)
        elif not is_unsafe:
            print(
                f"⚠️ Directory exists but doesn't look like a model directory. Will create alongside existing files: {args.save_dir}"
            )
        else:
            print(f"🚫 Refusing to remove directory that may contain important files: {args.save_dir}")
            print("Please specify a different save directory or manually remove the existing one.")
            return

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"💾 Saving complete merged model to {args.save_dir}...")

    # Move model back to CPU for saving to avoid memory issues
    model = model.cpu()

    # Save text model and tokenizer (now merged, no LoRA)
    model.text_model.save_pretrained(args.save_dir)
    model.text_tokenizer.save_pretrained(args.save_dir)

    # Save protein projection layer
    protein_projection_path = os.path.join(args.save_dir, "protein_projection.pt")
    torch.save(model.protein_projection.state_dict(), protein_projection_path)
    print(f"✅ Protein projection saved to {protein_projection_path}")

    # Save GO projection layer if it exists
    if hasattr(model, "go_projection") and model.go_projection is not None:
        go_projection_path = os.path.join(args.save_dir, "go_projection.pt")
        torch.save(model.go_projection.state_dict(), go_projection_path)
        print(f"✅ GO projection saved to {go_projection_path}")
    else:
        print("⚠️ GO projection component not found in model")

    # Save GO encoder if it exists
    if hasattr(model, "go_encoder") and model.go_encoder is not None:
        go_encoder_path = os.path.join(args.save_dir, "go_encoder.pt")
        torch.save(model.go_encoder.state_dict(), go_encoder_path)
        print(f"✅ GO encoder saved to {go_encoder_path}")

    # Save protein model separately if needed
    protein_model_path = os.path.join(args.save_dir, "protein_model")
    os.makedirs(protein_model_path, exist_ok=True)
    if hasattr(model.protein_model, "save_pretrained"):
        model.protein_model.save_pretrained(protein_model_path)
    else:
        torch.save(
            model.protein_model.state_dict(),
            os.path.join(protein_model_path, "pytorch_model.bin"),
        )

    print("✅ Complete merged model saved successfully!")
    print(f"📁 Model saved to: {args.save_dir}")

    # save the missing and unexpected keys to a file
    with open("missing_and_unexpected_keys.txt", "w") as f:
        f.write("Missing keys:\n")
        for key in result.missing_keys:
            f.write(f"{key}\n")
        f.write("\nUnexpected keys:\n")
        for key in result.unexpected_keys:
            f.write(f"{key}\n")
    print("💾 Saved missing and unexpected keys to missing_and_unexpected_keys.txt")

    # Report parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    text_params = sum(p.numel() for p in model.text_model.parameters())
    protein_params = sum(p.numel() for p in model.protein_model.parameters())
    go_encoder_params = (
        sum(p.numel() for p in model.go_encoder.parameters())
        if hasattr(model, "go_encoder") and model.go_encoder is not None
        else 0
    )
    go_projection_params = (
        sum(p.numel() for p in model.go_projection.parameters())
        if hasattr(model, "go_projection") and model.go_projection is not None
        else 0
    )
    print(
        f"✅ Loaded model with {total_params/1e6:.1f}M parameters "
        f"(text {text_params/1e6:.1f}M • protein {protein_params/1e6:.1f}M • GO encoder {go_encoder_params/1e6:.1f}M • GO projection {go_projection_params/1e6:.1f}M)"
    )

    model.eval()

    print(f"✅ Model loaded from: {args.checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepSpeed ZeRO checkpoint to HuggingFace format for ProteinLLMModel."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Text model name or path (e.g. Qwen/Qwen3-1.7B) - deprecated, use --text_model_name",
    )
    parser.add_argument(
        "--protein_model_name",
        type=str,
        required=True,
        help="Protein model name or path (e.g. esm3_t33_650M_UR50D)",
    )
    parser.add_argument(
        "--text_model_name",
        type=str,
        required=True,
        help="Text model name or path (e.g. Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for models",
    )
    parser.add_argument(
        "--max_length_text",
        type=int,
        default=4000,
        help="Maximum length of text sequences",
    )
    parser.add_argument(
        "--max_length_protein",
        type=int,
        default=2000,
        help="Maximum length of protein sequences",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=128,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=256,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--protein_embedding_layer",
        type=int,
        default=-1,
        help="ESM3 layer to extract embeddings from. Use -1 for final output (default), 0-N for specific transformer layers",
    )
    parser.add_argument("--go_obo_path", type=str, default=None, help="Path to GO ontology OBO file")
    parser.add_argument(
        "--precomputed_embeddings_path",
        type=str,
        default=None,
        help="Path to precomputed GO embeddings",
    )
    parser.add_argument(
        "--go_projection_path",
        type=str,
        default=None,
        help="Path to pretrained GO projection weights (.pt file)",
    )
    parser.add_argument(
        "--go_encoder_path",
        type=str,
        default=None,
        help="Path to pretrained GO encoder weights (.pt file or directory)",
    )
    parser.add_argument(
        "--go_hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension for GO GAT layers",
    )
    parser.add_argument(
        "--go_num_gat_layers",
        type=int,
        default=3,
        help="Number of GAT layers in GO encoder",
    )
    parser.add_argument(
        "--go_num_heads",
        type=int,
        default=8,
        help="Number of attention heads in GO GAT",
    )
    parser.add_argument(
        "--go_num_reduced_embeddings",
        type=int,
        default=200,
        help="Number of reduced embeddings per GO namespace",
    )
    parser.add_argument(
        "--go_embedding_dim",
        type=int,
        default=2560,
        help="GO embedding dimension",
    )
    parser.add_argument(
        "--unified_go_encoder",
        type=str2bool,
        default=False,
        help="If True, use unified GOGraphEncoderUnified; if False, use original GOGraphEncoder",
    )
    parser.add_argument(
        "--protein_model_finetune",
        type=str2bool,
        default=False,
        help="Whether to finetune the protein model",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to DeepSpeed ZeRO checkpoint directory (e.g. .../last.ckpt/)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the converted HuggingFace model",
    )
    args = parser.parse_args()

    # Handle backward compatibility
    if args.model_name_or_path and not args.text_model_name:
        args.text_model_name = args.model_name_or_path

    save_ckpt(args)


if __name__ == "__main__":
    main()
