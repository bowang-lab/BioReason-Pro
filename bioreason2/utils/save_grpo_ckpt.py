import os
import torch
from bioreason2.models.protein_llm import ProteinLLMModel, _get_target_modules
from pathlib import Path
import argparse
from bioreason2.utils.argparse_utils import str2bool
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _setup_lora_for_checkpoint_loading(
    model: ProteinLLMModel,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Setup LoRA for GRPO checkpoint loading"""
    print(f"🔧 Setting up LoRA (rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout})")
    
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
    
    model.text_model = prepare_model_for_kbit_training(model.text_model)
    model.text_model = get_peft_model(model.text_model, lora_config)
    
    return lora_config


def load_grpo_checkpoint(checkpoint_path: str):
    """Load GRPO checkpoint"""
    checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE, weights_only=False)
    print(f"✅ Loaded {len(checkpoint)} parameters")
    
    return checkpoint


def save_grpo_ckpt(args):
    """Convert GRPO checkpoint to HuggingFace format"""
    
    state_dict = load_grpo_checkpoint(args.checkpoint_path)
    
    print("🔧 Building ProteinLLMModel...")
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
        use_unsloth=False,
    ).to(DEVICE)
    
    # Check and resize vocabulary if needed
    checkpoint_vocab_size = None
    for k in state_dict.keys():
        if "embed_tokens" in k and "weight" in k:
            checkpoint_vocab_size = state_dict[k].shape[0]
            break
    
    if checkpoint_vocab_size:
        current_vocab_size = len(model.text_tokenizer)
        if current_vocab_size != checkpoint_vocab_size:
            print(f"⚠️  Vocab size mismatch: checkpoint={checkpoint_vocab_size}, model={current_vocab_size}")
    
    _setup_lora_for_checkpoint_loading(
        model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Resize embeddings if needed
    if checkpoint_vocab_size:
        if hasattr(model.text_model, "base_model") and hasattr(model.text_model.base_model, "model"):
            actual_model = model.text_model.base_model.model
            current_embed_size = actual_model.model.embed_tokens.weight.shape[0]
            current_lm_head_size = actual_model.lm_head.weight.shape[0]
        else:
            current_embed_size = model.text_model.model.embed_tokens.weight.shape[0]
            current_lm_head_size = model.text_model.lm_head.weight.shape[0]
        
        if current_embed_size != checkpoint_vocab_size or current_lm_head_size != checkpoint_vocab_size:
            print(f"🔧 Resizing embeddings to {checkpoint_vocab_size}")
            
            if hasattr(model.text_model, "base_model") and hasattr(model.text_model.base_model, "model"):
                model.text_model.base_model.model.resize_token_embeddings(checkpoint_vocab_size)
            else:
                model.text_model.resize_token_embeddings(checkpoint_vocab_size)
    
    print(f"📊 Loading {len(state_dict)} parameters...")
    result = model.load_state_dict(state_dict, strict=False)
    
    if result.missing_keys:
        print(f"⚠️  Missing {len(result.missing_keys)} keys")
    if result.unexpected_keys:
        print(f"⚠️  Unexpected {len(result.unexpected_keys)} keys")
    
    # Merge LoRA adapters before saving
    print("🔗 Merging LoRA adapters...")
    model.text_model = model.text_model.merge_and_unload()
    
    if os.path.exists(args.save_dir):
        raise FileExistsError(
            f"Save directory already exists: {args.save_dir}\n"
            f"Please remove it first or choose a different directory."
        )
    
    os.makedirs(args.save_dir, exist_ok=False)
    
    print(f"💾 Saving to {args.save_dir}...")
    model.text_model.save_pretrained(args.save_dir)
    model.text_tokenizer.save_pretrained(args.save_dir)
    torch.save(model.protein_projection.state_dict(), os.path.join(args.save_dir, "protein_projection.pt"))
    torch.save(model.go_projection.state_dict(), os.path.join(args.save_dir, "go_projection.pt"))
    torch.save(model.go_encoder.state_dict(), os.path.join(args.save_dir, "go_encoder.pt"))
    
    # Save key mismatch log
    with open(os.path.join(args.save_dir, "missing_and_unexpected_keys.txt"), "w") as f:
        f.write("Missing keys:\n")
        for key in result.missing_keys:
            f.write(f"{key}\n")
        f.write("\nUnexpected keys:\n")
        for key in result.unexpected_keys:
            f.write(f"{key}\n")
    
    # Print missing and unexpected keys to console
    if result.missing_keys:
        print(f"⚠️  Missing keys ({len(result.missing_keys)}):")
        for key in result.missing_keys:
            print(f"  - {key}")
    
    if result.unexpected_keys:
        print(f"⚠️  Unexpected keys ({len(result.unexpected_keys)}):")
        for key in result.unexpected_keys:
            print(f"  - {key}")
    
    total_params = sum(p.numel() for p in model.parameters())
    text_params = sum(p.numel() for p in model.text_model.parameters())
    go_enc_params = sum(p.numel() for p in model.go_encoder.parameters())
    print(f"✅ Saved {total_params/1e6:.1f}M params (text: {text_params/1e6:.1f}M, GO: {go_enc_params/1e6:.1f}M)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GRPO checkpoint to HuggingFace format for ProteinLLMModel."
    )
    parser.add_argument(
        "--text_model_name",
        type=str,
        required=True,
        help="Text model name or path (e.g. Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--protein_model_name",
        type=str,
        required=True,
        help="Protein model name (e.g. esm3_sm_open_v1)",
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
        default=10000,
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
        default=0.0,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--protein_embedding_layer",
        type=int,
        default=37,
        help="Protein embedding layer to extract from ESM3",
    )
    parser.add_argument(
        "--go_obo_path",
        type=str,
        required=True,
        help="Path to GO ontology OBO file",
    )
    parser.add_argument(
        "--precomputed_embeddings_path",
        type=str,
        required=True,
        help="Directory with precomputed GO embeddings",
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
        default=True,
        help="Whether to use unified GO encoder",
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
        help="Path to GRPO checkpoint directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the converted HuggingFace model",
    )
    args = parser.parse_args()
    
    save_grpo_ckpt(args)


if __name__ == "__main__":
    main()
