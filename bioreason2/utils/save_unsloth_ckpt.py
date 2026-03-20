import os
import torch
from bioreason2.models.protein_llm import ProteinLLMModel, _get_target_modules
from pathlib import Path
import argparse
from bioreason2.utils.argparse_utils import str2bool
from unsloth import FastLanguageModel


def _setup_lora_for_checkpoint_loading(
    model: ProteinLLMModel,
    lora_rank: int = 128,
    lora_alpha: int = 256,
    lora_dropout: float = 0.05,
):
    """Setup LoRA using Unsloth (matching training setup)"""
    print(f"🔧 Setting up Unsloth LoRA (rank={lora_rank}, alpha={lora_alpha})")
    
    target_modules = _get_target_modules(model)
    
    model.text_model = FastLanguageModel.get_peft_model(
        model.text_model,
        r=lora_rank,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("✅ Unsloth LoRA setup complete")


def save_lightning_ckpt(args):
    """Convert PyTorch Lightning checkpoint (with Unsloth) to HuggingFace format."""
    checkpoint_path = Path(args.checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📥 Loading Lightning checkpoint: {checkpoint_path}")
    
    # Load checkpoint and extract state_dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    
    print(f"📤 Extracted state_dict with {len(state_dict)} keys")
    print(f"📊 Epoch: {checkpoint.get('epoch', 'N/A')} | Global step: {checkpoint.get('global_step', 'N/A')}")
    
    print("🔧 Building ProteinLLMModel with Unsloth...")
    
    # Create model with Unsloth (matching training setup)
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
        use_unsloth=True,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"📍 Model on {device}")
    
    # Setup Unsloth LoRA
    _setup_lora_for_checkpoint_loading(
        model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Remap keys: Lightning saves with "model." prefix, we need to remove it
    print("🔧 Remapping keys...")
    
    # Filter to only keys with "model." prefix (Lightning wrapper)
    # The checkpoint has duplicates: both "model.X" and "X" for each key
    # We only want the "model." prefixed ones (these are the actual saved weights)
    lightning_keys = {k: v for k, v in state_dict.items() if k.startswith("model.")}
    print(f"📊 Filtering to Lightning-wrapped keys: {len(state_dict)} → {len(lightning_keys)}")
    
    remapped = {}
    for k, v in lightning_keys.items():
        # Remove "model." prefix from Lightning checkpoint
        # The PEFT structure (base_model.model) is already present in the checkpoint
        new_k = k[6:]  # Remove "model." prefix
        
        # Move tensors to device
        if isinstance(v, torch.Tensor):
            v = v.to(device)
        
        remapped[new_k] = v
    
    print(f"📊 After removing 'model.' prefix: {len(remapped)} keys")
    
    # Show sample remapped keys for validation
    print("\n🔍 Sample remapped keys:")
    text_model_keys_sample = [k for k in remapped.keys() if "text_model" in k]
    if text_model_keys_sample:
        for key in sorted(text_model_keys_sample)[:5]:
            print(f"  {key}")
        if len(text_model_keys_sample) > 5:
            print(f"  ... and {len(text_model_keys_sample) - 5} more text_model keys")
    
    # Load state dict
    result = model.load_state_dict(remapped, strict=False)
    print(f"\n📥 Loaded: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")
    
    # Analyze and validate loaded keys
    if result.missing_keys:
        print("\n⚠️  Missing keys analysis:")
        # Group by component
        missing_by_component = {}
        for key in result.missing_keys:
            component = key.split('.')[0] if '.' in key else key
            missing_by_component.setdefault(component, []).append(key)
        
        for component, keys in sorted(missing_by_component.items()):
            print(f"  {component}: {len(keys)} keys")
    
    if result.unexpected_keys:
        print("\n⚠️  Unexpected keys analysis:")
        # Group by component
        unexpected_by_component = {}
        for key in result.unexpected_keys:
            component = key.split('.')[0] if '.' in key else key
            unexpected_by_component.setdefault(component, []).append(key)
        
        for component, keys in sorted(unexpected_by_component.items()):
            print(f"  {component}: {len(keys)} keys")
            # Show samples for problematic components
            if component == "text_model" and len(keys) <= 5:
                for k in keys:
                    print(f"    - {k}")
    
    # Validate that critical text_model weights loaded correctly
    text_model_unexpected = [k for k in result.unexpected_keys if k.startswith('text_model.')]
    
    if text_model_unexpected:
        print(f"\n❌ ERROR: {len(text_model_unexpected)} text_model keys were unexpected!")
        print("These weights were NOT loaded from the checkpoint and will use random initialization:")
        for key in text_model_unexpected[:10]:
            print(f"  - {key}")
        if len(text_model_unexpected) > 10:
            print(f"  ... and {len(text_model_unexpected) - 10} more")
        raise RuntimeError(
            f"Failed to load {len(text_model_unexpected)} text_model weights from checkpoint. "
            "The model would contain untrained weights. Please check key remapping logic."
        )
    
    print("\n✅ All critical weights loaded successfully")
    
    # Save missing and unexpected keys to file (in current working directory)
    keys_log_path = "unsloth_checkpoint_keys.txt"
    with open(keys_log_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Save directory: {args.save_dir}\n")
        f.write(f"Total checkpoint keys: {len(state_dict)}\n")
        f.write(f"Total remapped keys: {len(remapped)}\n\n")
        
        f.write(f"Missing keys ({len(result.missing_keys)}):\n")
        for key in result.missing_keys:
            f.write(f"  {key}\n")
        
        f.write(f"\nUnexpected keys ({len(result.unexpected_keys)}):\n")
        for key in result.unexpected_keys:
            f.write(f"  {key}\n")
    print(f"💾 Saved key mapping log to {keys_log_path}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=False)
    print(f"💾 Saving to {args.save_dir}")
    
    # Merge LoRA adapters
    print("🔗 Merging LoRA adapters...")
    model.text_model = model.text_model.merge_and_unload()
    print("✅ LoRA merged into base model")
    model = model.cpu()
    
    # Save all components
    model.text_model.save_pretrained(args.save_dir)
    model.text_tokenizer.save_pretrained(args.save_dir)
    torch.save(model.protein_projection.state_dict(), f"{args.save_dir}/protein_projection.pt")
    torch.save(model.go_projection.state_dict(), f"{args.save_dir}/go_projection.pt")
    torch.save(model.go_encoder.state_dict(), f"{args.save_dir}/go_encoder.pt")
    
    protein_model_dir = f"{args.save_dir}/protein_model"
    os.makedirs(protein_model_dir, exist_ok=True)
    torch.save(model.protein_model.state_dict(), f"{protein_model_dir}/pytorch_model.bin")
    
    # Report parameter counts
    total = sum(p.numel() for p in model.parameters())
    text = sum(p.numel() for p in model.text_model.parameters())
    protein = sum(p.numel() for p in model.protein_model.parameters())
    go_enc = sum(p.numel() for p in model.go_encoder.parameters())
    go_proj = sum(p.numel() for p in model.go_projection.parameters())
    
    print(f"✅ Saved {total/1e6:.1f}M params (text {text/1e6:.1f}M • protein {protein/1e6:.1f}M • GO enc {go_enc/1e6:.1f}M • GO proj {go_proj/1e6:.1f}M)")


def main():
    parser = argparse.ArgumentParser(description="Convert Lightning+Unsloth checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--text_model_name", type=str, required=True)
    parser.add_argument("--protein_model_name", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_length_text", type=int, default=4000)
    parser.add_argument("--max_length_protein", type=int, default=2000)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--protein_embedding_layer", type=int, default=-1)
    parser.add_argument("--go_obo_path", type=str, default=None)
    parser.add_argument("--precomputed_embeddings_path", type=str, default=None)
    parser.add_argument("--go_hidden_dim", type=int, default=512)
    parser.add_argument("--go_num_gat_layers", type=int, default=3)
    parser.add_argument("--go_num_heads", type=int, default=8)
    parser.add_argument("--go_num_reduced_embeddings", type=int, default=200)
    parser.add_argument("--go_embedding_dim", type=int, default=2560)
    parser.add_argument("--unified_go_encoder", type=str2bool, default=False)
    parser.add_argument("--protein_model_finetune", type=str2bool, default=False)
    
    save_lightning_ckpt(parser.parse_args())


if __name__ == "__main__":
    main()

