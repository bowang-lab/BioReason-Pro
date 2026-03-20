import os
import sys

# Suppress pkg_resources deprecation warning before any imports
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*Deprecated call to `pkg_resources.declare_namespace.*")

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    Main training function with Hydra configuration.
    All experiment settings come from a single YAML file.
    """
    # Get the original working directory (before Hydra changes it)
    original_cwd = hydra.utils.get_original_cwd()
    
    # Add src to path for imports
    sys.path.append(os.path.join(original_cwd, "src"))
    
    # Print config for debugging
    print("="*50)
    print("Configuration:")
    print("="*50)
    print(OmegaConf.to_yaml(cfg))
    print("="*50)

    # Set seeds for reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    
    # Setup torch optimizations from config
    if 'pytorch' in cfg:
        # Set matrix multiplication precision
        matmul_precision = cfg.pytorch.get('matmul_precision', 'medium')
        torch.set_float32_matmul_precision(matmul_precision)
        
        # Enable/disable TF32
        enable_tf32 = cfg.pytorch.get('enable_tf32', True)
        torch.backends.cudnn.allow_tf32 = enable_tf32
    else:
        # Default settings if pytorch config not provided
        torch.set_float32_matmul_precision('medium')
        torch.backends.cudnn.allow_tf32 = True
    
    # Import here to ensure path is set
    from gogpt.data.dataset import load_preprocessed_data
    from gogpt.models.gogpt_lightning import LightningGOGPT
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    
    if not hasattr(cfg.data, 'preprocessed_path') or not cfg.data.preprocessed_path:
        raise ValueError("preprocessed_path must be specified in config. Run prepare_data.py first to preprocess the data.")
    
    tokenizer, train_loader, val_loader, tokenizer_info = load_preprocessed_data(
        preprocessed_path=cfg.data.preprocessed_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.get('num_workers', 8),
        pin_memory=cfg.data.get('pin_memory', False),
        persistent_workers=cfg.data.get('persistent_workers', True),
        prefetch_factor=cfg.data.get('prefetch_factor', 8)
    )
    
    # Build complete model args
    model_args = OmegaConf.to_container(cfg.model)
    model_args.update({
        'vocab_size': len(tokenizer.token_to_id),
        'pad_token_id': tokenizer_info['pad_token_id'],
        'mf_start_token_id': tokenizer_info['mf_start_token_id'],
        'mf_end_token_id': tokenizer_info['mf_end_token_id'],
        'bp_start_token_id': tokenizer_info['bp_start_token_id'],
        'bp_end_token_id': tokenizer_info['bp_end_token_id'],
        'cc_start_token_id': tokenizer_info['cc_start_token_id'],
        'cc_end_token_id': tokenizer_info['cc_end_token_id'],
        'organism_vocab_size': tokenizer_info['organism_vocab_size'],
    })
    
    # Create model
    print("Creating model...")
    hparams = {
        'learning_rate': cfg.training.learning_rate,
        'weight_decay': cfg.training.weight_decay,
        'warmup_fraction': cfg.training.get('warmup_fraction', 0.1),  # Default to 10% if not specified
        'min_lr_ratio': cfg.training.get('min_lr_ratio', 0.1),  # Default to 10% if not specified
        'log_generations': cfg.training.get('log_generations', True),  # Default to True if not specified
        'max_logged_generations': cfg.training.get('max_logged_generations', 15)  # Default to 15 if not specified
    }
    model = LightningGOGPT(
        model_args=model_args,
        hparams=hparams,
        tokenizer=tokenizer
    )
    
    # Setup callbacks
    callbacks = []

    # Checkpoint callbacks - handle both single and multiple checkpoint configs
    # Support backward compatibility with old 'checkpoint' config and new 'checkpoints' list
    if 'checkpoints' in cfg and cfg.checkpoints:
        # New format: list of checkpoint configs
        checkpoint_callbacks = []
        for ckpt_cfg in cfg.checkpoints:
            ckpt_config = OmegaConf.to_container(ckpt_cfg)
            # Checkpoint directory is relative to current directory (which is the Hydra output dir)
            ckpt_config['dirpath'] = ckpt_config.get('dirpath', 'checkpoints')
            checkpoint_cb = ModelCheckpoint(**ckpt_config)
            callbacks.append(checkpoint_cb)
            checkpoint_callbacks.append(checkpoint_cb)
        # Use the first callback with a monitor for best model tracking
        best_checkpoint_cb = next((cb for cb in checkpoint_callbacks if cb.monitor), checkpoint_callbacks[0])
    elif 'checkpoint' in cfg:
        # Old format: single checkpoint config (backward compatibility)
        checkpoint_config = OmegaConf.to_container(cfg.checkpoint)
        checkpoint_config.pop('every_n_epochs', None)
        checkpoint_config['dirpath'] = checkpoint_config.get('dirpath', 'checkpoints')
        checkpoint_cb = ModelCheckpoint(**checkpoint_config)
        callbacks.append(checkpoint_cb)
        best_checkpoint_cb = checkpoint_cb
    else:
        raise ValueError("No checkpoint configuration found in config")
    
    # Setup logger
    logger = None
    if cfg.wandb.enabled:
        if cfg.wandb.offline:
            os.environ["WANDB_MODE"] = "offline"
        
        wandb_config = OmegaConf.to_container(cfg.wandb, resolve=True)
        # Remove non-WandbLogger parameters
        wandb_config.pop('enabled', None)
        wandb_config.pop('offline', None)
        # Wandb saves to current directory (which is the Hydra output dir)
        wandb_config['save_dir'] = wandb_config.get('save_dir', 'wandb')
        
        logger = WandbLogger(**wandb_config)
        # Log hyperparameters
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    
    # Extract training config
    training_config = OmegaConf.to_container(cfg.training)
    # Remove non-Trainer arguments (these are model hyperparameters, not trainer arguments)
    non_trainer_keys = ['learning_rate', 'weight_decay', 'warmup_fraction', 'min_lr_ratio',
                        'log_generations', 'max_logged_generations']
    trainer_args = {k: v for k, v in training_config.items()
                   if k not in non_trainer_keys}
    
    # Add trainer settings
    trainer_args.update({
        'default_root_dir': ".",  # Current directory (Hydra output dir)
        'callbacks': callbacks,
        'logger': logger,
    })
    
    # Create trainer
    print("Creating trainer...")
    trainer = pl.Trainer(**trainer_args)
    
    # Handle resume
    ckpt_path = cfg.resume.checkpoint_path if cfg.resume.checkpoint_path else None
    if ckpt_path:
        # If checkpoint path is relative, it's relative to original cwd
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(original_cwd, ckpt_path)

        # Check if weights_only flag is set (for starting finetuning from frozen checkpoint)
        weights_only = cfg.resume.get('weights_only', False)

        if weights_only:
            print(f"Loading weights only from checkpoint: {ckpt_path}")
            print("Note: Optimizer and scheduler states will be reinitialized (useful for starting finetuning)")
            # Load checkpoint manually with weights_only
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            ckpt_path = None  # Don't pass to trainer.fit to avoid loading optimizer state
        else:
            print(f"Resuming from checkpoint: {ckpt_path}")
            print("Note: Will restore model, optimizer, and scheduler states")

    # Train
    print("Starting training...")
    print(f"Output directory: {os.getcwd()}")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    
    # Save final config for reference
    config_save_path = Path("final_config.yaml")
    OmegaConf.save(cfg, config_save_path)
    print(f"\nTraining complete!")
    print(f"Outputs saved to: {os.getcwd()}")
    print(f"Final config saved to: {config_save_path}")
    
    # Print best checkpoint info
    if best_checkpoint_cb.best_model_path:
        print(f"Best checkpoint: {best_checkpoint_cb.best_model_path}")
        print(f"Best score ({best_checkpoint_cb.monitor}): {best_checkpoint_cb.best_model_score}")

    # Print all checkpoint directories
    if 'checkpoints' in cfg and cfg.checkpoints:
        print("\nCheckpoint directories:")
        for i, ckpt_cfg in enumerate(cfg.checkpoints):
            dirpath = ckpt_cfg.get('dirpath', 'checkpoints')
            print(f"  Strategy {i+1}: {dirpath}/")

if __name__ == "__main__":
    train()