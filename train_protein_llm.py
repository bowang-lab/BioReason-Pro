# Set envioronment variables
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_CUMEM_ENABLE"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

# Import unsloth
import unsloth
from unsloth import FastLanguageModel

# Standard library imports
import os
import time
from argparse import ArgumentParser
from functools import partial
from typing import Optional, Dict

# Third-party imports
import pytorch_lightning as pl
import torch
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    DeviceStatsMonitor,
    Callback,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from transformers import logging
from datasets import Value, concatenate_datasets

# Local imports
from bioreason2.dataset.cafa5.collate import qwen_protein_collate_fn
from bioreason2.dataset.cafa5.generate import generate_single_response
from bioreason2.dataset.cafa5.load import load_cafa5_dataset
from bioreason2.models.pl.processing_pl import PLProcessor
from bioreason2.models.protein_llm import (
    ProteinLLMModel,
    _get_target_modules,
)
from bioreason2.utils import str2bool

# Set start method to 'spawn' for CUDA compatibility with multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
logging.set_verbosity_error()


class EpochCheckpointFromN(Callback):
    """Custom callback to save checkpoints every epoch starting from a specific epoch."""
    
    def __init__(self, checkpoint_dir, run_name, start_epoch):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.start_epoch = start_epoch
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Save checkpoint at the end of each epoch if epoch >= start_epoch."""
        if trainer.current_epoch >= self.start_epoch:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{self.run_name}-epoch={trainer.current_epoch:02d}.ckpt"
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"✓ Saved epoch checkpoint: {checkpoint_path}")


class ProteinLLMFineTuner(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning Protein-LLM models.
    """

    def __init__(self, hparams, train_dataset=None, val_dataset=None, test_dataset=None):
        """
        Initialize the ProteinLLMFineTuner.

        Args:
            hparams: Hyperparameters for the model and training
            train_dataset: Pre-loaded training dataset
            val_dataset: Pre-loaded validation dataset
            test_dataset: Pre-loaded test dataset
        """
        super().__init__()
        self.save_hyperparameters(hparams)

        self.text_model_name = self.hparams.text_model_name
        self.protein_model_name = self.hparams.protein_model_name
        self.cache_dir = self.hparams.cache_dir
        self.learning_rate = self.hparams.learning_rate
        self.weight_decay = self.hparams.weight_decay
        self.warmup_ratio = self.hparams.warmup_ratio
        self.text_model_finetune = self.hparams.text_model_finetune
        self.protein_model_finetune = self.hparams.protein_model_finetune
        self.protein_train_layer_start = self.hparams.protein_train_layer_start
        self.protein_embedding_layer = self.hparams.protein_embedding_layer
        self.go_model_finetune = self.hparams.go_model_finetune
        self.attn_implementation = self.hparams.attn_implementation
        self.go_obo_path = self.hparams.go_obo_path
        self.precomputed_embeddings_path = self.hparams.precomputed_embeddings_path
        self.go_hidden_dim = self.hparams.go_hidden_dim
        self.go_num_gat_layers = self.hparams.go_num_gat_layers
        self.go_num_heads = self.hparams.go_num_heads
        self.go_num_reduced_embeddings = self.hparams.go_num_reduced_embeddings
        self.go_embedding_dim = self.hparams.go_embedding_dim
        self.lora_rank = self.hparams.lora_rank
        self.lora_alpha = self.hparams.lora_alpha
        self.lora_dropout = self.hparams.lora_dropout
        self.max_length_protein = self.hparams.max_length_protein
        self.max_length_text = self.hparams.max_length_text
        self.return_answer_in_batch = self.hparams.return_answer_in_batch
        self.training_stage = self.hparams.training_stage
        self.projector_checkpoint_path = self.hparams.projector_checkpoint_path
        self.go_projection_checkpoint_path = self.hparams.go_projection_checkpoint_path
        self.go_encoder_checkpoint_path = self.hparams.go_encoder_checkpoint_path
        self.enable_sample_generation = self.hparams.enable_sample_generation
        self.verbose_sample_generation = self.hparams.verbose_sample_generation
        self.every_n_train_steps = self.hparams.every_n_train_steps
        self.unified_go_encoder = self.hparams.unified_go_encoder
        self.use_unsloth = self.hparams.use_unsloth

        # Store dataset configuration
        self.dataset_type = self.hparams.dataset_type

        # Store datasets
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset

        # Create quantization config if QLoRA is enabled
        quantization_config = None
        if self.hparams.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.hparams.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self.hparams.bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=self.hparams.bnb_4bit_use_double_quant,
            )
            print(f"🔧 QLoRA enabled with {self.hparams.bnb_4bit_quant_type} quantization")

        # Load model
        self.model = ProteinLLMModel(
            text_model_name=self.text_model_name,
            protein_model_name=self.protein_model_name,
            cache_dir=self.cache_dir,
            max_length_protein=self.max_length_protein,
            max_length_text=self.max_length_text,
            text_model_finetune=self.text_model_finetune,
            protein_model_finetune=self.protein_model_finetune,
            protein_train_layer_start=self.protein_train_layer_start,
            protein_embedding_layer=self.protein_embedding_layer,
            go_model_finetune=self.go_model_finetune,
            attn_implementation=self.attn_implementation,
            go_obo_path=self.go_obo_path,
            precomputed_embeddings_path=self.precomputed_embeddings_path,
            go_hidden_dim=self.go_hidden_dim,
            go_num_gat_layers=self.go_num_gat_layers,
            go_num_heads=self.go_num_heads,
            go_num_reduced_embeddings=self.go_num_reduced_embeddings,
            go_embedding_dim=self.go_embedding_dim,
            quantization_config=quantization_config,
            unified_go_encoder=self.unified_go_encoder,
            use_unsloth=self.use_unsloth,
        )

        # Load projector weights if provided (for stage 2)
        if self.training_stage == 2 and self.projector_checkpoint_path and not self.hparams.ckpt_path:
            print(f"Loading projector weights from: {self.projector_checkpoint_path}")
            projector_state_dict = torch.load(self.projector_checkpoint_path, map_location=self.device)
            self.model.protein_projection.load_state_dict(projector_state_dict)
            print("✓ Projector weights loaded successfully.")

            # Also load GO projection weights if available
            if (
                self.go_projection_checkpoint_path
                and os.path.exists(self.go_projection_checkpoint_path)
                and hasattr(self.model, "go_projection")
                and self.model.go_projection is not None
            ):
                print(f"Loading GO projection weights from: {self.go_projection_checkpoint_path}")
                go_projection_state_dict = torch.load(self.go_projection_checkpoint_path, map_location=self.device)
                self.model.go_projection.load_state_dict(go_projection_state_dict)
                print("✓ GO projection weights loaded successfully.")

            # Also load GO encoder weights if available
            if (
                self.go_encoder_checkpoint_path
                and os.path.exists(self.go_encoder_checkpoint_path)
                and hasattr(self.model, "go_encoder")
                and self.model.go_encoder is not None
            ):
                print(f"Loading GO encoder weights from: {self.go_encoder_checkpoint_path}")
                go_encoder_state_dict = torch.load(self.go_encoder_checkpoint_path, map_location=self.device)
                
                # Use strict=False to handle architecture changes (old vs new GO encoder)
                missing_keys, unexpected_keys = self.model.go_encoder.load_state_dict(go_encoder_state_dict, strict=False)
                
                if missing_keys:
                    print(f"⚠️  Missing keys in GO encoder checkpoint (will be randomly initialized): {len(missing_keys)}")
                    # Only show key names for new architecture components
                    new_arch_missing = [k for k in missing_keys if "all_cross_attention_reducer" in k]
                    if new_arch_missing:
                        print(f"   - New 'all' cross-attention reducer parameters: {len(new_arch_missing)} keys")
                    other_missing = [k for k in missing_keys if "all_cross_attention_reducer" not in k]
                    if other_missing:
                        print(f"   - Other missing keys: {other_missing[:3]}{'...' if len(other_missing) > 3 else ''}")
                
                if unexpected_keys:
                    print(f"⚠️  Unexpected keys in GO encoder checkpoint (ignored): {len(unexpected_keys)}")
                
                print("✓ GO encoder weights loaded successfully (with architecture compatibility).")

        self.text_model = self.model.text_model
        self.protein_model = self.model.protein_model
        self.protein_projection = self.model.protein_projection
        self.go_projection = self.model.go_projection
        self.go_encoder = self.model.go_encoder
        self.tokenizer = self.model.text_tokenizer
        self.lora_config = self._setup_training_strategy()

        # --- Detailed Parameter Count ---
        protein_trainable = sum(p.numel() for p in self.protein_model.parameters() if p.requires_grad)
        protein_total = sum(p.numel() for p in self.protein_model.parameters())
        projector_trainable = sum(p.numel() for p in self.protein_projection.parameters() if p.requires_grad)
        text_model_trainable = sum(p.numel() for p in self.text_model.parameters() if p.requires_grad)
        embed_tokens_trainable = sum(
            p.numel() for p in self.text_model.get_input_embeddings().weight if p.requires_grad
        )
        lm_head_trainable = sum(p.numel() for p in self.text_model.get_output_embeddings().weight if p.requires_grad)

        # Count GO encoder and GO projection parameters
        go_encoder_trainable = 0
        go_projection_trainable = 0
        if hasattr(self.model, "go_encoder") and self.model.go_encoder is not None:
            go_encoder_trainable = sum(p.numel() for p in self.model.go_encoder.parameters() if p.requires_grad)
        if hasattr(self.model, "go_projection") and self.model.go_projection is not None:
            go_projection_trainable = sum(p.numel() for p in self.model.go_projection.parameters() if p.requires_grad)

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"--- Trainable Parameters (Stage {self.training_stage}) ---")
        if protein_total > 0:
            protein_pct = (protein_trainable / protein_total) * 100
            print(f"  - Protein Model: {protein_trainable:,} / {protein_total:,} ({protein_pct:.1f}%)")
        else:
            print(f"  - Protein Model: {protein_trainable:,}")
        print(f"  - Projector MLP: {projector_trainable:,}")
        print(f"  - GO Encoder: {go_encoder_trainable:,}")
        print(f"  - GO Projection: {go_projection_trainable:,}")
        print(f"  - Text Model (LoRA): {text_model_trainable:,}")
        print(f"  - Embed Tokens: {embed_tokens_trainable:,}")
        print(f"  - LM Head: {lm_head_trainable:,}")
        print("  ----------------------------------")
        print(f"  - Total Trainable: {total_trainable:,}")
        print("------------------------------------")

    def _setup_projection_training(self):
        """
        Set up projection layers for training (always trainable).
        """
        # Protein projection: always trainable
        self.model.protein_projection.train()
        for param in self.model.protein_projection.parameters():
            param.requires_grad = True
        print("  - Protein projection: trainable")
        
        # GO projection: always trainable if available
        if hasattr(self.model, "go_projection") and self.model.go_projection is not None:
            self.model.go_projection.train()
            for param in self.model.go_projection.parameters():
                param.requires_grad = True
            print("  - GO projection: trainable")

    def _setup_training_strategy(self) -> Optional[LoraConfig]:
        """
        Configures the training strategy based on the current training stage.
        This involves freezing/unfreezing model parts and setting up LoRA if needed.
        """
        print("✓ Setting up training configuration...")
        
        # Protein encoder training is now handled automatically during model creation
        if self.protein_model_finetune:
            print(f"  - Protein encoder training enabled (layer start: {self.protein_train_layer_start})")
        else:
            print("  - Protein encoder: keeping frozen")

        # Setup GO encoder training
        if hasattr(self.model, "go_encoder") and self.model.go_encoder is not None:
            if self.go_model_finetune:
                print("  - Enabling GO encoder training")
                self.model.go_encoder.train()
                for param in self.model.go_encoder.parameters():
                    param.requires_grad = True
            else:
                print("  - GO encoder: keeping frozen")
        
        # Setup projection layers (always trainable)
        self._setup_projection_training()
        
        # Setup text model training
        if self.text_model_finetune:
            print("  - Enabling text model LoRA training")
        else:
            print("  - Text model: keeping frozen")

        # Stage 1: Train only the projectors
        if self.training_stage == 1:
            print("Setting up for Stage 1: Projector training only.")
            # Freeze the text model completely and keep in eval mode
            self.model.text_model.eval()
            for param in self.model.text_model.parameters():
                param.requires_grad = False
            if hasattr(self.model.text_model, "config"):
                self.model.text_model.config.use_cache = False
            if hasattr(self.model.text_model, "generation_config"):
                self.model.text_model.generation_config.use_cache = False
            print("     Text model: frozen")
            return None  # No LoRA config in stage 1

        # Stage 2: Full model fine-tuning (with optional LoRA)
        elif self.training_stage == 2:
            print("Setting up for Stage 2: Full model fine-tuning.")
            lora_config = None
            if self.text_model_finetune:
                target_modules = _get_target_modules(self.model)

                if self.use_unsloth:
                    self.model.text_model = FastLanguageModel.get_peft_model(
                        self.model.text_model,
                        r=self.lora_rank,
                        target_modules=target_modules,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout,
                        bias="none",
                        use_gradient_checkpointing = "unsloth",
                        random_state=self.hparams.seed,
                        use_rslora=False,
                        loftq_config=None,
                    )

                else:
                    lora_config = LoraConfig(
                        r=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout,
                        target_modules=target_modules,
                        init_lora_weights="gaussian",
                        bias="none",
                        task_type="CAUSAL_LM",
                    )
                    self.model.text_model = prepare_model_for_kbit_training(self.model.text_model)
                    self.model.text_model = get_peft_model(self.model.text_model, lora_config)
                
                self.model.text_model.train()
            
            else:
                # Keep text model frozen and in eval mode
                print("     Text model remaining frozen")
                for param in self.model.text_model.parameters():
                    param.requires_grad = False
                self.model.text_model.eval()

            return lora_config

        else:
            raise ValueError(f"Invalid training stage: {self.training_stage}")

    def _step(self, batch: Dict, batch_idx: int, prefix: str) -> torch.Tensor:
        """
        Performs a single step for training, validation, or testing.

        Args:
            batch: Dictionary containing the batch data
            batch_idx: Integer indicating the batch index
            prefix: String indicating the step type ('train', 'val', or 'test')

        Returns:
            torch.Tensor: The computed loss for this batch
        """
        # Get batch data from the collate function
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device) if "labels" in batch else None
        protein_sequences = batch.get("protein_sequences")
        batch_idx_map = batch.get("batch_idx_map")
        structure_coords = batch.get("structure_coords")
        go_aspects = batch.get("batch_go_aspects")

        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            protein_sequences=protein_sequences,
            batch_idx_map=batch_idx_map,
            structure_coords=structure_coords,
            labels=labels,
            go_aspects=go_aspects,
        )

        # Get the loss from model outputs
        loss = outputs.loss

        # Logging metrics
        self.log(
            f"{prefix}_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Rank-0 live per-step loss for progress bar without cross-GPU sync
        if self.trainer.is_global_zero:
            self.log(
                f"{prefix}_loss",
                loss.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                sync_dist=False,
            )
            self.log(
                "lr",
                self.lr_schedulers().get_last_lr()[0],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=False,
            )
            self.log(
                "step",
                self.global_step,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )

        # Sample generation for debugging and monitoring
        if self.enable_sample_generation and (
            (prefix == "train" and (self.global_step % 5_000 == 0)) or (prefix == "val" and (batch_idx % 5_000 == 0))
        ):
            self._log_sample_generation(
                batch,
                prefix,
                batch_idx,
                input_ids,
                attention_mask,
                labels,
                protein_sequences,
                structure_coords,
                batch_idx_map,
                go_aspects,
            )

        return loss

    def _log_sample_generation(
        self,
        batch: Dict,
        prefix: str,
        batch_idx: int,
        input_ids,
        attention_mask,
        labels,
        protein_sequences,
        structure_coords,
        batch_idx_map,
        go_aspects,
    ):
        """Generates, prints, and logs a single sample generation."""
        example_idx = 0  # Select first example from batch

        if self.verbose_sample_generation:
            print(
                f"\n=== Sample Generation {prefix} (step {self.global_step} / {self.trainer.estimated_stepping_batches}) ==="
            )

        if self.use_unsloth:
            # Unsloth does not support model.generate() during training
            return

        result = generate_single_response(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            protein_sequences=protein_sequences,
            structure_coords=structure_coords,
            batch_idx_map=batch_idx_map,
            go_aspects=go_aspects,
            example_idx=example_idx,
            max_new_tokens=64,
            do_sample=False,
        )

        if result["success"]:
            if self.verbose_sample_generation:
                print(
                    f"=====[Sample {prefix} | Batch {batch_idx} | Example {example_idx} | Step {self.global_step}]====="
                )
                print(f"=====[User input]=====\n{result['user_input']}")
                print(f"=====[Complete generation]=====\n{result['generation']}")
                print(f"=====[Ground truth]=====\n{result['ground_truth']}")

            # Log to wandb
            timestamp = time.time()
            step_id = f"gen_{self.global_step}-{timestamp}"
            wandb_logger = self.logger.experiment
            wandb_logger.log(
                {
                    step_id: wandb.Table(
                        columns=[
                            "timestamp",
                            "prefix",
                            "batch_idx",
                            "user_input",
                            "generation",
                            "ground_truth",
                        ],
                        data=[
                            [
                                timestamp,
                                prefix,
                                batch_idx,
                                result["user_input"],
                                result["generation"],
                                result["ground_truth"],
                            ]
                        ],
                    )
                }
            )
        elif self.verbose_sample_generation:
            print(f"=====[Generation failed for this example {example_idx}]=====")

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        return self._step(batch, batch_idx, prefix="train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single validation step."""
        return self._step(batch, batch_idx, prefix="val")

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Tuple[List, List]: A tuple containing a list of optimizers and schedulers
        """
        # In Stage 1, we optimize the projector and GO components (if available)
        if self.training_stage == 1:
            # Collect all trainable parameters for Stage 1
            trainable_params = list(self.model.protein_projection.parameters())
            components = ["protein projector"]

            # Add GO projection parameters if available
            if hasattr(self.model, "go_projection") and self.model.go_projection is not None:
                trainable_params.extend(list(self.model.go_projection.parameters()))
                components.append("GO projection")
            
            # Add GO encoder parameters if available and trainable
            if (hasattr(self.model, "go_encoder") and self.model.go_encoder is not None and 
                self.go_model_finetune):
                trainable_params.extend(list(self.model.go_encoder.parameters()))
                components.append("GO encoder")

            component_str = " + ".join(components)
            print(f"Optimizer configured for Stage 1 ({component_str}) with LR: {self.learning_rate}")

            optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)

        else:  # Stage 2 optimizes all trainable parameters (LoRA + projector)
            trainable_params = self.parameters()
            print(f"Optimizer configured for Stage 2 (full) with LR: {self.learning_rate}")
            optimizer = AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.warmup_ratio * total_steps)
        decay_steps = total_steps - warmup_steps

        warmup = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=self.learning_rate * 0.1,
        )
        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[warmup_steps],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


    def _create_dataloader(
        self, dataset, split: str, shuffle: bool = False, return_answers: bool = False
    ) -> DataLoader:
        """Helper function to create dataloaders with common logic."""
        if dataset is None or len(dataset) == 0:
            print("No dataset provided. Creating empty dataloader")
            return DataLoader([], batch_size=self.hparams.batch_size, shuffle=shuffle)

        try:
            print(f"Creating {split} dataloader: {len(dataset)} samples")

            # Create processor
            processor = PLProcessor(
                tokenizer=self.model.text_tokenizer,
                # protein_tokenizer=None,  # ESM3 handles this internally
            )

            # Create collate function
            collate_fn = partial(
                qwen_protein_collate_fn,
                processor=processor,
                max_length_text=self.max_length_text,
                max_length_protein=self.max_length_protein,
                return_answer_in_batch=return_answers or self.return_answer_in_batch,
            )

            return DataLoader(
                dataset,
                batch_size=self.hparams.batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
                num_workers=self.hparams.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )

        except Exception as e:
            print(f"Failed to create dataloader: {e}")
            return DataLoader([], batch_size=self.hparams.batch_size, shuffle=shuffle)

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader."""
        return self._create_dataloader(self._train_dataset, split="train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader."""
        return self._create_dataloader(self._val_dataset, split="val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Create and return the test DataLoader."""
        return self.val_dataloader()

    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        pass

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to ignore missing/unexpected keys from protein model.
        Since we don't care about protein model weights (frozen anyway), we can safely ignore these.
        """
        # Filter out structure encoder keys that might not be present
        filtered_state_dict = {}
        unexpected_keys = []

        for key, value in state_dict.items():
            if "_structure_encoder" in key:
                # Skip structure encoder keys - these are from ESM3 and we don't need them
                unexpected_keys.append(key)
            else:
                filtered_state_dict[key] = value

        if unexpected_keys:
            print(f"⚠️  Ignoring {len(unexpected_keys)} unexpected structure encoder keys from checkpoint")
            print("   These keys are from ESM3's structure encoder which may not be initialized in current model")
            for key in sorted(unexpected_keys)[:5]:  # Show first 5 sorted
                print(f"   - {key}")

        # Call parent's load_state_dict with strict=False to handle any other mismatches gracefully
        result = super().load_state_dict(filtered_state_dict, strict=False)

        # Log any other unexpected issues
        if result.missing_keys:
            print(f"⚠️  Missing keys in checkpoint: {len(result.missing_keys)} keys")
        if result.unexpected_keys:
            print(f"⚠️  Other unexpected keys (non-structure encoder): {len(result.unexpected_keys)} keys")

        print(f"Missing keys:\n{result.missing_keys}")
        print(f"Unexpected keys:\n{result.unexpected_keys}")

        return result


def main(args: ArgumentParser):
    """
    Main function to run the Protein-Text fine-tuning process.

    Args:
        args (ArgumentParser): Parsed command-line arguments
    """
    # Set random seed and environment variables
    pl.seed_everything(args.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # Load and split datasets
    print("Loading and splitting datasets...")

    if args.dataset_type == "cafa5":
        # Handle multiple dataset names (comma-separated or single)
        dataset_names = [name.strip() for name in args.cafa5_dataset_name.split(",")]

        # Parse dataset weights if provided
        dataset_weights = None
        if args.cafa5_dataset_weights:
            try:
                dataset_weights = [int(w.strip()) for w in args.cafa5_dataset_weights.split(",")]
                if len(dataset_weights) != len(dataset_names):
                    raise ValueError(
                        f"Number of weights ({len(dataset_weights)}) must match number of datasets ({len(dataset_names)})"
                    )
                print(f"Using dataset weights: {dict(zip(dataset_names, dataset_weights))}")
            except ValueError as e:
                print(f"Error parsing dataset weights: {e}")
                print("Using equal weights for all datasets")
                dataset_weights = None

        if dataset_weights is None:
            dataset_weights = [1] * len(dataset_names)

        print(f"Loading {len(dataset_names)} CAFA5 dataset(s): {dataset_names}")
        print(f"Dataset weights: {dataset_weights}")

        all_train_datasets = []
        all_val_datasets = []
        all_test_datasets = []

        for i, dataset_name in enumerate(dataset_names):
            weight = dataset_weights[i]
            print(f"Loading dataset: {dataset_name} (weight: {weight}x)")

            train_ds, val_ds, test_ds = load_cafa5_dataset(
                dataset=args.cafa5_dataset,
                dataset_name=dataset_name,
                cache_dir=args.dataset_cache_dir,
                dataset_subset=args.cafa5_dataset_subset,
                max_length=args.max_length_protein,
                seed=args.seed,
                val_split_ratio=args.val_split_ratio,
                return_as_chat_template=True,
                structure_dir=args.structure_dir,
                debug=args.debug,
                include_go_defs=args.include_go_defs,
                interpro_dataset_name=args.interpro_dataset_name,
                split_go_aspects=args.split_go_aspects,
                interpro_in_prompt=args.interpro_in_prompt,
                ppi_in_prompt=args.ppi_in_prompt,
                predict_interpro=args.predict_interpro,
                include_protein_function_summary=args.include_protein_function_summary,
                reasoning_dataset_name=args.reasoning_dataset_name,
                include_ground_truth_in_final_answer=args.include_ground_truth_in_final_answer,
                add_uniprot_summary=args.add_uniprot_summary,
                is_swissprot=args.is_swissprot,
                min_go_mf_freq=args.min_go_mf_freq,
                min_go_bp_freq=args.min_go_bp_freq,
                min_go_cc_freq=args.min_go_cc_freq,
                apply_go_filtering_to_val_test=args.apply_go_filtering_to_val_test,
                go_gpt_predictions_column=args.go_gpt_predictions_column,
            )

            print(f"  - Original sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)} samples")

            # Repeat datasets according to their weights
            for repeat_idx in range(weight):
                all_train_datasets.append(train_ds)
                all_val_datasets.append(val_ds)
                all_test_datasets.append(test_ds)

            print(
                f"  - After weighting ({weight}x) - Train: {len(train_ds) * weight}, Val: {len(val_ds) * weight}, Test: {len(test_ds) * weight} effective samples"
            )

        # Fix 'length' field type mismatch before concatenation
        def fix_length_type(dataset):
            """Convert 'length' field from float64 to int64."""
            if "length" in dataset.features:
                return dataset.cast_column("length", Value("int64"))
            return dataset

        # Apply the fix to all datasets
        all_train_datasets = [fix_length_type(ds) for ds in all_train_datasets]
        all_val_datasets = [fix_length_type(ds) for ds in all_val_datasets]
        all_test_datasets = [fix_length_type(ds) for ds in all_test_datasets]

        # Concatenate all datasets using HuggingFace datasets concatenate_datasets
        train_dataset = (
            concatenate_datasets(all_train_datasets) if len(all_train_datasets) > 1 else all_train_datasets[0]
        )
        val_dataset = concatenate_datasets(all_val_datasets) if len(all_val_datasets) > 1 else all_val_datasets[0]
        test_dataset = concatenate_datasets(all_test_datasets) if len(all_test_datasets) > 1 else all_test_datasets[0]

        # Re-shuffle the concatenated datasets to properly mix samples from different sources
        if len(dataset_names) > 1:
            print("Re-shuffling concatenated datasets to properly mix samples from different sources...")
            train_dataset = train_dataset.shuffle(seed=args.seed)
            val_dataset = val_dataset.shuffle(seed=args.seed)
            test_dataset = test_dataset.shuffle(seed=args.seed)

        print(
            f"Mixed dataset totals - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)} samples"
        )

    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{args.wandb_project}-{args.dataset_type}-{args.text_model_name.split('/')[-1]}"

    # Initialize model with pre-loaded datasets
    model = ProteinLLMFineTuner(
        args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    # Setup callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
    ]

    # Only enable model checkpointing for Stage 2, as we only need the projector weights from Stage 1
    if args.training_stage == 2:
        # 1) Keep the single best by lowest val_loss_epoch (saved at validation end)
        best_val_ckpt = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"{run_name}-best-epoch={{epoch:02d}}-val={{val_loss_epoch:.4f}}",
            monitor="val_loss_epoch",
            mode="min",
            save_top_k=1,
            save_last=False,
            save_on_train_epoch_end=False,
            verbose=True,
        )
        # 2) Keep the most recent, saved every N training steps
        recent_ckpts = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"{run_name}-recent-epoch={{epoch:02d}}-step={{step:06d}}",
            save_top_k=args.save_top_k,
            monitor="step",
            mode="max",
            save_last=True,
            every_n_train_steps=args.every_n_train_steps,
            save_on_train_epoch_end=False,
            save_weights_only=False,
            verbose=True,
        )
        # 3) Save every epoch from checkpoint_start_epoch onwards
        epoch_ckpt = EpochCheckpointFromN(
            checkpoint_dir=args.checkpoint_dir,
            run_name=run_name,
            start_epoch=args.checkpoint_start_epoch,
        )
        callbacks.extend([recent_ckpts, best_val_ckpt, epoch_ckpt])

    # Setup logger
    is_resuming = args.ckpt_path is not None
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        save_dir=args.log_dir,
        name=run_name,
        resume="allow" if is_resuming else None,  # Allow resuming existing run
        log_model=False,
    )

    # Configure Lightning AdvancedProfiler (simple and robust)
    profiler = None
    if args.enable_profiler:
        os.makedirs(args.profiler_dir, exist_ok=True)
        profiler = AdvancedProfiler(dirpath=args.profiler_dir, filename=args.profiler_filename)

    # Optionally add device stats monitor
    if args.enable_device_stats_monitor:
        callbacks.append(DeviceStatsMonitor(cpu_stats=args.device_stats_cpu))

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=(args.max_steps if (args.max_steps is not None and args.max_steps > 0) else -1),
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=args.strategy,
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        val_check_interval=args.val_check_interval,
        num_nodes=args.num_nodes,
        profiler=profiler,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        enable_model_summary=True,
        enable_progress_bar=True,
        sync_batchnorm=False,
    )

    # Start the training process
    trainer.fit(model, ckpt_path=args.ckpt_path)

    # After stage 1, save the projector weights
    if args.training_stage == 1 and trainer.global_rank == 0:
        projector_weights_path = os.path.join(args.checkpoint_dir, "projector_weights.pt")
        print(f"Stage 1 finished. Saving projector weights to {projector_weights_path}")
        torch.save(model.model.protein_projection.state_dict(), projector_weights_path)
        print("✓ Projector weights saved.")

        # Also save GO projection weights if available
        if hasattr(model.model, "go_projection") and model.model.go_projection is not None:
            go_projection_weights_path = os.path.join(args.checkpoint_dir, "go_projection_weights.pt")
            print(f"Saving GO projection weights to {go_projection_weights_path}")
            torch.save(model.model.go_projection.state_dict(), go_projection_weights_path)
            print("✓ GO projection weights saved.")

        # Also save GO encoder weights if available
        if hasattr(model.model, "go_encoder") and model.model.go_encoder is not None:
            go_encoder_weights_path = os.path.join(args.checkpoint_dir, "go_encoder_weights.pt")
            print(f"Saving GO encoder weights to {go_encoder_weights_path}")
            torch.save(model.model.go_encoder.state_dict(), go_encoder_weights_path)
            print("✓ GO encoder weights saved.")

    # trainer.test(model, ckpt_path=args.ckpt_path if args.ckpt_path else "best")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add command-line arguments
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs.")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.expanduser("~/.cache/huggingface/hub"),
        help="Directory for HuggingFace model cache.",
    )
    parser.add_argument("--text_model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--protein_model_name",
        type=str,
        default="esm3_sm_open_v1",
        choices=["esm3_sm_open_v1", "esmc_300m", "esmc_600m"],
        help="Protein model name. Supported models: ESM3 (esm3_sm_open_v1) and ESM-C (esmc_300m, esmc_600m).",
    )
    parser.add_argument("--model_type", type=str, default="protein-llm")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--max_length_protein", type=int, default=2000)
    parser.add_argument("--max_length_text", type=int, default=4000)
    parser.add_argument("--max_assistant_reasoning_length", type=int, default=4000)
    parser.add_argument("--text_model_finetune", type=str2bool, default=True)
    parser.add_argument("--protein_model_finetune", type=str2bool, default=False)
    parser.add_argument(
        "--protein_train_layer_start", 
        type=int, 
        default=36,
        help="ESM3 transformer layer to start training from. Default 36 trains last 25%% of layers. Use -1 or >=48 to train only output heads, 0 to train all transformer layers."
    )
    parser.add_argument(
        "--protein_embedding_layer",
        type=int,
        default=-1,
        help="ESM3 layer to extract embeddings from. Use -1 for final output (default), 0-N for specific transformer layers. Only works with ESM3 models."
    )
    parser.add_argument("--go_model_finetune", type=str2bool, default=True)

    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--go_obo_path", type=str, default=None)
    parser.add_argument("--precomputed_embeddings_path", type=str, default=None)
    parser.add_argument("--go_hidden_dim", type=int, default=512)
    parser.add_argument("--go_num_gat_layers", type=int, default=3)
    parser.add_argument("--go_num_heads", type=int, default=8)
    parser.add_argument("--go_num_reduced_embeddings", type=int, default=200)
    parser.add_argument("--go_embedding_dim", type=int, default=2560)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_qlora", type=str2bool, default=False, help="Enable QLoRA 4-bit quantization")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", help="Compute dtype for 4-bit quantization")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="Quantization type (nf4 or fp4)")
    parser.add_argument("--bnb_4bit_use_double_quant", type=str2bool, default=True, help="Use double quantization")
    parser.add_argument("--use_unsloth", type=str2bool, default=True, help="Use Unsloth for faster training")
    parser.add_argument("--strategy", type=str, default="auto")    # or use ddp_find_unused_parameters_false with unsloth
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="esm3-qwen-1.7B-finetune",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="adibvafa",
        help="WandB entity (username or team).",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["cafa5"],
        default="cafa5",
    )
    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.1,
        help="Ratio of training data to use for validation",
    )
    parser.add_argument("--cafa5_dataset", type=str, default="wanglab/cafa5")
    parser.add_argument(
        "--cafa5_dataset_name",
        type=str,
        default="cafa5_reasoning",
        help="CAFA5 dataset name(s). Use comma-separated values to mix multiple datasets (e.g., 'swissprot_reasoning,experiment_data')",
    )
    parser.add_argument(
        "--cafa5_dataset_weights",
        type=str,
        default=None,
        help="Dataset sampling weights (comma-separated). Controls how often each dataset appears. E.g., '5,1' means first dataset appears 5x more often than second. If not specified, all datasets weighted equally.",
    )
    parser.add_argument(
        "--interpro_dataset_name",
        type=str,
        default=None,
        help="Name of InterPro metadata dataset config. If None, InterPro data is not included.",
    )
    parser.add_argument(
        "--include_go_defs",
        type=str2bool,
        default=True,
        help="Whether to include GO term definitions in the training data",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--cafa5_dataset_subset", type=str, default=None)
    parser.add_argument(
        "--reasoning_dataset_name",
        type=str,
        default=None,
        help="Config name for reasoning traces dataset (e.g., 'experiment_data_reasoning'). If provided, uses reasoning data instead of generating assistant reasoning. Requires split_go_aspects=False since reasoning contains comprehensive analysis for all GO aspects together.",
    )
    
    parser.add_argument("--structure_dir", type=str, default=None)
    parser.add_argument("--return_answer_in_batch", type=str2bool, default=False)
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--val_check_interval", type=float, default=0.2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--include_ground_truth_in_final_answer", type=str2bool, default=True)
    parser.add_argument("--add_uniprot_summary", type=str2bool, default=False)
    parser.add_argument("--is_swissprot", type=str2bool, default=False)

    # Arguments for staged training
    parser.add_argument(
        "--training_stage",
        type=int,
        default=2,
        choices=[1, 2],
        help="Training stage: 1 for projector only, 2 for full model.",
    )
    parser.add_argument(
        "--projector_checkpoint_path",
        type=str,
        default=None,
        help="Path to a pretrained projector checkpoint for stage 2.",
    )
    parser.add_argument(
        "--go_projection_checkpoint_path",
        type=str,
        default=None,
        help="Path to a pretrained GO projection checkpoint for stage 2.",
    )
    parser.add_argument(
        "--go_encoder_checkpoint_path",
        type=str,
        default=None,
        help="Path to a pretrained GO encoder checkpoint for stage 2.",
    )
    parser.add_argument("--checkpoint_start_epoch", type=int, default=4, help="Epoch to start saving checkpoints from.")
    parser.add_argument("--every_n_train_steps", type=int, default=None)
    # Arguments for sample generation
    parser.add_argument(
        "--enable_sample_generation",
        type=str2bool,
        default=False,
        help="Enable generation of sample responses during training.",
    )
    parser.add_argument(
        "--verbose_sample_generation",
        type=str2bool,
        default=False,
        help="Print generated samples to the console.",
    )
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for the WandB run.")
    parser.add_argument(
        "--split_go_aspects",
        type=str2bool,
        default=False,
        help="Split each protein into separate examples for each GO aspect (MF, BP, CC)",
    )
    parser.add_argument(
        "--interpro_in_prompt",
        type=str2bool,
        default=False,
        help="Include InterPro data in user prompt instead of generation",
    )
    parser.add_argument(
        "--predict_interpro",
        type=str2bool,
        default=False,
        help="Ask model to predict InterPro terms as part of generation (when interpro_in_prompt=False)",
    )
    parser.add_argument(
        "--ppi_in_prompt",
        type=str2bool,
        default=False,
        help="Include PPI data in user prompt instead of generation",
    )
    parser.add_argument(
        "--include_protein_function_summary",
        type=str2bool,
        default=True,
        help="Include protein function summaries in the training data",
    )
    # GO term frequency filtering parameters
    parser.add_argument(
        "--min_go_mf_freq",
        type=int,
        default=1,
        help="Minimum frequency for Molecular Function GO terms to be included (default=1)",
    )
    parser.add_argument(
        "--min_go_bp_freq",
        type=int,
        default=1,
        help="Minimum frequency for Biological Process GO terms to be included (default=1)",
    )
    parser.add_argument(
        "--min_go_cc_freq",
        type=int,
        default=1,
        help="Minimum frequency for Cellular Component GO terms to be included (default=1)",
    )
    parser.add_argument(
        "--apply_go_filtering_to_val_test",
        type=str2bool,
        default=False,
        help="Whether to apply GO frequency filtering to validation/test sets. For pre-split datasets: controls whether val/test are filtered. For non-pre-split datasets: must be True if filtering is enabled.",
    )
    parser.add_argument(
        "--go_gpt_predictions_column",
        type=str,
        default=None,
        help="Column name containing pre-computed GO-GPT predictions (e.g., 'go_pred'). If the dataset has this column, predictions will be included in reasoning prompts. Only works with reasoning_dataset_name.",
    )
    parser.add_argument(
        "--unified_go_encoder",
        type=str2bool,
        default=False,
        help="Whether to use unified GOGraphEncoderUnified",
    )
    # Profiling controls (AdvancedProfiler only)
    parser.add_argument("--enable_profiler", type=str2bool, default=False)
    parser.add_argument("--profiler_dir", type=str, default="profiles")
    parser.add_argument("--profiler_filename", type=str, default="profile")
    # Short-run controls for profiling
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="If > 0, overrides to run a fixed number of steps",
    )
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--num_sanity_val_steps", type=int, default=2)
    # Device stats monitor controls
    parser.add_argument("--enable_device_stats_monitor", type=str2bool, default=False)
    parser.add_argument("--device_stats_cpu", type=str2bool, default=False)
    args = parser.parse_args()

    main(args)
