"""
Protein encoder implementations for ESM3 and ESM-C models.

This module provides a unified interface for protein embedding generation using different ESM models.
"""

import torch
from abc import ABC, abstractmethod
from typing import List, Optional

# ESM imports
from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, SamplingConfig, LogitsConfig
from esm.utils.sampling import _BatchedESMProteinTensor


class ProteinEncoder(ABC):
    """Abstract base class for protein encoders"""

    def __init__(self, model_name: str, inference_mode: bool = True):
        self.model_name = model_name
        self.model = None
        self._embedding_dim = None
        self.inference_mode = inference_mode
        self._protein_train_layer_start = 36  # Default value

    @abstractmethod
    def encode_sequences(
        self,
        protein_sequences: List[str],
        batch_idx_map: List[int],
        batch_size: int,
        structure_coords: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Encode protein sequences into embeddings.

        Args:
            protein_sequences: List of protein sequence strings
            batch_idx_map: Mapping of each sequence to its batch item
            batch_size: Number of items in the batch
            structure_coords: Optional tensor containing structure coordinates

        Returns:
            List of tensor embeddings for each batch item
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension"""
        pass

    @property
    @abstractmethod
    def supports_structure(self) -> bool:
        """Whether this encoder supports structure information"""
        pass

    @abstractmethod
    def setup_training(self, protein_train_layer_start: int = 36):
        """
        Set up training configuration for the protein encoder.
        
        Args:
            protein_train_layer_start: Layer to start training from. 
                                     -1 or >=total_blocks for output heads only.
        """
        pass

    def set_inference_mode(self, inference_mode: bool, protein_train_layer_start: int = 36):
        """
        Set the inference mode for the model.
        
        Args:
            inference_mode: If True, set to inference mode (eval). If False, set to training mode.
            protein_train_layer_start: Layer to start training from when enabling training mode.
        """
        self.inference_mode = inference_mode
        self._protein_train_layer_start = protein_train_layer_start  # Store for future use

        if inference_mode:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.setup_training(protein_train_layer_start)


class ESM3Encoder(ProteinEncoder):
    """ESM3 protein encoder with structure support"""

    def __init__(self, model_name: str, inference_mode: bool = True, embedding_layer: int = -1):
        super().__init__(model_name, inference_mode)

        self.model = ESM3.from_pretrained(model_name)
        self.sampling_config = SamplingConfig
        self._embedding_dim = self.model.encoder.sequence_embed.embedding_dim
        
        # Layer selection for embeddings - ESM3 always has transformer.blocks
        self.total_blocks = len(self.model.transformer.blocks)
        
        self.embedding_layer = self._validate_embedding_layer(embedding_layer)
        self.layer_outputs = {}
        self.hooks = []
        
        self.set_inference_mode(self.inference_mode)

    def _validate_embedding_layer(self, embedding_layer: int) -> int:
        """
        Validate and normalize the embedding layer parameter.
        
        Args:
            embedding_layer: Layer index to extract embeddings from.
                           -1 means final output (default behavior)
                           0 to total_blocks-1 means specific transformer layer
                           
        Returns:
            Validated layer index
        """
        if embedding_layer == -1:
            return -1  # Final output
        elif 0 <= embedding_layer < self.total_blocks:
            return embedding_layer
        else:
            raise ValueError(
                f"embedding_layer must be -1 (final) or 0-{self.total_blocks-1}, got {embedding_layer}"
            )
    
    def _register_layer_hook(self, layer_idx: int):
        """Register a forward hook to capture output from a specific layer."""
        def hook_fn(module, input, output):
            self.layer_outputs[layer_idx] = output
        
        block = self.model.transformer.blocks[layer_idx]
        hook = block.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        return hook
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.layer_outputs.clear()

    def encode_sequences(
        self,
        protein_sequences: List[str],
        batch_idx_map: List[int],
        batch_size: int,
        structure_coords: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Encode sequences using ESM3 with optional layer-specific extraction"""

        # Initialize result list
        result = [[] for _ in range(batch_size)]

        # Set up layer hook if using specific layer
        if self.embedding_layer != -1:
            self._register_layer_hook(self.embedding_layer)

        # Process each protein sequence individually (ESM3 doesn't support batching)
        for seq_idx, sequence in enumerate(protein_sequences):

            # Check if we have valid structure coordinates to use
            coords_truncated = None
            if structure_coords is not None and structure_coords.numel() > 0:
                coords = structure_coords[batch_idx_map[seq_idx]].to(dtype=torch.float32)
                seq_len = len(sequence)
                
                # Only use coordinates if we have enough and they contain valid data
                if (coords.shape[0] >= seq_len and 
                    not torch.isnan(coords[:seq_len]).all()):
                    coords_truncated = coords[:seq_len]
            
            # Create ESMProtein with or without structure coordinates
            if coords_truncated is not None:
                protein = ESMProtein(sequence=sequence, coordinates=coords_truncated)
            else:
                protein = ESMProtein(sequence=sequence)

            # Encode protein
            protein_tensor = self.model.encode(protein)

            # Get embeddings - respect finetune parameter and layer selection
            with torch.set_grad_enabled(not self.inference_mode):

                if self.embedding_layer == -1:
                    # Standard behavior: get final embeddings
                    seq_embeddings = self.model.forward_and_sample(
                        protein_tensor,
                        self.sampling_config(return_per_residue_embeddings=True),
                    ).per_residue_embedding

                else:
                    # Layer-specific extraction: run forward pass and extract from hook
                    self.model.forward_and_sample(
                        protein_tensor,
                        self.sampling_config(return_per_residue_embeddings=True),
                    )
                    
                    # Extract embeddings from the specific layer - ESM3 blocks return direct tensors
                    seq_embeddings = self.layer_outputs[self.embedding_layer]
                    
                    # Layer outputs have extra batch dimension that needs squeezing
                    if seq_embeddings.dim() == 3 and seq_embeddings.shape[0] == 1:
                        seq_embeddings = seq_embeddings.squeeze(0)

            # Get the batch index for this sequence
            batch_idx = batch_idx_map[seq_idx]

            # Add to appropriate batch result
            result[batch_idx].append(seq_embeddings)

        # Clean up hooks
        self._clear_hooks()

        # Concatenate embeddings for each batch item
        for i in range(batch_size):
            if result[i]:
                result[i] = torch.cat(result[i], dim=0)
            else:
                # Empty tensor for batch items with no proteins
                result[i] = torch.zeros((0, self.embedding_dim))

        return result

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def supports_structure(self) -> bool:
        return True

    def setup_training(self, protein_train_layer_start: int = 36):
        """
        Set up layer-wise training for ESM3 protein encoder.
        
        Training behavior:
        - protein_train_layer_start = -1: Only train output heads
        - protein_train_layer_start >= total_blocks: Only train output heads  
        - Otherwise: Train output heads + transformer blocks from protein_train_layer_start onward
        - Encoder components are always frozen (for embedding stability)
        """
        # Freeze everything first and keep base model in eval mode
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        print("✓ ESM3 protein encoder training setup:")
        
        # Always train output heads when protein fine-tuning is enabled
        self._enable_output_heads_training()
        
        # Train transformer blocks based on protein_train_layer_start
        self._enable_transformer_training(protein_train_layer_start)
        
        # Summary
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        pct_total = (total_trainable / total_params) * 100 if total_params > 0 else 0
        
        print(f"  - Total trainable: {total_trainable:,} / {total_params:,} ({pct_total:.1f}%)")

    def _enable_output_heads_training(self) -> int:
        """Enable training for output heads and return parameter count."""
        self.model.output_heads.train()  # Set to train mode
        params = 0
        for param in self.model.output_heads.parameters():
            param.requires_grad = True
            params += param.numel()
        print(f"  - Output heads: {params:,} params (trainable)")
        return params
    
    def _enable_transformer_training(self, protein_train_layer_start: int) -> int:
        """Enable training for transformer blocks based on protein_train_layer_start."""
        blocks = self.model.transformer.blocks
        total_blocks = len(blocks)
        
        # Determine training behavior
        if protein_train_layer_start == -1:
            print(f"  - Transformer blocks: {total_blocks} total, 0 trainable (protein_train_layer_start=-1, output heads only)")
            return 0
        
        if protein_train_layer_start >= total_blocks:
            print(f"  - Transformer blocks: {total_blocks} total, 0 trainable (protein_train_layer_start={protein_train_layer_start} >= {total_blocks})")
            return 0
        
        # Train blocks from start_layer onward
        start_layer = max(0, protein_train_layer_start)
        trainable_params = 0
        
        for i in range(start_layer, total_blocks):
            blocks[i].train()  # Set trainable blocks to train mode
            for param in blocks[i].parameters():
                param.requires_grad = True
                trainable_params += param.numel()
        
        trainable_blocks = total_blocks - start_layer
        pct_trainable = (trainable_blocks / total_blocks) * 100
        
        print(f"  - Transformer blocks: {total_blocks} total, {trainable_blocks} trainable (layers {start_layer}-{total_blocks-1}) ({pct_trainable:.1f}%)")
        print(f"  - Transformer params: {trainable_params:,}")
        
        return trainable_params
    


class ESMCEncoder(ProteinEncoder):
    """ESM-C protein encoder for efficient representation learning"""

    def __init__(self, model_name: str, inference_mode: bool = True):
        super().__init__(model_name, inference_mode)

        self.model = ESMC.from_pretrained(model_name)
        self.logits_config = LogitsConfig
        self._embedding_dim = self._get_embedding_dim()
        self.set_inference_mode(self.inference_mode)

    def _get_embedding_dim(self) -> int:
        """Get the embedding dimension accurately from model"""
        # Known dimensions for official ESM-C models
        known_dims = {
            "esmc_300m": 640,
            "esmc_600m": 1152,
        }

        # Check known models first for accuracy
        try:
            return known_dims[self.model_name]
        except Exception as e:
            raise RuntimeError(f"Could not determine embedding dimension for {self.model_name}: {e}")

    def encode_sequences(
        self,
        protein_sequences: List[str],
        batch_idx_map: List[int],
        batch_size: int,
        structure_coords: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Encode sequences using ESM-C"""
        # Initialize result list
        result = [[] for _ in range(batch_size)]

        # Process each protein sequence individually
        for seq_idx, sequence in enumerate(protein_sequences):
            protein_tensor = self.model.encode(ESMProtein(sequence=sequence))

            # Get embeddings - respect finetune parameter
            with torch.set_grad_enabled(not self.inference_mode):

                # Create batch dimension if necessary
                if not isinstance(protein_tensor, _BatchedESMProteinTensor):
                    protein_tensor = _BatchedESMProteinTensor.from_protein_tensor(protein_tensor)

                # Forward pass
                seq_embeddings = self.model.forward(sequence_tokens=protein_tensor.sequence).embeddings

                # ESM-C may return shape (1, seq_len, dim), squeeze if needed
                if seq_embeddings.dim() == 3 and seq_embeddings.shape[0] == 1:
                    seq_embeddings = seq_embeddings.squeeze(0)

            # Get the batch index for this sequence
            batch_idx = batch_idx_map[seq_idx]

            # Add to appropriate batch result
            result[batch_idx].append(seq_embeddings)

        # Concatenate embeddings for each batch item
        for i in range(batch_size):
            if result[i]:
                result[i] = torch.cat(result[i], dim=0)
            else:
                # Empty tensor for batch items with no proteins
                result[i] = torch.zeros((0, self.embedding_dim))

        return result

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def supports_structure(self) -> bool:
        return False

    def setup_training(self, protein_train_layer_start: int = 36):
        """
        Set up training for ESM-C protein encoder.
        ESM-C has a simpler architecture, so we just freeze/unfreeze the entire model.
        
        Args:
            protein_train_layer_start: Ignored for ESM-C (no layer-wise training support)
        """
        # For ESM-C, we don't have fine-grained layer control like ESM3
        # So we just enable training for the entire model when requested
        self.model.train()  # Correct: All params trainable, so entire model in train mode
        for param in self.model.parameters():
            param.requires_grad = True
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print("✓ ESM-C protein encoder training setup:")
        print(f"  - Total trainable: {total_params:,} params (full model)")
        print("  - Note: ESM-C does not support layer-wise training")


def create_protein_encoder(model_name: str, inference_mode: bool = True, embedding_layer: int = -1) -> ProteinEncoder:
    """
    Factory function to create the appropriate protein encoder based on model name.

    Args:
        model_name: Name of the protein model (e.g., "esm3_sm_open_v1", "esmc_600m")
        inference_mode: Whether to set model in inference mode
        embedding_layer: Layer to extract embeddings from (-1 for final output, 0-N for specific layers)
                        Only supported for ESM3 models.

    Returns:
        Appropriate protein encoder instance
    """
    model_name_lower = model_name.lower()

    # ESM-C models (efficient representation learning)
    if "esmc" in model_name_lower:
        if embedding_layer != -1:
            print(f"⚠️  Warning: embedding_layer parameter ignored for ESM-C model {model_name}")
        return ESMCEncoder(model_name, inference_mode)

    # ESM3 models (generative, structure-aware)
    elif "esm3" in model_name_lower:
        return ESM3Encoder(model_name, inference_mode, embedding_layer)

    # Default to ESM3 for backward compatibility
    else:
        raise ValueError(f"Unknown model name '{model_name}' - defaulting to ESM3 encoder")
