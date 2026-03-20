import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from unsloth import FastLanguageModel

from typing import Optional, List, Union

from bioreason2.models.pl.processing_pl import PLProcessor
from bioreason2.models.pl.chat_template_pl import get_chat_template
from bioreason2.models.protein_encoder import create_protein_encoder
from bioreason2.models.go_graph_encoder import create_go_graph_encoder_pipeline
from bioreason2.models.special_tokens import get_all_special_tokens, get_token


def _get_target_modules(model):
    """Get target modules for LoRA fine-tuning."""
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


class ProteinLLMModel(nn.Module):
    """
    A combined model that processes both protein sequences and text inputs.

    The model uses a protein encoder (ESM3 or ESM-C) to extract features from protein sequences
    and a text model (LLM) to process text inputs and generate responses. The protein features are
    projected to the text model's embedding space and prepended to the text embeddings.
    """

    def __init__(
        self,
        text_model_name: str,
        protein_model_name: str = "esm3_sm_open_v1",
        cache_dir: Optional[str] = None,
        max_length_protein: int = 2048,
        max_length_text: int = 4096,
        text_model_finetune: bool = True,
        protein_model_finetune: bool = False,
        protein_train_layer_start: int = 36,
        protein_embedding_layer: int = -1,
        go_model_finetune: bool = True,
        attn_implementation: str = "flash_attention_2",
        go_obo_path: Optional[str] = None,
        precomputed_embeddings_path: Optional[str] = None,
        go_hidden_dim: int = 512,
        go_num_gat_layers: int = 3,
        go_num_heads: int = 8,
        go_num_reduced_embeddings: int = 200,  # Update processing_pl.py to use this as well
        go_embedding_dim: int = 2560,
        quantization_config: Optional[object] = None,  # QLoRA quantization config
        load_in_4bit: bool = False,
        unified_go_encoder: bool = False,
        use_unsloth: bool = True,
    ):
        """
        Initialize the ProteinLLMModel.

        Args:
            text_model_name: Name of the text model to be used.
            protein_model_name: Name of the protein model to be used (ESM3 or ESM-C).
            cache_dir: Directory to cache the models.
            max_length_protein: Maximum length of protein sequences. Defaults to 2048.
            max_length_text: Maximum length of text sequences. Defaults to 4096.
            text_model_finetune: Whether to finetune the text model. Defaults to True.
            protein_model_finetune: Whether to finetune the protein model. Defaults to False.
            protein_train_layer_start: ESM3 layer to start training from. Use -1 or >=total_blocks for output heads only, 0 for all transformer layers. Defaults to 36.
            protein_embedding_layer: ESM3 layer to extract embeddings from. Use -1 for final output (default), 0-N for specific transformer layers. Only works with ESM3 models.
            go_model_finetune: Whether to finetune the GO graph encoder. Defaults to True.
            attn_implementation: Attention implementation to use. Defaults to "flash_attention_2".
            go_obo_path: Path to GO ontology OBO file. If None, GO encoder will be disabled.
            precomputed_embeddings_path: Directory with GO embeddings .safetensors files.
            go_hidden_dim: Hidden dimension for GO GAT layers. Defaults to 512.
            go_num_gat_layers: Number of GAT layers in GO encoder. Defaults to 3.
            go_num_heads: Number of attention heads in GO GAT. Defaults to 8.
            go_num_reduced_embeddings: Number of reduced embeddings per GO namespace. Defaults to 200.
            go_embedding_dim: GO embedding dimension. Defaults to 2560.
            quantization_config: QLoRA quantization configuration for 4-bit training. Defaults to None.
            load_in_4bit: Whether to load the model in 4-bit for unsloth. Defaults to False.
            unified_go_encoder: If True, use unified GOGraphEncoderUnified; if False, use original GOGraphEncoder
            use_unsloth: If True, use Unsloth for faster training. Defaults to True.
        """
        super().__init__()

        self.text_model_finetune = text_model_finetune
        self.protein_model_finetune = protein_model_finetune
        self.protein_train_layer_start = protein_train_layer_start
        self.protein_embedding_layer = protein_embedding_layer
        self.go_model_finetune = go_model_finetune
        self.max_length_protein = max_length_protein
        self.max_length_text = max_length_text
        self.unified_go_encoder = unified_go_encoder
        self.use_unsloth = use_unsloth

        if use_unsloth:
            self.text_model, self.text_tokenizer = FastLanguageModel.from_pretrained(
                model_name=text_model_name,
                max_seq_length=max_length_text + max_length_protein + go_num_reduced_embeddings + 8,    # Use 8 for special tokens
                dtype=torch.bfloat16,
                load_in_4bit=load_in_4bit,
                cache_dir=cache_dir,
                trust_remote_code=True,
                device_map={"": "cpu"},
            )
        else:
            text_model_kwargs = {
                "cache_dir": cache_dir,
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "attn_implementation": attn_implementation,
            }
            if quantization_config is not None:
                text_model_kwargs["quantization_config"] = quantization_config
            
            self.text_model = AutoModelForCausalLM.from_pretrained(
                text_model_name,
                **text_model_kwargs
            )
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

        self.text_config = self.text_model.config
        self.text_tokenizer.chat_template = get_chat_template(text_model_name)

        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        # Add special tokens from centralized module
        all_special_tokens = get_all_special_tokens()
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": all_special_tokens})
        self.protein_token_id = self.text_tokenizer.convert_tokens_to_ids(get_token("protein_pad"))
        self.go_token_id = self.text_tokenizer.convert_tokens_to_ids(get_token("go_graph_pad"))

        self.text_model.resize_token_embeddings(len(self.text_tokenizer))

        # Load the protein encoder (ESM3 or ESM-C)
        self.protein_encoder = create_protein_encoder(
            protein_model_name, 
            inference_mode=not protein_model_finetune,
            embedding_layer=protein_embedding_layer
        )
        self.protein_model = self.protein_encoder.model

        # Get embedding dimensions
        self.text_hidden_size = self.text_config.hidden_size
        self.protein_hidden_size = self.protein_encoder.embedding_dim

        # Initialize GO graph encoder if paths are provided
        self.go_encoder = None
        if go_obo_path is not None and precomputed_embeddings_path is not None:
            self.go_encoder = create_go_graph_encoder_pipeline(
                go_obo_path=go_obo_path,
                precomputed_embeddings_path=precomputed_embeddings_path,
                hidden_dim=go_hidden_dim,
                num_gat_layers=go_num_gat_layers,
                num_heads=go_num_heads,
                num_reduced_embeddings=go_num_reduced_embeddings,
                embedding_dim=go_embedding_dim,
                unified_go_encoder=unified_go_encoder
            )

            # Create projection layer for GO embeddings to text space
            self.go_projection = nn.Sequential(
                nn.Linear(go_embedding_dim, self.text_hidden_size),
                nn.GELU(),
                nn.Linear(self.text_hidden_size, self.text_hidden_size),
            )
        else:
            self.go_projection = None

        # Create projection layer to map protein embeddings to text model's embedding space
        self.protein_projection = nn.Sequential(
            nn.Linear(self.protein_hidden_size, self.text_hidden_size),
            nn.GELU(),
            nn.Linear(self.text_hidden_size, self.text_hidden_size),
        )

        # Initialize all models in eval mode with frozen parameters by default
        # Training setup will be handled by train_protein_llm.py
        self._setup_default_eval_mode()

        # Create processor for handling inputs
        self.processor = PLProcessor(tokenizer=self.text_tokenizer)

    def _setup_default_eval_mode(self):
        """
        Set all model components to eval mode with frozen parameters by default.
        Training setup will be handled by train_protein_llm.py.
        """
        # Text model: eval mode, frozen
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # Protein encoder: use proper API to set inference mode
        self.protein_encoder.set_inference_mode(inference_mode=not self.protein_model_finetune)
        
        # Protein projection: eval mode, frozen
        self.protein_projection.eval()
        for param in self.protein_projection.parameters():
            param.requires_grad = False
            
        # GO encoder: eval mode, frozen
        if self.go_encoder is not None:
            self.go_encoder.eval()
            for param in self.go_encoder.parameters():
                param.requires_grad = False
        
        # GO projection: eval mode, frozen
        if self.go_projection is not None:
            self.go_projection.eval()
            for param in self.go_projection.parameters():
                param.requires_grad = False

    def process_protein_embeddings(
        self,
        protein_sequences: List[str],
        batch_idx_map: List[int],
        batch_size: int,
        structure_coords: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Process protein sequences and structures to obtain embeddings using the protein encoder.

        Args:
            protein_sequences: List of protein sequence strings
            batch_idx_map: Mapping of each sequence to its batch item
            batch_size: Number of items in the batch
            structure_coords: Optional tensor containing the structure coordinates

        Returns:
            List of tensor embeddings for each batch item
        """
        # Use the protein encoder to get embeddings
        batch_protein_embeddings = self.protein_encoder.encode_sequences(
            protein_sequences=protein_sequences,
            batch_idx_map=batch_idx_map,
            batch_size=batch_size,
            structure_coords=structure_coords,
        )

        # Project all embeddings to text embedding space
        for i in range(batch_size):
            if batch_protein_embeddings[i].numel() > 0:  # Check if tensor is not empty
                batch_protein_embeddings[i] = batch_protein_embeddings[i].to(
                    device=self.protein_projection[0].weight.device,
                    dtype=self.protein_projection[0].weight.dtype,
                )
                batch_protein_embeddings[i] = self.protein_projection(batch_protein_embeddings[i])
            else:
                # Ensure empty tensors have correct dimensions
                batch_protein_embeddings[i] = torch.zeros(
                    (0, self.text_hidden_size),
                    device=self.protein_projection[0].weight.device,
                    dtype=self.protein_projection[0].weight.dtype,
                )

        return batch_protein_embeddings

    def process_go_aspects(
        self,
        go_aspects: Optional[List[str]] = None,
        batch_size: int = 1,
    ) -> Optional[List[torch.Tensor]]:
        """
        Process GO aspects to obtain embeddings using the GO graph encoder.
        Each example gets its own aspect-specific embeddings or all aspects combined.

        Args:
            go_aspects: List of GO aspect strings for each batch item. If None or
                       individual items are None, defaults to "all" aspect.
            batch_size: Number of items in the batch

        Returns:
            Optional list of tensors with GO embeddings, one per batch item.
            Each tensor has shape (200, text_hidden_size) for specific aspect
            or combined all aspects when aspect is None or "all".
            Returns None if no GO encoder is available.
        """
        if self.go_encoder is None or go_aspects is None:
            return None

        batch_go_embeddings = []

        if self.unified_go_encoder:
            # For unified encoder, do one forward pass and duplicate for all batch items
            # Namespace doesn't matter for unified encoder
            reduced_embeddings = self.go_encoder("all")  # (200, 2560)

            # Project to text embedding space
            if self.go_projection is not None:
                reduced_embeddings = reduced_embeddings.to(
                    device=self.go_projection[0].weight.device,
                    dtype=self.go_projection[0].weight.dtype,
                )
                reduced_embeddings = self.go_projection(reduced_embeddings)  # (200, text_hidden_size)

            # Duplicate for all batch items
            for i in range(batch_size):
                batch_go_embeddings.append(reduced_embeddings)
        else:
            # Process each example's aspect separately for non-unified encoder
            for i in range(batch_size):
                # Use default "all" aspect if no specific aspect is provided
                if i < len(go_aspects) and go_aspects[i] is not None:
                    aspect = go_aspects[i]
                else:
                    aspect = "all"

                # Get reduced embeddings for this specific aspect (200, 2560)
                reduced_embeddings = self.go_encoder(aspect)

                # Project to text embedding space
                if self.go_projection is not None:
                    reduced_embeddings = reduced_embeddings.to(
                        device=self.go_projection[0].weight.device,
                        dtype=self.go_projection[0].weight.dtype,
                    )
                    reduced_embeddings = self.go_projection(reduced_embeddings)  # (200, text_hidden_size)
                batch_go_embeddings.append(reduced_embeddings)

        return batch_go_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        protein_sequences: Optional[List[str]] = None,
        batch_idx_map: Optional[List[int]] = None,
        structure_coords: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        go_aspects: Optional[List[str]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            protein_sequences: List of protein sequence strings
            batch_idx_map: Batch mapping for protein sequences
            structure_coords: Optional tensor containing the structure coordinates
            labels: Labels for supervised fine-tuning
            go_aspects: GO aspects for protein sequences
            **kwargs: Additional arguments

        Returns:
            Outputs from the text model
        """
        # Ensure required inputs are available
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        batch_size = input_ids.shape[0]

        # Get text embeddings from the model's embedding layer
        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        # Process GO aspects if provided
        go_embeddings = None
        if go_aspects is not None:
            go_embeddings = self.process_go_aspects(go_aspects, batch_size)

        # Process protein sequences if provided
        if protein_sequences is not None and batch_idx_map is not None:
            # Find positions where protein tokens should be replaced
            mask = input_ids == self.protein_token_id
            n_protein_tokens = mask.sum().item()

            batch_protein_embeds = self.process_protein_embeddings(
                protein_sequences,
                batch_idx_map,
                batch_size,
                structure_coords=structure_coords,
            )
            protein_embeds_flat = torch.cat(batch_protein_embeds, dim=0)
            n_protein_features = protein_embeds_flat.shape[0]

            if n_protein_features != n_protein_tokens:
                raise ValueError(
                    f"Protein features and protein tokens do not match: "
                    f"features {n_protein_features}, tokens: {n_protein_tokens}"
                )

            # Replace protein tokens with actual protein embeddings (out-of-place to preserve autograd graph)
            if n_protein_tokens > 0:
                orig_shape = text_inputs_embeds.shape  # (B, L, H)
                hidden_size = orig_shape[-1]
                embeds_2d = text_inputs_embeds.view(-1, hidden_size)
                mask_flat = mask.view(-1)
                idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
                # Compute difference at masked positions and scatter-add it to get an updated tensor without in-place assignment
                diff = protein_embeds_flat - embeds_2d.index_select(0, idx)
                embeds_2d = embeds_2d.scatter_add(0, idx.unsqueeze(1).expand(-1, hidden_size), diff)
                text_inputs_embeds = embeds_2d.view(orig_shape)

        # Process GO embeddings if provided
        if go_embeddings is not None:
            # Find positions where GO tokens should be replaced
            go_mask = input_ids == self.go_token_id

            # Count tokens and embeddings
            n_go_tokens = go_mask.sum().item()
            go_embeds_flat = torch.cat([emb for emb in go_embeddings if emb.numel() > 0], dim=0)
            n_go_features = go_embeds_flat.shape[0] if go_embeds_flat.numel() > 0 else 0

            if n_go_features != n_go_tokens:
                raise ValueError(
                    f"GO embeddings and GO tokens do not match: " f"embeddings {n_go_features}, tokens: {n_go_tokens}"
                )

            if n_go_tokens > 0:
                # Ensure GO embeddings have the same dtype as text embeddings
                go_embeds_flat = go_embeds_flat.to(dtype=text_inputs_embeds.dtype)

                # Replace GO tokens with actual GO embeddings (out-of-place)
                orig_shape = text_inputs_embeds.shape  # (B, L, H)
                hidden_size = orig_shape[-1]
                embeds_2d = text_inputs_embeds.view(-1, hidden_size)
                mask_flat = go_mask.view(-1)
                idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
                diff = go_embeds_flat - embeds_2d.index_select(0, idx)
                embeds_2d = embeds_2d.scatter_add(0, idx.unsqueeze(1).expand(-1, hidden_size), diff)
                text_inputs_embeds = embeds_2d.view(orig_shape)

        # Forward pass through the text model
        outputs = self.text_model(
            inputs_embeds=text_inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        protein_sequences: Optional[List[str]] = None,
        batch_idx_map: Optional[List[int]] = None,
        structure_coords: Optional[torch.Tensor] = None,
        go_aspects: Optional[List[str]] = None,
        **generation_kwargs,
    ) -> Union[torch.Tensor, List[str]]:
        """
        Generate text based on protein and text inputs.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            protein_sequences: List of protein sequence strings
            batch_idx_map: Batch mapping for protein sequences
            structure_coords: Optional tensor containing the structure coordinates
            go_aspects: GO aspects for protein sequences
            **generation_kwargs: Additional arguments for generation

        Returns:
            Generated token IDs
        """
        # Ensure required inputs are available
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        batch_size = input_ids.shape[0]

        # Get text embeddings from the model's embedding layer
        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        # Process GO aspects if provided
        go_embeddings = None
        if go_aspects is not None:
            go_embeddings = self.process_go_aspects(go_aspects, batch_size)

        # Process protein sequences if provided
        if protein_sequences is not None and batch_idx_map is not None:
            batch_protein_embeds = self.process_protein_embeddings(
                protein_sequences,
                batch_idx_map,
                batch_size,
                structure_coords=structure_coords,
            )

            # Find positions where protein tokens should be replaced
            mask = input_ids == self.protein_token_id

            # Count tokens and embeddings
            n_protein_tokens = mask.sum().item()
            protein_embeds_flat = torch.cat(batch_protein_embeds, dim=0)
            n_protein_features = protein_embeds_flat.shape[0]

            if n_protein_features != n_protein_tokens:
                raise ValueError(
                    f"Protein features and protein tokens do not match: "
                    f"features {n_protein_features}, tokens: {n_protein_tokens}"
                )

            # Ensure protein embeddings have the same dtype as text embeddings
            protein_embeds_flat = protein_embeds_flat.to(dtype=text_inputs_embeds.dtype)

            # Replace protein tokens with actual protein embeddings (out-of-place)
            if n_protein_tokens > 0:
                orig_shape = text_inputs_embeds.shape  # (B, L, H)
                hidden_size = orig_shape[-1]
                embeds_2d = text_inputs_embeds.view(-1, hidden_size)
                mask_flat = mask.view(-1)
                idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
                diff = protein_embeds_flat - embeds_2d.index_select(0, idx)
                embeds_2d = embeds_2d.scatter_add(0, idx.unsqueeze(1).expand(-1, hidden_size), diff)
                text_inputs_embeds = embeds_2d.view(orig_shape)

        # Process GO embeddings if provided
        if go_embeddings is not None:
            # Find positions where GO tokens should be replaced
            go_mask = input_ids == self.go_token_id

            # Count tokens and embeddings
            n_go_tokens = go_mask.sum().item()
            go_embeds_flat = torch.cat([emb for emb in go_embeddings if emb.numel() > 0], dim=0)
            n_go_features = go_embeds_flat.shape[0] if go_embeds_flat.numel() > 0 else 0

            if n_go_features != n_go_tokens:
                raise ValueError(
                    f"GO embeddings and GO tokens do not match: " f"embeddings {n_go_features}, tokens: {n_go_tokens}"
                )

            if n_go_tokens > 0:
                # Ensure GO embeddings have the same dtype as text embeddings
                go_embeds_flat = go_embeds_flat.to(dtype=text_inputs_embeds.dtype)

                # Replace GO tokens with actual GO embeddings (out-of-place)
                orig_shape = text_inputs_embeds.shape  # (B, L, H)
                hidden_size = orig_shape[-1]
                embeds_2d = text_inputs_embeds.view(-1, hidden_size)
                mask_flat = go_mask.view(-1)
                idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
                diff = go_embeds_flat - embeds_2d.index_select(0, idx)
                embeds_2d = embeds_2d.scatter_add(0, idx.unsqueeze(1).expand(-1, hidden_size), diff)
                text_inputs_embeds = embeds_2d.view(orig_shape)

        # Generation with embeddings
        text_inputs_embeds = text_inputs_embeds.to(input_ids.device)
        attention_mask = attention_mask.to(input_ids.device)

        with torch.inference_mode():
            outputs = self.text_model.generate(
                inputs_embeds=text_inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                **generation_kwargs,
            )

        return outputs  
