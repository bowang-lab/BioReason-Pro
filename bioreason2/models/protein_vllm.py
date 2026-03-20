import os
import glob
from typing import Optional, List, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from safetensors import safe_open
from vllm import LLM, SamplingParams
from esm.models.esm3 import ESM3

from bioreason2.models.pl.processing_pl import PLProcessor
from bioreason2.models.pl.chat_template_pl import get_chat_template
from bioreason2.models.protein_encoder import create_protein_encoder
from bioreason2.models.go_graph_encoder import create_go_graph_encoder_pipeline
from bioreason2.models.special_tokens import get_all_special_tokens, get_token


class ProteinLLMModel(nn.Module):
    """
    A combined model that processes both protein sequences and text inputs.

    The model uses a protein encoder (ESM3 or ESM-C) to extract features from protein sequences
    and a text model (LLM) to process text inputs and generate responses. The protein features are
    projected to the text model's embedding space and prepended to the text embeddings.
    """

    def __init__(
        self,
        ckpt_dir: Optional[str] = None,
        text_model_name: str = "Qwen/Qwen3-4B-Thinking-2507",
        protein_model_name: str = "esm3_sm_open_v1",
        cache_dir: Optional[str] = None,
        max_length_protein: int = 2048,
        max_length_text: int = 4096,
        text_model_finetune: bool = False,
        protein_model_finetune: bool = False,
        protein_train_layer_start: int = 36,
        protein_embedding_layer: int = -1,
        go_model_finetune: bool = False,
        attn_implementation: str = "flash_attention_2",
        go_obo_path: Optional[str] = None,
        precomputed_embeddings_path: Optional[str] = None,
        go_hidden_dim: int = 512,
        go_num_gat_layers: int = 3,
        go_num_heads: int = 8,
        go_num_reduced_embeddings: int = 200,
        go_embedding_dim: int = 2560,
        unified_go_encoder: bool = False,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 32768,
        max_num_seqs: int = 256,
    ):
        """
        Initialize the ProteinLLMModel for vLLM inference.

        Args:
            ckpt_dir: Directory or identifier of the pre-trained text model.
            text_model_name: Name of the text model to be used.
            protein_model_name: Name of the protein model to be used (ESM3 or ESM-C).
            cache_dir: Directory to cache the models.
            max_length_protein: Maximum length of protein sequences. Defaults to 2048.
            max_length_text: Maximum length of text sequences. Defaults to 4096.
            text_model_finetune: Whether to finetune the text model. Defaults to False.
            protein_model_finetune: Whether to finetune the protein model. Defaults to False.
            protein_train_layer_start: ESM3 layer to start training from. Use -1 or >=total_blocks for output heads only, 0 for all transformer layers. Defaults to 36.
            protein_embedding_layer: ESM3 layer to extract embeddings from. Use -1 for final output (default), 0-N for specific transformer layers. Only works with ESM3 models.
            go_model_finetune: Whether to finetune the GO graph encoder. Defaults to False.
            attn_implementation: Attention implementation to use. Defaults to "flash_attention_2".
            go_obo_path: Path to GO ontology OBO file. If None, GO encoder will be disabled.
            precomputed_embeddings_path: Directory with GO embeddings .safetensors files.
            go_hidden_dim: Hidden dimension for GO GAT layers. Defaults to 512.
            go_num_gat_layers: Number of GAT layers in GO encoder. Defaults to 3.
            go_num_heads: Number of attention heads in GO GAT. Defaults to 8.
            go_num_reduced_embeddings: Number of reduced embeddings per GO namespace. Defaults to 200.
            go_embedding_dim: GO embedding dimension. Defaults to 2560.
            unified_go_encoder: If True, use unified GOGraphEncoderUnified; if False, use original GOGraphEncoder. Defaults to False.
            gpu_memory_utilization: GPU memory utilization for vLLM. Defaults to 0.4.
            max_model_len: Maximum length of the model. Defaults to 32768.
            max_num_seqs: Maximum number of sequences to process concurrently in vLLM. Defaults to 256.
        """
        super().__init__()

        self.text_model_finetune = text_model_finetune
        self.protein_model_finetune = protein_model_finetune
        self.protein_train_layer_start = protein_train_layer_start
        self.protein_embedding_layer = protein_embedding_layer
        self.go_model_finetune = go_model_finetune
        self.max_length_protein = max_length_protein
        self.max_length_text = max_length_text
        self.max_model_len = max_model_len
        self.unified_go_encoder = unified_go_encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        # Load the text model and tokenizer
        self.text_model = LLM(
            model=ckpt_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prompt_embeds=True,
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            max_num_seqs=max_num_seqs,
            dtype=self.dtype
        )

        self.text_tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        self.text_config = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
        self.text_tokenizer.chat_template = get_chat_template(text_model_name)
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        # Add special tokens from centralized module
        all_special_tokens = get_all_special_tokens()
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": all_special_tokens})

        self.protein_token_id = self.text_tokenizer.convert_tokens_to_ids(get_token("protein_pad"))
        self.go_token_id = self.text_tokenizer.convert_tokens_to_ids(get_token("go_graph_pad"))

        # # Resize text model embeddings to accommodate new tokens
        # self.text_model.resize_token_embeddings(len(self.text_tokenizer))

        # Load the protein encoder (ESM3 or ESM-C)
        self.protein_encoder = create_protein_encoder(
            protein_model_name, 
            inference_mode=not protein_model_finetune,
            embedding_layer=protein_embedding_layer
        )
        self.protein_model = self.protein_encoder.model
        self.protein_model = self.protein_model.to(self.device, dtype=self.dtype)

        # Get embedding dimensions
        self.text_hidden_size = self.text_config.hidden_size
        self.protein_hidden_size = self.protein_encoder.embedding_dim

        # Initialize GO graph encoder if paths are provided
        self.go_encoder = None
        self.go_embeddings_cache = {}  # Cache for GO embeddings when encoder is frozen

        if go_obo_path is not None and precomputed_embeddings_path is not None:
            self.go_encoder = create_go_graph_encoder_pipeline(
                go_obo_path=go_obo_path,
                precomputed_embeddings_path=precomputed_embeddings_path,
                hidden_dim=go_hidden_dim,
                num_gat_layers=go_num_gat_layers,
                num_heads=go_num_heads,
                num_reduced_embeddings=go_num_reduced_embeddings,
                embedding_dim=go_embedding_dim,
                embeddings_load_to=str(self.device),
                unified_go_encoder=unified_go_encoder,
            )

            # Create projection layer for GO embeddings to text space
            self.go_projection = nn.Sequential(
                nn.Linear(go_embedding_dim, self.text_hidden_size),
                nn.GELU(),
                nn.Linear(self.text_hidden_size, self.text_hidden_size),
            ).to(device=self.device, dtype=self.dtype)
        else:
            self.go_projection = None

        # Create projection layer to map protein embeddings to text model's embedding space
        self.protein_projection = nn.Sequential(
            nn.Linear(self.protein_hidden_size, self.text_hidden_size),
            nn.GELU(),
            nn.Linear(self.text_hidden_size, self.text_hidden_size),
        ).to(device=self.device, dtype=self.dtype)

        # load custom components
        self.load_custom_components(ckpt_dir)

        # Initialize all models in eval mode with frozen parameters by default
        # Training setup will be handled by train_protein_llm.py
        self._setup_default_eval_mode()

        # Create processor for handling inputs
        self.processor = PLProcessor(tokenizer=self.text_tokenizer)

        # Initialize embedding layer on CPU first for safer multi-GPU loading
        self._embedding_layer = nn.Embedding(self.text_config.vocab_size, self.text_config.hidden_size)
        print(
            f"🧬 Initialized local embedding layer with shape: ({self.text_config.vocab_size}, {self.text_config.hidden_size})"
        )

        # Load weights to CPU first, then move to device (safer for multi-GPU setups)
        try:
            safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
            if not os.path.exists(safetensors_path):
                raise FileNotFoundError("model.safetensors not found. Looking for sharded checkpoints...")

            # The tensor name 'model.embed_tokens.weight' is standard for Qwen and many other models.
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                embed_weights = f.get_tensor("model.embed_tokens.weight")
                self._embedding_layer.weight.data = embed_weights
                print(f"✅ Loaded embedding weights from '{safetensors_path}'")

        except Exception as e:
            print(f"⚠️ Could not load weights from primary safetensors file: {e}. Trying sharded files...")
            try:
                # Fallback for sharded models (e.g., model-00001-of-00002.safetensors)
                shard_files = sorted(glob.glob(os.path.join(ckpt_dir, "model-*.safetensors")))
                if not shard_files:
                    raise FileNotFoundError("No sharded safetensors files found.")

                print(f"🧬 Found {len(shard_files)} sharded checkpoint files.")
                loaded = False
                for shard_file in shard_files:
                    with safe_open(shard_file, framework="pt", device="cpu") as f:
                        if "model.embed_tokens.weight" in f.keys():
                            embed_weights = f.get_tensor("model.embed_tokens.weight")
                            self._embedding_layer.weight.data = embed_weights
                            print(f"✅ Loaded embedding weights from shard: '{shard_file}'")
                            loaded = True
                            break
                if not loaded:
                    raise RuntimeError("Embedding weights not found in any shard.")
            except Exception as e_shard:
                print(f"❌ CRITICAL: Failed to load embedding weights from checkpoint files: {e_shard}")
                print("   The model will use RANDOMLY INITIALIZED embeddings, leading to incorrect output.")

        # Move the embedding layer to the correct device and dtype
        self._embedding_layer = self._embedding_layer.to(self.device, dtype=self.dtype)
        print(f"🧬 Moved embedding layer to device: {self.device}")

    def load_custom_components(
        self,
        llm_dir: str,
    ) -> None:

        # ===== Protein projection =====
        projection_path = os.path.join(llm_dir, "protein_projection.pt")
        if os.path.exists(projection_path):
            state = torch.load(projection_path, map_location=self.device)
            self.protein_projection.load_state_dict(state, strict=True)
            self.protein_projection = self.protein_projection.to(device=self.device, dtype=self.dtype)
            print(f"✅ Loaded protein projection from {projection_path} (converted to bfloat16)")
        else:
            raise FileNotFoundError(f"Protein projection not found at {projection_path}")

        # ===== Optional local protein model (existing) =====
        protein_model_path = os.path.join(llm_dir, "protein_model")
        if os.path.exists(protein_model_path):
            print(f"📁 Found local protein model at {protein_model_path}")
            try:
                self.protein_model = ESM3.from_pretrained(protein_model_path).to(self.device, dtype=self.dtype)
                print("✅ Local protein model loaded successfully")
            except Exception as e:
                print(f"⚠️ Error loading local protein model: {e} (keeping original)")

        # ===== GO encoder + projection =====
        if self.go_encoder is not None and self.go_projection is not None:
            # Load GO encoder weights
            go_encoder_path = os.path.join(llm_dir, "go_encoder.pt")
            if os.path.exists(go_encoder_path):
                go_state = torch.load(go_encoder_path, map_location=self.device)
                self.go_encoder.load_state_dict(go_state, strict=True)
                self.go_encoder = self.go_encoder.to(device=self.device, dtype=self.dtype)
                print(f"✅ Loaded GO encoder from {go_encoder_path} (converted to bfloat16)")
            else:
                print(f"⚠️ GO encoder weights not found at {go_encoder_path}")

            # Load GO projection weights
            go_proj_path = os.path.join(llm_dir, "go_projection.pt")
            if os.path.exists(go_proj_path):
                state = torch.load(go_proj_path, map_location=self.device)
                self.go_projection.load_state_dict(state, strict=True)
                self.go_projection = self.go_projection.to(device=self.device, dtype=self.dtype)
                print(f"✅ Loaded GO projection from {go_proj_path} (converted to bfloat16)")
            else:
                print(f"⚠️ GO projection weights not found at {go_proj_path}")

            # Load pre-computed GO embeddings if available (skips GO encoder forward pass)
            go_embedding_path = os.path.join(llm_dir, "go_embedding.pt")
            if os.path.exists(go_embedding_path):
                cached_embedding = torch.load(go_embedding_path, map_location=self.device)
                cached_embedding = cached_embedding.to(device=self.device, dtype=self.dtype)
                self.go_embeddings_cache["all"] = cached_embedding
                print(f"✅ Loaded pre-computed GO embedding from {go_embedding_path} (will skip GO encoder)")
            else:
                print(f"ℹ️ No pre-computed GO embedding found at {go_embedding_path}, will use GO encoder")

    def _setup_default_eval_mode(self):
        """
        Set all model components to eval mode with frozen parameters by default.
        Training setup will be handled by train_protein_llm.py.
        Note: Text model parameter freezing is skipped since vLLM handles it.
        """
        # vLLM text model: Skip parameter freezing (vLLM handles this)
        # The vLLM LLM class manages its own parameters
        
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
            aspect_key = "all"
            
            # Check cache if not finetuning (cache contains encoder output)
            if not self.go_model_finetune and aspect_key in self.go_embeddings_cache:
                reduced_embeddings = self.go_embeddings_cache[aspect_key]
            else:
                reduced_embeddings = self.go_encoder("all")  # (200, 2560)
                # Cache encoder output if not finetuning
                if not self.go_model_finetune:
                    self.go_embeddings_cache[aspect_key] = reduced_embeddings
            
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

                # Check cache if not finetuning (cache contains encoder output)
                if not self.go_model_finetune and aspect in self.go_embeddings_cache:
                    reduced_embeddings = self.go_embeddings_cache[aspect]
                else:
                    # Get reduced embeddings for this specific aspect (200, 2560)
                    reduced_embeddings = self.go_encoder(aspect)
                    # Cache encoder output if not finetuning
                    if not self.go_model_finetune:
                        self.go_embeddings_cache[aspect] = reduced_embeddings

                batch_go_embeddings.append(reduced_embeddings)

        # Project all embeddings to text embedding space
        if self.go_projection is not None:
            for i in range(len(batch_go_embeddings)):
                batch_go_embeddings[i] = batch_go_embeddings[i].to(
                    device=self.go_projection[0].weight.device,
                    dtype=self.go_projection[0].weight.dtype,
                )
                batch_go_embeddings[i] = self.go_projection(batch_go_embeddings[i])  # (200, text_hidden_size)

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
        raise RuntimeError("HF forward is not supported with vLLM. Use .generate() with prompt_embeds.")

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

        # Ensure input_ids is on the same device as the embedding layer
        input_ids = input_ids.to(self.device)

        # Get text embeddings from the model's embedding layer
        text_inputs_embeds = self._embedding_layer(input_ids)

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

            # Count tokens and embeddings for validation
            n_protein_tokens = mask.sum().item()
            protein_embeds_flat = torch.cat(batch_protein_embeds, dim=0)
            n_protein_features = protein_embeds_flat.shape[0]

            if n_protein_features != n_protein_tokens:
                raise ValueError(
                    f"Protein features and protein tokens do not match: "
                    f"features {n_protein_features}, tokens: {n_protein_tokens}"
                )

            # Ensure protein embeddings have the same dtype as text embeddings
            protein_embeds_flat = protein_embeds_flat.to(
                dtype=text_inputs_embeds.dtype, device=text_inputs_embeds.device
            )

            # Replace protein tokens with actual protein embeddings
            text_inputs_embeds[mask] = protein_embeds_flat

        # Process GO embeddings if provided
        if go_embeddings is not None:
            # Find positions where GO tokens should be replaced
            go_mask = input_ids == self.go_token_id

            # Count tokens and embeddings for validation
            n_go_tokens = go_mask.sum().item()
            go_embeds_flat = torch.cat([emb for emb in go_embeddings if emb.numel() > 0], dim=0)
            n_go_features = go_embeds_flat.shape[0] if go_embeds_flat.numel() > 0 else 0

            if n_go_features != n_go_tokens:
                raise ValueError(
                    f"GO embeddings and GO tokens do not match: " f"embeddings {n_go_features}, tokens: {n_go_tokens}"
                )

            if n_go_tokens > 0:
                # Ensure GO embeddings have the same dtype and device as text embeddings
                go_embeds_flat = go_embeds_flat.to(
                    dtype=text_inputs_embeds.dtype, device=text_inputs_embeds.device
                )

                # Replace GO tokens with actual GO embeddings
                text_inputs_embeds[go_mask] = go_embeds_flat

        # Generation with embeddings using vLLM
        sampling_params = SamplingParams(
            temperature=generation_kwargs.get("temperature", 0),
            top_p=generation_kwargs.get("top_p", 0.95),
            max_tokens=generation_kwargs.get("max_new_tokens", 1000),
            stop=generation_kwargs.get("stop", ["<|im_end|>"]) + ["- Hypothesized Interaction Partners"],
        )

        # Build requests for vLLM generation
        requests = []
        for i in range(batch_size):
            requests.append({"prompt_embeds": text_inputs_embeds[i]})

        # Generate for the entire batch in one call
        vllm_outputs = self.text_model.generate(requests, sampling_params=sampling_params)

        # Extract the generated text from the output objects
        return [output.outputs[0].text for output in vllm_outputs]
