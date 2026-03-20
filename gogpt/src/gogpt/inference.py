"""
High-level inference API for GO-GPT model.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import yaml
from transformers import AutoTokenizer

from gogpt.models.gogpt import GOGPT
from gogpt.config.model_config import GOGPTConfig


@dataclass
class GOTokenizerJSON:
    """GO tokenizer loaded from JSON (portable, no pickle dependency)."""
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "GOTokenizerJSON":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            token_to_id=data["token_to_id"],
            id_to_token={int(k): v for k, v in data["id_to_token"].items()}
        )


@dataclass
class OrganismMapperJSON:
    """Organism mapper loaded from JSON (portable, no pickle dependency)."""
    organism_to_idx: Dict[str, int]
    vocab_size: int

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "OrganismMapperJSON":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            organism_to_idx=data["organism_to_idx"],
            vocab_size=data["vocab_size"]
        )

    def map_organism(self, organism: str) -> int:
        """Map organism name to index. Unknown organisms map to 0."""
        if organism is None:
            return 0
        return self.organism_to_idx.get(organism, 0)


class GOGPTPredictor:
    """
    High-level predictor for GO-GPT model.

    Handles model loading, preprocessing, and inference in a simple API.

    Example:
        predictor = GOGPTPredictor(
            checkpoint_path="artifacts/checkpoints/base/model.ckpt",
            artifacts_dir="artifacts/preprocessed/base",
            config_path="configs/experiment/default.yaml"
        )
        results = predictor.predict(sequence="MKTAYIAK...", organism="Homo sapiens")
        print(results)  # {"MF": ["GO:0003674", ...], "BP": [...], "CC": [...]}

    Example (HuggingFace Hub):
        predictor = GOGPTPredictor.from_pretrained("armansa1/gogpt-dev")
        results = predictor.predict(sequence="MKTAYIAK...", organism="Homo sapiens")
    """

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "armansa1/gogpt-dev",
        revision: str = "main",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> "GOGPTPredictor":
        """
        Load GO-GPT from HuggingFace Hub.

        Downloads model weights and artifacts, then initializes predictor.

        Args:
            model_id: HuggingFace repo ID (e.g., "armansa1/gogpt-dev")
            revision: Git revision for versioning (e.g., "main", "v1.0")
            cache_dir: Local directory to cache downloaded files
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            verbose: Whether to print loading progress

        Returns:
            Initialized GOGPTPredictor ready for inference

        Example:
            predictor = GOGPTPredictor.from_pretrained("armansa1/gogpt-dev")
            predictions = predictor.predict(sequence="MKTAYIAK...", organism="Homo sapiens")
        """
        from huggingface_hub import hf_hub_download

        if verbose:
            print(f"Downloading GO-GPT from {model_id} (revision: {revision})...")

        # Download all required files
        required_files = [
            "model.ckpt",
            "config.yaml",
            "tokenizer_info.json",
            "go_tokenizer.json",
            "organism_mapper.json",
        ]

        paths = {}
        for fname in required_files:
            paths[fname] = hf_hub_download(
                repo_id=model_id,
                filename=fname,
                revision=revision,
                cache_dir=cache_dir,
            )

        # Create instance using JSON loading (bypasses __init__)
        instance = object.__new__(cls)
        instance.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        instance.verbose = verbose
        instance.config_path = paths["config.yaml"]

        if verbose:
            print(f"Using device: {instance.device}")

        # Load artifacts from JSON
        with open(paths["tokenizer_info.json"], "r") as f:
            instance.tokenizer_info = json.load(f)

        instance.go_tokenizer = GOTokenizerJSON.from_json(paths["go_tokenizer.json"])
        instance.organism_mapper = OrganismMapperJSON.from_json(paths["organism_mapper.json"])

        # Load ESM2 protein tokenizer
        with open(paths["config.yaml"], "r") as f:
            yaml_config = yaml.safe_load(f)
        embed_model_path = yaml_config.get("model", {}).get(
            "embed_model_path", "facebook/esm2_t33_650M_UR50D"
        )
        instance.protein_tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
        instance._embed_model_path = embed_model_path

        # Load model
        instance._load_model(paths["model.ckpt"])

        return instance

    def __init__(
        self,
        checkpoint_path: str,
        artifacts_dir: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the predictor.

        Args:
            checkpoint_path: Path to model checkpoint (.ckpt file)
            artifacts_dir: Path to preprocessing artifacts directory
            config_path: Path to experiment config YAML (optional, uses defaults if not provided)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            verbose: Whether to print loading progress
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.verbose = verbose
        self.config_path = config_path

        if verbose:
            print(f"Using device: {self.device}")

        # Load artifacts and model
        self._load_artifacts(artifacts_dir)
        self._load_model(checkpoint_path)

    def _load_artifacts(self, artifacts_dir: str):
        """Load preprocessing artifacts."""
        artifacts_dir = Path(artifacts_dir)

        # Load tokenizer info
        with open(artifacts_dir / "tokenizer_info.json", "r") as f:
            self.tokenizer_info = json.load(f)

        # Load GO term tokenizer
        with open(artifacts_dir / "go_tokenizer.pkl", "rb") as f:
            self.go_tokenizer = pickle.load(f)

        # Load organism mapper
        with open(artifacts_dir / "organism_mapper.pkl", "rb") as f:
            self.organism_mapper = pickle.load(f)

        # Get embed model path from config if provided
        embed_model_path = "facebook/esm2_t33_650M_UR50D"  # default
        if self.config_path:
            with open(self.config_path, "r") as f:
                yaml_config = yaml.safe_load(f)
            embed_model_path = yaml_config.get("model", {}).get("embed_model_path", embed_model_path)

        # Load protein tokenizer
        self.protein_tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
        self._embed_model_path = embed_model_path

    def _load_model(self, checkpoint_path: str):
        """Load the GO-GPT model from checkpoint."""
        if self.verbose:
            print("Loading GO-GPT model...", end=" ", flush=True)

        # Load model config from YAML if provided, otherwise use defaults
        if self.config_path:
            with open(self.config_path, "r") as f:
                yaml_config = yaml.safe_load(f)
            model_cfg = yaml_config.get("model", {})
        else:
            model_cfg = {}

        # Model configuration (YAML overrides defaults)
        config = GOGPTConfig(
            n_layer=model_cfg.get("n_layer", 12),
            n_head=model_cfg.get("n_head", 12),
            n_embd=model_cfg.get("n_embd", 1080),
            block_size=model_cfg.get("block_size", 2048),
            dropout=0.0,  # Always 0 for inference
            bias=model_cfg.get("bias", True),
            vocab_size=self.tokenizer_info["vocab_size"],
            organism_vocab_size=self.tokenizer_info["organism_vocab_size"],
            pad_token_id=self.tokenizer_info["pad_token_id"],
            mf_start_token_id=self.tokenizer_info["mf_start_token_id"],
            mf_end_token_id=self.tokenizer_info["mf_end_token_id"],
            bp_start_token_id=self.tokenizer_info["bp_start_token_id"],
            bp_end_token_id=self.tokenizer_info["bp_end_token_id"],
            cc_start_token_id=self.tokenizer_info["cc_start_token_id"],
            cc_end_token_id=self.tokenizer_info["cc_end_token_id"],
            protein_embedding_dim=model_cfg.get("protein_embedding_dim", 1280),
            embed_model_path=self._embed_model_path,
            protein_layer_index=model_cfg.get("protein_layer_index", 30),
            use_gated_attention=model_cfg.get("use_gated_attention", True),
            freeze_esm=True,
            esm_num_unfrozen_layers=0,
        )

        # Initialize model
        self.model = GOGPT(config)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()

        if self.verbose:
            total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"done ({total_params:.1f}M parameters)")

    def _preprocess(self, sequence: str, organism: str) -> Dict[str, torch.Tensor]:
        """Preprocess a protein sequence for model input."""
        protein_encoded = self.protein_tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024,
        )

        organism_id = self.organism_mapper.map_organism(organism)

        return {
            "protein_tokens": protein_encoded["input_ids"].to(self.device),
            "protein_mask": protein_encoded["attention_mask"].to(self.device),
            "organism_id": torch.tensor([organism_id], device=self.device),
        }

    def _decode_tokens(self, tokens: torch.Tensor, aspect: str) -> List[str]:
        """Decode generated tokens to GO term IDs for a specific aspect."""
        token_list = tokens[0].tolist()

        # Get aspect-specific token IDs
        if aspect == "MF":
            start_id = self.tokenizer_info["mf_start_token_id"]
            end_id = self.tokenizer_info["mf_end_token_id"]
        elif aspect == "BP":
            start_id = self.tokenizer_info["bp_start_token_id"]
            end_id = self.tokenizer_info["bp_end_token_id"]
        else:  # CC
            start_id = self.tokenizer_info["cc_start_token_id"]
            end_id = self.tokenizer_info["cc_end_token_id"]

        # Find start and end positions
        start_pos = token_list.index(start_id) if start_id in token_list else -1
        end_pos = token_list.index(end_id) if end_id in token_list else len(token_list)

        # Extract GO term tokens
        if start_pos != -1:
            go_token_ids = token_list[start_pos + 1:end_pos]
        else:
            go_token_ids = []

        # Skip special tokens
        special_tokens = {
            self.tokenizer_info["pad_token_id"],
            self.tokenizer_info["mf_start_token_id"],
            self.tokenizer_info["mf_end_token_id"],
            self.tokenizer_info["bp_start_token_id"],
            self.tokenizer_info["bp_end_token_id"],
            self.tokenizer_info["cc_start_token_id"],
            self.tokenizer_info["cc_end_token_id"],
        }

        # Convert to GO terms
        go_terms = []
        seen = set()
        for token_id in go_token_ids:
            if token_id not in special_tokens:
                go_term = self.go_tokenizer.id_to_token.get(token_id)
                if go_term and go_term.startswith("GO:") and go_term not in seen:
                    go_terms.append(go_term)
                    seen.add(go_term)

        return go_terms

    @torch.no_grad()
    def predict(
        self,
        sequence: str,
        organism: str = "Unknown",
        max_new_tokens: int = 300,
        beam_size: int = 5,
        length_penalty: float = 0.3,
    ) -> Dict[str, List[str]]:
        """
        Predict GO terms for a protein sequence.

        Args:
            sequence: Amino acid sequence string
            organism: Organism name (e.g., "Homo sapiens")
            max_new_tokens: Maximum tokens to generate per aspect
            beam_size: Beam size for beam search
            length_penalty: Length penalty for beam search

        Returns:
            Dictionary with GO terms for each aspect:
            {"MF": ["GO:0003674", ...], "BP": [...], "CC": [...]}
        """
        inputs = self._preprocess(sequence, organism)
        results = {}

        for aspect in ["MF", "BP", "CC"]:
            # Get start token for this aspect
            if aspect == "MF":
                start_token_id = self.tokenizer_info["mf_start_token_id"]
            elif aspect == "BP":
                start_token_id = self.tokenizer_info["bp_start_token_id"]
            else:
                start_token_id = self.tokenizer_info["cc_start_token_id"]

            go_tokens = torch.tensor([[start_token_id]], device=self.device)

            generated = self.model.generate_beam_search(
                protein_tokens=inputs["protein_tokens"],
                protein_mask=inputs["protein_mask"],
                go_tokens=go_tokens,
                max_new_tokens=max_new_tokens,
                beam_size=beam_size,
                length_penalty=length_penalty,
                organism_id=inputs["organism_id"],
            )

            results[aspect] = self._decode_tokens(generated, aspect)

        return results
