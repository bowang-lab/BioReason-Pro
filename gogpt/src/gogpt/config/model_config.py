from dataclasses import dataclass

@dataclass
class GOGPTConfig:
    block_size: int = None
    vocab_size: int = None
    n_layer: int = None
    n_head: int = None
    n_embd: int = None
    dropout: float = None
    bias: bool = None
    pad_token_id: int = 0
    mf_start_token_id: int = None
    mf_end_token_id: int = None
    bp_start_token_id: int = None
    bp_end_token_id: int = None
    cc_start_token_id: int = None
    cc_end_token_id: int = None
    protein_embedding_dim: int = None
    embed_model_path: str = None
    organism_vocab_size: int = None
    protein_layer_index: int = -1  # -1 for last layer, 0-N for specific layer (ESM2 only)

    # Gated attention
    use_gated_attention: bool = True

    # ESM finetuning parameters
    freeze_esm: bool = True  # If False, enable ESM finetuning
    esm_num_unfrozen_layers: int = 0  # Number of top ESM layers to finetune
    esm_learning_rate: float = 1e-5  # Learning rate for ESM layers
