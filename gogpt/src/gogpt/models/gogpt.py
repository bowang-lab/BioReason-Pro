import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ProteinProjection(nn.Module):
    """Projects protein embeddings to model dimension with 2-layer MLP."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = (config.protein_embedding_dim + config.n_embd) // 2

        self.projection = nn.Sequential(
            nn.Linear(config.protein_embedding_dim, hidden_dim, bias=config.bias),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout)
        )

    def forward(self, protein_embeddings):
        return self.projection(protein_embeddings)


class PrefixCausalAttention(nn.Module):
    """Attention that allows both protein residues and GO terms to participate in attention."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.go_qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.protein_qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        self.go_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.protein_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.use_gated_attention = getattr(config, 'use_gated_attention', True)
        if self.use_gated_attention:
            self.go_gate_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.protein_gate_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention")

    def forward(self, go_states, protein_states, protein_mask=None, go_mask=None):
        B, T_go, C = go_states.size()
        _, T_protein, _ = protein_states.size()

        go_q, go_k, go_v = self.go_qkv(go_states).split(self.n_embd, dim=2)
        go_q = go_q.view(B, T_go, self.n_head, C // self.n_head).transpose(1, 2)
        go_k = go_k.view(B, T_go, self.n_head, C // self.n_head).transpose(1, 2)
        go_v = go_v.view(B, T_go, self.n_head, C // self.n_head).transpose(1, 2)

        if go_mask is not None:
            go_valid = go_mask.view(B, 1, T_go, 1).to(go_q.dtype)
            go_q = go_q * go_valid
            go_k = go_k * go_valid
            go_v = go_v * go_valid

        protein_q, protein_k, protein_v = self.protein_qkv(protein_states).split(self.n_embd, dim=2)
        protein_q = protein_q.view(B, T_protein, self.n_head, C // self.n_head).transpose(1, 2)
        protein_k = protein_k.view(B, T_protein, self.n_head, C // self.n_head).transpose(1, 2)
        protein_v = protein_v.view(B, T_protein, self.n_head, C // self.n_head).transpose(1, 2)

        k = torch.cat([protein_k, go_k], dim=2)
        v = torch.cat([protein_v, go_v], dim=2)

        if self.flash:
            protein_attention_mask = torch.ones((B, 1, 1, T_protein + T_go), dtype=torch.bool, device=protein_q.device)
            protein_attention_mask[:, :, :, T_protein:] = False

            if protein_mask is not None:
                protein_attention_mask[:, :, :, :T_protein] = protein_mask.unsqueeze(1).unsqueeze(2)

            protein_output = torch.nn.functional.scaled_dot_product_attention(
                protein_q, k, v,
                attn_mask=protein_attention_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False
            )

            go_attention_mask = torch.ones((B, 1, T_go, T_protein + T_go), dtype=torch.bool, device=go_q.device)

            if protein_mask is not None:
                go_attention_mask[:, :, :, :T_protein] = protein_mask.unsqueeze(1).unsqueeze(1)

            causal_mask = torch.tril(torch.ones(T_go, T_go, device=go_q.device))
            causal_mask = causal_mask.view(1, 1, T_go, T_go)

            go_attention_mask[:, :, :, T_protein:] = causal_mask

            if go_mask is not None:
                go_attention_mask[:, :, :, T_protein:] &= go_mask.unsqueeze(1).unsqueeze(1)

            go_output = torch.nn.functional.scaled_dot_product_attention(
                go_q, k, v,
                attn_mask=go_attention_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False
            )

            if go_mask is not None:
                go_output = go_output * go_mask.unsqueeze(1).unsqueeze(-1).to(go_output.dtype)

        protein_output = protein_output.transpose(1, 2).contiguous().view(B, T_protein, C)
        go_output = go_output.transpose(1, 2).contiguous().view(B, T_go, C)

        if self.use_gated_attention:
            go_gate = torch.sigmoid(self.go_gate_proj(go_states))
            go_output = go_output * go_gate

            protein_gate = torch.sigmoid(self.protein_gate_proj(protein_states))
            protein_output = protein_output * protein_gate

        if go_mask is not None:
            go_output = go_output * go_mask.unsqueeze(-1).to(go_output.dtype)

        protein_output = self.resid_dropout(self.protein_proj(protein_output))
        go_output = self.resid_dropout(self.go_proj(go_output))

        return protein_output, go_output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.fc(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1_protein = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_1_go = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = PrefixCausalAttention(config)
        self.ln_2_protein = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2_go = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp_protein = MLP(config)
        self.mlp_go = MLP(config)

    def forward(self, go_states, protein_states, protein_mask=None, go_mask=None):
        protein_attn, go_attn = self.attn(
            self.ln_1_go(go_states),
            self.ln_1_protein(protein_states),
            protein_mask, go_mask
        )

        protein_states = protein_states + protein_attn
        go_states = go_states + go_attn

        protein_states = protein_states + self.mlp_protein(self.ln_2_protein(protein_states))
        go_states = go_states + self.mlp_go(self.ln_2_go(go_states))

        return go_states, protein_states


class GOGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.protein_projection = ProteinProjection(config)

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f_protein=LayerNorm(config.n_embd, bias=config.bias),
            ln_f_go=LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.organism_embedding = nn.Embedding(
            config.organism_vocab_size,
            config.n_embd
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        self.embed_model_type = "esm2"
        self.esm = AutoModel.from_pretrained(config.embed_model_path)
        print(f"Loaded ESM2 model from {config.embed_model_path}")

        for param in self.esm.parameters():
            param.requires_grad = False

        if not config.freeze_esm and config.esm_num_unfrozen_layers > 0:
            self._unfreeze_esm_layers(config.esm_num_unfrozen_layers, config.protein_layer_index)
            self.esm.train()
        else:
            self.esm.eval()

        trainable_esm = sum(p.numel() for p in self.esm.parameters() if p.requires_grad)
        total_esm = sum(p.numel() for p in self.esm.parameters())
        print(f"ESM parameters: {trainable_esm/1e6:.2f}M trainable / {total_esm/1e6:.2f}M total")
        print(f"ESM training mode: {self.esm.training}")

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def train(self, mode=True):
        """Override train to handle ESM mode correctly."""
        super().train(mode)
        if self.config.freeze_esm:
            self.esm.eval()
        return self

    def eval(self):
        """Override eval to handle ESM mode correctly."""
        super().eval()
        self.esm.eval()
        return self

    def _unfreeze_esm_layers(self, num_layers, protein_layer_index=-1):
        """Unfreeze the top N layers of ESM2 encoder for finetuning."""
        for param in self.esm.parameters():
            param.requires_grad = False

        total_layers = len(self.esm.encoder.layer)
        extract_layer = protein_layer_index if protein_layer_index >= 0 else total_layers - 1

        if num_layers > extract_layer + 1:
            raise ValueError(
                f"Cannot unfreeze {num_layers} layers when only extracting from layer {extract_layer}. "
                f"Maximum unfrozen layers: {extract_layer + 1}"
            )

        start_layer = extract_layer - num_layers + 1
        layers_to_unfreeze = list(range(start_layer, extract_layer + 1))

        print(f"Unfreezing ESM2 layers: {layers_to_unfreeze} (extracting from layer {extract_layer})")

        for layer_idx in layers_to_unfreeze:
            for param in self.esm.encoder.layer[layer_idx].parameters():
                param.requires_grad = True

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, protein_tokens, go_tokens, targets=None,
                protein_mask=None, go_mask=None,
                organism_id=None,
                protein_embeddings=None):
        device = go_tokens.device
        b, t = go_tokens.size()

        if protein_embeddings is None:
            with torch.set_grad_enabled(self.training and not self.config.freeze_esm):
                use_hidden_states = hasattr(self.config, 'protein_layer_index') and self.config.protein_layer_index != -1

                protein_outputs = self.esm(
                    input_ids=protein_tokens,
                    attention_mask=protein_mask,
                    output_hidden_states=use_hidden_states
                )

                if use_hidden_states:
                    hidden_states = protein_outputs.hidden_states
                    layer_idx = self.config.protein_layer_index
                    if layer_idx < 0:
                        layer_idx = len(hidden_states) + layer_idx
                    if not hasattr(self, '_layer_logged'):
                        print(f"Using ESM2 layer {layer_idx} out of {len(hidden_states)} total layers")
                        self._layer_logged = True
                    protein_embeddings = hidden_states[layer_idx]
                else:
                    protein_embeddings = protein_outputs.last_hidden_state

                if protein_mask is not None:
                    mask = protein_mask.to(protein_embeddings.dtype).unsqueeze(-1)
                    protein_embeddings = protein_embeddings * mask

        protein_states = self.protein_projection(protein_embeddings)

        protein_length = protein_states.size(1)
        go_length = go_tokens.size(1)
        total_length = protein_length + go_length
        if total_length > self.config.block_size:
            raise ValueError(
                f"Total sequence length ({total_length} = {protein_length} protein + {go_length} GO) "
                f"exceeds block_size ({self.config.block_size}). Consider increasing block_size in config."
            )

        go_states = self.transformer.wte(go_tokens)

        positions = torch.arange(0, go_length, device=device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(b, -1)
        go_states = go_states + self.transformer.wpe(positions)

        if organism_id is not None:
            org_embeddings_go = self.organism_embedding(organism_id).unsqueeze(1).expand(-1, go_length, -1)
            go_states = go_states + org_embeddings_go

        go_states = self.transformer.drop(go_states)

        for block in self.transformer.h:
            go_states, protein_states = block(
                go_states, protein_states,
                protein_mask, go_mask
            )

        go_states = self.transformer.ln_f_go(go_states)

        if targets is not None:
            logits = self.lm_head(go_states)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                targets.view(-1),
                                ignore_index=0)
        else:
            logits = self.lm_head(go_states[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, protein_tokens, protein_mask, go_tokens, max_new_tokens,
                temperature=1.0, top_k=None, organism_id=None, protein_embeddings=None):
        """Generate GO terms for a batch of proteins."""
        if protein_embeddings is None:
            with torch.no_grad():
                use_hidden_states = hasattr(self.config, 'protein_layer_index') and self.config.protein_layer_index != -1

                protein_outputs = self.esm(
                    input_ids=protein_tokens,
                    attention_mask=protein_mask,
                    output_hidden_states=use_hidden_states
                )

                if use_hidden_states:
                    hidden_states = protein_outputs.hidden_states
                    layer_idx = self.config.protein_layer_index
                    if layer_idx < 0:
                        layer_idx = len(hidden_states) + layer_idx
                    if not hasattr(self, '_layer_logged'):
                        print(f"Using ESM2 layer {layer_idx} out of {len(hidden_states)} total layers")
                        self._layer_logged = True
                    protein_embeddings = hidden_states[layer_idx]
                else:
                    protein_embeddings = protein_outputs.last_hidden_state

                if protein_mask is not None:
                    mask = protein_mask.to(protein_embeddings.dtype).unsqueeze(-1)
                    protein_embeddings = protein_embeddings * mask

        finished_sequences = torch.zeros(go_tokens.size(0), dtype=torch.bool, device=go_tokens.device)
        pad_token_id = 0

        go_mask = torch.ones_like(go_tokens, dtype=torch.bool)

        end_token_ids = [
            self.config.mf_end_token_id if hasattr(self.config, 'mf_end_token_id') else 2,
            self.config.bp_end_token_id if hasattr(self.config, 'bp_end_token_id') else 4,
            self.config.cc_end_token_id if hasattr(self.config, 'cc_end_token_id') else 6,
        ]

        for _ in range(max_new_tokens):
            logits, _, = self(
                protein_tokens=protein_tokens,
                protein_mask=protein_mask,
                go_tokens=go_tokens,
                go_mask=go_mask,
                organism_id=organism_id,
                protein_embeddings=protein_embeddings
            )

            logits = logits[:, -1, :]

            for idx in range(logits.size(0)):
                if finished_sequences[idx]:
                    logits[idx, :] = float('-inf')
                    logits[idx, pad_token_id] = 0.0

            logits = torch.where(
                finished_sequences.unsqueeze(1),
                logits,
                logits / temperature
            )

            if top_k is not None:
                topk_mask = ~finished_sequences
                if topk_mask.any():
                    v, _ = torch.topk(logits[topk_mask], min(top_k, logits.size(-1)))
                    logits[topk_mask] = torch.where(
                        logits[topk_mask] < v[:, [-1]],
                        torch.tensor(float('-inf'), device=logits.device),
                        logits[topk_mask]
                    )

            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            go_tokens = torch.cat((go_tokens, next_tokens), dim=1)

            new_token_mask = ~finished_sequences
            go_mask = torch.cat((go_mask, new_token_mask.unsqueeze(1)), dim=1)

            for end_token_id in end_token_ids:
                finished_sequences = finished_sequences | (next_tokens.squeeze(-1) == end_token_id)

            if finished_sequences.all():
                break

        return go_tokens

    @torch.no_grad()
    def generate_beam_search(self, protein_tokens, protein_mask, go_tokens, max_new_tokens,
                             beam_size=5, length_penalty=0.6, organism_id=None, protein_embeddings=None):
        """Batched beam search generation for GO terms."""
        batch_size = go_tokens.size(0)
        device = go_tokens.device
        vocab_size = self.config.vocab_size
        pad_token_id = 0

        if protein_embeddings is None:
            with torch.no_grad():
                use_hidden_states = hasattr(self.config, 'protein_layer_index') and self.config.protein_layer_index != -1
                protein_outputs = self.esm(
                    input_ids=protein_tokens,
                    attention_mask=protein_mask,
                    output_hidden_states=use_hidden_states
                )
                if use_hidden_states:
                    hidden_states = protein_outputs.hidden_states
                    layer_idx = self.config.protein_layer_index
                    if layer_idx < 0:
                        layer_idx = len(hidden_states) + layer_idx
                    protein_embeddings = hidden_states[layer_idx]
                else:
                    protein_embeddings = protein_outputs.last_hidden_state
                if protein_mask is not None:
                    mask = protein_mask.to(protein_embeddings.dtype).unsqueeze(-1)
                    protein_embeddings = protein_embeddings * mask

        end_token_ids = [
            self.config.mf_end_token_id if hasattr(self.config, 'mf_end_token_id') else 2,
            self.config.bp_end_token_id if hasattr(self.config, 'bp_end_token_id') else 4,
            self.config.cc_end_token_id if hasattr(self.config, 'cc_end_token_id') else 6,
        ]
        end_token_set = set(end_token_ids)

        protein_tokens_expanded = protein_tokens.repeat_interleave(beam_size, dim=0)
        protein_mask_expanded = protein_mask.repeat_interleave(beam_size, dim=0)
        protein_embeddings_expanded = protein_embeddings.repeat_interleave(beam_size, dim=0)
        organism_id_expanded = organism_id.repeat_interleave(beam_size, dim=0) if organism_id is not None else None

        beam_sequences = go_tokens.repeat_interleave(beam_size, dim=0)
        go_mask = torch.ones_like(beam_sequences, dtype=torch.bool)

        beam_scores = torch.zeros(batch_size * beam_size, device=device)
        beam_scores[1::beam_size] = float('-inf')

        finished_beams = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=device)

        best_completed = [None] * batch_size
        best_completed_scores = [float('-inf')] * batch_size

        for step in range(max_new_tokens):
            logits, _ = self(
                protein_tokens=protein_tokens_expanded,
                protein_mask=protein_mask_expanded,
                go_tokens=beam_sequences,
                go_mask=go_mask,
                organism_id=organism_id_expanded,
                protein_embeddings=protein_embeddings_expanded
            )

            next_token_logits = logits[:, -1, :]

            log_probs = F.log_softmax(next_token_logits, dim=-1)

            log_probs[finished_beams] = float('-inf')
            log_probs[finished_beams, pad_token_id] = 0.0

            next_scores = beam_scores.unsqueeze(1) + log_probs

            next_scores = next_scores.view(batch_size, beam_size * vocab_size)

            top_scores, top_indices = torch.topk(next_scores, beam_size, dim=1)

            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * beam_size
            global_beam_indices = (batch_offsets + beam_indices).view(-1)

            new_sequences = beam_sequences[global_beam_indices]
            new_go_mask = go_mask[global_beam_indices]

            new_tokens = token_indices.view(-1, 1)
            beam_sequences = torch.cat([new_sequences, new_tokens], dim=1)

            beam_scores = top_scores.view(-1)

            new_finished = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=device)
            for end_token_id in end_token_ids:
                new_finished = new_finished | (new_tokens.squeeze(-1) == end_token_id)

            old_finished = finished_beams[global_beam_indices]
            finished_beams = old_finished | new_finished

            new_token_mask = ~finished_beams
            go_mask = torch.cat([new_go_mask, new_token_mask.unsqueeze(1)], dim=1)

            current_length = beam_sequences.size(1)
            for b in range(batch_size):
                for k in range(beam_size):
                    idx = b * beam_size + k
                    if new_finished[idx] and not old_finished[idx]:
                        seq_length = current_length - 1
                        normalized_score = beam_scores[idx].item() / (seq_length ** length_penalty)
                        if normalized_score > best_completed_scores[b]:
                            best_completed_scores[b] = normalized_score
                            best_completed[b] = beam_sequences[idx].clone()

            all_have_completed = all(bc is not None for bc in best_completed)
            if all_have_completed:
                can_improve = False
                for b in range(batch_size):
                    example_mask = ~finished_beams[b*beam_size:(b+1)*beam_size]
                    if example_mask.any():
                        best_active = beam_scores[b*beam_size:(b+1)*beam_size][example_mask].max().item()
                        max_future_length = current_length + (max_new_tokens - step - 1)
                        optimistic_normalized = best_active / (max_future_length ** length_penalty)
                        if optimistic_normalized > best_completed_scores[b]:
                            can_improve = True
                            break
                if not can_improve:
                    break

            if finished_beams.all():
                break

        results = []
        for b in range(batch_size):
            if best_completed[b] is not None:
                results.append(best_completed[b])
            else:
                example_scores = beam_scores[b*beam_size:(b+1)*beam_size]
                best_idx = b * beam_size + example_scores.argmax().item()
                results.append(beam_sequences[best_idx])

        max_len = max(seq.size(0) for seq in results)
        padded_results = []
        for seq in results:
            if seq.size(0) < max_len:
                padding = torch.full((max_len - seq.size(0),), pad_token_id, device=device, dtype=seq.dtype)
                seq = torch.cat([seq, padding])
            padded_results.append(seq)

        return torch.stack(padded_results, dim=0)
