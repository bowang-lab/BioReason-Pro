from typing import List, Set

class GOTermTokenizer:
    def __init__(self, go_terms: Set[str]):
        self.special_tokens = ["<pad>", "<|MF_START|>", "<|MF_END|>", "<|BP_START|>", "<|BP_END|>", "<|CC_START|>", "<|CC_END|>"]
        all_tokens = self.special_tokens + sorted(go_terms)

        self.token_to_id = {token: idx for idx, token in enumerate(all_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        self.pad_token_id = self.token_to_id["<pad>"]
        self.mf_start_token_id = self.token_to_id["<|MF_START|>"]
        self.mf_end_token_id = self.token_to_id["<|MF_END|>"]
        self.bp_start_token_id = self.token_to_id["<|BP_START|>"]
        self.bp_end_token_id = self.token_to_id["<|BP_END|>"]
        self.cc_start_token_id = self.token_to_id["<|CC_START|>"]
        self.cc_end_token_id = self.token_to_id["<|CC_END|>"]

        self.aspect_to_tokens = {
            "MF": (self.mf_start_token_id, self.mf_end_token_id),
            "BP": (self.bp_start_token_id, self.bp_end_token_id),
            "CC": (self.cc_start_token_id, self.cc_end_token_id),
        }

        print(f"Vocabulary size: {len(self.token_to_id)}")

    def encode(self, go_terms_list: List[str], aspect="MF") -> List[int]:
        """Convert list of GO terms to token IDs with aspect tokens."""
        aspect_start_id, aspect_end_id = self.aspect_to_tokens.get(
            aspect, (self.mf_start_token_id, self.mf_end_token_id)
        )

        ids = [aspect_start_id]
        ids.extend(
            self.token_to_id.get(term, self.pad_token_id) for term in go_terms_list
        )
        ids.append(aspect_end_id)

        return ids

    def decode(self, ids: List[int]) -> List[str]:
        """Convert token IDs back to GO terms."""
        tokens = [self.id_to_token[idx] for idx in ids]
        return tokens
