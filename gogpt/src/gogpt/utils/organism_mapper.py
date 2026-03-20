from collections import Counter

class OrganismMapper:
    def __init__(self, organisms, top_n_organisms=None):
        """
        Initialize organism mapper with optional top-N filtering.

        Args:
            organisms: List of organism names from the dataset
            top_n_organisms: If provided, only create embeddings for top N most frequent organisms.
                           All other organisms will be mapped to <UNKNOWN> token.
        """
        valid_organisms = [org for org in organisms if org is not None]

        if top_n_organisms is not None:
            organism_counts = Counter(valid_organisms)
            top_organisms = [org for org, count in organism_counts.most_common(top_n_organisms)]

            self.organism_to_idx = {"<UNKNOWN>": 0}
            for idx, organism in enumerate(sorted(top_organisms), 1):
                self.organism_to_idx[organism] = idx

            self.top_n_organisms = top_n_organisms
            self.total_original_organisms = len(set(valid_organisms))
            print(f"OrganismMapper: Using top {top_n_organisms} organisms out of {self.total_original_organisms} total organisms")
        else:
            unique_organisms = sorted(set(valid_organisms))
            self.organism_to_idx = {"<UNKNOWN>": 0}
            for idx, organism in enumerate(unique_organisms, 1):
                self.organism_to_idx[organism] = idx

            self.top_n_organisms = None
            self.total_original_organisms = len(unique_organisms)

        self.idx_to_organism = {idx: organism for organism, idx in self.organism_to_idx.items()}
        self.vocab_size = len(self.organism_to_idx)

    def map_organism(self, organism):
        """Map organism name to index. Unknown organisms are mapped to index 0."""
        if organism is None:
            return 0
        return self.organism_to_idx.get(organism, 0)

    def get_vocab_size(self):
        """Get the size of the organism vocabulary (including <UNKNOWN>)."""
        return self.vocab_size

    def get_mapping_stats(self):
        """Get statistics about the organism mapping."""
        return {
            'vocab_size': self.vocab_size,
            'top_n_organisms': self.top_n_organisms,
            'total_original_organisms': self.total_original_organisms,
            'mapped_organisms': list(self.organism_to_idx.keys())
        }
