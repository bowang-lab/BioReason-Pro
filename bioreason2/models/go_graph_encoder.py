"""
GO Graph Encoder for BioReason2

This module provides two graph encoder architectures for the Gene Ontology (GO) graph:

1. **GOGraphEncoderUnified** (NEW - Aspect-Agnostic):
   - Processes ALL GO terms in a single unified graph
   - No namespace separation or specialized embeddings
   - More parameter-efficient and generalization-friendly
   - Architecture: All GO terms → GAT → CrossAttn → 200 vectors
   - Maintains API compatibility with namespace parameters

2. **GOGraphEncoder** (Original - Namespace-Aware):
   - Uses shared GAT layers with learnable namespace embeddings
   - Processes namespaces separately then combines
   - Architecture: GAT(BP/MF/CC) → CrossAttn → 200 vectors per namespace

Both encoders:
- Load and process the GO ontology
- Use pre-computed embeddings for GO terms (2560 dimensions) 
- Encode GO terms using Graph Attention Networks
- Use cross-attention to reduce embeddings to 200 vectors
- Keep final embeddings at 2560 dimensions
- Support the same public API for compatibility

Use create_go_graph_encoder_pipeline(unified_go_encoder=True/False) to choose architecture.
"""

import os
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from torch_geometric.nn import GATConv
import networkx as nx
import obonet


BIOLOGICAL_PROCESS_NAMESPACE = "biological_process"  # Biological Process
MOLECULAR_FUNCTION_NAMESPACE = "molecular_function"  # Molecular Function
CELLULAR_COMPONENT_NAMESPACE = "cellular_component"  # Cellular Component

GO_BASIC_OBO_NAMESPACE_MAPPINGS = {
    "BP": "biological_process",
    "MF": "molecular_function",
    "CC": "cellular_component",
}


class CrossAttentionReducer(nn.Module):
    """
    Cross-attention module to reduce the number of GO embeddings to a fixed size.
    """

    def __init__(
        self,
        input_dim: int = 2560,
        num_queries: int = 200,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_queries = num_queries
        self.num_heads = num_heads

        # Learnable query embeddings
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, input_dim))

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.query_embeddings)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, go_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Reduce GO embeddings using cross-attention.

        Args:
            go_embeddings: Tensor of shape (batch_size, num_go_terms, input_dim)

        Returns:
            Tensor of shape (batch_size, num_queries, input_dim)
        """
        batch_size = go_embeddings.size(0)

        # Expand query embeddings to batch size
        queries = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: queries attend to GO embeddings
        attended_queries, _ = self.cross_attention(
            query=queries, key=go_embeddings, value=go_embeddings
        )

        # Residual connection and layer norm
        attended_queries = self.layer_norm1(queries + attended_queries)

        # Feed-forward network
        ffn_output = self.ffn(attended_queries)

        # Residual connection and layer norm
        output = self.layer_norm2(attended_queries + ffn_output)

        return output


class GOGraphEncoderUnified(nn.Module):
    """
    Unified GO Graph Encoder that processes all GO terms together.
    
    This encoder:
    1. Processes ALL GO terms in a single unified graph (no namespace separation)
    2. Uses GAT layers to process the complete GO ontology
    3. Uses cross-attention to reduce to a fixed number of embeddings
    4. Maintains compatibility by supporting namespace-specific queries through masking
    5. More parameter-efficient without namespace embeddings
    
    Architecture:
    - All GO terms → GAT → CrossAttn → 200 unified embeddings
    - Namespace requests handled via term indexing/masking
    """

    def __init__(
        self,
        go_obo_path: str,
        precomputed_embeddings_path: str,  # Directory with individual GO_*.safetensors files
        hidden_dim: int = 512,
        num_gat_layers: int = 3,
        num_heads: int = 8,
        num_reduced_embeddings: int = 200,
        embedding_dim: int = 2560,
        dropout: float = 0.1,
        embeddings_load_to: str = "cpu",
    ):
        super().__init__()

        self.go_obo_path = go_obo_path
        self.precomputed_embeddings_path = precomputed_embeddings_path
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads
        self.num_reduced_embeddings = num_reduced_embeddings
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.embeddings_load_to = embeddings_load_to

        # Load GO ontology
        self.go_graph = self._load_go_ontology(go_obo_path)

        # Store all terms with their namespace information (for compatibility)
        self.all_go_terms, self.term_to_namespace = self._get_all_terms_with_namespaces()

        # Create unified GAT layers
        self.gat_layers = nn.ModuleList()
        # First layer: embedding_dim -> hidden_dim
        self.gat_layers.append(
            GATConv(
                in_channels=self.embedding_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
            )
        )
        # Subsequent layers: hidden_dim -> hidden_dim
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                )
            )

        # Output projection (2-layer MLP with GeLU)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embedding_dim),
        )

        # Cross-attention reducer
        self.cross_attention_reducer = CrossAttentionReducer(
            input_dim=self.embedding_dim,
            num_queries=num_reduced_embeddings,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Load pre-computed embeddings (required)
        self.precomputed_embeddings = self._load_precomputed_embeddings(
            precomputed_embeddings_path,
            embeddings_load_to=self.embeddings_load_to,
        )

        # Initialize weights
        self._initialize_weights()

        print("Initialized Unified GO Graph Encoder")
        print(f"Total GO terms: {len(self.all_go_terms)}")
        for namespace in [BIOLOGICAL_PROCESS_NAMESPACE, MOLECULAR_FUNCTION_NAMESPACE, CELLULAR_COMPONENT_NAMESPACE]:
            count = sum(1 for term in self.all_go_terms if self.term_to_namespace[term] == namespace)
            print(f"  {namespace}: {count} terms")
        print(f"GAT layers: {len(self.gat_layers)}")

    def _get_device(self) -> torch.device:
        """Get the device of the model parameters, with fallback to CUDA/CPU."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            # No parameters yet, use CUDA if available
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_go_ontology(self, go_obo_path: str) -> nx.DiGraph:
        """Load GO ontology from OBO file."""
        if not os.path.exists(go_obo_path):
            raise FileNotFoundError(f"GO OBO file not found: {go_obo_path}")

        print(f"Loading GO ontology from {go_obo_path}")
        go_graph = obonet.read_obo(go_obo_path)
        print(
            f"Loaded {len(go_graph.nodes())} GO terms and {len(go_graph.edges())} relationships"
        )

        return go_graph

    def _load_precomputed_embeddings(
        self,
        embeddings_path: str,
        embeddings_load_to: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Load pre-computed embeddings for GO terms from a directory of .safetensors files.

        Args:
            embeddings_path: Directory containing files like GO_0006647.safetensors
            embeddings_load_to: "cpu" or "cuda" (no "auto")

        Returns:
            Dict mapping "GO:0006647" -> torch.Tensor(dtype=bfloat16) stored on the chosen device.
            If stored on CPU and CUDA is available, tensors are pinned for faster later H2D copies.
        """
        if not os.path.isdir(embeddings_path):
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_path}")

        if embeddings_load_to not in ("cpu", "cuda"):
            raise ValueError(f"embeddings_load_to must be 'cpu' or 'cuda', got: {embeddings_load_to}")
        if embeddings_load_to == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("embeddings_load_to='cuda' but CUDA is not available.")

        safetensor_files = [f for f in os.listdir(embeddings_path) if f.endswith(".safetensors")]
        if not safetensor_files:
            raise ValueError(f"No .safetensors files found in directory: {embeddings_path}")
        print(f"Found {len(safetensor_files)} GO embeddings files")

        target = embeddings_load_to
        embeddings: Dict[str, torch.Tensor] = {}

        for filename in safetensor_files:
            if not filename.startswith("GO_"):
                continue
            go_term_id = filename.replace("GO_", "GO:").replace(".safetensors", "")
            file_path = os.path.join(embeddings_path, filename)

            with safe_open(file_path, framework="torch", device=target) as f:
                keys = list(f.keys())
                if not keys:
                    continue
                t = f.get_tensor(keys[0]).to(torch.bfloat16)
                if target == "cpu" and torch.cuda.is_available():
                    t = t.pin_memory()
                embeddings[go_term_id] = t

        if not embeddings:
            raise ValueError("No embeddings were loaded")

        sample = next(iter(embeddings.values()))
        if sample.shape[0] != 2560:
            raise ValueError(f"Expected embedding dimension 2560, got {sample.shape[0]}")

        print(f"Loaded {len(embeddings)} GO embeddings with dimension {sample.shape[0]}, dtype {sample.dtype}, device {sample.device}")

        return embeddings

    def _get_all_terms_with_namespaces(self) -> Tuple[List[str], Dict[str, str]]:
        """Get all GO terms with their namespace mappings for compatibility."""
        all_terms = []
        term_to_namespace = {}

        for term in self.go_graph.nodes():
            node_data = self.go_graph.nodes[term]
            
            namespace = node_data.get("namespace", None)
            if namespace is None:
                namespace = GO_BASIC_OBO_NAMESPACE_MAPPINGS.get(term, None)

            if namespace is None:
                raise ValueError(f"Namespace not found for GO term: {term}")

            # Only include terms from the three main namespaces
            if namespace in [BIOLOGICAL_PROCESS_NAMESPACE, MOLECULAR_FUNCTION_NAMESPACE, CELLULAR_COMPONENT_NAMESPACE]:
                all_terms.append(term)
                term_to_namespace[term] = namespace

        return all_terms, term_to_namespace

    def _get_unified_edge_index(self) -> torch.Tensor:
        """Create edge index for all GO terms in unified graph."""
        # Create mapping from terms to indices
        term_to_idx = {term: idx for idx, term in enumerate(self.all_go_terms)}

        # Extract edges from the complete graph
        edges = []
        for edge in self.go_graph.edges():
            src, dst = edge[0], edge[1]
            
            # Only include edges where both nodes are in our term list
            if src in term_to_idx and dst in term_to_idx:
                src_idx = term_to_idx[src]
                dst_idx = term_to_idx[dst]
                edges.append([src_idx, dst_idx])

        if len(edges) == 0:
            # If no edges, create self-loops
            edges = [[i, i] for i in range(len(self.all_go_terms))]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Move edge index to the same device as the model
        edge_index = edge_index.to(self._get_device())

        return edge_index

    def _get_precomputed_embeddings(self, go_terms: List[str]) -> torch.Tensor:
        """Get pre-computed embeddings for GO terms."""
        embeddings = []
        missing_terms = []

        for term in go_terms:
            if term in self.precomputed_embeddings:
                embeddings.append(self.precomputed_embeddings[term])
            else:
                missing_terms.append(term)

        # Raise error if any terms are missing
        if missing_terms:
            raise ValueError(f"Missing embeddings for GO terms: {missing_terms}")

        # Stack embeddings directly as torch tensors
        embeddings_tensor = torch.stack(embeddings)

        # Move to the same device as the model
        embeddings_tensor = embeddings_tensor.to(self._get_device(), non_blocking=True)

        return embeddings_tensor

    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_and_map_namespace(self, namespace: str) -> str:
        """Validate and map namespace to long form for compatibility."""
        # Handle both short and long namespace forms
        if namespace == "all":
            return "all"
        elif namespace in [BIOLOGICAL_PROCESS_NAMESPACE, MOLECULAR_FUNCTION_NAMESPACE, CELLULAR_COMPONENT_NAMESPACE]:
            return namespace
        elif namespace in GO_BASIC_OBO_NAMESPACE_MAPPINGS:
            return GO_BASIC_OBO_NAMESPACE_MAPPINGS[namespace]
        else:
            valid_long = [BIOLOGICAL_PROCESS_NAMESPACE, MOLECULAR_FUNCTION_NAMESPACE, CELLULAR_COMPONENT_NAMESPACE]
            valid_short = list(GO_BASIC_OBO_NAMESPACE_MAPPINGS.keys())
            raise ValueError(
                f"Invalid namespace: {namespace}. Must be one of {valid_long} (long), {valid_short} (short), or 'all'"
            )

    def _get_namespace_mask(self, namespace: str) -> torch.Tensor:
        """Get boolean mask for terms belonging to a specific namespace."""
        if namespace == "all":
            return torch.ones(len(self.all_go_terms), dtype=torch.bool, device=self._get_device())
        
        mask = torch.zeros(len(self.all_go_terms), dtype=torch.bool, device=self._get_device())
        for i, term in enumerate(self.all_go_terms):
            if self.term_to_namespace[term] == namespace:
                mask[i] = True
        return mask

    def forward(self, namespace: str) -> torch.Tensor:
        """
        Forward pass through the unified graph encoder.

        Args:
            namespace: One of "biological_process", "molecular_function", "cellular_component", "all"
                      (short forms "BP", "MF", "CC" also accepted)

        Returns:
            reduced_embeddings: Tensor of shape (200, 2560)
        """
        # Get embeddings for all terms
        x = self._get_precomputed_embeddings(self.all_go_terms)
        
        # Get unified edge index
        edge_index = self._get_unified_edge_index()
        
        # Apply GAT layers
        for layer in self.gat_layers:
            x = layer(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final projection
        x = self.output_projection(x)
        
        # If specific namespace requested, we still process all terms but can use
        # the information for potential future namespace-aware pooling
        # For now, we apply cross-attention to all terms
        x_batch = x.unsqueeze(0)  # (1, num_terms, embedding_dim)
        
        # Apply cross-attention reduction
        reduced_embeddings = self.cross_attention_reducer(x_batch)
        reduced_embeddings = reduced_embeddings.squeeze(0)  # (num_reduced_embeddings, embedding_dim)
        
        return reduced_embeddings

    def encode_namespace(self, namespace: str) -> torch.Tensor:
        """
        Encode GO terms from unified graph (maintains compatibility).

        Args:
            namespace: One of "biological_process", "molecular_function", "cellular_component", "all"
                      (short forms "BP", "MF", "CC" also accepted)

        Returns:
            reduced_embeddings: Tensor of shape (200, 2560)
        """
        self.eval()
        with torch.no_grad():
            return self(namespace)

    def get_all_reduced_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Get unified embeddings for all namespaces (for compatibility).
        Note: In unified mode, all namespaces return the same embeddings.

        Returns:
            Dictionary mapping namespace to tensor of shape (200, 2560)
        """
        results = {}
        self.eval()
        with torch.no_grad():
            # In unified mode, we return the same embeddings for all namespaces
            unified_emb = self("all")
            for namespace in [BIOLOGICAL_PROCESS_NAMESPACE, MOLECULAR_FUNCTION_NAMESPACE, CELLULAR_COMPONENT_NAMESPACE]:
                results[namespace] = unified_emb
        return results

    def get_combined_reduced_embeddings(self) -> torch.Tensor:
        """
        Get unified reduced embeddings.

        Returns:
            Tensor of shape (200, 2560) - same as single namespace since we process everything unified
        """
        return self.encode_namespace("all")


class GOGraphEncoder(nn.Module):
    """
    GO Graph Encoder with shared GAT and shared cross-attention.

    This encoder:
    1. Uses shared GAT layers with learnable namespace embeddings for all namespaces
    2. Uses single shared cross-attention reducer for both individual and combined processing
    3. For individual namespaces: GAT(single namespace) → CrossAttn → 200 embeddings
    4. For "all" mode: GAT(BP) + GAT(MF) + GAT(CC) → Concat → CrossAttn → 200 embeddings
    5. Maintains namespace specialization in GAT while unifying cross-attention reduction
    """

    def __init__(
        self,
        go_obo_path: str,
        precomputed_embeddings_path: str,  # Directory with individual GO_*.safetensors files
        hidden_dim: int = 512,
        num_gat_layers: int = 3,
        num_heads: int = 8,
        num_reduced_embeddings: int = 200,
        embedding_dim: int = 2560,
        dropout: float = 0.1,
        embeddings_load_to: str = "cpu",
    ):
        super().__init__()

        self.go_obo_path = go_obo_path
        self.precomputed_embeddings_path = precomputed_embeddings_path
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads
        self.num_reduced_embeddings = num_reduced_embeddings
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.embeddings_load_to = embeddings_load_to

        # Load GO ontology
        self.go_graph = self._load_go_ontology(go_obo_path)

        # Separate terms by namespace
        self.namespace_terms = self._separate_by_namespace()

        # Create separate graphs for each namespace
        self.namespace_graphs = self._create_namespace_graphs()

        # Learnable namespace embeddings to distinguish terms by namespace
        self.namespace_embeddings = nn.Parameter(
            torch.randn(3, self.embedding_dim)  # 3 namespaces
        )
        # Mapping from namespace name to embedding index
        self.namespace_to_idx = {
            BIOLOGICAL_PROCESS_NAMESPACE: 0,  # biological_process -> 0
            MOLECULAR_FUNCTION_NAMESPACE: 1,  # molecular_function -> 1
            CELLULAR_COMPONENT_NAMESPACE: 2,  # cellular_component -> 2
        }

        # Create shared GAT layers for all namespaces
        self.shared_gat_layers = nn.ModuleList()
        # First layer: embedding_dim -> hidden_dim
        self.shared_gat_layers.append(
            GATConv(
                in_channels=self.embedding_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
            )
        )
        # Subsequent layers: hidden_dim -> hidden_dim
        for _ in range(num_gat_layers - 1):
            self.shared_gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                )
            )

        # Create shared output projection (2-layer MLP with GeLU)
        self.shared_output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embedding_dim),
        )

        # Create shared cross-attention reducer for individual namespaces
        self.shared_cross_attention_reducer = CrossAttentionReducer(
            input_dim=self.embedding_dim,
            num_queries=num_reduced_embeddings,
            num_heads=num_heads,
            dropout=dropout,
        )


        # Load pre-computed embeddings (required)
        self.precomputed_embeddings = self._load_precomputed_embeddings(
            precomputed_embeddings_path,
            embeddings_load_to=self.embeddings_load_to,
        )

        # Initialize weights
        self._initialize_weights()

        print(
            "Initialized GO Graph Encoder with shared GAT and unified cross-attention"
        )
        print(f"Namespaces: {list(self.namespace_terms.keys())}")
        for namespace, terms in self.namespace_terms.items():
            print(f"  {namespace}: {len(terms)} terms")
        print(f"Shared GAT layers: {len(self.shared_gat_layers)}")

    def _get_device(self) -> torch.device:
        """Get the device of the model parameters, with fallback to CUDA/CPU."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            # No parameters yet, use CUDA if available
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_go_ontology(self, go_obo_path: str) -> nx.DiGraph:
        """Load GO ontology from OBO file."""
        if not os.path.exists(go_obo_path):
            raise FileNotFoundError(f"GO OBO file not found: {go_obo_path}")

        print(f"Loading GO ontology from {go_obo_path}")
        go_graph = obonet.read_obo(go_obo_path)
        print(
            f"Loaded {len(go_graph.nodes())} GO terms and {len(go_graph.edges())} relationships"
        )

        return go_graph

    def _load_precomputed_embeddings(
        self,
        embeddings_path: str,
        embeddings_load_to: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Load pre-computed embeddings for GO terms from a directory of .safetensors files.

        Args:
            embeddings_path: Directory containing files like GO_0006647.safetensors
            embeddings_load_to: "cpu" or "cuda" (no "auto")

        Returns:
            Dict mapping "GO:0006647" -> torch.Tensor(dtype=bfloat16) stored on the chosen device.
            If stored on CPU and CUDA is available, tensors are pinned for faster later H2D copies.
        """
        if not os.path.isdir(embeddings_path):
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_path}")

        if embeddings_load_to not in ("cpu", "cuda"):
            raise ValueError(f"embeddings_load_to must be 'cpu' or 'cuda', got: {embeddings_load_to}")
        if embeddings_load_to == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("embeddings_load_to='cuda' but CUDA is not available.")

        safetensor_files = [f for f in os.listdir(embeddings_path) if f.endswith(".safetensors")]
        if not safetensor_files:
            raise ValueError(f"No .safetensors files found in directory: {embeddings_path}")
        print(f"Found {len(safetensor_files)} GO embeddings files")

        target = embeddings_load_to
        embeddings: Dict[str, torch.Tensor] = {}

        for filename in safetensor_files:
            if not filename.startswith("GO_"):
                continue
            go_term_id = filename.replace("GO_", "GO:").replace(".safetensors", "")
            file_path = os.path.join(embeddings_path, filename)

            with safe_open(file_path, framework="torch", device=target) as f:
                keys = list(f.keys())
                if not keys:
                    continue
                t = f.get_tensor(keys[0]).to(torch.bfloat16)
                if target == "cpu" and torch.cuda.is_available():
                    t = t.pin_memory()
                embeddings[go_term_id] = t

        if not embeddings:
            raise ValueError("No embeddings were loaded")


        sample = next(iter(embeddings.values()))
        if sample.shape[0] != 2560:
            raise ValueError(f"Expected embedding dimension 2560, got {sample.shape[0]}")

        print(f"Loaded {len(embeddings)} GO embeddings with dimension {sample.shape[0]}, dtype {sample.dtype}, device {sample.device}")

        return embeddings

    def _separate_by_namespace(self) -> Dict[str, List[str]]:
        """Separate GO terms by namespace."""
        namespace_terms = {
            BIOLOGICAL_PROCESS_NAMESPACE: [],
            MOLECULAR_FUNCTION_NAMESPACE: [],
            CELLULAR_COMPONENT_NAMESPACE: [],
        }

        for term in self.go_graph.nodes():
            node_data = self.go_graph.nodes[term]

            namespace = node_data.get("namespace", None)
            if namespace is None:
                namespace = GO_BASIC_OBO_NAMESPACE_MAPPINGS.get(term, None)

            if namespace is None:
                raise ValueError(f"Namespace not found for GO term: {term}")

            # Keep the original namespace from GO OBO (biological_process, etc.)
            if namespace in namespace_terms:
                namespace_terms[namespace].append(term)

        return namespace_terms

    def _create_namespace_graphs(self) -> Dict[str, nx.DiGraph]:
        """Create separate graphs for each namespace."""
        namespace_graphs = {}

        for namespace, terms in self.namespace_terms.items():
            # Create subgraph for this namespace
            subgraph = self.go_graph.subgraph(terms).copy()
            namespace_graphs[namespace] = subgraph
            print(
                f"Created {namespace} graph with {len(subgraph.nodes())} nodes and {len(subgraph.edges())} edges"
            )

        return namespace_graphs

    def _get_batch_edge_index(
        self, namespace: str, batch_terms: List[str]
    ) -> torch.Tensor:
        """Create edge index for a batch of nodes within a namespace."""
        # Get the namespace graph
        namespace_graph = self.namespace_graphs[namespace]

        # Create mapping from batch terms to local indices
        batch_to_local = {term: idx for idx, term in enumerate(batch_terms)}

        # Filter edges that connect nodes in the batch
        batch_edges = []
        for edge in namespace_graph.edges():
            src, dst = edge[0], edge[1]

            # Check if both source and destination are in the batch
            if src in batch_to_local and dst in batch_to_local:
                batch_src = batch_to_local[src]
                batch_dst = batch_to_local[dst]
                batch_edges.append([batch_src, batch_dst])

        if len(batch_edges) == 0:
            # If no edges in batch, create self-loops
            batch_edges = [[i, i] for i in range(len(batch_terms))]

        batch_edge_index = torch.tensor(batch_edges, dtype=torch.long).t().contiguous()

        # Move edge index to the same device as the model
        batch_edge_index = batch_edge_index.to(self._get_device())

        return batch_edge_index

    def _get_precomputed_embeddings(self, go_terms: List[str]) -> torch.Tensor:
        """Get pre-computed embeddings for GO terms."""
        embeddings = []
        missing_terms = []

        for term in go_terms:
            if term in self.precomputed_embeddings:
                embeddings.append(self.precomputed_embeddings[term])
            else:
                missing_terms.append(term)

        # Raise error if any terms are missing
        if missing_terms:
            raise ValueError(f"Missing embeddings for GO terms: {missing_terms}")

        # Stack embeddings directly as torch tensors
        embeddings_tensor = torch.stack(embeddings)

        # Move to the same device as the model
        embeddings_tensor = embeddings_tensor.to(self._get_device(), non_blocking=True)

        return embeddings_tensor

    def _initialize_weights(self):
        """Initialize model weights."""
        # Initialize namespace embeddings
        nn.init.xavier_uniform_(self.namespace_embeddings)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_and_map_namespace(self, namespace: str) -> str:
        """
        Validate and map namespace to long form.
        
        Args:
            namespace: Namespace string to validate and map
            
        Returns:
            Long-form namespace string
            
        Raises:
            ValueError: If namespace is invalid
        """
        # Handle both short and long namespace forms, always use long forms internally
        if namespace in self.namespace_terms:
            # Already in long form (biological_process, etc.)
            return namespace
        elif namespace in GO_BASIC_OBO_NAMESPACE_MAPPINGS:
            # Short form (BP, MF, CC) - map to long form
            return GO_BASIC_OBO_NAMESPACE_MAPPINGS[namespace]
        else:
            # Invalid namespace
            valid_long = list(self.namespace_terms.keys())  # biological_process, etc.
            valid_short = list(GO_BASIC_OBO_NAMESPACE_MAPPINGS.keys())  # BP, MF, CC
            raise ValueError(
                f"Invalid namespace: {namespace}. Must be one of {valid_long} (long), {valid_short} (short), or 'all'"
            )

    def forward(self, namespace: str) -> torch.Tensor:
        """
        Forward pass through the graph encoder for a single namespace or all namespaces.

        Args:
            namespace: One of "biological_process", "molecular_function", "cellular_component", "all"
                      (short forms "BP", "MF", "CC" also accepted)

        Returns:
            reduced_embeddings: Tensor of shape (200, 2560)
        """
        # Handle special case of "all" - process all terms from all namespaces together
        if namespace == "all":
            return self._forward_all_namespaces()
        
        # Use the internal method for single namespace processing
        mapped_namespace = self._validate_and_map_namespace(namespace)
        return self._forward_single_namespace(mapped_namespace)

    def _forward_all_namespaces(self) -> torch.Tensor:
        """
        Forward pass for all namespaces together (internal method).
        Processes each namespace separately through GAT, then concatenates and reduces.
        
        Returns:
            reduced_embeddings: Tensor of shape (200, 2560)
        """
        # Process each namespace separately through GAT
        namespace_embeddings = []
        for namespace in [BIOLOGICAL_PROCESS_NAMESPACE, MOLECULAR_FUNCTION_NAMESPACE, CELLULAR_COMPONENT_NAMESPACE]:
            ns_embeddings = self._forward_namespace_gat(namespace)
            namespace_embeddings.append(ns_embeddings)
        
        # Concatenate embeddings from all namespaces
        concatenated_embeddings = torch.cat(namespace_embeddings, dim=0)
        concatenated_embeddings_batch = concatenated_embeddings.unsqueeze(0)  # (1, total_terms, 2560)
        
        # Apply shared cross-attention reduction
        reduced_embeddings = self.shared_cross_attention_reducer(concatenated_embeddings_batch)
        reduced_embeddings = reduced_embeddings.squeeze(0)  # Remove batch dimension (200, 2560)
        
        return reduced_embeddings

    def _forward_namespace_gat(self, namespace: str) -> torch.Tensor:
        """
        Forward pass for a single namespace through GAT only (no cross-attention).
        
        Args:
            namespace: Must be one of the long-form namespaces: "biological_process", 
                      "molecular_function", "cellular_component"
                      
        Returns:
            embeddings: Tensor of shape (num_terms, 2560) after GAT processing
        """
        # Get all GO terms from this namespace
        namespace_terms = self.namespace_terms[namespace]
        
        # Get pre-computed embeddings for GO terms
        x = self._get_precomputed_embeddings(namespace_terms)
        
        # Add namespace embeddings to distinguish terms by namespace
        namespace_idx = self.namespace_to_idx[namespace]
        namespace_emb = self.namespace_embeddings[namespace_idx]  # (embedding_dim,)
        x = x + namespace_emb.unsqueeze(0)  # Broadcast to (num_terms, embedding_dim)
        
        # Get edge index for this batch within the namespace
        edge_index = self._get_batch_edge_index(namespace, namespace_terms)
        
        # Apply shared GAT layers
        for layer in self.shared_gat_layers:
            x = layer(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final projection using 2-layer MLP with GeLU
        full_embeddings = self.shared_output_projection(x)
        
        return full_embeddings

    def _forward_single_namespace(self, namespace: str) -> torch.Tensor:
        """
        Forward pass for a single namespace (internal method).

        Args:
            namespace: Must be one of the long-form namespaces: "biological_process", 
                      "molecular_function", "cellular_component"

        Returns:
            reduced_embeddings: Tensor of shape (200, 2560)
        """
        # Get GAT-processed embeddings for this namespace
        full_embeddings = self._forward_namespace_gat(namespace)
        full_embeddings_batch = full_embeddings.unsqueeze(0)  # (1, num_terms, 2560)

        # Apply shared cross-attention reduction
        reduced_embeddings = self.shared_cross_attention_reducer(full_embeddings_batch)
        reduced_embeddings = reduced_embeddings.squeeze(0)  # Remove batch dimension (200, 2560)

        return reduced_embeddings

    def get_all_reduced_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Get reduced embeddings for all namespaces.

        Returns:
            Dictionary mapping namespace to tensor of shape (200, 2560)
        """
        results = {}
        self.eval()
        with torch.no_grad():
            for namespace in self.namespace_terms.keys():
                reduced_emb = self(namespace)
                results[namespace] = reduced_emb
        return results

    def encode_namespace(self, namespace: str) -> torch.Tensor:
        """
        Encode all GO terms from a specific namespace or all namespaces.

        Args:
            namespace: One of "biological_process", "molecular_function", "cellular_component", "all"
                      (short forms "BP", "MF", "CC" also accepted)

        Returns:
            reduced_embeddings: Tensor of shape (200, 2560)
        """
        self.eval()
        with torch.no_grad():
            return self(namespace)

    def get_combined_reduced_embeddings(self) -> torch.Tensor:
        """
        Get all reduced embeddings combined into a single tensor.

        Returns:
            Tensor of shape (3 * num_reduced_embeddings, 2560) = (600, 2560)
        """
        reduced_embeddings_dict = self.get_all_reduced_embeddings()

        # Concatenate embeddings from all namespaces
        combined_embeddings = []
        for namespace in [
            BIOLOGICAL_PROCESS_NAMESPACE,
            MOLECULAR_FUNCTION_NAMESPACE,
            CELLULAR_COMPONENT_NAMESPACE,
        ]:
            if namespace in reduced_embeddings_dict:
                combined_embeddings.append(reduced_embeddings_dict[namespace])

        if combined_embeddings:
            return torch.cat(combined_embeddings, dim=0)
        else:
            # Return empty tensor if no embeddings
            return torch.empty(0, self.embedding_dim, device=self._get_device())


def create_go_graph_encoder_pipeline(
    go_obo_path: str,
    precomputed_embeddings_path: str,
    hidden_dim: int = 512,
    num_gat_layers: int = 3,
    num_heads: int = 8,
    num_reduced_embeddings: int = 200,
    embedding_dim: int = 2560,
    embeddings_load_to: str = "cpu",
    unified_go_encoder: bool = False,
):
    """
    Create a complete GO graph encoder with shared GAT and shared cross-attention.

    Args:
        go_obo_path: Path to GO OBO file
        precomputed_embeddings_path: Directory containing individual .safetensors files
            (e.g., GO_0006647.safetensors, GO_0034016.safetensors, etc.)
        hidden_dim: Hidden dimension for GAT layers
        num_gat_layers: Number of GAT layers
        num_heads: Number of attention heads in GAT
        num_reduced_embeddings: Number of reduced embeddings per namespace (default: 200)
        embedding_dim: Embedding dimension for GO terms (default: 2560)
        embeddings_load_to: "cpu" (train-like), "cuda" (eval-like)
        unified_go_encoder: If True, use unified GOGraphEncoderUnified; if False, use original GOGraphEncoder

    Returns:
        Configured GOGraphEncoder or GOGraphEncoderUnified
    """
    if unified_go_encoder:
        # Create unified GO encoder
        go_encoder = GOGraphEncoderUnified(
            go_obo_path=go_obo_path,
            precomputed_embeddings_path=precomputed_embeddings_path,
            hidden_dim=hidden_dim,
            num_gat_layers=num_gat_layers,
            num_heads=num_heads,
            num_reduced_embeddings=num_reduced_embeddings,
            embedding_dim=embedding_dim,
            embeddings_load_to=embeddings_load_to,
        )
    else:
        # Create original GO encoder
        go_encoder = GOGraphEncoder(
            go_obo_path=go_obo_path,
            precomputed_embeddings_path=precomputed_embeddings_path,
            hidden_dim=hidden_dim,
            num_gat_layers=num_gat_layers,
            num_heads=num_heads,
            num_reduced_embeddings=num_reduced_embeddings,
            embedding_dim=embedding_dim,
            embeddings_load_to=embeddings_load_to,
        )

    return go_encoder
