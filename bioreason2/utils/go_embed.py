#!/usr/bin/env python3
"""
GO Term Embedding Script

This script loads GO metadata from the wanglab/cafa5 dataset, creates embeddings
using the Qwen/Qwen3-Embedding-4B model, and saves each embedding as a .safetensors file.
"""

import os
import sys
from typing import List, Dict, Any, Optional
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from safetensors.torch import save_file
from tqdm import tqdm
import argparse
from bioreason2.utils import str2bool


class GOEmbeddingGenerator:
    """
    A class for generating embeddings for GO terms using Qwen3-Embedding-4B model.

    This class handles:
    - Loading GO metadata from HuggingFace dataset
    - Creating text representations for GO terms
    - Generating embeddings using the Qwen3 embedding model
    - Saving embeddings as .safetensors files
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        cache_dir: str = None,
        dataset_cache_dir: str = None,
        output_dir: str = "go_embeddings",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the GO embedding generator.

        Args:
            model_name: Name of the embedding model to use
            cache_dir: Directory for caching models
            dataset_cache_dir: Directory for caching datasets
            output_dir: Directory to save the generated embeddings
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.dataset_cache_dir = dataset_cache_dir
        self.output_dir = output_dir
        self.device = device

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize the embedding model
        self._load_model()

    def _load_model(self) -> None:
        """Load the Qwen3 embedding model."""
        print(f"Loading embedding model: {self.model_name}")
        print(f"Using device: {self.device}")

        # Load the model with appropriate configurations
        self.model = SentenceTransformer(
            self.model_name,
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "device_map": "auto" if self.device == "cuda" else None,
                "torch_dtype": torch.bfloat16,
            },
            tokenizer_kwargs={"padding_side": "left"},
        )

        print("✓ Model loaded successfully")

    def load_go_metadata(
        self, dataset_name: str = "wanglab/cafa5", subset: str = "go_metadata"
    ) -> List[Dict[str, Any]]:
        """
        Load GO metadata from the HuggingFace dataset.

        Args:
            dataset_name: Name of the dataset to load
            subset: Name of the subset to load

        Returns:
            List of dictionaries containing GO term metadata
        """
        print(f"Loading GO metadata from {dataset_name}/{subset}")

        # Load the dataset
        dataset = load_dataset(dataset_name, subset, cache_dir=self.dataset_cache_dir)

        # Convert to list of dictionaries
        go_metadata = dataset["metadata"].to_pandas().to_dict("records")

        print(f"✓ Loaded {len(go_metadata)} GO terms")

        # Filter out terms without required fields
        filtered_metadata = []
        for item in go_metadata:
            if item.get("go_id") and item.get("go_name") and item.get("go_def_clean"):
                filtered_metadata.append(item)
            else:
                print(f"⚠️  Skipping GO term with missing fields: {item.get('go_id', 'unknown')}")

        print(f"✓ Filtered to {len(filtered_metadata)} valid GO terms")
        return filtered_metadata

    def format_go_aspect(self, aspect: str) -> str:
        """
        Format the GO aspect with spaces instead of underscores and capitalize each word.

        Args:
            aspect: The raw GO aspect string (e.g., 'biological_process')

        Returns:
            Formatted aspect string (e.g., 'Biological Process')
        """
        if not aspect:
            return ""

        # Replace underscores with spaces and title case
        return aspect.replace("_", " ").title()

    def create_go_text(self, go_term: Dict[str, Any]) -> str:
        """
        Create the text representation for a GO term.

        Args:
            go_term: Dictionary containing GO term metadata

        Returns:
            Formatted text string for embedding
        """
        go_id = go_term["go_id"]
        go_name = go_term["go_name"]
        go_def_clean = go_term["go_def_clean"]
        go_aspect = go_term.get("go_aspect", "")
        formatted_aspect = self.format_go_aspect(go_aspect)

        return f"{go_id} ({formatted_aspect}): {go_name}. {go_def_clean}"

    def generate_embeddings_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Generate embeddings for multiple texts using the Qwen3 model in batch.

        Args:
            texts: List of input texts to embed

        Returns:
            Tensor containing embeddings with shape (batch_size, embedding_dim)
        """
        # The model.encode method returns normalized embeddings by default
        embeddings = self.model.encode(texts, convert_to_tensor=True, batch_size=len(texts))

        return embeddings

    def get_embedding_filename(self, go_id: str) -> str:
        """
        Get the filename for a GO term embedding.

        Args:
            go_id: GO term ID

        Returns:
            Filename for the embedding file
        """
        # Clean the GO ID for filename (remove colons and other special characters)
        safe_filename = go_id.replace(":", "_").replace("-", "_") + ".safetensors"
        return safe_filename

    def get_embedding_filepath(self, go_id: str) -> str:
        """
        Get the full filepath for a GO term embedding.

        Args:
            go_id: GO term ID

        Returns:
            Full path to the embedding file
        """
        filename = self.get_embedding_filename(go_id)
        return os.path.join(self.output_dir, filename)

    def embedding_exists(self, go_id: str) -> bool:
        """
        Check if an embedding file already exists for a GO term.

        Args:
            go_id: GO term ID

        Returns:
            True if the embedding file exists, False otherwise
        """
        filepath = self.get_embedding_filepath(go_id)
        return os.path.exists(filepath)

    def delete_all_embeddings(self) -> int:
        """
        Delete all existing embedding files in the output directory.

        Returns:
            Number of files deleted
        """
        if not os.path.exists(self.output_dir):
            return 0

        deleted_count = 0
        for filename in os.listdir(self.output_dir):
            if filename.endswith(".safetensors"):
                filepath = os.path.join(self.output_dir, filename)
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    print(f"Warning: Failed to delete {filepath}: {e}")

        print(f"🗑️  Deleted {deleted_count} existing embedding files")
        return deleted_count

    def save_embedding(self, go_id: str, embedding: torch.Tensor) -> str:
        """
        Save a single embedding as a .safetensors file.

        Args:
            go_id: GO term ID (used as filename)
            embedding: Embedding tensor to save

        Returns:
            Path to the saved file
        """
        filepath = self.get_embedding_filepath(go_id)

        # Save as safetensors
        embedding_dict = {"embedding": embedding.cpu()}
        save_file(embedding_dict, filepath)

        return filepath

    def process_go_terms(
        self,
        go_metadata: List[Dict[str, Any]],
        batch_size: int = 32,
        max_terms: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Process all GO terms and generate embeddings.

        Args:
            go_metadata: List of GO term metadata dictionaries
            batch_size: Number of terms to process at once
            max_terms: Maximum number of terms to process (None = all)

        Returns:
            Dictionary mapping GO IDs to their embedding file paths
        """
        # Limit the number of terms if specified
        if max_terms is not None and max_terms > 0:
            go_metadata = go_metadata[:max_terms]
            print(f"Limited to processing {len(go_metadata)} GO terms (max_terms={max_terms})")

        print(f"Processing {len(go_metadata)} GO terms in batches of {batch_size}")

        # Track successful embeddings and skipped terms
        processed_files = {}
        skipped_terms = []
        failed_terms = []

        # Check which terms already have embeddings
        terms_to_process = []
        for go_term in go_metadata:
            go_id = go_term["go_id"]
            if self.embedding_exists(go_id):
                skipped_terms.append(go_id)
                # Still add to processed_files with existing filepath
                processed_files[go_id] = self.get_embedding_filepath(go_id)
            else:
                terms_to_process.append(go_term)

        if skipped_terms:
            print(f"⏭️  Skipping {len(skipped_terms)} GO terms that already have embeddings")

        if not terms_to_process:
            print("✅ All GO terms already have embeddings!")
            return processed_files

        print(f"📝 Need to process {len(terms_to_process)} new GO terms")

        # Process terms that need embeddings in batches
        for i in tqdm(range(0, len(terms_to_process), batch_size), desc="Processing GO terms"):
            batch = terms_to_process[i : i + batch_size]

            try:
                # Create text representations for the batch
                batch_texts = []
                batch_go_ids = []

                for go_term in batch:
                    go_id = go_term["go_id"]
                    text = self.create_go_text(go_term)
                    batch_texts.append(text)
                    batch_go_ids.append(go_id)

                # Generate embeddings for the entire batch at once
                batch_embeddings = self.generate_embeddings_batch(batch_texts)

                # Save each embedding
                for j, (go_id, embedding) in enumerate(zip(batch_go_ids, batch_embeddings)):
                    filepath = self.save_embedding(go_id, embedding)
                    processed_files[go_id] = filepath

            except Exception as e:
                print(f"❌ Failed to process batch starting with GO term {batch[0].get('go_id', 'unknown')}: {e}")
                # Add all terms in the failed batch to failed_terms
                for go_term in batch:
                    failed_terms.append(go_term.get("go_id", "unknown"))

        # Calculate statistics
        newly_processed = len(processed_files) - len(skipped_terms)

        print(f"✓ Successfully processed {len(processed_files)} GO terms")
        print(f"  - Newly embedded: {newly_processed} terms")
        print(f"  - Skipped (already exist): {len(skipped_terms)} terms")

        if failed_terms:
            print(f"❌ Failed to process {len(failed_terms)} GO terms: {failed_terms[:5]}...")

        return processed_files

    def run(self, batch_size: int = 32, max_terms: Optional[int] = None, resume: bool = True) -> Dict[str, str]:
        """
        Run the complete GO embedding generation pipeline.

        Args:
            batch_size: Number of terms to process at once
            max_terms: Maximum number of terms to process (None = all)
            resume: If True, skip existing embeddings; if False, delete all existing embeddings first

        Returns:
            Dictionary mapping GO IDs to their embedding file paths
        """
        print("=" * 60)
        print("GO TERM EMBEDDING GENERATION")
        print("=" * 60)

        # Handle resume flag
        if not resume:
            print("🔄 Resume=False: Deleting all existing embeddings...")
            self.delete_all_embeddings()
        else:
            print("▶️  Resume=True: Will skip existing embeddings")

        # Load GO metadata
        go_metadata = self.load_go_metadata()

        if not go_metadata:
            print("❌ No valid GO terms found!")
            return {}

        # Process all GO terms
        processed_files = self.process_go_terms(go_metadata, batch_size=batch_size, max_terms=max_terms)

        print("=" * 60)
        print("EMBEDDING GENERATION COMPLETED")
        print("=" * 60)
        print(f"Total GO terms processed: {len(processed_files)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Model used: {self.model_name}")
        print(f"Files saved in: {self.output_dir}")

        return processed_files


def main():
    """Main function to run the GO embedding generation."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for GO terms using Qwen3-Embedding-4B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-Embedding-4B",
        help="Name of the embedding model to use",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching models",
    )

    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=None,
        help="Directory for caching datasets",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="go_embeddings",
        help="Directory to save the generated embeddings",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of GO terms to process at once",
    )

    parser.add_argument(
        "--max_terms",
        type=int,
        default=None,
        help="Maximum number of GO terms to process (for testing, None = all)",
    )

    parser.add_argument(
        "--resume",
        type=str2bool,
        default=True,
        help="Resume mode: if True, skip existing embeddings; if False, delete all existing embeddings first",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run the model on",
    )

    args = parser.parse_args()

    # Print configuration
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()

    # Initialize and run the generator
    generator = GOEmbeddingGenerator(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        dataset_cache_dir=args.dataset_cache_dir,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Run the pipeline
    processed_files = generator.run(batch_size=args.batch_size, max_terms=args.max_terms, resume=args.resume)

    # Print summary
    if processed_files:
        print("\nExample saved files:")
        for i, (go_id, filepath) in enumerate(list(processed_files.items())[:5]):
            print(f"  {go_id} -> {os.path.basename(filepath)}")
        if len(processed_files) > 5:
            print(f"  ... and {len(processed_files) - 5} more files")
    else:
        print("❌ No files were generated!")
        sys.exit(1)

    print("\n🎉 GO embedding generation completed successfully!")


if __name__ == "__main__":
    main()
