# GO-GPT

GO-GPT is a decoder-only transformer-based model for predicting Gene Ontology (GO) terms from protein sequences. It combines protein language model embeddings with an autoregressive decoder to generate GO term annotations across all three ontology aspects: Molecular Function (MF), Biological Process (BP), and Cellular Component (CC).

## Architecture

- **Protein Encoder**: ESM2 protein language model
- **Decoder**: GPT-style transformer with prefix causal attention
- **Cross-Modal Attention**: Protein embeddings serve as prefix context for GO term generation
- **Organism Conditioning**: Organism-specific embeddings

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd gogpt

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, PyTorch 2.0+, CUDA-capable GPU (recommended)

## Project Structure

```
├── configs/
│   ├── config.yaml              # Main Hydra config
│   ├── experiment/default.yaml  # Training configuration
│   └── preprocess/base.yaml     # Preprocessing configuration
├── data/
│   ├── go-basic.obo             # GO ontology file
│   ├── IA.txt                   # Information Accretion weights
│   └── evals/                   # Evaluation datasets
├── scripts/
│   ├── prepare_data.py          # Data preprocessing
│   └── train.py                 # Model training
├── evals/
│   ├── eval.py                  # Evaluation script
│   ├── eval_probab-scores.py    # Probability-based evaluation
│   ├── run_eval.sh              # Evaluation runner
│   └── run_eval_probab_scores.sh
├── src/gogpt/
│   ├── models/
│   │   ├── gogpt.py             # Core model
│   │   └── gogpt_lightning.py   # PyTorch Lightning wrapper
│   ├── data/
│   │   ├── dataset.py           # Dataset classes
│   │   ├── tokenizer.py         # GO term tokenizer
│   │   └── preprocessing_utils.py
│   ├── config/
│   │   └── model_config.py      # Model configuration dataclass
│   └── utils/
│       └── organism_mapper.py   # Organism ID mapping
├── notebooks/
│   └── inference.ipynb          # Inference example notebook
└── artifacts/
    ├── checkpoints/base/        # Model checkpoints
    └── preprocessed/base/       # Tokenizer, organism mapper
```

## Usage

### 1. Data Preprocessing

Prepare training data from HuggingFace datasets:

```bash
python scripts/prepare_data.py
```

Configuration options in `configs/preprocess/base.yaml`:
- `hf_dataset_name`: Source dataset (default: `wanglab/cafa5`)
- `embed_model_path`: ESM2 model for tokenization
- `vocab_limits`: Minimum frequency thresholds per GO aspect
- `top_n_organisms`: Number of organisms for conditioning
- `output_dir`: Output directory for processed data

### 2. Training

Train the model using PyTorch Lightning:

```bash
python scripts/train.py experiment=default
```

Key training parameters in `configs/experiment/default.yaml`:
- `model.n_layer`, `model.n_head`, `model.n_embd`: Transformer dimensions
- `model.embed_model_path`: ESM2 model path
- `model.use_gated_attention`: Enable gated attention mechanism
- `training.learning_rate`: Learning rate
- `training.devices`: Number of GPUs
- `training.precision`: Mixed precision setting

Resume from checkpoint:
```bash
python scripts/train.py experiment=default resume.checkpoint_path=/path/to/checkpoint.ckpt
```

### 3. Evaluation

Evaluate on external datasets with CAFA metrics:

```bash
# Beam search (deterministic)
bash evals/run_eval.sh beam_search

# Sampling (stochastic)
bash evals/run_eval.sh sampling
```

Evaluation with frequency-based probability scores:
```bash
bash evals/run_eval_probab_scores.sh
```

Update paths in the shell scripts before running:
- `CHECKPOINT_PATH`: Trained model checkpoint
- `EVAL_DIR`: Directory containing FASTA and TSV annotation files
- `PREPROCESSED_ARTIFACTS`: Path to tokenizer and organism mapper

#### Evaluation Data Format

Place evaluation data in `data/evals/<dataset_name>/`:
- `sequences.fasta`: Protein sequences
- `annotations.tsv`: Ground truth GO annotations (columns: protein_id, GO_term, aspect)

### 4. Inference

```python
from gogpt.models.gogpt import GOGPT
from gogpt.config.model_config import GOGPTConfig

# Load model
config = GOGPTConfig(...)
model = GOGPT(config)
model.load_state_dict(torch.load("checkpoint.ckpt")["state_dict"])
model.eval()

# Generate GO terms
go_tokens = model.generate(
    protein_tokens=protein_input,
    protein_mask=attention_mask,
    go_tokens=start_tokens,
    max_new_tokens=300,
    temperature=0.7,
    top_k=50
)

# Or use beam search
go_tokens = model.generate_beam_search(
    protein_tokens=protein_input,
    protein_mask=attention_mask,
    go_tokens=start_tokens,
    max_new_tokens=300,
    beam_size=5,
    length_penalty=0.3
)
```

## Configuration

All configurations use [Hydra](https://hydra.cc/). Override parameters via command line:

```bash
python scripts/train.py experiment=default \
    model.n_layer=16 \
    training.learning_rate=5e-5 \
    training.devices=8
```

### Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_layer` | Number of transformer layers | 12 |
| `n_head` | Number of attention heads | 12 |
| `n_embd` | Embedding dimension | 1080 |
| `block_size` | Maximum sequence length | 2048 |
| `dropout` | Dropout rate | 0.1 |
| `protein_embedding_dim` | ESM2 output dimension | 1280 |
| `use_gated_attention` | Enable gated attention | true |
| `freeze_esm` | Freeze ESM2 parameters | true |

## Outputs

Training outputs are saved to `artifacts/training/<experiment_name>/`:
- `checkpoints/`: Model checkpoints
- `wandb/`: Weights & Biases logs (if enabled)

Evaluation outputs are saved to `artifacts/eval_results/`:
- `predictions.tsv`: Raw predictions
- `cafa_results/`: CAFA evaluation metrics

