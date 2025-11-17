# Language Embeddings Model with MoE Architecture

A from-scratch implementation of a language embeddings model with a modular architecture designed for **Mixture of Experts (MoE)** integration. Built using PyTorch and managed with `uv` for dependency management.

## Project Overview

This project implements a complete sentence/text embedding model capable of learning semantic representations through contrastive learning. The architecture is intentionally modular to facilitate future MoE enhancements, where experts can specialize in different linguistic domains or patterns.

### Key Features

- **From-Scratch Implementation**: Transformer encoder, attention mechanisms, and embeddings built from the ground up
- **Modular Architecture**: Clean separation between encoder, pooling, and expert components
- **Contrastive Learning**: Multiple loss functions including InfoNCE, Triplet Loss, and standard contrastive loss
- **Comprehensive Evaluation**: Similarity metrics, retrieval evaluation, and visualization tools
- **Production-Ready**: Training infrastructure with checkpointing, scheduling, and monitoring
- **MoE-Ready**: Designed with clear extension points for expert networks and gating mechanisms

## Architecture

```
Input Text
    ↓
[Tokenizer]
    ↓
Token IDs → [Token Embedding] + [Positional Embedding]
    ↓
[Transformer Encoder]
├── Multi-Head Attention
├── Feed-Forward Network ← (MoE replacement point)
└── Layer Normalization
    ↓
[Pooling Layer] (Mean/Max/CLS/Mean-Max)
    ↓
[L2 Normalization]
    ↓
Final Embeddings (normalized vectors)
```

### Components

#### 1. **Models** (`src/models/`)
- `encoder.py`: Transformer encoder with multi-head attention and feed-forward layers
- `pooling.py`: Multiple pooling strategies (mean, max, CLS, mean-max)
- `embeddings.py`: Complete embedding model combining encoder and pooling

#### 2. **Data** (`src/data/`)
- `tokenizer.py`: Simple word-based tokenizer (ready for BPE/WordPiece upgrade)
- `dataset.py`: Dataset classes for pairs, triplets, and in-batch negatives

#### 3. **Training** (`src/training/`)
- `losses.py`: Multiple loss functions (ContrastiveLoss, TripletLoss, MultipleNegativesRankingLoss)
- `trainer.py`: Complete training loop with validation, checkpointing, and scheduling

#### 4. **Evaluation** (`src/evaluation/`)
- `metrics.py`: Similarity computation, retrieval metrics (P@K, R@K, MRR, MAP), classification

#### 5. **Experts** (`src/experts/`)
- Placeholder for future MoE implementation
- Will contain expert networks, gating mechanisms, and MoE layers

## Installation

This project uses `uv` for fast, reliable Python package management.

### Prerequisites
- Python 3.13+
- uv (install from: https://github.com/astral-sh/uv)

### Setup

```bash
# Clone the repository
cd MoE-Embeddings-Research

# Install dependencies (already initialized with uv)
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

### Installed Dependencies

Core packages:
- `torch`: Deep learning framework
- `transformers`: Tokenizers and model utilities
- `datasets`: Dataset loading and processing
- `numpy`, `scipy`: Numerical computing
- `scikit-learn`: ML utilities and metrics
- `matplotlib`, `seaborn`: Visualization
- `pandas`: Data manipulation
- `jupyter`, `ipykernel`: Interactive notebooks
- `tqdm`: Progress bars

## Quick Start

### 1. Test the Pipeline

```bash
uv run python test_pipeline.py
```

This will verify that all components work correctly.

### 2. Run the Jupyter Notebook Demo

```bash
uv run jupyter notebook notebooks/demo_training_evaluation.ipynb
```

The notebook includes:
- Data preparation and tokenization
- Model training with contrastive learning
- Evaluation metrics and similarity search
- t-SNE visualization of embeddings
- Similarity heatmaps
- Model saving and inference examples

### 3. Train Your Own Model

```python
from src.models import EmbeddingModel
from src.data import SimpleTokenizer, PairDataset
from src.training import MultipleNegativesRankingLoss, EmbeddingTrainer
from torch.utils.data import DataLoader
import torch

# Prepare data
sentences = ["your", "training", "sentences"]
pairs = [("sent1", "sent2"), ...]

# Build tokenizer
tokenizer = SimpleTokenizer(vocab_size=10000)
tokenizer.fit(sentences)

# Create dataset
dataset = PairDataset(pairs, tokenizer)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = EmbeddingModel(
    vocab_size=len(tokenizer),
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    ff_dim=1024,
    pooling_mode="mean"
)

# Setup training
loss_fn = MultipleNegativesRankingLoss(temperature=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

trainer = EmbeddingTrainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Train
history = trainer.train(
    train_loader=train_loader,
    num_epochs=20,
    save_path="models/my_model.pt"
)
```

## Project Structure

```
MoE-Embeddings-Research/
├── src/
│   ├── models/              # Model architecture
│   │   ├── encoder.py       # Transformer encoder
│   │   ├── pooling.py       # Pooling strategies
│   │   └── embeddings.py    # Complete model
│   ├── data/                # Data utilities
│   │   ├── tokenizer.py     # Tokenization
│   │   └── dataset.py       # Dataset classes
│   ├── training/            # Training infrastructure
│   │   ├── losses.py        # Loss functions
│   │   └── trainer.py       # Training loop
│   ├── evaluation/          # Evaluation metrics
│   │   └── metrics.py       # Similarity & retrieval
│   └── experts/             # MoE components (future)
│       └── __init__.py      # Placeholder
├── notebooks/
│   └── demo_training_evaluation.ipynb  # Complete demo
├── models/                  # Saved models (created during training)
├── test_pipeline.py         # Pipeline test script
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

## Model Configuration

Default hyperparameters (customizable):

```python
model_config = {
    "vocab_size": 10000,        # Vocabulary size
    "hidden_dim": 256,          # Embedding dimension
    "num_layers": 6,            # Transformer layers
    "num_heads": 8,             # Attention heads
    "ff_dim": 1024,             # Feed-forward dimension
    "max_seq_len": 512,         # Max sequence length
    "dropout": 0.1,             # Dropout rate
    "pooling_mode": "mean",     # Pooling strategy
    "normalize_embeddings": True # L2 normalization
}
```

## Loss Functions

### 1. Multiple Negatives Ranking Loss (Recommended)
```python
from src.training import MultipleNegativesRankingLoss

loss_fn = MultipleNegativesRankingLoss(temperature=0.05)
```
- Uses in-batch negatives (efficient)
- Same loss as CLIP, SimCLR, sentence-transformers
- Best for large batches

### 2. Triplet Loss
```python
from src.training import TripletLoss

loss_fn = TripletLoss(margin=1.0)
```
- Requires (anchor, positive, negative) triplets
- Good for fine-grained similarity

### 3. Contrastive Loss
```python
from src.training import ContrastiveLoss

loss_fn = ContrastiveLoss(margin=1.0)
```
- Requires explicit positive/negative labels
- Classic approach

## Evaluation Metrics

### Semantic Similarity
```python
from src.evaluation import evaluate_semantic_similarity

results = evaluate_semantic_similarity(
    model, sentence_pairs, similarity_scores, tokenizer
)
# Returns: {"pearson": 0.85, "spearman": 0.83}
```

### Retrieval Performance
```python
from src.evaluation import evaluate_retrieval

results = evaluate_retrieval(
    query_embeddings, doc_embeddings, relevance_labels, k_values=[1, 5, 10]
)
# Returns: precision@k, recall@k, MRR, MAP
```

### Classification (k-NN)
```python
from src.evaluation import evaluate_classification

results = evaluate_classification(
    train_embeddings, train_labels, test_embeddings, test_labels, k=5
)
# Returns: accuracy, precision, recall, f1
```

## Next Steps: MoE Integration

The current architecture is ready for MoE enhancement. Here's the roadmap:

### Phase 1: Expert Networks
1. Implement expert modules in `src/experts/expert.py`
   - Specialized feed-forward networks
   - Domain-specific transformations

### Phase 2: Gating Mechanism
2. Design routing in `src/experts/gating.py`
   - Top-K gating (route to K experts)
   - Softmax gating
   - Learnable router network

### Phase 3: MoE Layer
3. Create MoE layer in `src/experts/moe_layer.py`
   - Replace feed-forward in transformer
   - Load balancing mechanisms
   - Expert utilization tracking

### Phase 4: Training & Evaluation
4. Add auxiliary losses for load balancing
5. Analyze expert specialization
6. Compare MoE vs dense baseline

### Integration Points

**Current FeedForward layer** (`src/models/encoder.py:66-80`):
```python
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout):
        # Can be replaced with MoE layer
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        # ...
```

**Future MoE layer**:
```python
class MoEFeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim, num_experts, top_k):
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ff_dim) for _ in range(num_experts)
        ])
        self.gate = GatingNetwork(hidden_dim, num_experts)
        # ...
```

## Performance Tips

1. **Batch Size**: Larger batches improve contrastive learning (more negatives)
2. **Learning Rate**: Start with 1e-3, use warmup and cosine decay
3. **Pooling**: Mean pooling works best for general similarity
4. **Normalization**: Always use L2 normalization for cosine similarity
5. **Data Quality**: High-quality positive pairs are crucial

## Datasets for Training

Recommended public datasets:

1. **SNLI/MultiNLI**: Natural language inference
2. **STS Benchmark**: Semantic textual similarity
3. **MS MARCO**: Information retrieval
4. **Natural Questions**: Question answering
5. **WikiAnswers**: Paraphrase pairs
6. **Quora Question Pairs**: Duplicate questions

## Citation

If you use this code for research, please cite:

```bibtex
@software{moe_embeddings_research,
  title = {Language Embeddings Model with MoE Architecture},
  author = {Arjun Mulchandani},
  year = {2025},
  url = {https://github.com/FloopyFlop/MoE-Embeddings-Research}
}
```