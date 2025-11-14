# Project Structure

```
MoE-Embeddings-Research/
│
├── 📄 README.md                          # Main documentation
├── 📄 PROJECT_OVERVIEW.md                # Detailed project summary
├── 📄 STRUCTURE.md                       # This file
├── 📄 pyproject.toml                     # Dependencies (managed by uv)
├── 📄 uv.lock                           # Lock file
├── 📄 .gitignore                        # Git ignore rules
├── 📄 .python-version                   # Python version
│
├── 🧪 test_pipeline.py                  # Automated pipeline tests
├── 💡 example_usage.py                  # Usage examples
│
├── 📁 src/                              # Source code (1,804 lines)
│   ├── __init__.py
│   ├── utils.py                         # Utility functions
│   │
│   ├── 📁 models/                       # Model architecture
│   │   ├── __init__.py
│   │   ├── encoder.py                   # Transformer encoder
│   │   │   ├── MultiHeadAttention       # Self-attention mechanism
│   │   │   ├── FeedForward              # ← MoE INTEGRATION POINT
│   │   │   ├── TransformerLayer         # Single transformer layer
│   │   │   └── TransformerEncoder       # Full encoder
│   │   ├── pooling.py                   # Pooling strategies
│   │   │   └── Pooler                   # Mean/Max/CLS/MeanMax pooling
│   │   └── embeddings.py                # Complete model
│   │       └── EmbeddingModel           # Main model class
│   │
│   ├── 📁 data/                         # Data processing
│   │   ├── __init__.py
│   │   ├── tokenizer.py                 # Simple tokenizer
│   │   │   └── SimpleTokenizer          # Word-based tokenization
│   │   └── dataset.py                   # Dataset classes
│   │       ├── PairDataset              # For contrastive pairs
│   │       ├── TripletDataset           # For triplet loss
│   │       └── InBatchNegativesDataset  # For in-batch negatives
│   │
│   ├── 📁 training/                     # Training infrastructure
│   │   ├── __init__.py
│   │   ├── losses.py                    # Loss functions
│   │   │   ├── ContrastiveLoss          # Classic contrastive
│   │   │   ├── TripletLoss              # Triplet margin loss
│   │   │   ├── MultipleNegativesRankingLoss  # InfoNCE/NT-Xent
│   │   │   └── CosineSimilarityLoss     # Direct similarity
│   │   └── trainer.py                   # Training loop
│   │       └── EmbeddingTrainer         # Complete trainer
│   │
│   ├── 📁 evaluation/                   # Evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py                   # Comprehensive metrics
│   │       ├── compute_similarity       # Cosine/Euclidean similarity
│   │       ├── evaluate_retrieval       # P@K, R@K, MRR, MAP
│   │       ├── evaluate_classification  # k-NN classification
│   │       ├── evaluate_semantic_similarity  # STS-style evaluation
│   │       └── compute_embedding_statistics  # Analysis tools
│   │
│   └── 📁 experts/                      # MoE components (FUTURE)
│       └── __init__.py                  # Placeholder with roadmap
│
├── 📁 notebooks/                        # Jupyter notebooks
│   └── demo_training_evaluation.ipynb   # Complete demo
│       ├── 1. Data Preparation
│       ├── 2. Tokenizer Building
│       ├── 3. Model Initialization
│       ├── 4. Training (20 epochs)
│       ├── 5. Loss Visualization
│       ├── 6. Similarity Evaluation
│       ├── 7. t-SNE Visualization
│       ├── 8. Heatmaps
│       └── 9. Inference Examples
│
├── 📁 models/                           # Saved models (created at runtime)
│   └── (model checkpoints go here)
│
└── 📁 .venv/                            # Virtual environment (managed by uv)
    └── (Python packages)
```

## Component Breakdown

### Core Model Components (src/models/)

```
EmbeddingModel
│
├── Token Embedding (vocab_size → hidden_dim)
├── Position Embedding (max_seq_len → hidden_dim)
│
├── TransformerEncoder (6 layers)
│   ├── Layer 1
│   │   ├── MultiHeadAttention (8 heads)
│   │   ├── LayerNorm
│   │   ├── FeedForward (hidden → ff_dim → hidden) ← MoE HERE
│   │   └── LayerNorm
│   ├── Layer 2
│   │   └── ...
│   └── Layer N
│       └── ...
│
├── Pooler (sequence → fixed vector)
│   ├── Mean Pooling (default)
│   ├── Max Pooling
│   ├── CLS Pooling
│   └── MeanMax Pooling
│
└── L2 Normalization (for cosine similarity)
```

### Data Flow

```
Input Text
    ↓
[Tokenizer] → Token IDs [batch, seq_len]
    ↓
[Embedding] → Embedded Tokens [batch, seq_len, hidden_dim]
    ↓
[TransformerEncoder] → Contextualized [batch, seq_len, hidden_dim]
    ↓
[Pooler] → Sentence Vector [batch, hidden_dim]
    ↓
[L2 Norm] → Final Embedding [batch, hidden_dim]
```

### Training Pipeline

```
Dataset (Pairs/Triplets)
    ↓
DataLoader (Batching)
    ↓
Model (Forward Pass)
    ↓
Loss Function (Contrastive/Triplet/InfoNCE)
    ↓
Backpropagation
    ↓
Optimizer (AdamW)
    ↓
Scheduler (Cosine/ReduceLR)
    ↓
Validation
    ↓
Checkpointing
```

## File Sizes & Complexity

| File | Lines | Purpose | Complexity |
|------|-------|---------|-----------|
| `models/encoder.py` | 214 | Transformer architecture | High |
| `models/pooling.py` | 96 | Pooling strategies | Medium |
| `models/embeddings.py` | 143 | Complete model | Medium |
| `data/tokenizer.py` | 140 | Tokenization | Medium |
| `data/dataset.py` | 123 | Dataset classes | Low |
| `training/losses.py` | 139 | Loss functions | Medium |
| `training/trainer.py` | 219 | Training loop | High |
| `evaluation/metrics.py` | 221 | Evaluation metrics | High |
| `utils.py` | 297 | Utility functions | Medium |
| **Total** | **~1,800** | | |

## MoE Integration Roadmap

### Current Architecture
```
FeedForward(x):
    x = Linear(hidden_dim → ff_dim)
    x = GELU()
    x = Dropout()
    x = Linear(ff_dim → hidden_dim)
    x = Dropout()
    return x
```

### Future MoE Architecture
```
MoEFeedForward(x):
    # Gating
    router_logits = RouterNetwork(x)      # [batch, seq, num_experts]
    expert_weights, expert_indices = TopK(router_logits, k=2)

    # Expert computation
    expert_outputs = []
    for expert_idx in expert_indices:
        expert_output = Expert[expert_idx](x)
        expert_outputs.append(expert_output)

    # Combine
    output = WeightedSum(expert_outputs, expert_weights)

    # Load balancing
    aux_loss = LoadBalancingLoss(router_logits)

    return output, aux_loss
```

## Quick Commands

```bash
# Setup
uv sync

# Test
uv run python test_pipeline.py

# Examples
uv run python example_usage.py

# Jupyter
uv run jupyter notebook notebooks/demo_training_evaluation.ipynb

# Train (custom)
uv run python your_training_script.py
```

## Dependencies Overview

```toml
[project]
name = "moe-embeddings-research"
version = "0.1.0"
requires-python = ">=3.13"

dependencies = [
    "torch",           # Deep learning
    "transformers",    # NLP utilities
    "datasets",        # Dataset management
    "numpy",           # Numerical computing
    "scipy",           # Scientific computing
    "scikit-learn",    # ML utilities
    "matplotlib",      # Plotting
    "seaborn",         # Statistical viz
    "pandas",          # Data manipulation
    "jupyter",         # Notebooks
    "ipykernel",       # Jupyter kernel
    "tqdm",            # Progress bars
]
```

## Next Steps

1. ✅ **Phase 1 Complete**: Base architecture implemented
2. ⏳ **Phase 2 Starting**: MoE implementation
   - Implement Expert class
   - Implement Gating mechanism
   - Create MoEFeedForward layer
   - Add load balancing
3. ⏳ **Phase 3**: Training & Evaluation
   - Train on diverse domains
   - Analyze expert specialization
   - Compare vs dense baseline
4. ⏳ **Phase 4**: Research & Publication
   - Write paper
   - Create visualizations
   - Benchmark results

---

**Status**: ✅ Ready for MoE integration
**Test Coverage**: ✅ All components tested
**Documentation**: ✅ Comprehensive
**Code Quality**: ✅ Production-ready
