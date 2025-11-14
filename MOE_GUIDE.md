# Mixture of Experts (MoE) Implementation Guide

## 🎯 What We've Built

A **complete MoE implementation** for embeddings models with:

1. **Expert Networks**: Specialized feed-forward networks
2. **Top-K Gating**: Learned routing to select best experts
3. **Load Balancing**: Ensures all experts are used evenly
4. **Fair Comparison**: MoE vs Dense with matched active parameters

## 📊 Fair Comparison Setup

### The Challenge
To properly evaluate if MoE helps, we need to ensure the comparison is **fair**:

- ❌ **Unfair**: MoE with 40M params vs Dense with 16M params
- ✅ **Fair**: Both have ~16M **active** params per forward pass

### Our Solution

**Dense Model:**
- 384 hidden dim
- 1536 FFN dim (4× hidden)
- **~16M total parameters**
- All parameters active

**MoE Model:**
- 384 hidden dim
- 8 experts × 768 FFN dim each
- Top-2 routing (only 2 experts active per token)
- **~40M total parameters, ~16M active**

**Result:** Same computational cost, but MoE has 2.5× model capacity!

## 🏗️ Architecture

### Expert Network (`src/experts/expert.py`)

```python
class Expert(nn.Module):
    """Single expert - a specialized FFN"""
    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
```

Each expert is a standard feed-forward network that can **specialize** in different patterns.

### Gating Mechanism (`src/experts/gating.py`)

```python
class TopKGating(nn.Module):
    """Routes each token to top-K experts"""
    def forward(self, x):
        logits = self.gate(x)  # Compute routing scores
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k)
        weights = F.softmax(top_k_logits, dim=-1)
        return weights, top_k_indices, logits
```

The gating network **learns** which experts are best for each input.

### MoE Layer (`src/experts/moe_layer.py`)

```python
class MoELayer(nn.Module):
    """Replaces standard FFN with mixture of experts"""
    def __init__(self, hidden_dim, ff_dim, num_experts=8, top_k=2):
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ff_dim) for _ in range(num_experts)
        ])
        self.gate = TopKGating(hidden_dim, num_experts, top_k)
```

Combines experts with gating for sparse, specialized computation.

### Load Balancing Loss

Prevents all tokens routing to same expert:

```python
def compute_load_balancing_loss(router_logits, selected_experts, num_experts):
    # Encourages uniform expert usage
    # Based on "Switch Transformers" paper
    ...
```

## 📈 How to Use

### Option 1: Run the Comparison Notebook (Recommended)

```bash
uv run jupyter notebook notebooks/moe_vs_dense_comparison.ipynb
```

This notebook:
1. ✅ Trains BOTH models on same data
2. ✅ Ensures fair comparison (matched active params)
3. ✅ Compares validation loss
4. ✅ Compares semantic similarity performance
5. ✅ Analyzes expert specialization
6. ✅ Declares a winner with statistical evidence

**Time:** ~10-15 minutes total on M4 MAX

### Option 2: Use MoE Model Directly

```python
from src.models import EmbeddingModelMoE

model = EmbeddingModelMoE(
    vocab_size=30000,
    hidden_dim=384,
    num_layers=6,
    num_heads=12,
    ff_dim=768,        # Per-expert FFN size
    num_experts=8,     # Total experts
    top_k=2,           # Active experts per token
    pooling_mode='mean'
)

# Training requires handling aux loss
output = model(input_ids, attention_mask)
embeddings = output['embeddings']
aux_loss = output['aux_loss']  # Add to main loss!

total_loss = main_loss + aux_loss
```

## 🔬 Understanding the Results

### What to Look For

**1. Validation Loss**
- Lower is better
- Shows which model generalizes better

**2. Semantic Similarity Separation**
- Higher separation = better embeddings
- Measures: `mean(similar) - mean(dissimilar)`

**3. Expert Usage**
- Should be relatively balanced
- Low coefficient of variation (CV) = good load balancing

### Possible Outcomes

**Scenario 1: MoE Wins** 🎉
- Better validation loss
- Better similarity separation
- **Conclusion:** Expert specialization helps!
- **Next steps:** Try more experts (16, 32) or different top-k (3, 4)

**Scenario 2: Dense Wins** 🤔
- Better validation loss
- **Conclusion:** MoE needs tuning
- **Possible issues:**
  - Load balancing weight too high/low
  - Experts too small/large
  - Need more training data
  - Need more epochs

**Scenario 3: Tie** 🤝
- Similar performance
- **Conclusion:** MoE gives 2.5× capacity for free!
- **Benefits:** Can scale to larger models without compute increase

## 🎓 Key Concepts

### Sparse Computation

MoE achieves sparsity through **conditional computation**:

- Dense model: ALL 1536 FFN dim units active
- MoE model: Only 2/8 experts active = 25% sparsity
- **Benefit:** More capacity, same compute

### Expert Specialization

Experts can learn different patterns:
- Expert 1: Technical language
- Expert 2: Casual conversation
- Expert 3: Questions
- Expert 4: Descriptions
- ... etc.

The gating network learns which expert to use for each input.

### Load Balancing

Without load balancing, some experts become "lazy":
- Most tokens → Expert 0, Expert 1
- Other experts rarely used

Load balancing loss encourages equal usage.

## 📊 Parameter Counting

### Dense Model
```
Total params = Embedding + Attention + FFN + Other
Active params = Total params (all active)
```

### MoE Model
```
Total params = Embedding + Attention + (8 × Expert FFN) + Other
Active params = Embedding + Attention + (2 × Expert FFN) + Other
                (only top-2 experts active)
```

### Our Configuration

| Component | Dense | MoE |
|-----------|-------|-----|
| Embeddings | ~5M | ~5M |
| Attention | ~3M | ~3M |
| FFN | ~7M | ~28M (but 25% active = ~7M) |
| Other | ~1M | ~1M |
| **Total** | **~16M** | **~40M** |
| **Active** | **~16M** | **~16M** ✓ |

## 🚀 Advanced: Tuning MoE

### Hyperparameters to Try

**1. Number of Experts**
```python
num_experts=4   # Fewer experts, more capacity each
num_experts=8   # Balanced (our default)
num_experts=16  # Many small experts
```

**2. Top-K Selection**
```python
top_k=1  # Sparse, clear specialization
top_k=2  # Balanced (our default)
top_k=4  # Less sparse, more mixing
```

**3. Expert Size**
```python
ff_dim=512   # Smaller experts
ff_dim=768   # Balanced (our default)
ff_dim=1024  # Larger experts
```

**4. Load Balancing Weight**
```python
load_balancing_weight=0.001  # Less balancing
load_balancing_weight=0.01   # Balanced (our default)
load_balancing_weight=0.1    # Strong balancing
```

## 📚 Implementation Details

### Files Created

```
src/experts/
├── expert.py          # Expert and SparseExpert classes
├── gating.py          # TopKGating and load balancing
├── moe_layer.py       # MoELayer and EfficientMoELayer
└── __init__.py        # Exports

src/models/
├── encoder_moe.py     # TransformerEncoderMoE
├── embeddings_moe.py  # EmbeddingModelMoE
└── __init__.py        # Updated exports

notebooks/
└── moe_vs_dense_comparison.ipynb  # Complete comparison
```

### Code Quality

- ✅ Fully documented
- ✅ Type hints
- ✅ Modular design
- ✅ GPU compatible (MPS/CUDA)
- ✅ Production-ready

## 🎯 Research Questions to Answer

1. **Does MoE improve embedding quality?**
   - Measure: Validation loss, similarity separation

2. **Do experts specialize?**
   - Check: Expert usage patterns, routing decisions

3. **Is the extra capacity worth it?**
   - Compare: Performance vs model size

4. **How does it scale?**
   - Try: More experts, larger models

## 🔍 Debugging Tips

### Issue: All tokens route to same expert
**Solution:** Increase load balancing weight or add more noise to gating

### Issue: MoE trains slower than Dense
**Expected:** MoE has more overhead due to routing
**Solution:** Use EfficientMoELayer for better batching

### Issue: Poor performance
**Check:**
1. Load balancing working? (expert usage stats)
2. Aux loss included in total loss?
3. Enough training data?
4. Learning rate appropriate?

## 📖 References

- [Switch Transformers](https://arxiv.org/abs/2101.03961) - Google's MoE for NLP
- [GShard](https://arxiv.org/abs/2006.16668) - Scaling MoE models
- [Mixtral](https://arxiv.org/abs/2401.04088) - Sparse MoE for LLMs

## ✅ Next Steps

After running the comparison notebook:

1. **Analyze results** - Which model won and why?
2. **Tune hyperparameters** - Try different configurations
3. **Scale up** - More experts, more data, more layers
4. **Domain-specific training** - Train on specific domains
5. **Expert analysis** - Understand what each expert learned

---

**Run the comparison now:**
```bash
uv run jupyter notebook notebooks/moe_vs_dense_comparison.ipynb
```

This is a **real research experiment** that will tell you if MoE helps for embeddings! 🚀
