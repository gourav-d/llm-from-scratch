# Module 04 - Complete Implementation Guide

**Purpose**: This document provides everything needed to recreate all examples and exercises from scratch.

---

## 📋 What Needs to Be Created

### Examples (5 remaining)
1. **example_02_self_attention.py** (~220 lines)
2. **example_03_multi_head.py** (~280 lines)
3. **example_04_positional.py** (~250 lines)
4. **example_05_transformer_block.py** (~280 lines)
5. **example_06_mini_gpt.py** (~470 lines)

### Exercises (3 total)
1. **exercise_01_attention.py** (~200 lines)
2. **exercise_02_self_attention.py** (~250 lines)
3. **exercise_03_transformer.py** (~300 lines)

---

## 🎯 Implementation Pattern (Based on example_01)

All files follow this structure:

```python
"""
Example/Exercise Title

Brief description of what this demonstrates.

What you'll see:
1. Point 1
2. Point 2
3. Point 3
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print("=" * 70)
print("TITLE")
print("=" * 70)

# ==============================================================================
# PART 1: Description
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Title")
print("=" * 70)

# Implementation with extensive comments
# C#/.NET analogies
# Educational print statements

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - What We Learned")
print("=" * 70)

print("""
✓ Key point 1
✓ Key point 2
✓ Key point 3

Next Steps:
- Next example/exercise
""")
```

---

## 📚 Content Specifications

### Example 02: Self-Attention Layer

**Core Components:**
```python
class SelfAttention:
    def __init__(self, d_model):
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1

    def forward(self, X):
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        scores = Q @ K.T / np.sqrt(d_k)
        weights = softmax(scores)
        output = weights @ V
        return output, weights
```

**Visualizations:**
- Attention weight heatmap
- Bar chart showing how one word attends to others

**Key Concepts:**
- Learned weight matrices W_q, W_k, W_v
- Projection from input to Q, K, V
- Context-aware representations

---

### Example 03: Multi-Head Attention

**Core Components:**
```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Weight matrices for all heads

    def split_heads(self, x):
        # Reshape to (batch, num_heads, seq_len, d_k)

    def forward(self, X):
        # 1. Linear projections
        # 2. Split into heads
        # 3. Apply attention per head
        # 4. Concatenate heads
        # 5. Output projection
```

**Visualizations:**
- Grid of heatmaps (one per head)
- Head specialization analysis

---

### Example 04: Positional Encoding

**Core Function:**
```python
def positional_encoding(max_seq_len, d_model):
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    PE = np.zeros((max_seq_len, d_model))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)
    return PE
```

**Visualizations:**
- Full positional encoding heatmap
- Encoding curves for different dimensions
- Position fingerprints
- 2D position space
- Frequency spectrum

---

### Example 05: Transformer Block

**Core Components:**
```python
class LayerNorm:
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + epsilon)
        return gamma * x_norm + beta

class FeedForward:
    def forward(self, x):
        hidden = np.maximum(0, x @ W1 + b1)  # ReLU
        return hidden @ W2 + b2

class TransformerBlock:
    def forward(self, x):
        # Sublayer 1: Attention + Residual + Norm
        attn_output = self.attention.forward(x)
        x = self.norm1.forward(x + attn_output)

        # Sublayer 2: FFN + Residual + Norm
        ffn_output = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_output)
        return x
```

---

### Example 06: Mini-GPT (CAPSTONE)

**Complete Architecture:**
```python
class MiniGPT:
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len):
        # Token embeddings
        # Positional encoding
        # Transformer blocks (stacked)
        # Language modeling head

    def forward(self, token_ids):
        # 1. Token embeddings
        # 2. Add positional encoding
        # 3. Causal mask
        # 4. Pass through transformer blocks
        # 5. Final layer norm
        # 6. LM head projection
        return logits

    def generate(self, start_tokens, max_new_tokens, strategy):
        # Greedy, sampling, or top-k generation
```

**Features:**
- Causal masking for autoregressive generation
- Multiple generation strategies
- Parameter counting
- Model statistics

---

## 🎓 Exercise Specifications

### Exercise 01: Implementing Attention

**TODOs:**
1. Compute attention scores: `scores = (Q @ K.T) / sqrt(d_k)`
2. Implement softmax function
3. Apply softmax to get weights
4. Compute output: `output = weights @ V`
5. Visualize attention weights

**Solutions:** Commented out with `# SOLUTION (uncomment...)`

---

### Exercise 02: Building Self-Attention

**TODOs:**
1. Initialize W_q, W_k, W_v matrices
2. Project input to Q, K, V
3. Implement self-attention forward pass
4. Create SelfAttention class
5. BONUS: Multi-head capability

---

### Exercise 03: Complete Transformer

**TODOs:**
1. Implement LayerNorm class
2. Implement FeedForward class
3. Build TransformerBlock class
4. Stack multiple blocks
5. BONUS: Analyze representation evolution

---

## 🚀 Quick Recreation Commands

When ready to recreate all files, say:

**"Create Module 4 examples and exercises following the specifications in COMPLETE_IMPLEMENTATION_GUIDE.md"**

The AI will:
1. Read this guide
2. Read example_01_attention.py as template
3. Read lesson documentation for concepts
4. Generate all 8 files following the pattern

---

## ✅ Verification Steps

After creation:
```bash
# Test examples
python examples/example_01_attention.py
python examples/example_02_self_attention.py
python examples/example_03_multi_head.py
python examples/example_04_positional.py
python examples/example_05_transformer_block.py
python examples/example_06_mini_gpt.py

# Test exercises
python exercises/exercise_01_attention.py
python exercises/exercise_02_self_attention.py
python exercises/exercise_03_transformer.py
```

All should run without errors and display visualizations.

---

## 📖 Reference Documents

When recreating, refer to:
1. **Lesson files** (01-06) - Concepts and theory
2. **example_01_attention.py** - Code structure pattern
3. **CLAUDE.md** - Teaching standards
4. **This guide** - Component specifications

Everything needed is in the documentation!

---

**This guide ensures Module 4 can be completed even after clearing context.** ✅
