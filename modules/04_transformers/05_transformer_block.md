# Lesson 5: Complete Transformer Block

**Putting it all together - the building block of GPT and BERT!**

---

## 🎯 What You'll Learn

- ✅ The complete transformer block architecture
- ✅ Feed-forward networks in transformers
- ✅ Layer normalization
- ✅ Residual connections (skip connections)
- ✅ Building a complete transformer layer

**Time:** 2-3 hours
**Difficulty:** ⭐⭐⭐⭐⭐

**Prerequisites:** Lessons 1-4 (All attention and positional encoding)

---

## 🏗️ The Complete Architecture

### Transformer Block Components

**A transformer block has 4 main components:**

```
Input (with positional encoding)
  ↓
1. Multi-Head Self-Attention ← Lesson 3
  ↓
2. Add & Normalize (Residual + LayerNorm)
  ↓
3. Feed-Forward Network (NEW!)
  ↓
4. Add & Normalize (Residual + LayerNorm)
  ↓
Output (ready for next block or final layer)
```

**Let's learn the missing pieces!**

---

## 🧠 Feed-Forward Networks (FFN)

### What Are They?

**Simple 2-layer neural network applied to each position independently:**

```python
FFN(x) = max(0, x @ W1 + b1) @ W2 + b2
       = ReLU(x @ W1 + b1) @ W2 + b2
```

**Where:**
- W1: (d_model, d_ff) - expands dimension
- W2: (d_ff, d_model) - projects back
- d_ff = 4 × d_model (typical in transformers)
- ReLU: max(0, x) activation

---

### Why Do We Need FFN?

**Attention alone isn't enough!**

**What attention does:**
- ✅ Aggregates information from other positions
- ✅ Learns relationships between words
- ❌ But it's just weighted averaging!

**What FFN adds:**
- ✅ Non-linear transformations
- ✅ Position-wise processing
- ✅ Adds model capacity (most parameters here!)

**Together: Attention = communication, FFN = computation!**

---

### Analogy: Team Meeting

```
Multi-Head Attention = Team Discussion
- Everyone shares information
- Each person learns from others
- Communication phase

Feed-Forward Network = Individual Thinking
- Each person processes independently
- Thinks about what they learned
- Computation phase

Both are needed for effective teamwork!
```

---

## 🔗 Residual Connections

### The Problem: Deep Networks Are Hard to Train

**Without residual connections:**

```
Block 1 → Block 2 → Block 3 → ... → Block 12

Problem: Gradients vanish!
- Block 12's gradient is strong
- Block 6's gradient is weak
- Block 1's gradient ~0 (can't learn!)
```

**GPT-3 has 96 layers - impossible to train without residuals!**

---

### The Solution: Skip Connections

**Add input directly to output:**

```python
# Without residual
output = Block(input)

# With residual (skip connection)
output = Block(input) + input
         ↑              ↑
      transformed    original
```

**Benefit:**
- Gradient can flow directly through addition
- Each layer only learns the "residual" (difference)
- Much easier to train deep networks!

---

### Visual Representation

```
Input x
  │
  ├────────────────────┐  (skip connection)
  │                    │
  ↓                    │
Multi-Head           │
Attention            │
  │                    │
  ↓                    │
  └──────→ ADD ←───────┘
            ↓
          Output
```

**The input "skips" the block and gets added back!**

---

## 📊 Layer Normalization

### What Is Layer Norm?

**Normalize activations to have mean=0, variance=1:**

```python
LayerNorm(x) = γ * (x - mean) / std + β

Where:
- mean = average across features (for each sample)
- std = standard deviation across features
- γ, β = learned scale and shift parameters
```

---

### Why Normalize?

**Stabilizes training of deep networks:**

```python
Without LayerNorm:
- Activations can explode or vanish
- Learning rate is sensitive
- Training is unstable

With LayerNorm:
- Activations stay in reasonable range
- More stable gradients
- Faster convergence
```

---

### Layer Norm vs. Batch Norm

**C# developers: This is like input validation!**

| Feature | Batch Norm (CNNs) | Layer Norm (Transformers) |
|---------|-------------------|---------------------------|
| **Normalizes across** | Batch dimension | Feature dimension |
| **Works with** | Fixed batch sizes | Any batch size (even 1!) |
| **Used in** | CNNs (images) | Transformers (sequences) |
| **Benefits** | Faster training | Stable across batch sizes |

**Transformers use Layer Norm because sequence lengths vary!**

---

## 🏗️ Complete Transformer Block

### Architecture Diagram

```
Input (seq_len, d_model)
  ↓
[Positional Encoding added]
  ↓
  ├─────────────────────────────┐
  │                             │
  ↓                             │
Multi-Head Attention            │
  ↓                             │
  └──→ ADD ←────────────────────┘
       ↓
  Layer Norm
       ↓
  ├─────────────────────────────┐
  │                             │
  ↓                             │
Feed-Forward Network            │
  ↓                             │
  └──→ ADD ←────────────────────┘
       ↓
  Layer Norm
       ↓
Output (seq_len, d_model)
```

**This block is repeated N times (12 in GPT-2, 96 in GPT-3)!**

---

## 💻 Complete Implementation

### Transformer Block Class

```python
import numpy as np

class FeedForward:
    """
    Position-wise feed-forward network.

    Two linear layers with ReLU activation.
    """

    def __init__(self, d_model, d_ff=None):
        """
        Initialize FFN.

        Args:
            d_model: Model dimension
            d_ff: Hidden dimension (default: 4 * d_model)
        """
        self.d_model = d_model
        self.d_ff = d_ff or (4 * d_model)

        # Initialize weights
        scale = 1.0 / np.sqrt(d_model)
        self.W1 = np.random.randn(d_model, self.d_ff) * scale
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)

    def forward(self, X):
        """
        Forward pass.

        Args:
            X: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        # First layer with ReLU
        hidden = np.maximum(0, X @ self.W1 + self.b1)

        # Second layer (no activation)
        output = hidden @ self.W2 + self.b2

        return output


class LayerNorm:
    """
    Layer normalization.

    Normalizes across the feature dimension.
    """

    def __init__(self, d_model, eps=1e-6):
        """
        Initialize layer norm.

        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones(d_model)   # Scale
        self.beta = np.zeros(d_model)   # Shift

    def forward(self, X):
        """
        Normalize input.

        Args:
            X: (batch, seq_len, d_model)

        Returns:
            Normalized X with same shape
        """
        # Compute mean and std across feature dimension
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True)

        # Normalize
        X_norm = (X - mean) / (std + self.eps)

        # Scale and shift
        return self.gamma * X_norm + self.beta


class TransformerBlock:
    """
    Complete transformer block.

    This is the building block of GPT, BERT, and all transformers!
    """

    def __init__(self, d_model, num_heads, d_ff=None):
        """
        Initialize transformer block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: FFN hidden dimension (default: 4 * d_model)
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff or (4 * d_model)

        # Sub-layers
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, self.d_ff)

        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, X, mask=None):
        """
        Forward pass through transformer block.

        Args:
            X: Input (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output (batch, seq_len, d_model)
        """
        # 1. Multi-head attention with residual connection
        attn_output, attn_weights = self.attention.forward(X, mask)
        X = X + attn_output  # Residual connection
        X = self.norm1.forward(X)  # Layer normalization

        # 2. Feed-forward network with residual connection
        ff_output = self.feed_forward.forward(X)
        X = X + ff_output  # Residual connection
        X = self.norm2.forward(X)  # Layer normalization

        return X, attn_weights
```

---

### Usage Example

```python
# Configuration (GPT-2 Small)
batch_size = 2
seq_len = 10
d_model = 768
num_heads = 12
d_ff = 3072  # 4 * d_model

# Create input (normally from embedding layer)
X = np.random.randn(batch_size, seq_len, d_model)

print("Input shape:", X.shape)  # (2, 10, 768)

# Create transformer block
block = TransformerBlock(d_model, num_heads, d_ff)

# Forward pass
output, attn_weights = block.forward(X)

print("Output shape:", output.shape)  # (2, 10, 768)
print("Attention weights shape:", attn_weights.shape)  # (2, 12, 10, 10)

# Stack multiple blocks for deep transformer
num_layers = 12
X_current = X

for layer in range(num_layers):
    block = TransformerBlock(d_model, num_heads, d_ff)
    X_current, _ = block.forward(X_current)
    print(f"Layer {layer + 1} output shape:", X_current.shape)

# Final output ready for task-specific head (classification, generation, etc.)
```

---

## 📊 Parameter Count

### How Many Parameters in a Transformer Block?

**For d_model = 768, num_heads = 12, d_ff = 3072:**

```python
Multi-Head Attention:
- W_Q: 768 × 768 = 589,824
- W_K: 768 × 768 = 589,824
- W_V: 768 × 768 = 589,824
- W_O: 768 × 768 = 589,824
  Subtotal: 2,359,296 parameters

Feed-Forward Network:
- W1: 768 × 3072 = 2,359,296
- b1: 3072 = 3,072
- W2: 3072 × 768 = 2,359,296
- b2: 768 = 768
  Subtotal: 4,722,432 parameters

Layer Normalization (×2):
- γ1, β1: 768 × 2 = 1,536
- γ2, β2: 768 × 2 = 1,536
  Subtotal: 3,072 parameters

Total per block: 7,084,800 parameters (~7M)
```

**For GPT-2 (12 blocks): 12 × 7M ≈ 85M parameters!**

**Note: Most parameters are in FFN! (~67%)**

---

## 🎯 GPT vs. BERT: Block Differences

### GPT (Decoder-only)

```
Input (with positional encoding)
  ↓
Masked Multi-Head Attention ← Can't see future!
  ↓
Add & Norm
  ↓
Feed-Forward Network
  ↓
Add & Norm
  ↓
Output

Used for: Text generation, completion
```

---

### BERT (Encoder-only)

```
Input (with positional encoding)
  ↓
Multi-Head Attention ← Bidirectional (sees all words)
  ↓
Add & Norm
  ↓
Feed-Forward Network
  ↓
Add & Norm
  ↓
Output

Used for: Classification, understanding
```

---

### Full Transformer (Original Paper)

```
Encoder:                    Decoder:
Input                       Input
  ↓                          ↓
Multi-Head Attn            Masked Multi-Head Attn
  ↓                          ↓
Add & Norm                 Add & Norm
  ↓                          ↓
FFN                        Cross-Attention (to encoder)
  ↓                          ↓
Add & Norm                 Add & Norm
  ↓                          ↓
Output ─────────────────→  FFN
                            ↓
                          Add & Norm
                            ↓
                          Output

Used for: Translation, seq2seq tasks
```

---

## 🧪 Practice Problems

### Problem 1: Calculate Parameters

For a transformer with:
- d_model = 512
- num_heads = 8
- d_ff = 2048
- num_layers = 6

Calculate total parameters.

<details>
<summary>Solution</summary>

```python
Per block:
- Attention: 4 × (512 × 512) = 1,048,576
- FFN: (512 × 2048) + 2048 + (2048 × 512) + 512 = 2,099,712
- LayerNorm: 2 × (512 + 512) = 2,048
  Total per block: 3,150,336

For 6 layers: 6 × 3,150,336 = 18,902,016 (~19M parameters)

Plus embedding and output layers!
```
</details>

---

### Problem 2: Implement Pre-Norm

Modern transformers use "Pre-Norm" (normalize before sub-layer):

```python
# Post-Norm (original paper)
output = LayerNorm(X + SubLayer(X))

# Pre-Norm (modern, more stable)
output = X + SubLayer(LayerNorm(X))
```

Modify the TransformerBlock class to use Pre-Norm.

<details>
<summary>Solution</summary>

```python
def forward_prenorm(self, X, mask=None):
    """Pre-norm variant (more stable training)."""
    # Attention with pre-norm
    X_norm = self.norm1.forward(X)
    attn_output, attn_weights = self.attention.forward(X_norm, mask)
    X = X + attn_output

    # FFN with pre-norm
    X_norm = self.norm2.forward(X)
    ff_output = self.feed_forward.forward(X_norm)
    X = X + ff_output

    return X, attn_weights
```
</details>

---

## 🔑 Key Takeaways

### Remember These Points

1. **Transformer block = Attention + FFN + Residuals + LayerNorm**
   - 4 key components working together
   - Each has a specific role

2. **FFN adds non-linear computation**
   - Attention aggregates information
   - FFN processes it
   - Both are necessary!

3. **Residual connections enable deep networks**
   - Direct gradient flow
   - Each layer learns residual (difference)
   - Critical for 12+ layers

4. **Layer Norm stabilizes training**
   - Normalizes activations
   - More stable than batch norm for sequences
   - Applied after each sub-layer

5. **Most parameters are in FFN**
   - d_ff = 4 × d_model (typically)
   - FFN has ~67% of total parameters
   - This is where model capacity comes from!

---

## ✅ Self-Check

Before moving to Lesson 6 (Complete Transformer), ensure you can:

- [ ] Explain all 4 components of transformer block
- [ ] Describe role of feed-forward networks
- [ ] Understand residual connections
- [ ] Implement layer normalization
- [ ] Build complete transformer block from scratch
- [ ] Compare GPT vs. BERT block structure

**If you checked all boxes:** Ready for the complete transformer! 🎉

**If not:** Review this lesson, implement each component separately!

---

## 💬 Common Questions

**Q: Why is d_ff = 4 × d_model?**
A: Empirically found to work well! Gives model more capacity. Some modern models use different ratios (2×, 8×).

**Q: Can I use GELU instead of ReLU?**
A: Yes! GPT uses GELU activation. BERT uses GELU too. More modern than ReLU for transformers.

**Q: Why normalize after addition, not before?**
A: Both work! "Pre-norm" (before) is more stable. "Post-norm" (after) is original but can be less stable for deep networks.

**Q: What if I remove residual connections?**
A: Training becomes very difficult/impossible for deep networks. Gradients vanish rapidly.

---

## 📖 Further Reading

### Next Steps

1. **Run example code:**
   - `examples/example_05_transformer_block.py` (when available)
   - Build and test complete blocks!

2. **Next lesson:**
   - `06_complete_transformer.md`
   - Stack blocks + embeddings + output layer!
   - Build mini-GPT!

3. **Research papers:**
   - "Attention Is All You Need" (Section 3.1-3.3)
   - "On Layer Normalization in the Transformer Architecture"
   - "Understanding the Difficulty of Training Transformers"

---

## 🎊 Congratulations!

**You've learned the complete transformer block!**

**Key achievement:**
> You understand every component of the architecture that powers modern AI!

**Progress:**
- ✅ Attention mechanism
- ✅ Self-attention
- ✅ Multi-head attention
- ✅ Positional encoding
- ✅ Complete transformer block
- ⏳ Next: Full model architecture!

**One more lesson!** You're about to understand the complete GPT architecture! 🚀

---

**Next:** Lesson 6 - Complete Transformer Architecture (GPT, BERT, and beyond!)

**Almost there!** The complete picture awaits! 💪
