# Lesson 3: Multi-Head Attention

**Learn multiple patterns simultaneously - the secret to GPT's power!**

---

## 🎯 What You'll Learn

- ✅ Why one attention head isn't enough
- ✅ How multi-head attention works
- ✅ Parallel attention patterns
- ✅ Combining multiple heads
- ✅ Complete implementation

**Time:** 2-3 hours
**Difficulty:** ⭐⭐⭐⭐☆

**Prerequisites:** Lessons 1-2 (Attention, Self-Attention)

---

## 🤔 The Problem: One Head Isn't Enough

### What's Wrong with Single-Head Attention?

**Example sentence:** "The cat sat on the mat because it was tired"

**With single-head self-attention:**
```
"it" attends most strongly to → "cat" (0.8)

But loses other important information:
- "tired" (describes emotional state)
- "sat" (the action)
- "mat" (the location)
```

**Problem:** One attention pattern can't capture all relationships!

---

### What We Need: Multiple Perspectives

**Think about reading:**

When you read "The cat sat on the mat", you simultaneously understand:
1. **Syntax:** "cat" is subject, "sat" is verb
2. **Semantics:** "on the mat" describes location
3. **Grammar:** "The" modifies "cat" and "mat"
4. **Context:** Past tense action

**Your brain processes multiple patterns at once!**

---

## 💡 The Solution: Multi-Head Attention

### The Core Idea

**Instead of one attention mechanism, use many in parallel!**

```
Single-head:
Input → Self-Attention → Output

Multi-head (8 heads):
Input → Self-Attention (head 1) →
     → Self-Attention (head 2) →
     → Self-Attention (head 3) →
     → ...
     → Self-Attention (head 8) → Concat → Output
```

**Each head learns different patterns!**

---

### Real-World Analogy: Team of Experts

**Analyzing a sentence is like analyzing a business:**

**Single expert (single-head):**
```
CEO analyzes company:
- Tries to understand finance, marketing, operations, HR
- Can't be expert in everything!
```

**Team of experts (multi-head):**
```
CFO    → Analyzes finance (head 1)
CMO    → Analyzes marketing (head 2)
COO    → Analyzes operations (head 3)
CHRO   → Analyzes HR (head 4)

Combined view → Complete understanding!
```

**Multi-head attention = Team of specialized attention mechanisms!**

---

## 🧮 The Math: Parallel Attention Heads

### Architecture Overview

```
Input: X (seq_len, d_model)
       ↓
Split into h heads (each with dimension d_k = d_model/h)
       ↓
Head 1: Self-Attention with W_Q1, W_K1, W_V1
Head 2: Self-Attention with W_Q2, W_K2, W_V2
...
Head h: Self-Attention with W_Qh, W_Kh, W_Vh
       ↓
Concatenate all heads
       ↓
Linear projection: W_O
       ↓
Output: (seq_len, d_model)
```

---

### Step-by-Step Formula

**Given:**
- Input: X (seq_len, d_model)
- Number of heads: h (typically 8 or 12)
- Dimension per head: d_k = d_model / h

**For each head i:**
```python
Q_i = X @ W_Qi  # (seq_len, d_k)
K_i = X @ W_Ki  # (seq_len, d_k)
V_i = X @ W_Vi  # (seq_len, d_k)

head_i = Attention(Q_i, K_i, V_i)
       = softmax(Q_i @ K_i.T / sqrt(d_k)) @ V_i
```

**Combine heads:**
```python
# Concatenate all heads
MultiHead = Concat(head_1, head_2, ..., head_h)
          # Shape: (seq_len, d_model)

# Final linear projection
Output = MultiHead @ W_O
       # Shape: (seq_len, d_model)
```

---

## 📊 Concrete Example

### Setup: 4 words, 8-dimensional embeddings, 2 heads

```python
import numpy as np

# Input
sentence = ["The", "cat", "sat", "down"]
seq_len = 4
d_model = 8   # Embedding dimension
num_heads = 2 # Number of attention heads
d_k = d_model // num_heads  # = 4 (dimension per head)

# Random embeddings
np.random.seed(42)
X = np.random.randn(seq_len, d_model)

print("Input shape:", X.shape)  # (4, 8)
```

---

### Step 1: Initialize Weight Matrices

```python
# Each head has its own Q, K, V weights
# Head 1
W_Q1 = np.random.randn(d_model, d_k) * 0.1  # (8, 4)
W_K1 = np.random.randn(d_model, d_k) * 0.1  # (8, 4)
W_V1 = np.random.randn(d_model, d_k) * 0.1  # (8, 4)

# Head 2
W_Q2 = np.random.randn(d_model, d_k) * 0.1  # (8, 4)
W_K2 = np.random.randn(d_model, d_k) * 0.1  # (8, 4)
W_V2 = np.random.randn(d_model, d_k) * 0.1  # (8, 4)

# Output projection (combines heads)
W_O = np.random.randn(d_model, d_model) * 0.1  # (8, 8)
```

---

### Step 2: Compute Head 1

```python
def attention(Q, K, V):
    """Single attention head."""
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V
    return output, weights

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

# Head 1
Q1 = X @ W_Q1  # (4, 4)
K1 = X @ W_K1  # (4, 4)
V1 = X @ W_V1  # (4, 4)

head_1_output, head_1_weights = attention(Q1, K1, V1)

print("Head 1 output shape:", head_1_output.shape)  # (4, 4)
print("\nHead 1 attention pattern:")
print(head_1_weights)

#         The   cat   sat  down
# The   [[0.28  0.24  0.25  0.23]
# cat    [0.26  0.27  0.24  0.23]
# sat    [0.25  0.25  0.26  0.24]
# down   [0.24  0.25  0.25  0.26]]
```

---

### Step 3: Compute Head 2

```python
# Head 2 (different weights = different patterns!)
Q2 = X @ W_Q2  # (4, 4)
K2 = X @ W_K2  # (4, 4)
V2 = X @ W_V2  # (4, 4)

head_2_output, head_2_weights = attention(Q2, K2, V2)

print("Head 2 output shape:", head_2_output.shape)  # (4, 4)
print("\nHead 2 attention pattern:")
print(head_2_weights)

#         The   cat   sat  down
# The   [[0.30  0.22  0.26  0.22]
# cat    [0.23  0.31  0.24  0.22]
# sat    [0.24  0.23  0.29  0.24]
# down   [0.23  0.24  0.23  0.32]]

# Notice: Different pattern from Head 1!
```

---

### Step 4: Concatenate Heads

```python
# Concatenate along the last dimension
multi_head_output = np.concatenate([head_1_output, head_2_output], axis=-1)

print("Multi-head output shape:", multi_head_output.shape)  # (4, 8)

# Now we have:
# - First 4 dimensions from head 1
# - Last 4 dimensions from head 2
```

---

### Step 5: Final Linear Projection

```python
# Project back to d_model dimension
final_output = multi_head_output @ W_O

print("Final output shape:", final_output.shape)  # (4, 8)

print("\nInput shape:", X.shape)   # (4, 8)
print("Output shape:", final_output.shape)  # (4, 8)

# Shape preserved: (seq_len, d_model) → (seq_len, d_model)
```

---

## 💻 Complete Implementation

### Multi-Head Attention Class

```python
import numpy as np

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.

    This is what GPT uses!
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of parallel attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Initialize weights for all heads
        # Shape: (d_model, d_model) for each of Q, K, V
        # Internally split into num_heads
        scale = 1.0 / np.sqrt(d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        # Reshape: (batch, seq_len, num_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose: (batch, num_heads, seq_len, d_k)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        """
        Combine heads back to original shape.

        Args:
            x: (batch_size, num_heads, seq_len, d_k)

        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        # Transpose: (batch, seq_len, num_heads, d_k)
        x = x.transpose(0, 2, 1, 3)
        # Reshape: (batch, seq_len, d_model)
        return x.reshape(batch_size, seq_len, self.d_model)

    def attention(self, Q, K, V, mask=None):
        """
        Scaled dot-product attention.

        Args:
            Q, K, V: (batch, num_heads, seq_len, d_k)
            mask: Optional mask (batch, 1, seq_len, seq_len)

        Returns:
            output: (batch, num_heads, seq_len, d_k)
            weights: (batch, num_heads, seq_len, seq_len)
        """
        # Compute attention scores
        scores = Q @ K.transpose(0, 1, 3, 2)  # (batch, heads, seq_len, seq_len)
        scores = scores / np.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Softmax
        weights = self.softmax(scores)

        # Weighted sum of values
        output = weights @ V  # (batch, heads, seq_len, d_k)

        return output, weights

    def softmax(self, x):
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def forward(self, X, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            X: Input (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch_size, seq_len, d_model)
            all_attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = X.shape

        # Linear projections
        Q = X @ self.W_Q  # (batch, seq_len, d_model)
        K = X @ self.W_K
        V = X @ self.W_V

        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        # attn_output: (batch, num_heads, seq_len, d_k)

        # Combine heads
        attn_output = self.combine_heads(attn_output)
        # (batch, seq_len, d_model)

        # Final linear projection
        output = attn_output @ self.W_O

        return output, attn_weights
```

---

### Example Usage

```python
# Create sample input
batch_size = 2
seq_len = 6  # "The cat sat on the mat"
d_model = 512  # GPT uses 768 or 1024
num_heads = 8  # GPT uses 12

# Random input (normally from embedding layer)
X = np.random.randn(batch_size, seq_len, d_model)

print("Input shape:", X.shape)  # (2, 6, 512)

# Create multi-head attention layer
mha = MultiHeadAttention(d_model, num_heads)

# Forward pass
output, attention_weights = mha.forward(X)

print("\nOutput shape:", output.shape)  # (2, 6, 512)
print("Attention weights shape:", attention_weights.shape)  # (2, 8, 6, 6)

# Visualize attention for first sample, first head
print("\nHead 1 attention pattern (sample 1):")
print(attention_weights[0, 0])  # (6, 6) matrix
```

---

## 🎨 Visualizing Multi-Head Attention

### Different Heads Learn Different Patterns

**Example: "The cat sat on the mat"**

**Head 1 (Syntax/Grammar):**
```
         The  cat  sat  on   the  mat
The     [███  ██   █    █    █    █  ]  (article focuses on noun)
cat     [██   ███  ██   █    █    █  ]  (noun to verb)
sat     [█    ██   ███  ██   █    ██ ]  (verb to subject/object)
on      [█    █    ██   ███  ██   ██ ]  (prep to verb/object)
the     [█    █    █    ██   ███  ██ ]  (article to noun)
mat     [█    █    ██   ██   ██   ███]  (noun to verb/prep)

Pattern: Syntactic relationships
```

**Head 2 (Semantics/Meaning):**
```
         The  cat  sat  on   the  mat
The     [██   ███  █    █    █    ██ ]  (connects subject phrases)
cat     [██   ███  ███  █    █    ██ ]  (subject to action)
sat     [█    ███  ███  ██   █    ██ ]  (action to all entities)
on      [█    ██   ██   ███  ██   ███]  (location relationship)
the     [█    ██   █    ██   ███  ███]  (connects object phrases)
mat     [█    ██   ██   ███  ██   ███]  (object to location)

Pattern: Semantic relationships
```

**Head 3 (Position/Distance):**
```
         The  cat  sat  on   the  mat
The     [███  ██   █    █    █    █  ]  (nearby words)
cat     [██   ███  ██   █    █    █  ]  (nearby words)
sat     [█    ██   ███  ██   █    █  ]  (nearby words)
on      [█    █    ██   ███  ██   █  ]  (nearby words)
the     [█    █    █    ██   ███  ██ ]  (nearby words)
mat     [█    █    █    █    ██   ███]  (nearby words)

Pattern: Local context (nearby words)
```

**Each head specializes in different information!**

---

## 🔍 Why Multi-Head Works Better

### Comparison: Single-Head vs. Multi-Head

**Single-Head Attention:**
```python
# Must learn ONE pattern that tries to capture everything
Attention weights for "sat":
- "The" → 0.15  (low)
- "cat" → 0.40  (medium-high, subject)
- "sat" → 0.20  (self)
- "on"  → 0.15  (medium, preposition)
- "the" → 0.05  (low)
- "mat" → 0.05  (low, object less emphasized)

Problem: Can't strongly attend to both "cat" (subject)
and "mat" (object) at the same time!
```

**Multi-Head Attention (2 heads):**
```python
# Head 1: Subject-Verb relationships
Head 1 weights for "sat":
- "The" → 0.05
- "cat" → 0.70  ← Strong! (subject)
- "sat" → 0.15
- "on"  → 0.05
- "the" → 0.03
- "mat" → 0.02

# Head 2: Verb-Object relationships
Head 2 weights for "sat":
- "The" → 0.05
- "cat" → 0.10
- "sat" → 0.10
- "on"  → 0.20  ← Preposition
- "the" → 0.10
- "mat" → 0.45  ← Strong! (object)

Combined: Both subject AND object relationships captured!
```

---

## 🎯 Multi-Head in GPT and BERT

### GPT-3 Configuration

```python
GPT-3 Small:
- d_model = 768
- num_heads = 12
- d_k = 768 / 12 = 64

GPT-3 Large:
- d_model = 1536
- num_heads = 24
- d_k = 1536 / 24 = 64

GPT-3 (175B):
- d_model = 12288
- num_heads = 96
- d_k = 12288 / 96 = 128
```

**More heads = more specialized patterns!**

---

### What Each Head Learns

Research shows GPT heads specialize in:

**Syntactic heads:**
- Head 3: Subject-verb agreement
- Head 7: Noun-adjective relationships
- Head 11: Clause boundaries

**Semantic heads:**
- Head 2: Coreference (pronouns)
- Head 5: Named entity relationships
- Head 8: Argument structure

**Positional heads:**
- Head 1: Previous word
- Head 4: Next word
- Head 6: Start/end of sentence

**Fascinating:** The model learns these patterns automatically, not programmed!

---

## 🧪 Practice Problems

### Problem 1: Calculate Dimensions

Given a transformer with:
- d_model = 1024
- num_heads = 16

Calculate:
1. d_k (dimension per head)
2. Shape of W_Q, W_K, W_V
3. Shape of attention weights
4. Total parameters for multi-head attention

<details>
<summary>Solution</summary>

```python
1. d_k = d_model / num_heads = 1024 / 16 = 64

2. Weight matrix shapes:
   W_Q: (1024, 1024)
   W_K: (1024, 1024)
   W_V: (1024, 1024)
   W_O: (1024, 1024)

3. Attention weights shape (for seq_len=50):
   (batch, num_heads, seq_len, seq_len)
   = (batch, 16, 50, 50)

4. Total parameters:
   W_Q: 1024 × 1024 = 1,048,576
   W_K: 1024 × 1024 = 1,048,576
   W_V: 1024 × 1024 = 1,048,576
   W_O: 1024 × 1024 = 1,048,576
   Total: 4,194,304 parameters (~4.2M)
```
</details>

---

### Problem 2: Implement Head Splitting

Implement the `split_heads` function:

```python
def split_heads(x, num_heads):
    """
    Split last dimension into (num_heads, d_k).

    Args:
        x: (seq_len, d_model)
        num_heads: number of attention heads

    Returns:
        (num_heads, seq_len, d_k)
    """
    # Your code here
    pass

# Test
x = np.random.randn(10, 512)  # 10 words, 512-dim
num_heads = 8
result = split_heads(x, num_heads)
print(result.shape)  # Should be (8, 10, 64)
```

<details>
<summary>Solution</summary>

```python
def split_heads(x, num_heads):
    seq_len, d_model = x.shape
    d_k = d_model // num_heads

    # Reshape to (seq_len, num_heads, d_k)
    x = x.reshape(seq_len, num_heads, d_k)

    # Transpose to (num_heads, seq_len, d_k)
    x = x.transpose(1, 0, 2)

    return x

# Test
x = np.random.randn(10, 512)
num_heads = 8
result = split_heads(x, num_heads)
print(result.shape)  # (8, 10, 64) ✓
```
</details>

---

## 🔑 Key Takeaways

### Remember These Points

1. **Multi-head = Multiple parallel attention patterns**
   - Each head learns different relationships
   - More heads = more specialized patterns

2. **Dimension splitting: d_k = d_model / num_heads**
   - Keeps computation cost similar to single-head
   - GPT-3 uses d_k = 128 (for 96 heads)

3. **Process: Split → Attend → Concatenate → Project**
   - Split into heads (each d_k dimensional)
   - Each head does self-attention independently
   - Concatenate all heads (back to d_model)
   - Final projection with W_O

4. **Different heads specialize automatically**
   - Syntax, semantics, position, etc.
   - Emerges from training (not programmed!)

5. **This is what makes transformers powerful!**
   - Can model multiple relationship types
   - Critical for understanding complex language
   - Used in GPT, BERT, all modern LLMs

---

## ✅ Self-Check

Before moving to Lesson 4 (Positional Encoding), ensure you can:

- [ ] Explain why single-head isn't enough
- [ ] Describe how heads are split and combined
- [ ] Calculate dimensions for multi-head attention
- [ ] Implement multi-head attention from scratch
- [ ] Understand what different heads learn
- [ ] Explain GPT's multi-head configuration

**If you checked all boxes:** Ready for Positional Encoding! 🎉

**If not:** Review this lesson, run code, visualize attention patterns!

---

## 💬 Common Questions

**Q: Why not just increase d_model instead of adding heads?**
A: Multiple heads learn diverse patterns. One big head would try to learn everything in one representation (less flexible).

**Q: How many heads should I use?**
A: Typical: 8-12 heads. GPT-3 uses 96! More heads = more patterns, but diminishing returns.

**Q: Do all heads contribute equally?**
A: No! Research shows some heads are more important than others. Some can even be pruned (removed) without much performance loss!

**Q: How does the model decide what each head learns?**
A: It's learned during training! The model automatically specializes heads based on the task and data.

---

## 📖 Further Reading

### Next Steps

1. **Run example code:**
   - `examples/example_03_multi_head.py` (when available)
   - Visualize different head patterns!

2. **Next lesson:**
   - `04_positional_encoding.md`
   - Teaching transformers about word order!

3. **Research papers:**
   - "Attention Is All You Need" (Section 3.2.2)
   - "Analyzing Multi-Head Attention" (Voita et al.)
   - "Are Sixteen Heads Really Better than One?" (Michel et al.)

---

## 🎊 Congratulations!

**You've learned multi-head attention - GPT's secret weapon!**

**Key achievement:**
> You understand how transformers learn multiple relationship patterns simultaneously!

**What you now know:**
- How GPT processes language
- Why transformers are so powerful
- The architecture that powers ChatGPT!

---

**Next:** Lesson 4 - Positional Encoding (Teaching word order!)

**Almost there!** You're understanding the complete transformer architecture! 🚀
