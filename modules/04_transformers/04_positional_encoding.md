# Lesson 4: Positional Encoding

**Teaching transformers about word order using sine and cosine waves!**

---

## 🎯 What You'll Learn

- ✅ Why transformers need positional information
- ✅ The problem with pure attention
- ✅ Sinusoidal positional encoding
- ✅ How it works mathematically
- ✅ Implementation from scratch

**Time:** 2 hours
**Difficulty:** ⭐⭐⭐☆☆

**Prerequisites:** Lessons 1-3 (Attention mechanisms)

---

## 🤔 The Problem: Order Blindness

### Attention Doesn't Know Word Order!

**Critical problem:**

```python
Sentence 1: "The cat sat on the mat"
Sentence 2: "The mat sat on the cat"

With pure self-attention:
→ SAME attention patterns!
→ SAME output!

But meanings are completely different!
```

**Why does this happen?**

---

### Self-Attention is Permutation-Invariant

**Remember the attention formula:**
```python
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

**This formula doesn't care about position!**

**Example:**
```python
Words: ["The", "cat", "sat"]
Shuffled: ["cat", "The", "sat"]

# If we shuffle input AND shuffle output the same way:
# → Attention gives IDENTICAL results!

# Attention is like a "bag of words" with fancy averaging!
```

---

### Real-World Impact

**Without positional information:**

```
Input: "dog bites man"
Attention sees: {dog, bites, man} (unordered set)
Output: Could mean "dog bites man" OR "man bites dog"!

THIS IS A PROBLEM!
```

**We need to tell the model WHERE each word is!**

---

## 💡 The Solution: Positional Encoding

### The Core Idea

**Add position information to embeddings:**

```python
# Without positional encoding
word_embedding["cat"] = [0.2, 0.8, 0.3, ...]

# With positional encoding
# Position 0: "The"
word_at_pos_0 = word_embedding + positional_encoding(0)

# Position 1: "cat"
word_at_pos_1 = word_embedding + positional_encoding(1)

# Position 2: "sat"
word_at_pos_2 = word_embedding + positional_encoding(2)

# Now "cat" at position 1 is different from "cat" at position 5!
```

**Same word + different position = different representation!**

---

### Design Requirements

**Good positional encoding should:**

1. ✅ **Unique:** Different positions get different encodings
2. ✅ **Consistent:** Same position always gets same encoding
3. ✅ **Bounded:** Values don't grow unboundedly
4. ✅ **Generalizable:** Can handle sequences longer than seen in training
5. ✅ **Smooth:** Nearby positions have similar encodings

**How do we achieve all this?**

---

## 🌊 Sinusoidal Positional Encoding

### The Famous Formula

**For position `pos` and dimension `i`:**

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Where:**
- pos = position in sequence (0, 1, 2, ...)
- i = dimension index (0, 1, 2, ..., d_model/2)
- d_model = embedding dimension

**Each position gets a unique pattern of sine/cosine values!**

---

### Why Sine and Cosine?

**Analogy: Clock Hands**

```
Position encoding is like a clock with MANY hands:

Hand 1 (fast):  completes circle every 2 positions
Hand 2:         completes circle every 4 positions
Hand 3:         completes circle every 8 positions
...
Hand n (slow):  completes circle every 10000 positions

Each position = unique combination of hand positions!
```

**Benefits:**
- ✅ Unique encoding for each position
- ✅ Values stay between -1 and 1 (bounded)
- ✅ Smooth transitions between positions
- ✅ Can extrapolate to longer sequences

---

## 📊 Visual Understanding

### Example: d_model = 4, max_length = 8

```python
import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(max_len, d_model):
    """
    Create positional encoding matrix.

    Args:
        max_len: Maximum sequence length
        d_model: Embedding dimension

    Returns:
        PE: (max_len, d_model) matrix
    """
    PE = np.zeros((max_len, d_model))

    for pos in range(max_len):
        for i in range(0, d_model, 2):
            # Calculate the denominators
            denominator = np.power(10000, (2 * i) / d_model)

            # Apply sin to even indices
            PE[pos, i] = np.sin(pos / denominator)

            # Apply cos to odd indices
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / denominator)

    return PE

# Generate encoding
PE = get_positional_encoding(max_len=8, d_model=4)

print("Positional Encoding Matrix:")
print("Shape:", PE.shape)  # (8, 4)
print("\nValues:")
print(PE)
```

**Output:**
```
Position  Dim0(sin)  Dim1(cos)  Dim2(sin)  Dim3(cos)
0         0.00       1.00       0.00       1.00
1         0.84       0.54       0.01       1.00
2         0.91      -0.42       0.02       1.00
3         0.14      -0.99       0.03       1.00
4        -0.76      -0.65       0.04       1.00
5        -0.96       0.28       0.05       1.00
6        -0.28       0.96       0.06       1.00
7         0.66       0.75       0.07       1.00

Notice:
- Each row (position) is unique!
- Values oscillate between -1 and 1
- Different dimensions have different frequencies
```

---

### Heatmap Visualization

```
Positional Encoding Heatmap (8 positions × 4 dimensions):

Dim →  0     1     2     3
Pos ↓
0     [    ███              ███ ]
1     [███  ██              ███ ]
2     [███      █           ███ ]
3     [    ██████           ███ ]
4     [█   █████     █     ███ ]
5     [██       ██   █     ███ ]
6     [    ███       █     ███ ]
7     [██   ███       █    ███ ]

Legend: Dark = +1, Light = -1, Medium = ~0

Observations:
- Column 0,1: High frequency (many changes)
- Column 2,3: Low frequency (smooth changes)
- Each row (position) has unique pattern!
```

---

## 🧮 Mathematical Deep Dive

### Frequency Decomposition

**Each dimension pair (sin, cos) has a different wavelength:**

```python
For d_model = 512:

Dimension 0-1:
- Wavelength = 2π × 10000^(0/512) = 2π
- Changes very quickly (every ~6 positions)

Dimension 10-11:
- Wavelength = 2π × 10000^(10/512) = ~8π
- Changes moderately

Dimension 510-511:
- Wavelength = 2π × 10000^(510/512) ≈ 62,832
- Changes very slowly

Together: Unique fingerprint for each position!
```

---

### Why This Formula Works

**Key properties:**

**1. Linear relationships:**
```python
# For any fixed offset k:
PE(pos + k) can be expressed as linear function of PE(pos)

# This means the model can learn to attend to relative positions!
PE(pos + k, 2i) = sin((pos + k) / 10000^(2i/d))
                = sin(pos/10000^(2i/d)) * cos(k/10000^(2i/d))
                  + cos(pos/10000^(2i/d)) * sin(k/10000^(2i/d))
                = A * PE(pos, 2i) + B * PE(pos, 2i+1)

Where A, B are constants (depend only on k and i)!
```

**This is why transformers can learn relative positions!**

---

**2. Boundedness:**
```python
-1 ≤ sin(x) ≤ 1
-1 ≤ cos(x) ≤ 1

# Values never explode (unlike simple indexing: 0, 1, 2, ..., 10000)
```

---

**3. Uniqueness (for practical lengths):**
```python
# Wavelength of slowest dimension (d_model=512):
λ_max = 2π × 10000 ≈ 62,832 positions

# For sequences up to ~60k tokens, every position is unique!
# (GPT-3 max context: 2048 tokens - well within limits)
```

---

## 💻 Complete Implementation

### Positional Encoding Class

```python
import numpy as np

class PositionalEncoding:
    """
    Sinusoidal positional encoding for transformers.

    This is the exact method used in the original
    "Attention Is All You Need" paper!
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding.

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length to pre-compute
        """
        self.d_model = d_model
        self.max_len = max_len

        # Pre-compute positional encodings
        self.pe = self._create_encoding()

    def _create_encoding(self):
        """
        Create the positional encoding matrix.

        Returns:
            pe: (max_len, d_model) matrix
        """
        pe = np.zeros((self.max_len, self.d_model))

        # Create position indices
        position = np.arange(0, self.max_len).reshape(-1, 1)
        # Shape: (max_len, 1)

        # Create dimension indices
        div_term = np.exp(
            np.arange(0, self.d_model, 2) *
            -(np.log(10000.0) / self.d_model)
        )
        # Shape: (d_model/2,)

        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)

        # Apply cos to odd indices
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def forward(self, X):
        """
        Add positional encoding to input.

        Args:
            X: Input embeddings (batch_size, seq_len, d_model)

        Returns:
            X + PE: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = X.shape

        # Get positional encodings for this sequence length
        pos_encoding = self.pe[:seq_len, :]  # (seq_len, d_model)

        # Add to input (broadcasting over batch dimension)
        return X + pos_encoding
```

---

### Usage Example

```python
# Setup
batch_size = 2
seq_len = 10
d_model = 512

# Create random word embeddings
word_embeddings = np.random.randn(batch_size, seq_len, d_model)

print("Word embeddings shape:", word_embeddings.shape)
# (2, 10, 512)

# Create positional encoding layer
pos_encoder = PositionalEncoding(d_model)

# Add positional encoding
embeddings_with_pos = pos_encoder.forward(word_embeddings)

print("Embeddings with position shape:", embeddings_with_pos.shape)
# (2, 10, 512)

# Visualize effect on first word
print("\nFirst word (position 0):")
print("Original:", word_embeddings[0, 0, :5])
print("With PE:", embeddings_with_pos[0, 0, :5])
print("Difference (PE):", pos_encoder.pe[0, :5])
```

---

### Visualizing the Encoding

```python
import matplotlib.pyplot as plt

# Generate encoding for visualization
max_len = 100
d_model = 64
pe = PositionalEncoding(d_model, max_len).pe

# Plot heatmap
plt.figure(figsize=(15, 8))
plt.imshow(pe, cmap='RdBu', aspect='auto')
plt.colorbar(label='Value')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.title('Positional Encoding Heatmap (100 positions × 64 dimensions)')
plt.tight_layout()
plt.show()

# Observations:
# - Left side: High frequency oscillations
# - Right side: Low frequency (smoother)
# - Each row (position) is unique!
```

---

## 🎨 Alternative: Learned Positional Embeddings

### Different Approach

**Instead of fixed sine/cosine, learn positions:**

```python
class LearnedPositionalEmbedding:
    """
    Learned positional embeddings (used in BERT, GPT-2).

    Positions are learned during training like word embeddings.
    """

    def __init__(self, max_len, d_model):
        """
        Initialize learned position embeddings.

        Args:
            max_len: Maximum sequence length
            d_model: Embedding dimension
        """
        self.max_len = max_len
        self.d_model = d_model

        # Random initialization (would be trained)
        self.position_embeddings = np.random.randn(max_len, d_model) * 0.02

    def forward(self, X):
        """
        Add learned positional embeddings.

        Args:
            X: Input (batch, seq_len, d_model)

        Returns:
            X + position embeddings
        """
        batch_size, seq_len, _ = X.shape
        return X + self.position_embeddings[:seq_len, :]
```

---

### Comparison: Sinusoidal vs. Learned

| Feature | Sinusoidal (Original Transformer) | Learned (BERT, GPT-2) |
|---------|-----------------------------------|----------------------|
| **Fixed/Learned** | Fixed formula | Learned from data |
| **Parameters** | 0 (no training needed) | max_len × d_model |
| **Extrapolation** | Can handle longer sequences | Limited to max_len |
| **Performance** | Good | Slightly better in practice |
| **Used in** | Original Transformer, T5 | BERT, GPT-2, GPT-3 |

**Modern trend:** Learned embeddings (slightly better performance)

**Original paper:** Sinusoidal (elegant, parameter-free)

---

## 🧪 Practice Problems

### Problem 1: Calculate Wavelengths

For d_model = 512, calculate the wavelength of:
1. Dimensions 0-1 (fastest)
2. Dimensions 256-257 (middle)
3. Dimensions 510-511 (slowest)

<details>
<summary>Solution</summary>

```python
Formula: wavelength = 2π × 10000^(2i/d_model)

1. Dimensions 0-1:
   wavelength = 2π × 10000^(0/512) = 2π × 1 ≈ 6.28 positions

2. Dimensions 256-257:
   wavelength = 2π × 10000^(256/512) = 2π × 10000^0.5
              = 2π × 100 ≈ 628 positions

3. Dimensions 510-511:
   wavelength = 2π × 10000^(510/512) = 2π × 10000^0.996
              ≈ 2π × 9886 ≈ 62,095 positions

Interpretation:
- Fast dimensions change every few positions
- Slow dimensions change very gradually
- Together they create unique encodings!
```
</details>

---

### Problem 2: Implement RoPE

Implement **Rotary Positional Encoding** (used in newer models like LLaMA):

```python
def rotary_positional_encoding(X, positions):
    """
    Apply rotary position encoding.

    Rotates pairs of dimensions based on position.
    More recent alternative to sinusoidal PE.

    Args:
        X: Input (batch, seq_len, d_model)
        positions: Position indices (seq_len,)

    Returns:
        Rotated X
    """
    # Your code here
    # Hint: Apply rotation matrix to pairs of dimensions
    pass
```

---

## 🔑 Key Takeaways

### Remember These Points

1. **Pure attention is position-blind**
   - Permutation invariant
   - Needs positional information

2. **Positional encoding adds position info**
   - Added to word embeddings
   - Different positions → different representations

3. **Sinusoidal encoding uses sin/cosine waves**
   - Different frequencies for different dimensions
   - Creates unique fingerprint for each position

4. **Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))**
   - Bounded between -1 and 1
   - Can extrapolate to longer sequences
   - Allows learning relative positions

5. **Modern alternatives exist**
   - Learned embeddings (BERT, GPT)
   - Rotary encodings (RoPE)
   - Relative position encodings

---

## ✅ Self-Check

Before moving to Lesson 5 (Feed-Forward Networks), ensure you can:

- [ ] Explain why transformers need positional encoding
- [ ] Describe the sinusoidal encoding formula
- [ ] Understand different frequency components
- [ ] Implement positional encoding from scratch
- [ ] Compare sinusoidal vs. learned embeddings
- [ ] Explain how it preserves relative positions

**If you checked all boxes:** Ready for Feed-Forward Networks! 🎉

**If not:** Review this lesson, plot the encodings, experiment!

---

## 💬 Common Questions

**Q: Why not just use simple integers: 0, 1, 2, 3, ...?**
A: They grow unboundedly and don't have smooth relationships. The model would need to learn that position 100 and 101 are close (hard!).

**Q: Why sine AND cosine, not just sine?**
A: Using both provides a unique 2D representation for each position. It's like specifying a point on a circle using (x, y).

**Q: Can I use different formulas?**
A: Yes! Many alternatives exist (RoPE, ALiBi, etc.). Sinusoidal is just the original method.

**Q: Do I add or concatenate PE to embeddings?**
A: Add (element-wise addition). This preserves the embedding dimension.

---

## 📖 Further Reading

### Next Steps

1. **Run example code:**
   - `examples/example_04_positional.py` (when available)
   - Visualize different encodings!

2. **Next lesson:**
   - `05_feed_forward_networks.md`
   - The other half of transformer blocks!

3. **Research papers:**
   - "Attention Is All You Need" (Section 3.5)
   - "RoFormer: Enhanced Transformer with Rotary Position Embedding"
   - "Train Short, Test Long" (ALiBi position encoding)

---

## 🎊 Congratulations!

**You've learned how transformers encode position information!**

**Key achievement:**
> You understand how transformers know word order without recurrence!

**Progress:**
- ✅ Attention mechanism (Lesson 1)
- ✅ Self-attention (Lesson 2)
- ✅ Multi-head attention (Lesson 3)
- ✅ Positional encoding (Lesson 4)
- ⏳ Next: Feed-forward networks + Complete architecture!

**You're 80% there!** Almost understand the complete transformer! 🚀

---

**Next:** Lesson 5 - Feed-Forward Networks

**Keep going!** The complete picture is almost here! 💪
