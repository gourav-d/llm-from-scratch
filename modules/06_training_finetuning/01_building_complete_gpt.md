# Lesson 6.1: Building a Complete GPT Model from Scratch

**Assemble all the pieces into a working GPT architecture!**

---

## 🎯 Learning Objectives

By the end of this lesson, you will:
- ✅ Understand the complete GPT architecture
- ✅ Assemble all components from previous modules
- ✅ Implement a full GPT model in NumPy
- ✅ Count parameters like a pro
- ✅ Debug shape mismatches confidently
- ✅ Run forward pass through the entire model
- ✅ Compare your implementation to GPT-2

**Time Required:** 4-5 hours

---

## 📚 What is GPT?

### GPT = Generative Pre-trained Transformer

**Breaking it down:**
- **Generative:** Creates new text (generates sequences)
- **Pre-trained:** Trained on massive text corpus first
- **Transformer:** Uses transformer architecture (Module 4!)

### The Big Picture

```
GPT is a language model that:
1. Takes text as input
2. Predicts the next word
3. Repeats this process to generate sequences

That's it! Everything else is implementation details.
```

---

## 🧱 GPT Architecture Overview

### The Complete Stack

```
Input: "The cat sat on"
         ↓
┌────────────────────────────────────────┐
│  1. TOKEN EMBEDDING                    │
│     "The" → [0.25, -0.1, 0.8, ...]     │
│     (Lookup table)                      │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  2. POSITIONAL ENCODING                │
│     Add position information            │
│     pos[0] + pos[1] + pos[2] + pos[3]  │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  3. TRANSFORMER BLOCK 1                │
│     ├─ Multi-Head Attention            │
│     ├─ Add & Norm                      │
│     ├─ Feed-Forward Network            │
│     └─ Add & Norm                      │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  4. TRANSFORMER BLOCK 2                │
│     (Same structure, different weights) │
└────────────────────────────────────────┘
         ↓
       ... (more blocks)
         ↓
┌────────────────────────────────────────┐
│  5. FINAL LAYER NORM                   │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  6. OUTPUT PROJECTION                  │
│     (embed_dim → vocab_size)           │
│     [512] → [50,257]                   │
└────────────────────────────────────────┘
         ↓
Output: Probabilities for next token
        ["the": 0.35, "mat": 0.25, ...]
```

### C#/.NET Analogy

Think of GPT like a **pipeline in ASP.NET Core**:
```csharp
// ASP.NET Core middleware pipeline
app.Use(Embedding);           // Input processing
app.Use(PositionalEncoding);  // Add context
app.Use(TransformerBlock1);   // Processing layer 1
app.Use(TransformerBlock2);   // Processing layer 2
// ... more middleware
app.Use(OutputLayer);         // Final response
```

Each layer processes the input and passes it to the next!

---

## 🔧 Component 1: Configuration

### GPT Configuration Class

First, let's define the architecture configuration:

```python
class GPTConfig:
    """
    Configuration for GPT model architecture.

    Think of this like appsettings.json in .NET!
    """
    def __init__(
        self,
        vocab_size=50257,      # Size of vocabulary (like GPT-2)
        max_seq_len=256,       # Maximum sequence length
        embed_dim=512,         # Embedding dimension
        n_layers=6,            # Number of transformer blocks
        n_heads=8,             # Number of attention heads
        ff_dim=2048,           # Feed-forward hidden dimension
        dropout=0.1,           # Dropout probability
        activation='gelu'      # Activation function (GPT uses GELU)
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.activation = activation

        # Validation (like IValidateOptions in .NET!)
        assert embed_dim % n_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
```

**Line-by-line explanation:**

1. **`vocab_size=50257`** - How many different tokens we can handle
   - GPT-2 uses 50,257 tokens (256 bytes + 50,000 BPE merges + 1 special)
   - Like having 50,257 different "words" in our vocabulary

2. **`max_seq_len=256`** - Maximum input length
   - We can process up to 256 tokens at once
   - Longer sequences = more memory
   - GPT-2 uses 1024, GPT-3 uses 2048

3. **`embed_dim=512`** - Dimension of embeddings
   - Each token becomes a vector of 512 numbers
   - GPT-2 small uses 768, GPT-2 large uses 1280

4. **`n_layers=6`** - Number of transformer blocks
   - More layers = more capacity to learn
   - GPT-2 small: 12 layers, GPT-2 large: 36 layers, GPT-3: 96 layers!

5. **`n_heads=8`** - Number of attention heads
   - Multi-head attention splits into 8 parallel heads
   - Each head learns different patterns

6. **`ff_dim=2048`** - Feed-forward network size
   - Typically 4× the embedding dimension
   - GPT uses 4× (512 → 2048 → 512)

7. **`dropout=0.1`** - Regularization (prevent overfitting)
   - Randomly drop 10% of connections during training
   - Set to 0.0 during inference

8. **`activation='gelu'`** - Activation function
   - GPT uses GELU (Gaussian Error Linear Unit)
   - More sophisticated than ReLU

**C#/.NET Comparison:**
```csharp
// In C#, this would be:
public class GPTConfig
{
    public int VocabSize { get; init; } = 50257;
    public int MaxSeqLen { get; init; } = 256;
    public int EmbedDim { get; init; } = 512;
    // ... etc
}

// Or use IOptions pattern:
services.Configure<GPTConfig>(options => {
    options.VocabSize = 50257;
    options.EmbedDim = 512;
});
```

---

## 🔧 Component 2: Token Embedding

### From Token IDs to Dense Vectors

```python
import numpy as np

class TokenEmbedding:
    """
    Convert token IDs to dense vectors.

    This is a lookup table (like a dictionary in C#).
    Each token ID maps to a learned vector.
    """
    def __init__(self, vocab_size, embed_dim):
        """
        Initialize embedding matrix.

        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim: Dimension of each embedding vector
        """
        # Initialize embedding matrix with small random values
        # Shape: (vocab_size, embed_dim)
        # Like: Dictionary<int, float[]> in C#
        self.weight = np.random.randn(vocab_size, embed_dim) * 0.02

        # Store dimensions
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def forward(self, token_ids):
        """
        Look up embeddings for given token IDs.

        Args:
            token_ids: Array of token IDs, shape (batch_size, seq_len)

        Returns:
            Embeddings, shape (batch_size, seq_len, embed_dim)
        """
        # Simple lookup! Just index into the weight matrix
        # This is like: embeddings[token_id] in Python
        # Or: dictionary[tokenId] in C#
        embeddings = self.weight[token_ids]

        return embeddings

    def backward(self, token_ids, grad_output):
        """
        Compute gradients for embeddings.

        Args:
            token_ids: Token IDs used in forward pass
            grad_output: Gradient from next layer

        Returns:
            None (updates self.weight_grad)
        """
        # Initialize gradient matrix (same shape as weight)
        grad_weight = np.zeros_like(self.weight)

        # Accumulate gradients for each token
        # np.add.at handles multiple occurrences of same token
        np.add.at(grad_weight, token_ids, grad_output)

        return grad_weight
```

**How it works:**

```python
# Example
vocab_size = 1000
embed_dim = 128

embedding = TokenEmbedding(vocab_size, embed_dim)

# Input: token IDs
token_ids = np.array([[42, 156, 89]])  # Shape: (1, 3)

# Forward pass
embeddings = embedding.forward(token_ids)
# Output shape: (1, 3, 128)
# Each token ID → 128-dimensional vector

print(embeddings.shape)  # (1, 3, 128)
```

**Visual representation:**
```
Token IDs:    [42,    156,    89]
                ↓       ↓       ↓
Embedding:  [0.25]  [0.10]  [-0.5]
            [-0.1]  [0.30]  [0.23]
            [0.80]  [-0.2]  [0.11]
            [...]   [...]   [...]
            (128)   (128)   (128)
```

---

## 🔧 Component 3: Positional Encoding

### Adding Position Information

Transformers don't inherently understand position. We must add it!

```python
class PositionalEncoding:
    """
    Add positional information to embeddings.

    Uses sine/cosine functions to encode positions.
    This is the same method as the original "Attention is All You Need" paper!
    """
    def __init__(self, max_seq_len, embed_dim):
        """
        Pre-compute positional encodings.

        Args:
            max_seq_len: Maximum sequence length
            embed_dim: Embedding dimension
        """
        # Create positional encoding matrix
        # Shape: (max_seq_len, embed_dim)
        pe = np.zeros((max_seq_len, embed_dim))

        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        position = np.arange(0, max_seq_len)[:, np.newaxis]

        # Dimension indices: [0, 2, 4, ..., embed_dim-2]
        div_term = np.exp(
            np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim)
        )

        # Apply sine to even indices
        pe[:, 0::2] = np.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = np.cos(position * div_term)

        # Store the positional encoding
        self.pe = pe

    def forward(self, embeddings):
        """
        Add positional encoding to embeddings.

        Args:
            embeddings: Token embeddings, shape (batch_size, seq_len, embed_dim)

        Returns:
            Embeddings with position info, same shape
        """
        batch_size, seq_len, embed_dim = embeddings.shape

        # Add positional encoding (broadcasting handles batch dimension)
        # self.pe[:seq_len] gets positions 0 to seq_len-1
        embeddings = embeddings + self.pe[:seq_len]

        return embeddings
```

**Why sine/cosine?**

1. **Uniqueness:** Each position gets a unique encoding
2. **Relative positions:** Model can learn relative positions (e.g., "3 words before")
3. **Extrapolation:** Can handle sequences longer than max_seq_len (somewhat)

**Visual:**
```
Position 0: [sin(0/10000^(0/512)), cos(0/10000^(0/512)), sin(0/10000^(2/512)), ...]
Position 1: [sin(1/10000^(0/512)), cos(1/10000^(0/512)), sin(1/10000^(2/512)), ...]
Position 2: [sin(2/10000^(0/512)), cos(2/10000^(0/512)), sin(2/10000^(2/512)), ...]
...
```

Each position gets a different pattern of sine/cosine values!

---

## 🔧 Component 4: Transformer Block

### The Core Processing Unit

A transformer block contains:
1. Multi-head attention (from Module 4)
2. Add & normalize
3. Feed-forward network (from Module 3)
4. Add & normalize

```python
class TransformerBlock:
    """
    One transformer block (like one layer in a neural network).

    GPT stacks multiple blocks sequentially.
    """
    def __init__(self, embed_dim, n_heads, ff_dim, dropout=0.1):
        """
        Initialize transformer block components.

        Args:
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        # Multi-head attention (from Module 4!)
        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout)

        # Layer normalization after attention
        self.norm1 = LayerNorm(embed_dim)

        # Feed-forward network (from Module 3!)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout)

        # Layer normalization after FFN
        self.norm2 = LayerNorm(embed_dim)

        # Dropout for regularization
        self.dropout = dropout

    def forward(self, x, mask=None):
        """
        Forward pass through transformer block.

        Args:
            x: Input, shape (batch_size, seq_len, embed_dim)
            mask: Attention mask (optional)

        Returns:
            Output, same shape as input
        """
        # 1. Multi-head self-attention with residual connection
        # --------------------------------------------------
        # Residual connection: Like skip connections in ResNet!
        # Helps gradients flow during backpropagation
        attn_output = self.attention.forward(x, x, x, mask)

        # Apply dropout
        attn_output = self._apply_dropout(attn_output)

        # Add & Norm (residual + layer norm)
        x = self.norm1.forward(x + attn_output)

        # 2. Feed-forward network with residual connection
        # ---------------------------------------------
        ffn_output = self.ffn.forward(x)

        # Apply dropout
        ffn_output = self._apply_dropout(ffn_output)

        # Add & Norm
        x = self.norm2.forward(x + ffn_output)

        return x

    def _apply_dropout(self, x):
        """Apply dropout during training."""
        if self.training and self.dropout > 0:
            # Randomly zero out elements
            mask = np.random.binomial(1, 1 - self.dropout, size=x.shape)
            return x * mask / (1 - self.dropout)
        return x
```

**Residual connections visualized:**
```
Input
  │
  ├─────────────────────┐
  │                     │
  ↓                     │
Attention               │
  │                     │
  ↓                     │
Dropout                 │
  │                     │
  ↓                     │
  ├──ADD←───────────────┘  (Residual!)
  ↓
Layer Norm
  │
  ├─────────────────────┐
  │                     │
  ↓                     │
Feed-Forward            │
  │                     │
  ↓                     │
Dropout                 │
  │                     │
  ↓                     │
  ├──ADD←───────────────┘  (Residual!)
  ↓
Layer Norm
  ↓
Output
```

**Why residual connections?**
- Help gradients flow in deep networks
- Allow model to learn identity function (do nothing if needed)
- Enable training very deep models (96 layers in GPT-3!)

---

## 🔧 Component 5: Complete GPT Model

### Putting It All Together!

```python
class GPT:
    """
    Complete GPT model - the full stack!

    This combines ALL the components from Modules 3, 4, and 5.
    """
    def __init__(self, config):
        """
        Initialize GPT model.

        Args:
            config: GPTConfig object with architecture settings
        """
        self.config = config

        # 1. Token embedding layer (Module 5)
        self.token_embedding = TokenEmbedding(
            config.vocab_size,
            config.embed_dim
        )

        # 2. Positional encoding (Module 4)
        self.positional_encoding = PositionalEncoding(
            config.max_seq_len,
            config.embed_dim
        )

        # 3. Stack of transformer blocks (Module 4)
        self.transformer_blocks = [
            TransformerBlock(
                config.embed_dim,
                config.n_heads,
                config.ff_dim,
                config.dropout
            )
            for _ in range(config.n_layers)
        ]

        # 4. Final layer normalization
        self.final_norm = LayerNorm(config.embed_dim)

        # 5. Output projection (embed_dim → vocab_size)
        # This converts embeddings back to vocabulary space
        self.output_projection = Linear(config.embed_dim, config.vocab_size)

        # Initialize training mode
        self.training = True

    def forward(self, token_ids, targets=None):
        """
        Forward pass through entire GPT model.

        Args:
            token_ids: Input token IDs, shape (batch_size, seq_len)
            targets: Target token IDs for training (optional)

        Returns:
            logits: Output logits, shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided
        """
        batch_size, seq_len = token_ids.shape

        # Step 1: Token embeddings
        # ----------------------
        # Convert token IDs to dense vectors
        # Input:  (batch_size, seq_len)
        # Output: (batch_size, seq_len, embed_dim)
        x = self.token_embedding.forward(token_ids)

        # Step 2: Add positional encoding
        # ---------------------------
        # Add position information to embeddings
        # Output: (batch_size, seq_len, embed_dim)
        x = self.positional_encoding.forward(x)

        # Step 3: Create causal mask
        # ----------------------
        # GPT is autoregressive - can only attend to previous tokens!
        # This prevents "looking into the future"
        mask = self._create_causal_mask(seq_len)

        # Step 4: Pass through transformer blocks
        # -----------------------------------
        # Each block processes the sequence
        # Output shape stays: (batch_size, seq_len, embed_dim)
        for block in self.transformer_blocks:
            x = block.forward(x, mask=mask)

        # Step 5: Final layer normalization
        # -----------------------------
        x = self.final_norm.forward(x)

        # Step 6: Project to vocabulary
        # -------------------------
        # Convert embeddings to logits over vocabulary
        # Input:  (batch_size, seq_len, embed_dim)
        # Output: (batch_size, seq_len, vocab_size)
        logits = self.output_projection.forward(x)

        # Step 7: Calculate loss if targets provided
        # --------------------------------------
        loss = None
        if targets is not None:
            loss = self._compute_loss(logits, targets)

        return logits, loss

    def _create_causal_mask(self, seq_len):
        """
        Create causal (autoregressive) attention mask.

        Prevents attending to future positions.

        Args:
            seq_len: Sequence length

        Returns:
            Mask, shape (seq_len, seq_len)
        """
        # Create lower triangular matrix
        # 1 = can attend, 0 = cannot attend
        mask = np.tril(np.ones((seq_len, seq_len)))

        # Convert to -inf for softmax
        # -inf becomes 0 after softmax
        mask = np.where(mask == 0, -np.inf, 0.0)

        return mask

    def _compute_loss(self, logits, targets):
        """
        Compute cross-entropy loss.

        Args:
            logits: Model predictions, shape (batch_size, seq_len, vocab_size)
            targets: Target token IDs, shape (batch_size, seq_len)

        Returns:
            loss: Scalar loss value
        """
        # Reshape for loss calculation
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten: (batch_size * seq_len, vocab_size)
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Compute cross-entropy loss
        # This is the same as Module 3!
        loss = cross_entropy_loss(logits_flat, targets_flat)

        return loss

    def count_parameters(self):
        """
        Count total number of parameters.

        Returns:
            Total parameters
        """
        total = 0

        # Token embeddings
        total += self.token_embedding.weight.size

        # Transformer blocks (attention + FFN weights)
        for block in self.transformer_blocks:
            total += self._count_block_params(block)

        # Output projection
        total += self.output_projection.weight.size
        total += self.output_projection.bias.size

        return total

    def _count_block_params(self, block):
        """Count parameters in one transformer block."""
        count = 0

        # Attention weights (Q, K, V, output)
        count += block.attention.count_parameters()

        # FFN weights
        count += block.ffn.count_parameters()

        # Layer norm parameters
        count += block.norm1.count_parameters()
        count += block.norm2.count_parameters()

        return count
```

**The forward pass visualized:**

```
token_ids: [42, 156, 89, 12]  (batch_size=1, seq_len=4)
    ↓
┌─────────────────────────────────────┐
│ Token Embedding                     │
│ [42] → [0.25, -0.1, 0.8, ... ]     │  (4, 512)
│ [156] → [0.10, 0.30, -0.2, ...]    │
│ ...                                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Positional Encoding                 │
│ Add position 0, 1, 2, 3             │  (4, 512)
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Transformer Block 1                 │
│  ├ Attention (tokens talk to each)  │  (4, 512)
│  ├ Add & Norm                       │
│  ├ Feed-Forward (process each)      │
│  └ Add & Norm                       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Transformer Block 2                 │
│  (same structure)                   │  (4, 512)
└─────────────────────────────────────┘
    ↓
    ... (more blocks)
    ↓
┌─────────────────────────────────────┐
│ Final Layer Norm                    │  (4, 512)
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Output Projection                   │
│ (512 → 50,257)                      │  (4, 50257)
└─────────────────────────────────────┘
    ↓
logits: Predictions for each position
    Position 0: probabilities over 50,257 tokens
    Position 1: probabilities over 50,257 tokens
    Position 2: probabilities over 50,257 tokens
    Position 3: probabilities over 50,257 tokens ← "Next token" prediction
```

---

## 📊 Parameter Counting

### Where Do All the Parameters Come From?

Let's count for our configuration:
- vocab_size = 50,257
- embed_dim = 512
- n_layers = 6
- n_heads = 8
- ff_dim = 2048

**Token Embeddings:**
```
50,257 × 512 = 25,731,584 parameters
```

**Each Transformer Block:**
```
Multi-Head Attention:
  - Q, K, V projections: 3 × (512 × 512) = 786,432
  - Output projection: 512 × 512 = 262,144
  - Total attention: 1,048,576

Feed-Forward Network:
  - First layer: 512 × 2048 = 1,048,576
  - Bias: 2048 = 2,048
  - Second layer: 2048 × 512 = 1,048,576
  - Bias: 512 = 512
  - Total FFN: 2,099,712

Layer Norms (2):
  - 2 × (512 + 512) = 2,048

Block Total: 3,150,336 parameters
```

**All 6 Blocks:**
```
6 × 3,150,336 = 18,902,016 parameters
```

**Final Layer Norm:**
```
512 + 512 = 1,024 parameters
```

**Output Projection:**
```
512 × 50,257 = 25,731,584 parameters
```

**Grand Total:**
```
Token Embeddings:     25,731,584
Transformer Blocks:   18,902,016
Final Layer Norm:          1,024
Output Projection:    25,731,584
─────────────────────────────────
TOTAL:                70,366,208 parameters
                     ≈ 70M parameters!
```

### Comparing to GPT-2 and GPT-3

| Model | Parameters | Layers | Embed Dim | Heads | Vocab Size |
|-------|-----------|--------|-----------|-------|-----------|
| **Our GPT** | 70M | 6 | 512 | 8 | 50,257 |
| **GPT-2 Small** | 124M | 12 | 768 | 12 | 50,257 |
| **GPT-2 Medium** | 355M | 24 | 1024 | 16 | 50,257 |
| **GPT-2 Large** | 774M | 36 | 1280 | 20 | 50,257 |
| **GPT-2 XL** | 1.5B | 48 | 1600 | 25 | 50,257 |
| **GPT-3 Small** | 125M | 12 | 768 | 12 | 50,257 |
| **GPT-3 Large** | 175B | 96 | 12,288 | 96 | 50,257 |

**Our model is smaller than GPT-2 Small, but uses the SAME architecture!**

---

## 🐛 Debugging Shape Mismatches

### Common Shape Issues and Solutions

**Issue 1: Attention dimension mismatch**
```python
# ERROR: embed_dim must be divisible by n_heads
embed_dim = 513  # ❌ Not divisible by 8
n_heads = 8

# SOLUTION: Use valid dimension
embed_dim = 512  # ✅ 512 / 8 = 64 (head dimension)
```

**Issue 2: Sequence length mismatch**
```python
# ERROR: Input longer than max_seq_len
token_ids = np.random.randint(0, 1000, size=(1, 300))  # 300 tokens
max_seq_len = 256  # ❌ Only supports 256

# SOLUTION: Truncate or increase max_seq_len
token_ids = token_ids[:, :256]  # ✅ Truncate to 256
```

**Issue 3: Batch dimension confusion**
```python
# ERROR: Missing batch dimension
token_ids = np.array([42, 156, 89])  # Shape: (3,) ❌

# SOLUTION: Add batch dimension
token_ids = np.array([[42, 156, 89]])  # Shape: (1, 3) ✅
```

**Debugging tip:**
```python
# Print shapes at each step!
print(f"Input shape: {token_ids.shape}")
x = self.token_embedding.forward(token_ids)
print(f"After embedding: {x.shape}")
x = self.positional_encoding.forward(x)
print(f"After pos encoding: {x.shape}")
# etc...
```

---

## ✅ Summary

### What You Built

You assembled a **complete GPT model** with:
1. ✅ Token embeddings (vocab → dense vectors)
2. ✅ Positional encodings (add position info)
3. ✅ Stacked transformer blocks (attention + FFN)
4. ✅ Output projection (predictions over vocabulary)

### Key Insights

1. **GPT is modular** - Each component is independent
2. **Residual connections are critical** - Help gradients flow
3. **Parameters scale quickly** - 70M parameters in small model!
4. **Same architecture, different scale** - GPT-3 is just bigger

### Connection to .NET

```csharp
// GPT is like a processing pipeline:
public class GPTPipeline
{
    public Tensor Process(int[] tokenIds)
    {
        var embeddings = TokenEmbedding(tokenIds);
        embeddings = AddPositionalEncoding(embeddings);

        foreach (var block in transformerBlocks)
        {
            embeddings = block.Process(embeddings);
        }

        var normalized = LayerNorm(embeddings);
        var logits = OutputProjection(normalized);

        return logits;
    }
}
```

---

## 🎯 Practice Exercises

### Exercise 1: Build Mini-GPT
Create a tiny GPT with:
- vocab_size = 1000
- embed_dim = 128
- n_layers = 2
- n_heads = 4

Count the parameters!

### Exercise 2: Debug Shapes
Fix the shape mismatches in the provided buggy code.

### Exercise 3: Compare to GPT-2
Calculate how many parameters GPT-2 Small would have.

---

## 📚 Next Steps

✅ **You've built a complete GPT architecture!**

**Next lesson:** `02_text_generation.md`
- Generate text with your model
- Implement sampling strategies
- Control creativity vs coherence

**Up next:** Making your GPT talk! 🗣️

---

**Congratulations!** You now understand the exact architecture used in GPT-2, GPT-3, and ChatGPT! 🎉
