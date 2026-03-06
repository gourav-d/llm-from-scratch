"""
Example 03: Multi-Head Attention

This example demonstrates multi-head attention - running multiple attention
mechanisms in parallel and combining their outputs. This is THE key innovation
that makes transformers so powerful!

What you'll see:
1. Why single-head attention has limitations
2. How to split embeddings across multiple heads
3. Parallel attention computation for each head
4. Concatenating and projecting head outputs
5. Visualization showing how different heads learn different patterns

Think of it like a team of experts:
- Single-head: One generalist trying to understand everything
- Multi-head: Team of specialists, each focusing on different aspects
- CFO sees finances, CMO sees marketing, CTO sees technology
- Combined view gives complete understanding!
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("MULTI-HEAD ATTENTION")
print("=" * 70)

# ==============================================================================
# PART 1: The Problem with Single-Head Attention
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Why One Attention Head Isn't Enough")
print("=" * 70)

print("""
PROBLEM: One attention pattern can't capture all relationships!

Example: "The cat sat on the mat because it was tired"

Different types of relationships:
1. Syntax: "cat" (subject) ← "sat" (verb)
2. Position: "sat" → "on the mat" (location)
3. Reference: "it" → "cat" (pronoun resolution)
4. Reason: "sat" → "tired" (causation)

Single-head attention can focus on ONE dominant pattern.
Multi-head attention captures MULTIPLE patterns in parallel!

Like C# LINQ vs Multiple LINQ queries:
  // Single pattern
  var result = data.Select(x => x.BestMatch);

  // Multiple patterns (multi-head concept)
  var syntaxPattern = data.Select(x => x.SyntaxMatch);
  var semanticPattern = data.Select(x => x.SemanticMatch);
  var positionPattern = data.Select(x => x.PositionMatch);
  var combined = Combine(syntaxPattern, semanticPattern, positionPattern);
""")

# ==============================================================================
# PART 2: Multi-Head Attention Architecture
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Multi-Head Attention Architecture")
print("=" * 70)

print("""
ARCHITECTURE:

Input (d_model=512)
    ↓
Split into H heads (e.g., 8 heads × 64 dims each = 512)
    ↓
Head 1 → Attention (64-dim) ──┐
Head 2 → Attention (64-dim) ──┤
Head 3 → Attention (64-dim) ──┤ Concatenate
Head 4 → Attention (64-dim) ──┤ → (512-dim)
Head 5 → Attention (64-dim) ──┤
Head 6 → Attention (64-dim) ──┤
Head 7 → Attention (64-dim) ──┤
Head 8 → Attention (64-dim) ──┘
    ↓
Linear projection (W_o)
    ↓
Output (d_model=512)

Each head learns DIFFERENT patterns!
""")

# ==============================================================================
# PART 3: Setting Up Parameters
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Configuration and Input")
print("=" * 70)

# Configuration
sentence = ["The", "cat", "sat", "on", "the", "mat"]
seq_len = len(sentence)
d_model = 8  # Total embedding dimension
num_heads = 4  # Number of attention heads
d_k = d_model // num_heads  # Dimension per head

print(f"Sentence: {' '.join(sentence)}")
print(f"Sequence length: {seq_len}")
print(f"Model dimension (d_model): {d_model}")
print(f"Number of heads: {num_heads}")
print(f"Dimension per head (d_k): {d_k}")

print(f"\nSplitting: {d_model} dimensions → {num_heads} heads × {d_k} dims/head")

# Create input embeddings
# Shape: (seq_len, d_model) = (6, 8)
X = np.random.randn(seq_len, d_model) * 0.5

print(f"\nInput shape: {X.shape}")

# ==============================================================================
# PART 4: Weight Matrices for All Heads
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Weight Matrices for Multiple Heads")
print("=" * 70)

print("""
Each head has its OWN set of W_q, W_k, W_v matrices!

Traditional approach:
  - Head 1: W_q1, W_k1, W_v1 (d_model × d_k)
  - Head 2: W_q2, W_k2, W_v2 (d_model × d_k)
  - ...

Efficient approach (used in practice):
  - One large W_q, W_k, W_v (d_model × d_model)
  - Split outputs into heads
  - Mathematically equivalent, computationally faster!

Similar to C# batch processing vs individual operations.
""")

# Initialize weight matrices for all heads combined
# Shape: (d_model, d_model) for each
# We'll split the output into heads
W_q = np.random.randn(d_model, d_model) * 0.1
W_k = np.random.randn(d_model, d_model) * 0.1
W_v = np.random.randn(d_model, d_model) * 0.1

# Output projection (combines all heads)
W_o = np.random.randn(d_model, d_model) * 0.1

print(f"W_q shape: {W_q.shape} (projects to all heads)")
print(f"W_k shape: {W_k.shape} (projects to all heads)")
print(f"W_v shape: {W_v.shape} (projects to all heads)")
print(f"W_o shape: {W_o.shape} (output projection)")

# ==============================================================================
# PART 5: Linear Projections
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Projecting Input to Q, K, V")
print("=" * 70)

# Project input to Q, K, V
Q = X @ W_q  # Shape: (6, 8)
K = X @ W_k  # Shape: (6, 8)
V = X @ W_v  # Shape: (6, 8)

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

# ==============================================================================
# PART 6: Splitting Into Multiple Heads
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Splitting Q, K, V Into Heads")
print("=" * 70)

print("""
Now we split Q, K, V into multiple heads:

Original: (seq_len, d_model) = (6, 8)
Reshape:  (seq_len, num_heads, d_k) = (6, 4, 2)
          ↑        ↑           ↑
       6 words  4 heads   2 dims/head

This is like splitting an array in C#:
  float[6][8] → float[6][4][2]
  Total elements stay the same, just reorganized!
""")

def split_heads(x, num_heads):
    """
    Split last dimension into (num_heads, d_k).

    Args:
        x: shape (seq_len, d_model)
        num_heads: number of attention heads

    Returns:
        shape (seq_len, num_heads, d_k)

    Similar to C#:
        float[][][] SplitHeads(float[][] x, int num_heads) {
            int seq_len = x.Length;
            int d_k = x[0].Length / num_heads;
            var result = new float[seq_len][][];
            // ... reshaping logic ...
            return result;
        }
    """
    seq_len, d_model = x.shape
    d_k = d_model // num_heads

    # Reshape: (seq_len, d_model) → (seq_len, num_heads, d_k)
    return x.reshape(seq_len, num_heads, d_k)

# Split into heads
Q_heads = split_heads(Q, num_heads)  # (6, 4, 2)
K_heads = split_heads(K, num_heads)  # (6, 4, 2)
V_heads = split_heads(V, num_heads)  # (6, 4, 2)

print(f"Q split into heads: {Q_heads.shape}")
print(f"K split into heads: {K_heads.shape}")
print(f"V split into heads: {V_heads.shape}")

print(f"\nInterpretation: {seq_len} words, {num_heads} heads, {d_k} dims per head")

# ==============================================================================
# PART 7: Attention for Each Head
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: Computing Attention for Each Head")
print("=" * 70)

print("""
Now we compute attention SEPARATELY for each head.
Each head will learn different patterns!
""")

def softmax(x, axis=-1):
    """Apply softmax function."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Store outputs for each head
head_outputs = []
head_weights = []

print("\nProcessing each head:")
for h in range(num_heads):
    print(f"\n  Head {h+1}:")

    # Extract Q, K, V for this head
    # Shape: (seq_len, d_k) = (6, 2)
    Q_h = Q_heads[:, h, :]
    K_h = K_heads[:, h, :]
    V_h = V_heads[:, h, :]

    print(f"    Q_h shape: {Q_h.shape}")

    # Compute attention scores
    scores_h = Q_h @ K_h.T / np.sqrt(d_k)  # (6, 6)

    # Apply softmax
    weights_h = softmax(scores_h, axis=-1)  # (6, 6)

    # Compute output
    output_h = weights_h @ V_h  # (6, 2)

    head_outputs.append(output_h)
    head_weights.append(weights_h)

    print(f"    Attention weights shape: {weights_h.shape}")
    print(f"    Output shape: {output_h.shape}")

# ==============================================================================
# PART 8: Concatenating Heads
# ==============================================================================

print("\n" + "=" * 70)
print("PART 8: Concatenating Head Outputs")
print("=" * 70)

print("""
Now we concatenate all head outputs back together:

Head 1 output: (6, 2)  ┐
Head 2 output: (6, 2)  ├─ Concatenate
Head 3 output: (6, 2)  │  along last axis
Head 4 output: (6, 2)  ┘
        ↓
Combined: (6, 8)

Like C# LINQ SelectMany or Concat:
  var combined = heads.SelectMany(h => h.output).ToArray();
""")

# Stack along head dimension and reshape
# List of (6, 2) → Stack → (6, 4, 2) → Reshape → (6, 8)
multi_head_output = np.concatenate(head_outputs, axis=-1)

print(f"Concatenated output shape: {multi_head_output.shape}")
print("Back to original d_model dimension!")

# ==============================================================================
# PART 9: Output Projection
# ==============================================================================

print("\n" + "=" * 70)
print("PART 9: Final Output Projection")
print("=" * 70)

print("""
Finally, we apply one more linear transformation:
  Output = MultiHeadOutput @ W_o

This allows the model to:
1. Combine information from all heads
2. Learn how to weight different heads
3. Project to final representation

Similar to C# final aggregation:
  var final = combined.Select(x => x * W_o).ToArray();
""")

# Apply output projection
final_output = multi_head_output @ W_o

print(f"Final output shape: {final_output.shape}")
print("\nMulti-head attention complete! ✓")

# ==============================================================================
# PART 10: Multi-Head Attention Class
# ==============================================================================

print("\n" + "=" * 70)
print("PART 10: Multi-Head Attention as a Reusable Class")
print("=" * 70)

class MultiHeadAttention:
    """
    Multi-Head Attention layer.

    Equivalent C# structure:
        class MultiHeadAttention {
            private Matrix W_q, W_k, W_v, W_o;
            private int num_heads, d_model, d_k;

            public (Matrix output, Matrix[] weights) Forward(Matrix X) {
                // 1. Linear projections
                // 2. Split into heads
                // 3. Attention per head
                // 4. Concatenate
                // 5. Output projection
            }
        }
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Embedding dimension
            num_heads: Number of attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def split_heads(self, x):
        """Split into multiple heads."""
        seq_len, _ = x.shape
        return x.reshape(seq_len, self.num_heads, self.d_k)

    def forward(self, X):
        """
        Forward pass.

        Args:
            X: Input, shape (seq_len, d_model)

        Returns:
            output: shape (seq_len, d_model)
            attention_weights: list of (seq_len, seq_len) for each head
        """
        # 1. Linear projections
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        # 2. Split into heads
        Q_heads = self.split_heads(Q)
        K_heads = self.split_heads(K)
        V_heads = self.split_heads(V)

        # 3. Attention for each head
        head_outputs = []
        attention_weights = []

        for h in range(self.num_heads):
            Q_h = Q_heads[:, h, :]
            K_h = K_heads[:, h, :]
            V_h = V_heads[:, h, :]

            scores = Q_h @ K_h.T / np.sqrt(self.d_k)
            weights = softmax(scores, axis=-1)
            output = weights @ V_h

            head_outputs.append(output)
            attention_weights.append(weights)

        # 4. Concatenate heads
        concat_output = np.concatenate(head_outputs, axis=-1)

        # 5. Output projection
        final_output = concat_output @ self.W_o

        return final_output, attention_weights

# Test the class
print("\nTesting MultiHeadAttention class:")
mha = MultiHeadAttention(d_model=8, num_heads=4)
output, attn_weights = mha.forward(X)

print(f"  Input shape: {X.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Number of attention weight matrices: {len(attn_weights)}")
print(f"  Each attention matrix shape: {attn_weights[0].shape}")
print("\n✓ Multi-head attention class works!")

# ==============================================================================
# PART 11: Visualizing Different Head Patterns
# ==============================================================================

print("\n" + "=" * 70)
print("PART 11: Visualizing What Different Heads Learn")
print("=" * 70)

# Create a grid of heatmaps for all heads
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for h in range(num_heads):
    sns.heatmap(head_weights[h],
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                xticklabels=sentence,
                yticklabels=sentence,
                cbar_kws={'label': 'Weight'},
                ax=axes[h])

    axes[h].set_title(f'Head {h+1} Attention Pattern', fontsize=12, fontweight='bold')
    axes[h].set_xlabel('Attending TO')
    axes[h].set_ylabel('Attending FROM')

plt.suptitle('Multi-Head Attention: Each Head Learns Different Patterns',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

print("\nNotice how each head has DIFFERENT attention patterns!")
print("This is the power of multi-head attention:")
print("  - Head 1 might focus on nearby words (local patterns)")
print("  - Head 2 might focus on specific word types (syntax)")
print("  - Head 3 might focus on semantic relationships")
print("  - Head 4 might capture long-range dependencies")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - What We Learned")
print("=" * 70)

print("""
✓ Multi-Head Attention: Multiple attention mechanisms in parallel
✓ Splitting: Divide d_model into num_heads × d_k
✓ Parallel Processing: Each head computes attention independently
✓ Different Patterns: Each head learns different relationships
✓ Concatenation: Combine all head outputs
✓ Output Projection: Final linear transformation with W_o

The Complete Algorithm:
  1. Linear projections: Q, K, V = X @ W_q, X @ W_k, X @ W_v
  2. Split into heads: reshape to (seq_len, num_heads, d_k)
  3. Attention per head: for each head, compute attention
  4. Concatenate: combine all heads back to d_model
  5. Output projection: final @ W_o

Why Multi-Head?
  - Single head: Limited to one pattern
  - Multi-head: Captures multiple relationships simultaneously
  - Like having a team of experts vs one generalist

Typical Configurations:
  - GPT-2: d_model=768, num_heads=12, d_k=64
  - GPT-3: d_model=12288, num_heads=96, d_k=128
  - BERT: d_model=768, num_heads=12, d_k=64

In C#/.NET Terms:
  - split_heads() is like reshaping arrays
  - Each head is like parallel LINQ queries
  - Concatenation is like SelectMany
  - The whole process is like parallel processing with Aggregate

Next Steps:
  - example_04: Positional encoding (adding position info)
  - example_05: Complete transformer block
  - example_06: Building a mini-GPT!
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 03")
print("=" * 70)
