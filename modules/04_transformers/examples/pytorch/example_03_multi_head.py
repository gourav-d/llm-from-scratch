"""
Example 03: Multi-Head Attention - PyTorch Version

SAME example as NumPy version, now using PyTorch's built-in nn.MultiheadAttention.

KEY NEW CONCEPT: nn.MultiheadAttention
  In NumPy:   We coded everything from scratch (split, loop, concat)
  In PyTorch: nn.MultiheadAttention does it ALL in one line!

  NumPy approach (6+ steps):
    1. W_q, W_k, W_v = ... (create weight matrices)
    2. Q, K, V = X @ W_q, X @ W_k, X @ W_v
    3. Q_heads, K_heads, V_heads = split_heads(Q, K, V, num_heads)
    4. for h in range(num_heads): ... (loop over each head)
    5. multi_head_output = concatenate(head_outputs)
    6. final = multi_head_output @ W_o

  PyTorch approach (1 step!):
    self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    output, weights = self.attn(Q, K, V)

  In C# terms: Like switching from writing your own LINQ operators
               to using the built-in LINQ library.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(42)

print("=" * 70)
print("MULTI-HEAD ATTENTION - PyTorch Version")
print("=" * 70)

# ==============================================================================
# PART 1: Why Multiple Heads?
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Why One Attention Head Isn't Enough")
print("=" * 70)

print("""
SAME CONCEPT as NumPy version - no change here!

Multi-head attention captures MULTIPLE types of relationships:
  - Head 1: Nearby word patterns (syntax)
  - Head 2: Pronoun-to-noun links (reference)
  - Head 3: Verb-to-subject patterns (grammar)
  - Head 4: Long-range meaning (semantics)

The ONLY difference is HOW we implement it in code.
PyTorch provides nn.MultiheadAttention as a ready-made class!
""")

# ==============================================================================
# PART 2: Setup
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Configuration and Input")
print("=" * 70)

sentence = ["The", "cat", "sat", "on", "the", "mat"]
seq_len   = len(sentence)
d_model   = 8
num_heads = 4
d_k       = d_model // num_heads   # 8 // 4 = 2 dims per head

print(f"Sentence: {' '.join(sentence)}")
print(f"d_model: {d_model} | num_heads: {num_heads} | d_k (per head): {d_k}")

# NumPy:   X = np.random.randn(seq_len, d_model) * 0.5
# PyTorch: X = torch.randn(seq_len, d_model) * 0.5
X = torch.randn(seq_len, d_model) * 0.5
print(f"\nInput shape: {X.shape}  → (seq_len={seq_len}, d_model={d_model})")

# ==============================================================================
# PART 3: Manual Multi-Head Attention (NumPy-style, to understand)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Manual Implementation (Understanding the Math)")
print("=" * 70)

print("""
First, let's do it the manual way (same as NumPy) to see the math clearly.
Then we'll use nn.MultiheadAttention to do it in 1 line!
""")

# Weight matrices
W_q = torch.randn(d_model, d_model) * 0.1
W_k = torch.randn(d_model, d_model) * 0.1
W_v = torch.randn(d_model, d_model) * 0.1
W_o = torch.randn(d_model, d_model) * 0.1

# Project to Q, K, V
Q_all = X @ W_q   # (6, 8)
K_all = X @ W_k   # (6, 8)
V_all = X @ W_v   # (6, 8)

# Split into heads
# NumPy:   Q_heads = Q.reshape(seq_len, num_heads, d_k)
# PyTorch: Q_heads = Q.view(seq_len, num_heads, d_k)  (same idea)
Q_heads = Q_all.view(seq_len, num_heads, d_k)   # (6, 4, 2)
K_heads = K_all.view(seq_len, num_heads, d_k)
V_heads = V_all.view(seq_len, num_heads, d_k)

print(f"After split: Q_heads shape = {Q_heads.shape}  (seq, heads, d_k)")

# Compute attention for each head
head_outputs = []
head_weights = []

for h in range(num_heads):
    Q_h = Q_heads[:, h, :]   # (6, 2)
    K_h = K_heads[:, h, :]
    V_h = V_heads[:, h, :]

    scores_h  = Q_h @ K_h.T / (d_k ** 0.5)
    weights_h = F.softmax(scores_h, dim=-1)
    output_h  = weights_h @ V_h

    head_outputs.append(output_h)
    head_weights.append(weights_h)

# Concatenate heads
# NumPy:   np.concatenate(head_outputs, axis=-1)
# PyTorch: torch.cat(head_outputs, dim=-1)
concat_output = torch.cat(head_outputs, dim=-1)   # (6, 8)
final_output_manual = concat_output @ W_o          # (6, 8)

print(f"Final output shape (manual): {final_output_manual.shape}")
print("✓ Manual implementation works!")

# ==============================================================================
# PART 4: Using nn.MultiheadAttention (The PyTorch Way)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Using nn.MultiheadAttention - One Line vs Six Steps!")
print("=" * 70)

print("""
PyTorch has ALL of that math built into ONE class:

  nn.MultiheadAttention(
      embed_dim    = d_model,    # Total embedding dimension
      num_heads    = num_heads,  # Number of heads
      batch_first  = True,       # Input shape: (batch, seq, features)
      bias         = False       # No bias term
  )

  IMPORTANT: batch_first=True  (new default in modern PyTorch)
    With batch_first=True:  Input shape = (batch_size, seq_len, d_model)
    With batch_first=False: Input shape = (seq_len, batch_size, d_model)

    For our 1-sequence example, we add a batch dimension of 1:
      X.unsqueeze(0)  → shape (1, 6, 8)  = (batch=1, seq=6, d=8)

    This is like wrapping a single item in a List<T>:
      List<Item> batch = new List<Item> { singleItem };
""")

# Create built-in multi-head attention
mha = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=num_heads,
    batch_first=True,
    bias=False
)

# Add batch dimension: (seq_len, d_model) → (1, seq_len, d_model)
# .unsqueeze(0) adds a new dimension at position 0
# NumPy equivalent: X[np.newaxis, :]  or  X.reshape(1, seq_len, d_model)
X_batched = X.unsqueeze(0)    # Shape: (1, 6, 8)
print(f"X_batched shape: {X_batched.shape}  (batch=1, seq=6, d_model=8)")

# Run multi-head attention - ALL 6 steps in ONE call!
# Returns: (output, attention_weights_averaged_over_heads)
with torch.no_grad():
    output_builtin, attn_weights_builtin = mha(X_batched, X_batched, X_batched)
    # Passing X three times = Q, K, V all from same input (self-attention)

print(f"\nOutput shape: {output_builtin.shape}  (batch=1, seq=6, d_model=8)")
print(f"Attention weights shape: {attn_weights_builtin.shape}")

# Remove batch dimension for easier handling
# .squeeze(0) removes the batch dimension we added
output_final = output_builtin.squeeze(0)         # (6, 8)
attn_weights = attn_weights_builtin.squeeze(0)   # (6, 6)

print(f"\nAfter removing batch dim:")
print(f"  Output: {output_final.shape}")
print(f"  Attention weights: {attn_weights.shape}")

# ==============================================================================
# PART 5: Full Multi-Head Attention Module
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Clean MultiHeadAttention as nn.Module")
print("=" * 70)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention using nn.MultiheadAttention.

    PyTorch handles all the head splitting/concatenation internally!

    C# equivalent:
        class MultiHeadAttention : NeuralNetworkBase {
            MultiheadAttention attn;
            int num_heads, d_model;

            public Tensor Forward(Tensor X) {
                return attn.Forward(X, X, X);
            }
        }
    """

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads

        # Built-in multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            bias=False
        )

    def forward(self, X):
        """
        Forward pass.

        Args:
            X: shape (seq_len, d_model)  or  (batch_size, seq_len, d_model)

        Returns:
            output: shape (seq_len, d_model)
            weights: attention weights shape (seq_len, seq_len)
        """
        # Add batch dimension if not present
        if X.dim() == 2:
            X = X.unsqueeze(0)     # (seq_len, d) → (1, seq_len, d)
            squeeze_output = True
        else:
            squeeze_output = False

        # Q=X, K=X, V=X (self-attention - all from same source)
        output, weights = self.attention(X, X, X)

        if squeeze_output:
            output  = output.squeeze(0)
            weights = weights.squeeze(0)

        return output, weights

# Test
mha_module = MultiHeadAttention(d_model=8, num_heads=4)
X_test = torch.randn(seq_len, d_model) * 0.5

with torch.no_grad():
    out, wts = mha_module(X_test)

print(f"Input:  {X_test.shape}")
print(f"Output: {out.shape}")
print(f"Attention weights: {wts.shape}")

total_params = sum(p.numel() for p in mha_module.parameters())
print(f"\nTotal trainable parameters: {total_params:,}")
print("✓ MultiHeadAttention module works!")

# ==============================================================================
# PART 6: Visualization - Comparing Head Patterns
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Visualizing Attention Patterns per Head")
print("=" * 70)

print("""
NOTE: nn.MultiheadAttention returns AVERAGED attention weights
      (averaged across all heads). To see individual heads, we use
      the manual implementation from Part 3.
""")

# Use manual head weights from Part 3 for visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for h in range(num_heads):
    weights_np = head_weights[h].detach().numpy()
    sns.heatmap(weights_np,
                annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=sentence, yticklabels=sentence,
                cbar_kws={'label': 'Weight'},
                ax=axes[h])
    axes[h].set_title(f'Head {h+1} Attention Pattern (PyTorch)', fontsize=12, fontweight='bold')
    axes[h].set_xlabel('Attending TO')
    axes[h].set_ylabel('Attending FROM')

plt.suptitle('Multi-Head Attention: Each Head Learns Different Patterns',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - Multi-Head Attention: NumPy vs PyTorch")
print("=" * 70)

print("""
APPROACH COMPARISON:

NumPy (6 manual steps):            │ PyTorch (1 built-in class):
───────────────────────────────────┼──────────────────────────────────────
W_q, W_k, W_v = randn(...)        │ self.attn = nn.MultiheadAttention(
Q = X @ W_q                        │     embed_dim=d_model,
...                                │     num_heads=num_heads)
for h in range(num_heads):         │
    Q_h = Q_heads[:, h, :]         │
    ...                            │
head_outputs.append(output_h)      │
concat = concatenate(head_outputs) │ output, weights = self.attn(X, X, X)
final = concat @ W_o               │

KEY PyTorch TOOLS USED:
  ✓ torch.cat(tensors, dim=)  → like np.concatenate but for tensors
  ✓ tensor.view(shape)        → like np.reshape (must be contiguous in memory)
  ✓ tensor.unsqueeze(dim)     → add new dimension (like np.newaxis)
  ✓ tensor.squeeze(dim)       → remove dimension of size 1
  ✓ nn.MultiheadAttention     → full multi-head attention built-in!

Next:
  example_04: Positional encoding in PyTorch
  example_05: Full transformer block (nn.TransformerEncoderLayer)
  example_06: Mini-GPT (nn.Transformer, nn.Embedding)
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 03 - PyTorch Version")
print("=" * 70)
