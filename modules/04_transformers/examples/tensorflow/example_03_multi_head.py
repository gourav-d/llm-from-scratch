"""
Example 03: Multi-Head Attention - TensorFlow Version

SAME example as NumPy/PyTorch versions, using Keras MultiHeadAttention.

KEY: tf.keras.layers.MultiHeadAttention vs nn.MultiheadAttention

  PyTorch:
    self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    output, weights = self.attn(Q, K, V)

  TensorFlow:
    self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_k)
    output = self.attn(Q, K, V)   ← weights not returned by default

  PARAMETER DIFFERENCES:
    PyTorch: embed_dim=d_model, num_heads=num_heads
    TF:      num_heads=num_heads, key_dim=d_k  (d_k = d_model // num_heads)

  PyTorch uses total dimension (d_model).
  TF uses per-head dimension (key_dim = d_model / num_heads).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import MultiHeadAttention, Dense, LayerNormalization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tf.random.set_seed(42)

print("=" * 70)
print("MULTI-HEAD ATTENTION - TensorFlow Version")
print("=" * 70)

# ==============================================================================
# PART 1: Setup
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Configuration")
print("=" * 70)

sentence  = ["The", "cat", "sat", "on", "the", "mat"]
seq_len   = len(sentence)
d_model   = 8
num_heads = 4
d_k       = d_model // num_heads    # 8 // 4 = 2 per head

print(f"Sentence: {' '.join(sentence)}")
print(f"d_model: {d_model} | num_heads: {num_heads} | d_k (per head): {d_k}")

X = tf.random.normal([seq_len, d_model]) * 0.5
print(f"\nInput shape: {X.shape}")

# ==============================================================================
# PART 2: Manual Multi-Head Attention (to understand the math)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Manual Implementation (NumPy-style in TensorFlow)")
print("=" * 70)

print("""
First: manual implementation to see the math clearly.
Then: using MultiHeadAttention built-in class.
""")

W_q_man = tf.Variable(tf.random.normal([d_model, d_model]) * 0.1)
W_k_man = tf.Variable(tf.random.normal([d_model, d_model]) * 0.1)
W_v_man = tf.Variable(tf.random.normal([d_model, d_model]) * 0.1)
W_o_man = tf.Variable(tf.random.normal([d_model, d_model]) * 0.1)

# Project to Q, K, V
Q_all = X @ W_q_man
K_all = X @ W_k_man
V_all = X @ W_v_man

# Split into heads
# NumPy:   Q.reshape(seq_len, num_heads, d_k)
# PyTorch: Q.view(seq_len, num_heads, d_k)
# TF:      tf.reshape(Q, [seq_len, num_heads, d_k])
Q_heads = tf.reshape(Q_all, [seq_len, num_heads, d_k])   # (6, 4, 2)
K_heads = tf.reshape(K_all, [seq_len, num_heads, d_k])
V_heads = tf.reshape(V_all, [seq_len, num_heads, d_k])

print(f"After split: Q_heads shape = {Q_heads.shape}")

# Compute attention per head
head_outputs = []
head_weights = []

for h in range(num_heads):
    Q_h = Q_heads[:, h, :]    # (6, 2)
    K_h = K_heads[:, h, :]
    V_h = V_heads[:, h, :]

    scores_h  = Q_h @ tf.transpose(K_h) / (d_k ** 0.5)
    weights_h = tf.nn.softmax(scores_h, axis=-1)
    output_h  = weights_h @ V_h

    head_outputs.append(output_h)
    head_weights.append(weights_h)

# Concatenate heads
# NumPy:   np.concatenate(head_outputs, axis=-1)
# PyTorch: torch.cat(head_outputs, dim=-1)
# TF:      tf.concat(head_outputs, axis=-1)
concat_output        = tf.concat(head_outputs, axis=-1)    # (6, 8)
final_output_manual  = concat_output @ W_o_man             # (6, 8)

print(f"Final output (manual): {final_output_manual.shape}")
print("✓ Manual implementation works!")

# ==============================================================================
# PART 3: Using tf.keras.layers.MultiHeadAttention
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Built-in MultiHeadAttention")
print("=" * 70)

print("""
TensorFlow built-in (different parameters from PyTorch!):

  PyTorch: nn.MultiheadAttention(
               embed_dim=d_model,  ← total dimension
               num_heads=num_heads)

  TF:      MultiHeadAttention(
               num_heads=num_heads,
               key_dim=d_k,        ← PER-HEAD dimension (not total!)
               value_dim=d_k)      ← per-head value dimension

  FORMULA: key_dim = d_model // num_heads

  TF also expects different input format:
    PyTorch: X_batched.shape = (batch, seq, d_model)  with batch_first=True
    TF:      X.shape = (seq, d_model)  OR  (batch, seq, d_model)
             TF handles both!

  Attention output (return_attention_scores):
    PyTorch: always returns weights  → output, weights = attn(Q, K, V)
    TF:      weights optional        → output = attn(Q, K, V)
             with weights:            output, weights = attn(Q, K, V,
                                         return_attention_scores=True)
""")

# Create built-in multi-head attention
# key_dim = dimension per head (d_k = d_model // num_heads = 2)
mha = MultiHeadAttention(
    num_heads=num_heads,
    key_dim=d_k,       # Per-head dimension (NOT total d_model!)
    value_dim=d_k
)

# TF MultiHeadAttention accepts (seq, d_model) directly (no need to add batch dim!)
# But we CAN add batch dimension too - both work
X_batched = tf.expand_dims(X, axis=0)   # (1, 6, 8) - optional

# Run with return_attention_scores=True to get attention weights
output_tf, attn_weights_tf = mha(
    query=X_batched,
    key=X_batched,
    value=X_batched,
    return_attention_scores=True
)

# Remove batch dimension
output_tf      = output_tf[0]         # (6, 8)
attn_weights_tf = attn_weights_tf[0]  # (num_heads, 6, 6) - per head!

print(f"Output shape: {output_tf.shape}")
print(f"Attention weights shape: {attn_weights_tf.shape}")
print(f"  → TF returns weights per HEAD: (num_heads, seq, seq)")
print(f"  → PyTorch averages over heads: (seq, seq)")

# ==============================================================================
# PART 4: Full Multi-Head Attention as Keras Layer
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Clean MultiHeadAttention as a Keras Layer")
print("=" * 70)

class MultiHeadAttentionTF(keras.layers.Layer):
    """
    Multi-Head Attention using TF's built-in layer.

    Wraps tf.keras.layers.MultiHeadAttention in a clean Keras Layer.
    """

    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)

        assert d_model % num_heads == 0

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        # Built-in multi-head attention
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.d_k,
            value_dim=self.d_k
        )

    def call(self, X, training=False, return_scores=False):
        """
        Forward pass.

        X: (seq_len, d_model) or (batch, seq_len, d_model)
        """
        # Add batch dim if needed
        squeeze = (X.shape.rank == 2)
        if squeeze:
            X = tf.expand_dims(X, axis=0)    # → (1, seq, d)

        if return_scores:
            output, attn_weights = self.attention(
                X, X, X,
                return_attention_scores=True,
                training=training
            )
        else:
            output = self.attention(X, X, X, training=training)
            attn_weights = None

        if squeeze:
            output = output[0]
            if attn_weights is not None:
                attn_weights = attn_weights[0]

        return output, attn_weights

# Test
mha_layer = MultiHeadAttentionTF(d_model=8, num_heads=4)
X_test = tf.random.normal([seq_len, d_model]) * 0.5

out, wts = mha_layer(X_test, return_scores=True)
print(f"Input:  {X_test.shape}")
print(f"Output: {out.shape}")
print(f"Attention weights per head: {wts.shape}")

total_params = sum(v.numpy().size for v in mha_layer.trainable_variables)
print(f"Total trainable parameters: {total_params:,}")
print("✓ TF MultiHeadAttention layer works!")

# ==============================================================================
# PART 5: Visualization - Different Head Patterns
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Visualizing Attention Per Head")
print("=" * 70)

# attn_weights_tf has shape: (num_heads, seq, seq) from Part 3
print(f"TF returns attention per head: {attn_weights_tf.shape}")
print(f"  → Shape: (num_heads={num_heads}, seq={seq_len}, seq={seq_len})")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for h in range(num_heads):
    # Get weights for head h
    head_wts = attn_weights_tf[h].numpy()   # Shape: (6, 6)
    sns.heatmap(head_wts,
                annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=sentence, yticklabels=sentence,
                cbar_kws={'label': 'Weight'},
                ax=axes[h])
    axes[h].set_title(f'Head {h+1} Attention Pattern (TF)', fontsize=12, fontweight='bold')
    axes[h].set_xlabel('Attending TO')
    axes[h].set_ylabel('Attending FROM')

plt.suptitle('Multi-Head Attention: Different Patterns Per Head (TensorFlow)',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - Multi-Head Attention: TF vs PyTorch")
print("=" * 70)

print("""
PARAMETER DIFFERENCES (easy to get wrong!):

  PyTorch nn.MultiheadAttention:       TF MultiHeadAttention:
  ────────────────────────────────     ─────────────────────────────────
  embed_dim = d_model (total)          key_dim = d_k (per HEAD!)
  num_heads = num_heads                num_heads = num_heads
  batch_first = True                   (no batch_first needed - TF handles it)

  EXAMPLE:
    d_model=8, num_heads=4, d_k=2

    PyTorch: nn.MultiheadAttention(embed_dim=8, num_heads=4)
    TF:      MultiHeadAttention(num_heads=4, key_dim=2)  ← key_dim = d_k!

ATTENTION WEIGHT SHAPE DIFFERENCE:
  PyTorch: (seq, seq)          - averaged over heads
  TF:      (num_heads, seq, seq) - one matrix per head!

TF TENSOR OPERATIONS:
  NumPy/PyTorch  │ TensorFlow
  ───────────────┼──────────────────────────
  np.concatenate │ tf.concat
  np.reshape     │ tf.reshape
  np.expand_dims │ tf.expand_dims  or  [tf.newaxis, :]
  arr[:, h, :]   │ tensor[:, h, :]  (SAME slicing!)

Next:
  example_04: Positional encoding in TensorFlow
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 03 - TensorFlow Version")
print("=" * 70)
