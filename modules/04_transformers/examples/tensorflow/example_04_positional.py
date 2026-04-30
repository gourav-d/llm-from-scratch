"""
Example 04: Positional Encoding - TensorFlow Version

SAME example as NumPy/PyTorch versions.
The math is identical - only the tensor API changes.

NumPy vs PyTorch vs TensorFlow for positional encoding:
  ┌────────────────────────────────┬─────────────────────────┬────────────────────────────┐
  │ NumPy                          │ PyTorch                 │ TensorFlow                 │
  ├────────────────────────────────┼─────────────────────────┼────────────────────────────┤
  │ np.arange(n)                   │ torch.arange(n)         │ tf.range(n, dtype=tf.float32)│
  │ arr[:, np.newaxis]             │ t.unsqueeze(1)          │ t[:, tf.newaxis]           │
  │ np.zeros((n, d))               │ torch.zeros(n, d)       │ tf.zeros([n, d])           │
  │ np.exp(x)                      │ torch.exp(x)            │ tf.exp(x)                  │
  │ np.log(10000.0)                │ torch.log(tensor(1e4)) │ tf.math.log(tf.constant(1e4))│
  │ np.sin(x)                      │ torch.sin(x)            │ tf.sin(x)                  │
  │ np.cos(x)                      │ torch.cos(x)            │ tf.cos(x)                  │
  │ PE[:, 0::2] = np.sin(...)     │ PE[:, 0::2] = sin(...)  │ tf.concat / tf.TensorArray │
  └────────────────────────────────┴─────────────────────────┴────────────────────────────┘

NOTE: TensorFlow tensors are IMMUTABLE by default!
  You cannot do: PE[:, 0::2] = tf.sin(...)
  Instead, you build PE by concatenating columns, OR use tf.Variable.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(42)

print("=" * 70)
print("POSITIONAL ENCODING - TensorFlow Version")
print("=" * 70)

# ==============================================================================
# PART 1: The Problem (same as NumPy/PyTorch)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Attention is Order-Blind - Solution: Positional Encoding")
print("=" * 70)

print("""
SAME CONCEPT as other versions:
  Pure attention ignores word order!
  We add sinusoidal encodings to tell the model WHERE each word is.

  Formula:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

The math is IDENTICAL - only the code changes.
""")

# ==============================================================================
# PART 2: Positional Encoding in TensorFlow
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Implementing Positional Encoding in TensorFlow")
print("=" * 70)

print("""
IMPORTANT TF DIFFERENCE: Tensors are IMMUTABLE!

  NumPy/PyTorch can do in-place assignment:
    PE = np.zeros((n, d))
    PE[:, 0::2] = np.sin(...)    ← WORKS in NumPy
    PE[:, 0::2] = torch.sin(...) ← WORKS in PyTorch

  TensorFlow CANNOT do this (tensors are read-only by default):
    PE = tf.zeros([n, d])
    PE[:, 0::2] = tf.sin(...)   ← ERROR! TF tensors are immutable!

  TF Solutions:
  Option A: Use tf.Variable (mutable tensor) and assign
  Option B: Build PE using tf.concat (concatenate sine and cosine columns)
  Option C: Use Python lists and convert to tensor at the end (simplest!)

  We will use Option B (tf.concat) - the most "TF native" approach.
""")

def positional_encoding_tf(max_seq_len, d_model):
    """
    Generate sinusoidal positional encodings using TensorFlow.

    Returns a tf.Tensor of shape (max_seq_len, d_model).

    Key TF functions:
      tf.range()         = np.arange() in NumPy
      tf.cast()          = type casting (like (float) in C#)
      tf.exp(), tf.sin() = same math as NumPy
      tf.math.log()      = natural log
      tf.concat()        = np.concatenate() / torch.cat()
    """
    # Position indices: [0, 1, 2, ..., max_seq_len-1]
    # tf.range = same as np.arange or torch.arange
    # We use tf.float32 for math compatibility
    position = tf.cast(tf.range(max_seq_len), tf.float32)    # (max_seq_len,)
    position = position[:, tf.newaxis]                         # (max_seq_len, 1)
    # tf.newaxis = same as np.newaxis or unsqueeze(1) in PyTorch

    # Dimension indices for even positions: [0, 1, 2, ..., d_model/2-1]
    i = tf.cast(tf.range(0, d_model, 2), tf.float32)         # (d_model/2,)

    # Division term: 10000^(2i/d_model) = exp(2i * log(10000) / d_model)
    # tf.math.log() = natural log in TF (same as np.log or torch.log)
    div_term = tf.exp(i * -(tf.math.log(tf.constant(10000.0)) / d_model))

    # Compute sine and cosine columns
    # position * div_term: (max_seq_len, 1) * (d_model/2,) = (max_seq_len, d_model/2)
    sine_values   = tf.sin(position * div_term)    # (max_seq_len, d_model/2)
    cosine_values = tf.cos(position * div_term)    # (max_seq_len, d_model/2)

    # TF tensors are IMMUTABLE, so we cannot do PE[:, 0::2] = sine_values
    # Instead, we interleave sine and cosine columns using tf.concat and reshape

    # Stack sine and cosine alternately:
    # Stack: (max_seq_len, d_model/2) and (max_seq_len, d_model/2)
    #   → (max_seq_len, d_model/2, 2)  [pairs: (sin, cos)]
    # Reshape to: (max_seq_len, d_model)
    PE = tf.reshape(
        tf.stack([sine_values, cosine_values], axis=-1),    # Stack along last dim
        [max_seq_len, d_model]                               # Flatten pairs
    )
    # Result: [sin(pos,0), cos(pos,0), sin(pos,1), cos(pos,1), ...]
    # This is equivalent to: PE[:, 0::2] = sin, PE[:, 1::2] = cos

    return PE

# Generate positional encodings
max_seq_len = 50
d_model     = 128

PE = positional_encoding_tf(max_seq_len, d_model)
print(f"Generated PE shape: {PE.shape}")
print(f"  Type: {PE.dtype}")
print(f"\nFirst 8 dimensions for position 0:")
print(PE[0, :8].numpy())
print(f"\nFirst 8 dimensions for position 10:")
print(PE[10, :8].numpy())

# ==============================================================================
# PART 3: PositionalEncoding as a Keras Layer
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: PositionalEncoding as a Keras Layer")
print("=" * 70)

print("""
In TF/Keras, we add positional encoding as a non-trainable Keras Layer.

Instead of PyTorch's register_buffer(), we simply store the PE tensor
as an attribute (it's not a trainable variable, so no special registration).

  PyTorch:  self.register_buffer('PE', PE_tensor)
  TF/Keras: self.PE = PE_tensor   ← just a regular attribute, not a tf.Variable!
             (TF doesn't train it because it's not a tf.Variable)
""")

class PositionalEncodingTF(keras.layers.Layer):
    """
    Positional Encoding as a non-trainable Keras Layer.

    Compare to PyTorch version:
      PyTorch: register_buffer() → moved to GPU, saved in state_dict
      TF:      plain attribute    → for GPU, use strategy.run(); saves with model
    """

    def __init__(self, d_model, max_seq_len=512, **kwargs):
        super().__init__(**kwargs)

        # Generate positional encoding and store as plain attribute
        # NOT a tf.Variable → not trainable
        self.PE = positional_encoding_tf(max_seq_len, d_model)    # (max_seq_len, d_model)

    def call(self, x, training=False):
        """
        Add positional encoding to input.

        x: (seq_len, d_model) or (batch_size, seq_len, d_model)
        """
        if len(x.shape) == 2:
            seq_len = tf.shape(x)[0]
            return x + self.PE[:seq_len]
        else:
            seq_len = tf.shape(x)[1]
            return x + self.PE[:seq_len][tf.newaxis, :, :]

# Test
pe_layer = PositionalEncodingTF(d_model=8, max_seq_len=100)

sentence = ["The", "cat", "sat", "on", "the", "mat"]
word_emb = tf.random.normal([len(sentence), 8]) * 0.5

enhanced = pe_layer(word_emb)

print(f"Input shape:  {word_emb.shape}")
print(f"Enhanced shape: {enhanced.shape}")
print(f"\nTrainable variables in PE layer: {len(pe_layer.trainable_variables)}")
print("  → 0 parameters! PE is fixed (not learned)")

# ==============================================================================
# PART 4: Applying to Word Embeddings
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Adding Positional Encoding to Word Embeddings")
print("=" * 70)

d_model_small = 8
PE_small = positional_encoding_tf(len(sentence), d_model_small)

# Element-wise addition
# NumPy:   enhanced = word_embeddings + PE_small
# PyTorch: enhanced = word_embeddings + PE_small   (SAME)
# TF:      enhanced = word_embeddings + PE_small   (SAME!)
enhanced_embeddings = word_emb + PE_small

print(f"Word embeddings shape: {word_emb.shape}")
print(f"PE shape:              {PE_small.shape}")
print(f"Enhanced shape:        {enhanced_embeddings.shape}")

word_idx = 2   # "sat"
print(f"\nFor word '{sentence[word_idx]}' at position {word_idx}:")
print(f"  Word embedding: {word_emb[word_idx].numpy().tolist()}")
print(f"  Position enc:   {PE_small[word_idx].numpy().tolist()}")
print(f"  Combined:       {enhanced_embeddings[word_idx].numpy().tolist()}")

# ==============================================================================
# PART 5: Visualizations
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Visualizations")
print("=" * 70)

PE_np = PE.numpy()

fig = plt.figure(figsize=(16, 10))

# Full PE heatmap
ax1 = plt.subplot(2, 3, 1)
plt.imshow(PE_np, cmap='RdBu', aspect='auto')
plt.colorbar(label='Encoding Value')
plt.title('Positional Encoding (TensorFlow)\n(50 positions x 128 dims)',
          fontsize=11, fontweight='bold')
plt.xlabel('Dimension')
plt.ylabel('Position')

# Encoding patterns per position
ax2 = plt.subplot(2, 3, 2)
for pos in [0, 5, 10, 20, 30, 40]:
    plt.plot(PE_np[pos], label=f'Position {pos}', alpha=0.7)
plt.title('Different Position Patterns', fontsize=11, fontweight='bold')
plt.xlabel('Dimension')
plt.legend(fontsize=8)
plt.grid(alpha=0.3)

# Dimensions varying across positions
ax3 = plt.subplot(2, 3, 3)
for dim in [0, 1, 2, 3]:
    plt.plot(PE_np[:, dim], label=f'Dim {dim}', alpha=0.7)
plt.title('Dims Varying Across Positions', fontsize=11, fontweight='bold')
plt.xlabel('Position')
plt.legend(fontsize=8)
plt.grid(alpha=0.3)

# Zoomed fingerprint
ax4 = plt.subplot(2, 3, 4)
plt.imshow(PE_np[:20, :20], cmap='RdBu', aspect='auto')
plt.colorbar(label='Value')
plt.title('Position Fingerprints (Zoomed)', fontsize=11, fontweight='bold')
plt.xlabel('Dimension')
plt.ylabel('Position')

# Before vs after for sentence
ax5 = plt.subplot(2, 3, 5)
plt.imshow(word_emb.numpy(), cmap='RdBu', aspect='auto')
plt.colorbar(label='Value')
plt.title('Word Embeddings Only', fontsize=11, fontweight='bold')
plt.yticks(range(len(sentence)), sentence)
plt.xlabel('Dimension')

ax6 = plt.subplot(2, 3, 6)
plt.imshow(enhanced_embeddings.numpy(), cmap='RdBu', aspect='auto')
plt.colorbar(label='Value')
plt.title('Word + Positional Encoding', fontsize=11, fontweight='bold')
plt.yticks(range(len(sentence)), sentence)
plt.xlabel('Dimension')

plt.tight_layout()
plt.show()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - Positional Encoding: TF vs NumPy vs PyTorch")
print("=" * 70)

print("""
KEY TF DIFFERENCES:

1. Tensor Immutability:
   NumPy/PyTorch: PE[:, 0::2] = np.sin(...)   ← in-place OK
   TensorFlow:    PE[:, 0::2] = tf.sin(...)   ← ERROR! Use tf.concat instead

2. Range and Arange:
   NumPy:   np.arange(n)
   PyTorch: torch.arange(n)
   TF:      tf.range(n)   (shorter name!)

3. Type Casting:
   NumPy:   arr.astype(float)
   PyTorch: tensor.float()
   TF:      tf.cast(tensor, tf.float32)

4. Log function:
   NumPy:   np.log(x)
   PyTorch: torch.log(x)
   TF:      tf.math.log(x)   (under tf.math namespace)

5. Concatenation:
   NumPy:   np.concatenate(arrays, axis=0)
   PyTorch: torch.cat(tensors, dim=0)
   TF:      tf.concat(tensors, axis=0)

6. Stack:
   NumPy:   np.stack(arrays, axis=0)
   PyTorch: torch.stack(tensors, dim=0)
   TF:      tf.stack(tensors, axis=0)

7. Non-trainable storage:
   PyTorch: register_buffer() (moves to GPU, saved in checkpoint)
   TF:      plain Python attribute (not tf.Variable → not trained)

Next:
  example_05: Complete transformer block (tf.keras.layers.MultiHeadAttention +
              tf.keras.layers.LayerNormalization)
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 04 - TensorFlow Version")
print("=" * 70)
