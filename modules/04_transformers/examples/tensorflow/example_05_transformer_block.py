"""
Example 05: Complete Transformer Block - TensorFlow Version

SAME example as NumPy/PyTorch versions, using Keras built-in components.

TF Keras provides even more built-in components than PyTorch!

NumPy (custom code)          │ PyTorch (built-in)                  │ TF/Keras (built-in)
─────────────────────────────┼─────────────────────────────────────┼──────────────────────────────
class LayerNorm: ... (25 ln)│ nn.LayerNorm(d_model)               │ LayerNormalization(axis=-1)
class FeedForward: ...       │ nn.Sequential(Linear, ReLU, Linear) │ Dense + Dense (in call())
class MultiHeadAttn: ...     │ nn.MultiheadAttention(d, h)         │ MultiHeadAttention(h, d_k)
class TransformerBlock: ...  │ nn.TransformerEncoderLayer(...)     │ Custom (no exact equivalent)

NOTE: TF does not have a single nn.TransformerEncoderLayer like PyTorch.
Instead, you build it from components (which is actually more flexible!).
TF 2.x has tf.keras.layers.MultiHeadAttention, LayerNormalization, Dense.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    MultiHeadAttention,
    Dense,
    LayerNormalization,
    Dropout
)
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(42)

print("=" * 70)
print("COMPLETE TRANSFORMER BLOCK - TensorFlow Version")
print("=" * 70)

# ==============================================================================
# PART 1: Feed-Forward Network
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Feed-Forward Network (FFN)")
print("=" * 70)

print("""
SAME CONCEPT:
  FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

PyTorch uses nn.Sequential:
  nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Linear(d_ff, d))

TF/Keras uses two Dense layers in the call() method:
  self.dense1 = Dense(d_ff, activation='relu')   ← ReLU built into Dense!
  self.dense2 = Dense(d_model)

  KEY DIFFERENCE: TF's Dense can include the activation function!
    PyTorch: nn.ReLU() is a separate layer
    TF:      Dense(units, activation='relu') combines both

  In C# terms:
    PyTorch: separate Transform() and Activate() pipeline stages
    TF:      combined TransformAndActivate() stage
""")

d_model = 8
d_ff    = 32
seq_len = 6

class FeedForwardTF(keras.layers.Layer):
    """
    Feed-Forward Network as a Keras Layer.

    Two Dense layers with ReLU in between.
    ReLU is built into the first Dense layer (activation='relu').
    """

    def __init__(self, d_model, d_ff, **kwargs):
        super().__init__(**kwargs)

        # Layer 1: Expand + ReLU (activation='relu' combines nn.Linear + nn.ReLU)
        # PyTorch: nn.Linear(d_model, d_ff) + nn.ReLU() (2 objects)
        # TF:      Dense(d_ff, activation='relu')         (1 object!)
        self.dense1 = Dense(d_ff, activation='relu')    # Expand and activate

        # Layer 2: Contract back (no activation)
        self.dense2 = Dense(d_model)                    # Contract to d_model

    def call(self, x, training=False):
        x = self.dense1(x, training=training)    # (seq, d_model) → (seq, d_ff) with ReLU
        x = self.dense2(x, training=training)    # (seq, d_ff) → (seq, d_model)
        return x

ffn = FeedForwardTF(d_model, d_ff)

x_test = tf.random.normal([seq_len, d_model])
output = ffn(x_test)

print(f"Input shape:  {x_test.shape}")
print(f"Output shape: {output.shape}")

for var in ffn.trainable_variables:
    print(f"  {var.name}: {var.shape} ({var.numpy().size:,} values)")

total_ffn = sum(v.numpy().size for v in ffn.trainable_variables)
print(f"Total FFN parameters: {total_ffn:,}")
print("✓ FeedForward works!")

# ==============================================================================
# PART 2: Layer Normalization
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Layer Normalization")
print("=" * 70)

print("""
Layer Normalization comparison:

  NumPy:   custom class (25 lines)
  PyTorch: nn.LayerNorm(d_model)           ← 1 line
  TF:      LayerNormalization(axis=-1)     ← 1 line (different parameter name!)

  DIFFERENCE: 'normalized_shape' vs 'axis'
    PyTorch: nn.LayerNorm([d_model])          ← specify WHICH dimensions to normalize
    TF:      LayerNormalization(axis=-1)      ← specify AXIS to normalize across

  Both normalize across the last dimension (features/d_model).
  Just different ways to say the same thing.
""")

# PyTorch: nn.LayerNorm(d_model)
# TF:      LayerNormalization(axis=-1)
layer_norm = LayerNormalization(axis=-1)

x_large = tf.random.normal([seq_len, d_model]) * 10.0    # Large values

print("BEFORE normalization:")
print(f"  Mean per position: {tf.reduce_mean(x_large, axis=-1).numpy()}")
print(f"  Std  per position: {tf.math.reduce_std(x_large, axis=-1).numpy()}")

output_norm = layer_norm(x_large)

print("\nAFTER normalization:")
print(f"  Mean per position: {tf.reduce_mean(output_norm, axis=-1).numpy()}")
print(f"  Std  per position: {tf.math.reduce_std(output_norm, axis=-1).numpy()}")

print(f"\nLayerNorm parameters: {len(layer_norm.trainable_variables)}")
for var in layer_norm.trainable_variables:
    print(f"  {var.name}: {var.shape}  initialized to: {var.numpy()[:4].tolist()}")
print("✓ LayerNormalization works!")

# ==============================================================================
# PART 3: Complete Transformer Block
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Complete Transformer Block")
print("=" * 70)

print("""
Architecture (same as NumPy/PyTorch):
  Input → MultiHeadAttention → Add+Norm → FeedForward → Add+Norm → Output

TF does NOT have a single built-in TransformerEncoderLayer like PyTorch.
But we can easily build one from components!

Compare:
  PyTorch (1 line):
    block = nn.TransformerEncoderLayer(d_model, nhead, d_ff, batch_first=True)

  TF (build from parts):
    self.attention = MultiHeadAttention(num_heads, key_dim)
    self.ffn = FeedForwardTF(d_model, d_ff)
    self.norm1 = LayerNormalization(axis=-1)
    self.norm2 = LayerNormalization(axis=-1)

This is actually MORE TRANSPARENT - you can see exactly what's happening!
""")

class TransformerBlockTF(keras.layers.Layer):
    """
    Complete Transformer Block in TensorFlow/Keras.

    Components:
      1. Multi-head self-attention
      2. Add + LayerNorm (residual connection + normalization)
      3. Feed-forward network
      4. Add + LayerNorm (residual connection + normalization)

    C# equivalent:
        class TransformerBlock : KerasLayerBase {
            MultiHeadAttention attention;
            FeedForwardTF ffn;
            LayerNormalization norm1, norm2;

            public Tensor Call(Tensor x) {
                var attn_out = attention.Call(x, x, x);
                x = norm1.Call(x + attn_out);
                var ffn_out = ffn.Call(x);
                x = norm2.Call(x + ffn_out);
                return x;
            }
        }
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)

        d_k = d_model // num_heads    # Per-head dimension

        # Multi-head attention
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_k            # TF uses per-head dimension (not total d_model!)
        )

        # Feed-forward network
        self.ffn = FeedForwardTF(d_model, d_ff)

        # Layer normalization (×2)
        self.norm1 = LayerNormalization(axis=-1)
        self.norm2 = LayerNormalization(axis=-1)

        # Dropout (optional regularization during training)
        self.dropout = Dropout(dropout_rate) if dropout_rate > 0 else None

    def call(self, x, training=False, mask=None):
        """
        Forward pass.

        x:       (seq_len, d_model) or (batch, seq_len, d_model)
        training: bool - training or inference mode
        mask:     optional attention mask (for causal masking)
        """
        # Handle 2D input
        squeeze = (len(x.shape) == 2)
        if squeeze:
            x = x[tf.newaxis, :, :]    # → (1, seq, d)

        # Sub-layer 1: Multi-head attention with residual + norm
        # Note: TF MultiHeadAttention uses (query, key, value) separately
        attn_out = self.attention(
            query=x, key=x, value=x,
            attention_mask=mask,
            training=training
        )
        if self.dropout:
            attn_out = self.dropout(attn_out, training=training)

        # Residual connection + layer norm
        # NumPy:   x = layer_norm(x + attn_output)
        # PyTorch: x = self.norm1(x + attn_output)
        # TF:      x = self.norm1(x + attn_out)     (SAME STRUCTURE!)
        x = self.norm1(x + attn_out)

        # Sub-layer 2: Feed-forward with residual + norm
        ffn_out = self.ffn(x, training=training)
        if self.dropout:
            ffn_out = self.dropout(ffn_out, training=training)

        x = self.norm2(x + ffn_out)

        if squeeze:
            x = x[0]    # Remove batch dim

        return x

# Test complete block
block = TransformerBlockTF(d_model=8, num_heads=2, d_ff=32)

x_input = tf.random.normal([6, 8])
output_block = block(x_input, training=False)

print(f"\nTesting Complete Transformer Block:")
print(f"  Input shape:  {x_input.shape}")
print(f"  Output shape: {output_block.shape}")

total_params = sum(v.numpy().size for v in block.trainable_variables)
print(f"  Total parameters: {total_params:,}")
print("  ✓ Complete transformer block works!")

# ==============================================================================
# PART 4: Stacking Multiple Blocks
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Stacking Multiple Transformer Blocks")
print("=" * 70)

print("""
NumPy:   blocks = [TransformerBlock(...) for _ in range(N)]
PyTorch: nn.TransformerEncoder(layer, num_layers=N)
TF:      manually stack in a list (or use tf.keras.Sequential / custom)

TF approach: store blocks in a list as a layer attribute.
  self.blocks = [TransformerBlockTF(...) for _ in range(num_blocks)]

  Keras automatically tracks parameters in lists of layers!
  (Works because Keras inspects all attributes for tf.Variables)
""")

num_blocks = 3

class TransformerStackTF(keras.layers.Layer):
    """Stack of transformer blocks."""

    def __init__(self, num_blocks, d_model, num_heads, d_ff, **kwargs):
        super().__init__(**kwargs)

        # List of transformer blocks - Keras tracks their parameters!
        self.blocks = [
            TransformerBlockTF(d_model, num_heads, d_ff)
            for _ in range(num_blocks)
        ]

    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        return x

stack = TransformerStackTF(num_blocks=3, d_model=8, num_heads=2, d_ff=32)
x_stacked = tf.random.normal([6, 8])
output_stacked = stack(x_stacked)

print(f"Stacking {num_blocks} transformer blocks:")
print(f"  Input:  {x_stacked.shape}")
print(f"  Output: {output_stacked.shape}")
print(f"  Total parameters: {sum(v.numpy().size for v in stack.trainable_variables):,}")
print("✓ Stacked transformer blocks work!")

# ==============================================================================
# PART 5: Visualization
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Visualization - Before and After")
print("=" * 70)

sentence = ["The", "cat", "sat", "on", "the", "mat"]
x_vis    = tf.random.normal([len(sentence), 8])
out_vis  = block(x_vis, training=False)

x_vis_np = x_vis.numpy()
out_np   = out_vis.numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].imshow(x_vis_np, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
axes[0].set_title('Input Embeddings (TensorFlow)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Position')
axes[0].set_yticks(range(len(sentence)))
axes[0].set_yticklabels(sentence)
axes[0].set_xlabel('Dimension')
plt.colorbar(im1, ax=axes[0], label='Value')

im2 = axes[1].imshow(out_np, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
axes[1].set_title('After Transformer Block\n(Attention + FFN + LayerNorm + Residuals)',
                  fontsize=12, fontweight='bold')
axes[1].set_ylabel('Position')
axes[1].set_yticks(range(len(sentence)))
axes[1].set_yticklabels(sentence)
axes[1].set_xlabel('Dimension')
plt.colorbar(im2, ax=axes[1], label='Value')

plt.tight_layout()
plt.show()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - Transformer Block: Full Comparison")
print("=" * 70)

print("""
THREE-WAY COMPARISON:

Component          │ NumPy (custom)           │ PyTorch (built-in)              │ TF (built-in)
───────────────────┼──────────────────────────┼─────────────────────────────────┼───────────────────────────────
FFN                │ class FeedForward (20 ln)│ nn.Sequential(Linear,ReLU,Lin.) │ Dense(d_ff, activation='relu')
LayerNorm          │ class LayerNorm (25 ln)  │ nn.LayerNorm(d_model)           │ LayerNormalization(axis=-1)
Multi-Head Attn    │ class MultiHeadAttn (40) │ nn.MultiheadAttention(d, h)     │ MultiHeadAttention(h, d_k)
Transformer Block  │ class TransformerBlock   │ nn.TransformerEncoderLayer(...)  │ Custom (from parts)
Stacked blocks     │ Python list of blocks    │ nn.TransformerEncoder(lyr, N)   │ Python list of blocks

ACTIVATION FUNCTION DIFFERENCE:
  PyTorch: nn.ReLU()  is a SEPARATE layer in nn.Sequential
  TF:      Dense(units, activation='relu')  includes activation INSIDE Dense

  PyTorch:  nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Linear(d_ff, d))
  TF:       Dense(d_ff, activation='relu')   +   Dense(d)

  Both compute the exact same math!

NAMING DIFFERENCES:
  PyTorch nn.LayerNorm(d_model)        → normalized_shape = d_model
  TF LayerNormalization(axis=-1)       → normalize across last axis

  PyTorch nn.MultiheadAttention(embed_dim=d_model, num_heads=H)
  TF MultiHeadAttention(num_heads=H, key_dim=d_model//H)

TF ADVANTAGES:
  ✓ Dense layer includes activation (cleaner code)
  ✓ Keras model.summary() shows full architecture
  ✓ Easy to convert to TensorFlow Lite for mobile deployment
  ✓ Google Cloud TPU support built-in

Next:
  example_06: Mini-GPT using tf.keras.layers.Embedding + full model
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 05 - TensorFlow Version")
print("=" * 70)
