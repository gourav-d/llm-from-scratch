"""
Example 05: Complete Transformer Block - PyTorch Version

SAME example as NumPy version, using PyTorch's built-in components.

The BIGGEST benefit of PyTorch here: many components are built-in!

NumPy (you wrote everything):       │ PyTorch (all built-in!):
───────────────────────────────────┼──────────────────────────────────────
class LayerNorm: ... (25 lines)    │ nn.LayerNorm(d_model)
class FeedForward: ... (20 lines)  │ nn.Linear + nn.ReLU + nn.Linear
class TransformerBlock: ...        │ nn.TransformerEncoderLayer(...)

In C# terms:
  NumPy  = writing your own Pipeline, Normalizer, and LinearLayer classes
  PyTorch = using .NET built-in middleware, normalization, and ML layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

print("=" * 70)
print("COMPLETE TRANSFORMER BLOCK - PyTorch Version")
print("=" * 70)

# ==============================================================================
# PART 1: Feed-Forward Network with PyTorch
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Feed-Forward Network (FFN)")
print("=" * 70)

print("""
SAME CONCEPT as NumPy version:
  FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

PYTORCH IMPLEMENTATION: Use nn.Sequential and nn.Linear

  NumPy way:
    class FeedForward:
        def __init__(self, d_model, d_ff):
            self.W1 = np.random.randn(d_model, d_ff) * 0.1
            self.b1 = np.zeros(d_ff)
            ...
        def forward(self, x):
            hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
            return hidden @ self.W2 + self.b2

  PyTorch way (using built-in layers):
    nn.Linear(d_model, d_ff)     # Replaces W1 + b1 + matrix multiply
    nn.ReLU()                    # Replaces np.maximum(0, x)
    nn.Linear(d_ff, d_model)     # Replaces W2 + b2 + matrix multiply

  nn.Sequential: chains layers together, like LINQ method chaining in C#:
    data.Select(...).Where(...).GroupBy(...)
    becomes:
    nn.Sequential(nn.Linear(...), nn.ReLU(), nn.Linear(...))
""")

class FeedForward(nn.Module):
    """
    Feed-Forward Network using PyTorch built-in layers.

    FFN(x) = ReLU(Linear(x)) → Linear(...)
    """

    def __init__(self, d_model, d_ff):
        super().__init__()

        # nn.Sequential: runs layers one after another
        # Like a pipeline: input → Linear → ReLU → Linear → output
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),    # Expand:   (seq, d_model) → (seq, d_ff)
            nn.ReLU(),                    # Activate: max(0, x) - no parameters
            nn.Linear(d_ff, d_model)     # Contract: (seq, d_ff) → (seq, d_model)
        )

    def forward(self, x):
        return self.network(x)    # One call runs all three layers!

# Test FFN
d_model = 8
d_ff    = 32
seq_len = 6

ffn = FeedForward(d_model, d_ff)

x_test      = torch.randn(seq_len, d_model)
output_ffn  = ffn(x_test)

print(f"Input shape:  {x_test.shape}")
print(f"Output shape: {output_ffn.shape}")

print(f"\nFFN parameters:")
for name, param in ffn.named_parameters():
    print(f"  {name}: {param.shape} ({param.numel():,} values)")

total_ffn = sum(p.numel() for p in ffn.parameters())
print(f"Total FFN parameters: {total_ffn:,}")
print("✓ FeedForward works!")

# ==============================================================================
# PART 2: Layer Normalization with PyTorch
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Layer Normalization")
print("=" * 70)

print("""
SAME CONCEPT as NumPy version:
  LayerNorm(x) = γ × (x - mean) / √(variance + ε) + β

PYTORCH IMPLEMENTATION: nn.LayerNorm - completely built-in!

  NumPy way (25 lines of code):
    class LayerNorm:
        def __init__(self, d_model, epsilon=1e-6):
            self.gamma = np.ones(d_model)
            self.beta = np.zeros(d_model)
        def forward(self, x):
            mean = np.mean(x, axis=-1, keepdims=True)
            variance = np.var(x, axis=-1, keepdims=True)
            x_norm = (x - mean) / np.sqrt(variance + self.epsilon)
            return self.gamma * x_norm + self.beta

  PyTorch way (1 line!):
    layer_norm = nn.LayerNorm(d_model)
    output = layer_norm(x)
""")

# NumPy: custom LayerNorm class (25 lines)
# PyTorch: ONE LINE!
layer_norm = nn.LayerNorm(d_model)

# Test normalization
x_large = torch.randn(seq_len, d_model) * 10.0    # Large scale input

print(f"BEFORE normalization:")
print(f"  Mean per position: {x_large.mean(dim=-1).tolist()}")
print(f"  Std  per position: {x_large.std(dim=-1).tolist()}")

output_norm = layer_norm(x_large.detach())

print(f"\nAFTER normalization:")
print(f"  Mean per position: {output_norm.mean(dim=-1).tolist()}")
print(f"  Std  per position: {output_norm.std(dim=-1).tolist()}")

print(f"\nLayerNorm parameters (gamma and beta):")
for name, param in layer_norm.named_parameters():
    print(f"  {name}: {param.shape}  initialized to: {param[:4].tolist()}")

print("✓ LayerNorm works!")

# ==============================================================================
# PART 3: Residual Connections
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Residual Connections")
print("=" * 70)

print("""
SAME CONCEPT as NumPy version:
  With residual: output = x + Layer(x)

PyTorch: Residual connections are just ADDITION of tensors.
  IDENTICAL to NumPy! No special function needed.

  NumPy:   output = x + layer_output
  PyTorch: output = x + layer_output    (EXACTLY THE SAME!)

The key difference: in PyTorch, this addition is tracked for gradients
automatically, so backpropagation works through residual connections.
""")

x_demo = torch.tensor([1.0, 2.0, 3.0, 4.0])
layer_transform = torch.tensor([0.1, -0.2, 0.3, -0.1])

output_with_residual = x_demo + layer_transform    # That's it!

print(f"Input:                 {x_demo.tolist()}")
print(f"Layer output:          {layer_transform.tolist()}")
print(f"With residual (x+out): {output_with_residual.tolist()}")
print("\n✓ Residual = just tensor addition!")

# ==============================================================================
# PART 4: Complete Transformer Block - Two Ways
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Complete Transformer Block")
print("=" * 70)

print("""
We'll show TWO approaches:
  A) Manual implementation (to see the structure clearly)
  B) Using nn.TransformerEncoderLayer (PyTorch's built-in!)

Architecture (same as NumPy):
  Input → MultiHeadAttention → Add+Norm → FeedForward → Add+Norm → Output
""")

print("\n--- Approach A: Manual Implementation ---")

class TransformerBlockManual(nn.Module):
    """
    Transformer block built from individual components.
    Same structure as the NumPy version, but using PyTorch layers.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff)

        # Layer normalization (×2 - one after attention, one after FFN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Forward pass.

        x shape: (seq_len, d_model) or (batch, seq_len, d_model)
        """
        # Handle 2D input (add batch dimension)
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)   # → (1, seq_len, d_model)

        # Sub-layer 1: Multi-head attention with residual + norm
        attn_out, _ = self.attention(x, x, x)         # Q=K=V=x (self-attention)
        x = self.norm1(x + attn_out)                   # Residual + LayerNorm

        # Sub-layer 2: Feed-forward with residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)                    # Residual + LayerNorm

        if squeeze:
            x = x.squeeze(0)    # Remove batch dim

        return x

block_manual = TransformerBlockManual(d_model=8, num_heads=2, d_ff=32)
x_input = torch.randn(6, 8)

with torch.no_grad():
    output_manual = block_manual(x_input)

print(f"  Input shape:  {x_input.shape}")
print(f"  Output shape: {output_manual.shape}")
print(f"  Parameters: {sum(p.numel() for p in block_manual.parameters()):,}")
print("  ✓ Manual transformer block works!")

# ==============================================================================

print("\n--- Approach B: nn.TransformerEncoderLayer (Built-in!) ---")

print("""
PyTorch provides nn.TransformerEncoderLayer which does everything!

  NumPy (120 lines): Custom LayerNorm + FeedForward + SimplifiedMHA + TransformerBlock
  PyTorch (1 line!): nn.TransformerEncoderLayer(d_model, nhead, d_ff, batch_first=True)
""")

# ONE LINE replaces all our custom code!
block_builtin = nn.TransformerEncoderLayer(
    d_model=d_model,       # Embedding dimension
    nhead=2,               # Number of attention heads
    dim_feedforward=d_ff,  # FFN hidden dimension (same as d_ff in NumPy)
    dropout=0.0,           # No dropout for demo
    batch_first=True       # Input: (batch, seq, features)
)

x_batched = x_input.unsqueeze(0)    # Add batch dim: (1, 6, 8)

with torch.no_grad():
    output_builtin = block_builtin(x_batched)

output_builtin = output_builtin.squeeze(0)   # Remove batch dim

print(f"  Input shape:  {x_input.shape}")
print(f"  Output shape: {output_builtin.shape}")
print(f"  Parameters: {sum(p.numel() for p in block_builtin.parameters()):,}")
print("  ✓ Built-in TransformerEncoderLayer works!")

# ==============================================================================
# PART 5: Stacking Multiple Blocks
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Stacking Multiple Transformer Blocks")
print("=" * 70)

print("""
NumPy way: blocks = [TransformerBlock(...) for _ in range(num_blocks)]

PyTorch way (using nn.TransformerEncoder):
  single_layer = nn.TransformerEncoderLayer(...)
  full_encoder = nn.TransformerEncoder(single_layer, num_layers=num_blocks)

  nn.TransformerEncoder automatically stacks num_layers copies!
  Like calling .Repeat(n) on a middleware pipeline in C#.
""")

num_blocks = 3

# Create one encoder layer template
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=2,
    dim_feedforward=d_ff,
    dropout=0.0,
    batch_first=True
)

# Stack num_blocks copies
transformer_encoder = nn.TransformerEncoder(
    encoder_layer=encoder_layer,
    num_layers=num_blocks
)

x_batched = x_input.unsqueeze(0)   # (1, 6, 8)

with torch.no_grad():
    output_stacked = transformer_encoder(x_batched)

output_stacked = output_stacked.squeeze(0)   # (6, 8)

print(f"Stacking {num_blocks} transformer blocks:")
print(f"  Input:  {x_input.shape}")
print(f"  Output: {output_stacked.shape}")
print(f"  Total parameters: {sum(p.numel() for p in transformer_encoder.parameters()):,}")
print("✓ Stacked transformer blocks work!")

# ==============================================================================
# PART 6: Visualization - Before and After
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Visualization - Information Flow")
print("=" * 70)

sentence = ["The", "cat", "sat", "on", "the", "mat"]
seq_len  = len(sentence)
x_vis    = torch.randn(seq_len, d_model)

with torch.no_grad():
    output_vis = block_manual(x_vis)

# Convert to NumPy for plotting
x_vis_np   = x_vis.numpy()
output_np  = output_vis.numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].imshow(x_vis_np, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
axes[0].set_title('Input Embeddings (PyTorch)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Position')
axes[0].set_yticks(range(seq_len))
axes[0].set_yticklabels(sentence)
axes[0].set_xlabel('Dimension')
plt.colorbar(im1, ax=axes[0], label='Value')

im2 = axes[1].imshow(output_np, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
axes[1].set_title('After Transformer Block\n(Attention + FFN + LayerNorm + Residuals)',
                  fontsize=12, fontweight='bold')
axes[1].set_ylabel('Position')
axes[1].set_yticks(range(seq_len))
axes[1].set_yticklabels(sentence)
axes[1].set_xlabel('Dimension')
plt.colorbar(im2, ax=axes[1], label='Value')

plt.tight_layout()
plt.show()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - Transformer Block: NumPy vs PyTorch")
print("=" * 70)

print("""
COMPONENT COMPARISON:

Component              │ NumPy (custom code)          │ PyTorch (built-in)
───────────────────────┼──────────────────────────────┼────────────────────────────
Feed-Forward           │ class FeedForward: (20 lines)│ nn.Linear + nn.ReLU + nn.Linear
Layer Normalization    │ class LayerNorm: (25 lines)  │ nn.LayerNorm(d_model)
Multi-head Attention   │ class MultiHeadAttention:... │ nn.MultiheadAttention(...)
Transformer Block      │ class TransformerBlock:...   │ nn.TransformerEncoderLayer(...)
Stacked blocks         │ [TransformerBlock() ×N]     │ nn.TransformerEncoder(..., N)

PYTORCH TOOLS LEARNED:
  ✓ nn.Sequential       → chain layers into a pipeline
  ✓ nn.LayerNorm        → built-in layer normalization
  ✓ nn.ReLU             → activation function layer
  ✓ nn.TransformerEncoderLayer → complete transformer block
  ✓ nn.TransformerEncoder     → stacked transformer blocks

KEY OBSERVATION:
  PyTorch drastically reduces the amount of code you need to write!
  This means less bugs, better tested code, and GPU-ready automatically.

Next:
  example_06: Mini-GPT using nn.Embedding + full transformer
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 05 - PyTorch Version")
print("=" * 70)
