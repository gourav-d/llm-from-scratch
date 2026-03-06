"""
Example 05: Complete Transformer Block

This example builds a COMPLETE transformer block - the fundamental building
block of GPT, BERT, and all modern transformers. We'll combine everything
we've learned so far!

What you'll see:
1. Feed-Forward Network (position-wise neural network)
2. Layer Normalization (stabilizes training)
3. Residual Connections (helps gradient flow)
4. Complete Transformer Block (putting it all together)
5. Stacking multiple blocks

Think of it like a factory assembly line:
- Attention: Workers communicate and share information
- Feed-Forward: Each worker processes what they learned
- Layer Norm: Quality control checkpoints
- Residual: Safety net that preserves original information
- Stacking: Multiple processing stages for complex transformations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("COMPLETE TRANSFORMER BLOCK")
print("=" * 70)

# ==============================================================================
# PART 1: Feed-Forward Network (FFN)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Feed-Forward Network (FFN)")
print("=" * 70)

print("""
FEED-FORWARD NETWORK: A simple 2-layer neural network applied to each
position independently.

Architecture:
  Input (d_model) → Expand (d_ff) → ReLU → Contract (d_model) → Output

Formula:
  FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

Where:
  - W1: (d_model, d_ff) - expands to larger dimension
  - b1: (d_ff,) - bias for first layer
  - W2: (d_ff, d_model) - projects back to original dimension
  - b2: (d_model,) - bias for second layer
  - ReLU: max(0, x) - activation function
  - d_ff = 4 × d_model (typical in transformers)

Why FFN?
  - Attention: aggregates information (communication)
  - FFN: transforms information (computation)
  - Both are needed!

In C# terms:
  class FeedForward {
      Matrix W1, W2;
      Vector b1, b2;

      Vector Forward(Vector x) {
          var hidden = ReLU(x * W1 + b1);  // Expand & activate
          return hidden * W2 + b2;           // Contract
      }
  }
""")

class FeedForward:
    """
    Position-wise Feed-Forward Network.

    Applied to each position (word) independently.
    Like LINQ Select(): processes each element separately.
    """

    def __init__(self, d_model, d_ff):
        """
        Initialize FFN.

        Args:
            d_model: Input/output dimension
            d_ff: Hidden layer dimension (typically 4× d_model)
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Initialize weights with small random values
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)

        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input, shape (seq_len, d_model) or (d_model,)

        Returns:
            Output, same shape as input
        """
        # First layer: expand and activate
        # x @ W1: (seq_len, d_model) @ (d_model, d_ff) = (seq_len, d_ff)
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU activation

        # Second layer: contract back to d_model
        # hidden @ W2: (seq_len, d_ff) @ (d_ff, d_model) = (seq_len, d_model)
        output = hidden @ self.W2 + self.b2

        return output

# Test FFN
d_model = 8
d_ff = 32  # 4× d_model
seq_len = 6

print(f"\nConfiguration:")
print(f"  d_model: {d_model}")
print(f"  d_ff: {d_ff} (4× d_model)")
print(f"  Sequence length: {seq_len}")

# Create FFN
ffn = FeedForward(d_model, d_ff)

# Test input
x_test = np.random.randn(seq_len, d_model)
output_ffn = ffn.forward(x_test)

print(f"\nTest forward pass:")
print(f"  Input shape: {x_test.shape}")
print(f"  Output shape: {output_ffn.shape}")
print("  ✓ FFN works! Each position processed independently")

# ==============================================================================
# PART 2: Layer Normalization
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Layer Normalization")
print("=" * 70)

print("""
LAYER NORMALIZATION: Normalizes features across the embedding dimension
for each position (word) independently.

Formula:
  LayerNorm(x) = γ × (x - mean) / √(variance + ε) + β

Where:
  - mean: average across embedding dimension
  - variance: variance across embedding dimension
  - ε (epsilon): small value to prevent division by zero (1e-6)
  - γ (gamma): learned scaling parameter (initialized to 1)
  - β (beta): learned shift parameter (initialized to 0)

Why Layer Norm?
  ✓ Stabilizes training (prevents exploding/vanishing gradients)
  ✓ Allows higher learning rates
  ✓ Reduces sensitivity to initialization
  ✓ Applied AFTER attention and FFN in transformers

Similar to C# normalization:
  double[] LayerNorm(double[] x) {
      double mean = x.Average();
      double variance = x.Select(v => Math.Pow(v - mean, 2)).Average();
      return x.Select(v => (v - mean) / Math.Sqrt(variance + epsilon))
              .ToArray();
  }
""")

class LayerNorm:
    """
    Layer Normalization.

    Normalizes across the feature dimension (not batch dimension).
    Each position is normalized independently.
    """

    def __init__(self, d_model, epsilon=1e-6):
        """
        Initialize LayerNorm.

        Args:
            d_model: Dimension to normalize across
            epsilon: Small value for numerical stability
        """
        self.d_model = d_model
        self.epsilon = epsilon

        # Learnable parameters (initialized to identity transform)
        self.gamma = np.ones(d_model)   # Scale
        self.beta = np.zeros(d_model)   # Shift

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input, shape (seq_len, d_model) or (d_model,)

        Returns:
            Normalized output, same shape as input
        """
        # Compute mean and variance across last dimension (d_model)
        # keepdims=True preserves shape for broadcasting
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / np.sqrt(variance + self.epsilon)

        # Scale and shift (learnable)
        output = self.gamma * x_norm + self.beta

        return output

# Test LayerNorm
print(f"\nTesting Layer Normalization:")

ln = LayerNorm(d_model)

# Create test input with large values
x_unnorm = np.random.randn(seq_len, d_model) * 10  # Large scale

output_norm = ln.forward(x_unnorm)

print(f"  Input mean per position: {np.mean(x_unnorm, axis=-1)}")
print(f"  Input std per position: {np.std(x_unnorm, axis=-1)}")
print(f"\n  Output mean per position: {np.mean(output_norm, axis=-1)}")
print(f"  Output std per position: {np.std(output_norm, axis=-1)}")
print("\n  ✓ Layer Norm normalizes to ~mean=0, ~std=1!")

# ==============================================================================
# PART 3: Residual Connections
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Residual Connections (Skip Connections)")
print("=" * 70)

print("""
RESIDUAL CONNECTIONS: Add the input to the output of a layer.

Without residual:
  output = Layer(x)

With residual:
  output = x + Layer(x)

Why residuals?
  ✓ Helps gradient flow in deep networks
  ✓ Preserves original information
  ✓ Allows training very deep networks (100+ layers!)
  ✓ Provides "highway" for information flow

Analogy: Safety Net
  - Layer transformation is risky (might destroy information)
  - Residual connection = safety net
  - If layer fails, original signal still flows through
  - Layer only needs to learn the "delta" (change)

In transformers:
  1. output = x + MultiHeadAttention(x)
  2. output = x + FeedForward(x)

Similar to C# error handling:
  try {
      result = RiskyTransformation(input);
  } catch {
      result = input;  // Fallback to original
  }
  // Residual is like: result = input + RiskyTransformation(input)
""")

def residual_connection(x, layer_output):
    """
    Apply residual connection.

    Args:
        x: Original input
        layer_output: Output from layer (attention or FFN)

    Returns:
        x + layer_output
    """
    return x + layer_output

# Demonstrate residual
print("\nResidual Connection Example:")
x_input = np.array([1.0, 2.0, 3.0, 4.0])
layer_transform = np.array([0.1, -0.2, 0.3, -0.1])  # Small changes

output_no_residual = layer_transform
output_with_residual = residual_connection(x_input, layer_transform)

print(f"  Original input: {x_input}")
print(f"  Layer output (without residual): {output_no_residual}")
print(f"  Layer output (with residual): {output_with_residual}")
print("\n  ✓ Residual preserves original signal while adding learned changes!")

# ==============================================================================
# PART 4: Complete Transformer Block
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Complete Transformer Block")
print("=" * 70)

print("""
COMPLETE TRANSFORMER BLOCK: Combines all components!

Architecture:
  Input
    ↓
  Multi-Head Attention
    ↓
  Add (residual) & Normalize
    ↓
  Feed-Forward Network
    ↓
  Add (residual) & Normalize
    ↓
  Output

Pseudocode:
  def TransformerBlock(x):
      # Sub-layer 1: Multi-head attention
      attn_output = MultiHeadAttention(x)
      x = LayerNorm(x + attn_output)  # Residual + Norm

      # Sub-layer 2: Feed-forward
      ffn_output = FeedForward(x)
      x = LayerNorm(x + ffn_output)  # Residual + Norm

      return x

This is ONE transformer block. GPT-3 has 96 of these stacked!
""")

# Simple multi-head attention for completeness (simplified version)
def softmax(x, axis=-1):
    """Softmax function."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class SimplifiedMultiHeadAttention:
    """Simplified multi-head attention for demonstration."""

    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def forward(self, x):
        # Simplified single-head attention for demo
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        scores = Q @ K.T / np.sqrt(self.d_model)
        weights = softmax(scores, axis=-1)
        output = weights @ V

        return output @ self.W_o

class TransformerBlock:
    """
    Complete Transformer Block.

    Combines:
      - Multi-head attention
      - Feed-forward network
      - Layer normalization (×2)
      - Residual connections (×2)
    """

    def __init__(self, d_model, num_heads, d_ff):
        """
        Initialize transformer block.

        Args:
            d_model: Embedding dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
        """
        self.attention = SimplifiedMultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x):
        """
        Forward pass through transformer block.

        Args:
            x: Input, shape (seq_len, d_model)

        Returns:
            Output, same shape as input
        """
        # Sub-layer 1: Multi-head attention with residual and norm
        attn_output = self.attention.forward(x)
        x = self.norm1.forward(x + attn_output)  # Residual + LayerNorm

        # Sub-layer 2: Feed-forward with residual and norm
        ffn_output = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_output)  # Residual + LayerNorm

        return x

# Test complete block
print("\nTesting Complete Transformer Block:")

transformer_block = TransformerBlock(d_model=8, num_heads=2, d_ff=32)

x_input = np.random.randn(6, 8)  # 6 words, 8-dim embeddings
output_block = transformer_block.forward(x_input)

print(f"  Input shape: {x_input.shape}")
print(f"  Output shape: {output_block.shape}")
print("  ✓ Complete transformer block works!")

# ==============================================================================
# PART 5: Stacking Multiple Blocks
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Stacking Multiple Transformer Blocks")
print("=" * 70)

print("""
DEEP TRANSFORMERS: Stack multiple transformer blocks!

Architecture:
  Input
    ↓
  Block 1
    ↓
  Block 2
    ↓
  Block 3
    ↓
  ...
    ↓
  Block N
    ↓
  Output

Each block processes and refines the representations.

Model sizes:
  - GPT-2 Small: 12 blocks
  - GPT-2 Large: 36 blocks
  - GPT-3: 96 blocks
  - GPT-4: Unknown, but likely 100+ blocks!
""")

# Create a stack of transformer blocks
num_blocks = 3
blocks = [TransformerBlock(d_model=8, num_heads=2, d_ff=32)
          for _ in range(num_blocks)]

# Pass input through all blocks
x_stacked = np.random.randn(6, 8)
print(f"\nPassing through {num_blocks} stacked blocks:")
print(f"  Initial input shape: {x_stacked.shape}")

for i, block in enumerate(blocks):
    x_stacked = block.forward(x_stacked)
    print(f"  After block {i+1}: {x_stacked.shape}")

print("\n  ✓ Stacked transformer blocks work!")
print("    Each block refines the representations further!")

# ==============================================================================
# PART 6: Visualization - Information Flow
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Visualizing Transformer Block")
print("=" * 70)

# Process sentence through one block and visualize
sentence = ["The", "cat", "sat", "on", "the", "mat"]
seq_len_vis = len(sentence)
d_model_vis = 8

# Create input embeddings
x_vis = np.random.randn(seq_len_vis, d_model_vis)

# Create block
vis_block = TransformerBlock(d_model=d_model_vis, num_heads=2, d_ff=32)

# Process
output_vis = vis_block.forward(x_vis)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Input
im1 = axes[0].imshow(x_vis, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
axes[0].set_title('Input Embeddings', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Position')
axes[0].set_yticks(range(seq_len_vis))
axes[0].set_yticklabels(sentence)
axes[0].set_xlabel('Dimension')
plt.colorbar(im1, ax=axes[0], label='Value')

# Output after transformer block
im2 = axes[1].imshow(output_vis, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
axes[1].set_title('After Transformer Block\n(Attention + FFN + Norms + Residuals)',
                  fontsize=12, fontweight='bold')
axes[1].set_ylabel('Position')
axes[1].set_yticks(range(seq_len_vis))
axes[1].set_yticklabels(sentence)
axes[1].set_xlabel('Dimension')
plt.colorbar(im2, ax=axes[1], label='Value')

plt.tight_layout()
plt.show()

print("\nVisualization shows:")
print("  - Left: Input word embeddings")
print("  - Right: After transformer block processing")
print("  - Embeddings are refined to be context-aware!")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - What We Learned")
print("=" * 70)

print("""
✓ Feed-Forward Network (FFN):
  - 2-layer neural network
  - Applied to each position independently
  - Formula: FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
  - Adds computation to attention's communication

✓ Layer Normalization:
  - Normalizes across feature dimension
  - Formula: (x - mean) / √variance
  - Stabilizes training, allows deeper networks

✓ Residual Connections:
  - Add input to layer output: x + Layer(x)
  - Helps gradient flow
  - Preserves information through deep networks

✓ Complete Transformer Block:
  1. Multi-head attention + residual + norm
  2. Feed-forward + residual + norm
  - This is the FUNDAMENTAL building block!

✓ Stacking Blocks:
  - Real transformers stack 12-96+ blocks
  - Each block refines representations
  - Deeper = more powerful patterns

Component Breakdown:
  - Attention: 20-30% of computation
  - FFN: 70-80% of computation (most parameters!)
  - Layer Norm: <1% of computation
  - Residuals: Free (just addition)

In C#/.NET Terms:
  - FFN is like pipeline stages: Expand → Transform → Contract
  - LayerNorm is like data normalization
  - Residual is like try-catch with fallback
  - Stacking is like middleware pipeline in ASP.NET

Parameter Count (typical):
  For d_model=768, d_ff=3072:
    - Attention: 4 × 768² = 2.4M parameters
    - FFN: 768×3072 + 3072×768 = 4.7M parameters
    - Total per block: ~7M parameters
    - GPT-2 (12 blocks): ~85M parameters
    - GPT-3 (96 blocks): ~175B parameters!

Next Steps:
  - example_06: Building a mini-GPT (THE FINALE!)
    - Token embeddings
    - Causal masking
    - Language model head
    - Text generation!
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 05")
print("=" * 70)
