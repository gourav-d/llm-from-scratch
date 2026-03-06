"""
Exercise 03: Building a Complete Transformer Block

🎯 GOAL: Build a complete transformer block - the real deal!

This is the FINAL exercise where you combine everything to build a production-ready
transformer block. This is THE building block of GPT, BERT, and all modern LLMs!

Components you'll build:
  1. LayerNorm class
  2. FeedForward network class
  3. TransformerBlock class (combines everything!)
  4. Stack multiple blocks
  5. BONUS: Analyze representation evolution

Tasks:
  1. Implement LayerNorm class
  2. Implement FeedForward class
  3. Build TransformerBlock class
  4. Stack and test multiple blocks
  5. BONUS: Visualize how representations evolve

Hints for .NET developers:
  - Classes similar to C# class hierarchy
  - Forward methods like pipeline stages
  - Normalization like data standardization
  - Stacking like middleware pipeline

This is ADVANCED - take your time! Good luck! 🚀
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print("=" * 70)
print("EXERCISE 03: Build Complete Transformer Block")
print("=" * 70)

# ==============================================================================
# GIVEN: Setup and Helper Functions
# ==============================================================================

print("\n" + "=" * 70)
print("SETUP: Given Information")
print("=" * 70)

sentence = ["The", "cat", "sat", "on", "the", "mat"]
d_model = 8
d_ff = 32  # Feed-forward dimension (typically 4× d_model)
seq_len = len(sentence)

print(f"Sentence: {' '.join(sentence)}")
print(f"Embedding dimension (d_model): {d_model}")
print(f"Feed-forward dimension (d_ff): {d_ff}")

# Create input embeddings (given)
X = np.random.randn(seq_len, d_model)

print(f"\nInput shape: {X.shape}")

# Softmax function (from previous exercises)
def softmax(x, axis=-1):
    """Apply softmax function."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Simplified self-attention (given for this exercise)
class SimpleSelfAttention:
    """Simplified self-attention for exercise."""

    def __init__(self, d_model):
        self.d_model = d_model
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1

    def forward(self, x):
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        scores = Q @ K.T / np.sqrt(self.d_model)
        weights = softmax(scores, axis=-1)
        return weights @ V

# ==============================================================================
# TODO 1: Implement LayerNorm Class
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 1: Implement Layer Normalization")
print("=" * 70)

print("""
TASK: Implement Layer Normalization class.

LayerNorm normalizes features across the embedding dimension for each
position independently.

Formula:
  LayerNorm(x) = γ × (x - mean) / √(variance + ε) + β

Where:
  - mean: average across last dimension (embedding dim)
  - variance: variance across last dimension
  - ε (epsilon): small constant for numerical stability (1e-6)
  - γ (gamma): learnable scale parameter (initialized to 1)
  - β (beta): learnable shift parameter (initialized to 0)

Steps:
  1. In __init__:
     - Store epsilon
     - Initialize gamma to ones: np.ones(d_model)
     - Initialize beta to zeros: np.zeros(d_model)

  2. In forward(x):
     - Compute mean: np.mean(x, axis=-1, keepdims=True)
     - Compute variance: np.var(x, axis=-1, keepdims=True)
     - Normalize: (x - mean) / sqrt(variance + epsilon)
     - Scale and shift: gamma * normalized + beta

Hint for C# developers:
  - Like data normalization: (value - avg) / stddev
  - keepdims=True preserves shape for broadcasting
  - gamma/beta are like trainable multiplier/offset
""")

# TODO: Your code here

# SOLUTION (uncomment to check):
# class LayerNorm:
#     """Layer Normalization."""
#
#     def __init__(self, d_model, epsilon=1e-6):
#         """
#         Initialize LayerNorm.
#
#         Args:
#             d_model: Dimension to normalize across
#             epsilon: Small value for numerical stability
#         """
#         self.d_model = d_model
#         self.epsilon = epsilon
#
#         # Learnable parameters (initialized to identity transform)
#         self.gamma = np.ones(d_model)   # Scale
#         self.beta = np.zeros(d_model)   # Shift
#
#     def forward(self, x):
#         """
#         Apply layer normalization.
#
#         Args:
#             x: Input, shape (seq_len, d_model) or (d_model,)
#
#         Returns:
#             Normalized output, same shape as input
#         """
#         # Compute statistics across last dimension
#         mean = np.mean(x, axis=-1, keepdims=True)
#         variance = np.var(x, axis=-1, keepdims=True)
#
#         # Normalize
#         x_norm = (x - mean) / np.sqrt(variance + self.epsilon)
#
#         # Scale and shift
#         return self.gamma * x_norm + self.beta

# Test your LayerNorm (uncomment after implementing):
# print("\nTesting LayerNorm:")
# ln = LayerNorm(d_model)
# x_test = np.random.randn(seq_len, d_model) * 10
# x_normalized = ln.forward(x_test)
# print(f"  Input mean per position: {np.mean(x_test, axis=-1)}")
# print(f"  Output mean per position: {np.mean(x_normalized, axis=-1)}")
# print("  ✓ LayerNorm works!")

# ==============================================================================
# TODO 2: Implement FeedForward Class
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 2: Implement Feed-Forward Network")
print("=" * 70)

print("""
TASK: Implement Feed-Forward Network class.

FFN is a simple 2-layer neural network applied to each position independently.

Architecture:
  Input (d_model) → Expand (d_ff) → ReLU → Contract (d_model) → Output

Formula:
  FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

Where:
  - W1: (d_model, d_ff) - expands to larger dimension
  - b1: (d_ff,) - bias for first layer
  - W2: (d_ff, d_model) - contracts back to d_model
  - b2: (d_model,) - bias for second layer
  - ReLU: max(0, x) - activation function

Steps:
  1. In __init__:
     - Initialize W1: np.random.randn(d_model, d_ff) * 0.1
     - Initialize b1: np.zeros(d_ff)
     - Initialize W2: np.random.randn(d_ff, d_model) * 0.1
     - Initialize b2: np.zeros(d_model)

  2. In forward(x):
     - First layer: hidden = max(0, x @ W1 + b1)  # ReLU
     - Second layer: output = hidden @ W2 + b2
     - Return output

Hint for C# developers:
  - Like two-stage transformation pipeline
  - ReLU is: Math.Max(0, x) for each element
  - np.maximum(0, x) applies element-wise
  - @ is matrix multiplication
""")

# TODO: Your code here

# SOLUTION (uncomment to check):
# class FeedForward:
#     """Feed-Forward Network."""
#
#     def __init__(self, d_model, d_ff):
#         """
#         Initialize FFN.
#
#         Args:
#             d_model: Input/output dimension
#             d_ff: Hidden layer dimension (typically 4× d_model)
#         """
#         self.d_model = d_model
#         self.d_ff = d_ff
#
#         # Initialize weights
#         self.W1 = np.random.randn(d_model, d_ff) * 0.1
#         self.b1 = np.zeros(d_ff)
#         self.W2 = np.random.randn(d_ff, d_model) * 0.1
#         self.b2 = np.zeros(d_model)
#
#     def forward(self, x):
#         """
#         Forward pass.
#
#         Args:
#             x: Input, shape (seq_len, d_model) or (d_model,)
#
#         Returns:
#             Output, same shape as input
#         """
#         # First layer: expand and activate
#         hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
#
#         # Second layer: contract back
#         output = hidden @ self.W2 + self.b2
#
#         return output

# Test your FeedForward (uncomment after implementing):
# print("\nTesting FeedForward:")
# ffn = FeedForward(d_model, d_ff)
# x_test = np.random.randn(seq_len, d_model)
# x_transformed = ffn.forward(x_test)
# print(f"  Input shape: {x_test.shape}")
# print(f"  Output shape: {x_transformed.shape}")
# print("  ✓ FeedForward works!")

# ==============================================================================
# TODO 3: Build Complete TransformerBlock Class
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 3: Build Complete Transformer Block")
print("=" * 70)

print("""
TASK: Build the complete TransformerBlock class!

This combines:
  - Multi-head attention (we'll use simplified version)
  - Feed-forward network
  - Layer normalization (×2)
  - Residual connections (×2)

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

Steps:
  1. In __init__:
     - Create self.attention = SimpleSelfAttention(d_model)
     - Create self.ffn = FeedForward(d_model, d_ff)
     - Create self.norm1 = LayerNorm(d_model)
     - Create self.norm2 = LayerNorm(d_model)

  2. In forward(x):
     - Sub-layer 1:
       * attn_output = self.attention.forward(x)
       * x = self.norm1.forward(x + attn_output)  # Residual + Norm

     - Sub-layer 2:
       * ffn_output = self.ffn.forward(x)
       * x = self.norm2.forward(x + ffn_output)  # Residual + Norm

     - Return x

Hint for C# developers:
  - Like middleware pipeline in ASP.NET
  - Each component processes sequentially
  - Residual: x + Layer(x) preserves information
  - Norm: stabilizes values
""")

# TODO: Your code here

# SOLUTION (uncomment to check):
# class TransformerBlock:
#     """Complete Transformer Block."""
#
#     def __init__(self, d_model, d_ff):
#         """
#         Initialize transformer block.
#
#         Args:
#             d_model: Embedding dimension
#             d_ff: Feed-forward hidden dimension
#         """
#         self.attention = SimpleSelfAttention(d_model)
#         self.ffn = FeedForward(d_model, d_ff)
#         self.norm1 = LayerNorm(d_model)
#         self.norm2 = LayerNorm(d_model)
#
#     def forward(self, x):
#         """
#         Forward pass through transformer block.
#
#         Args:
#             x: Input, shape (seq_len, d_model)
#
#         Returns:
#             Output, same shape as input
#         """
#         # Sub-layer 1: Multi-head attention with residual and norm
#         attn_output = self.attention.forward(x)
#         x = self.norm1.forward(x + attn_output)  # Residual + LayerNorm
#
#         # Sub-layer 2: Feed-forward with residual and norm
#         ffn_output = self.ffn.forward(x)
#         x = self.norm2.forward(x + ffn_output)  # Residual + LayerNorm
#
#         return x

# Test your TransformerBlock (uncomment after implementing):
# print("\nTesting TransformerBlock:")
# block = TransformerBlock(d_model, d_ff)
# x_test = np.random.randn(seq_len, d_model)
# x_output = block.forward(x_test)
# print(f"  Input shape: {x_test.shape}")
# print(f"  Output shape: {x_output.shape}")
# print("  ✓ TransformerBlock works!")

# ==============================================================================
# TODO 4: Stack Multiple Blocks
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 4: Stack Multiple Transformer Blocks")
print("=" * 70)

print("""
TASK: Create and test a stack of transformer blocks.

Real transformers stack many blocks:
  - GPT-2 Small: 12 blocks
  - GPT-3: 96 blocks
  - GPT-4: Unknown, likely 100+

Steps:
  1. Create a list of 3 TransformerBlock instances
  2. Pass input through all blocks sequentially
  3. Print shape after each block

Hint:
  blocks = [TransformerBlock(d_model, d_ff) for _ in range(3)]

  x = input
  for i, block in enumerate(blocks):
      x = block.forward(x)
      print(f"After block {i+1}: {x.shape}")
""")

# TODO: Your code here

# SOLUTION (uncomment to check):
# print("\nStacking 3 Transformer Blocks:")
# num_blocks = 3
# blocks = [TransformerBlock(d_model, d_ff) for _ in range(num_blocks)]
#
# x_stacked = np.random.randn(seq_len, d_model)
# print(f"Input shape: {x_stacked.shape}")
#
# for i, block in enumerate(blocks):
#     x_stacked = block.forward(x_stacked)
#     print(f"After block {i+1}: {x_stacked.shape}")
#
# print("\n✓ Stacked transformer blocks work!")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

print("\n" + "=" * 70)
print("VISUALIZATION: Before and After Transformer")
print("=" * 70)

# Uncomment after completing all TODOs:
# x_input = np.random.randn(seq_len, d_model)
# block_vis = TransformerBlock(d_model, d_ff)
# x_output = block_vis.forward(x_input)
#
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# # Input
# im1 = axes[0].imshow(x_input, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
# axes[0].set_title('Input Embeddings', fontsize=12, fontweight='bold')
# axes[0].set_ylabel('Position')
# axes[0].set_yticks(range(seq_len))
# axes[0].set_yticklabels(sentence)
# axes[0].set_xlabel('Dimension')
# plt.colorbar(im1, ax=axes[0])
#
# # Output
# im2 = axes[1].imshow(x_output, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
# axes[1].set_title('After Transformer Block', fontsize=12, fontweight='bold')
# axes[1].set_ylabel('Position')
# axes[1].set_yticks(range(seq_len))
# axes[1].set_yticklabels(sentence)
# axes[1].set_xlabel('Dimension')
# plt.colorbar(im2, ax=axes[1])
#
# plt.tight_layout()
# plt.show()

# ==============================================================================
# BONUS CHALLENGE: Representation Evolution
# ==============================================================================

print("\n" + "=" * 70)
print("BONUS CHALLENGE: Visualize Representation Evolution")
print("=" * 70)

print("""
BONUS: Visualize how representations change through multiple blocks!

Steps:
  1. Create 4 transformer blocks
  2. Store output after each block
  3. Create a 2×2 grid showing evolution
  4. Title each subplot: "After Block 1", "After Block 2", etc.

This shows how transformers progressively refine representations!

Hint:
  outputs = [input]
  for block in blocks:
      x = block.forward(x)
      outputs.append(x)

  # Then plot outputs[0], outputs[1], outputs[2], outputs[3]
""")

# BONUS SOLUTION (uncomment to check):
# print("\nBONUS: Visualizing evolution through 4 blocks:")
#
# # Create 4 blocks
# blocks_bonus = [TransformerBlock(d_model, d_ff) for _ in range(4)]
#
# # Process through blocks and store outputs
# x_evolution = np.random.randn(seq_len, d_model)
# outputs_evolution = [x_evolution.copy()]
#
# for block in blocks_bonus:
#     x_evolution = block.forward(x_evolution)
#     outputs_evolution.append(x_evolution.copy())
#
# # Visualize
# fig, axes = plt.subplots(2, 3, figsize=(16, 10))
# axes = axes.ravel()
#
# titles = ['Input', 'After Block 1', 'After Block 2', 'After Block 3', 'After Block 4']
#
# for i, (output, title) in enumerate(zip(outputs_evolution, titles)):
#     im = axes[i].imshow(output, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
#     axes[i].set_title(title, fontsize=12, fontweight='bold')
#     axes[i].set_ylabel('Position')
#     axes[i].set_yticks(range(seq_len))
#     axes[i].set_yticklabels(sentence)
#     axes[i].set_xlabel('Dimension')
#     plt.colorbar(im, ax=axes[i])
#
# # Hide the 6th subplot
# axes[5].axis('off')
#
# plt.suptitle('Representation Evolution Through Transformer Blocks',
#              fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.show()
#
# print("✓ Evolution visualization complete!")

# ==============================================================================
# VERIFICATION
# ==============================================================================

print("\n" + "=" * 70)
print("VERIFICATION CHECKLIST")
print("=" * 70)

print("""
Check that you've completed:

□ TODO 1: Implemented LayerNorm class
  - __init__ with epsilon, gamma, beta
  - forward() with mean, variance, normalization

□ TODO 2: Implemented FeedForward class
  - __init__ with W1, b1, W2, b2
  - forward() with expand, ReLU, contract

□ TODO 3: Built TransformerBlock class
  - __init__ with attention, ffn, norm1, norm2
  - forward() with two sub-layers (attention + ffn)
  - Both with residual connections and layer norms

□ TODO 4: Stacked multiple blocks
  - Created list of blocks
  - Passed input through all sequentially

□ Visualized before/after transformation

BONUS:
□ Visualized representation evolution through 4+ blocks

Once all checked, YOU'VE MASTERED TRANSFORMERS! 🎉🎉🎉
""")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("🎉 EXERCISE SUMMARY - YOU DID IT! 🎉")
print("=" * 70)

print("""
CONGRATULATIONS! You've built a complete transformer block!

What You Built:
  ✓ LayerNorm: Stabilizes training
  ✓ FeedForward: Transforms representations
  ✓ TransformerBlock: Complete architecture
  ✓ Stacked blocks: Deep transformer

Components Breakdown:
  1. LayerNorm:
     - Normalizes features to mean=0, std=1
     - Has learnable gamma (scale) and beta (shift)
     - Applied after attention and FFN

  2. FeedForward:
     - 2-layer network: expand → ReLU → contract
     - Typical: d_ff = 4× d_model
     - Contains MOST parameters in transformer!

  3. TransformerBlock:
     - Attention + Residual + Norm
     - FFN + Residual + Norm
     - THIS IS THE FUNDAMENTAL BUILDING BLOCK!

  4. Stacking:
     - Real models stack 12-96+ blocks
     - Each block refines representations
     - Deeper = more complex patterns

You Now Understand:
  ✅ How attention works (queries, keys, values)
  ✅ How self-attention adds learning (W matrices)
  ✅ How multi-head captures different patterns
  ✅ How positional encoding adds order
  ✅ How layer norm stabilizes training
  ✅ How feed-forward transforms features
  ✅ How residuals help deep networks
  ✅ How complete transformers are built!

THIS IS EXACTLY HOW GPT/BERT WORK!

In C#/.NET Terms:
  - Classes: Similar to C# class hierarchy
  - Layers: Like middleware pipeline in ASP.NET
  - Residuals: Like error handling fallbacks
  - Stacking: Like nested service calls
  - Forward pass: Like data flow through pipeline

Real-World Scale:
  Your implementation:
    - d_model: 8
    - Layers: 2-4
    - Parameters: ~2,000

  GPT-2 Small:
    - d_model: 768
    - Layers: 12
    - Parameters: 124M

  GPT-3:
    - d_model: 12,288
    - Layers: 96
    - Parameters: 175B

  YOUR CODE SCALES TO GPT-3/4 WITH MORE LAYERS/DIMS!

Next Steps:
  - Train this on real text data
  - Add more sophisticated attention (multi-head)
  - Implement dropout for regularization
  - Add learning rate scheduling
  - BUILD YOUR OWN CHATBOT!

You've completed ALL transformer exercises!
You now have the skills to:
  ✓ Build transformers from scratch
  ✓ Understand research papers
  ✓ Contribute to ML projects
  ✓ Build LLM applications

Module 4 COMPLETE! Ready for Module 5! 🚀

AMAZING WORK! You're now a transformer expert! 🎊🎉
""")

print("\n" + "=" * 70)
print("END OF EXERCISE 03 - FINAL EXERCISE COMPLETE!")
print("=" * 70)
