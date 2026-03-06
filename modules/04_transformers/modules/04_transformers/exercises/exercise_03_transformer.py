"""
Exercise 03: Building a Complete Transformer Block

This is the FINAL exercise! You'll build a complete transformer block that
combines everything you've learned:
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Residual connections

This is what powers GPT, BERT, and all modern transformers!

Tasks:
1. Implement layer normalization
2. Implement feed-forward network
3. Combine into a transformer block
4. Stack multiple blocks
5. Test on sequences

Make sure you completed Exercises 01 and 02 first!
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print("=" * 70)
print("EXERCISE 03: BUILDING A TRANSFORMER BLOCK")
print("=" * 70)

# ==============================================================================
# HELPERS: Functions from Previous Exercises
# ==============================================================================

def softmax(x):
    """Softmax function."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def self_attention_forward(Q, K, V):
    """Self-attention forward pass."""
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V
    return output, weights


class MultiHeadAttention:
    """Multi-Head Attention (simplified for this exercise)."""

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def forward(self, X):
        """Forward pass through multi-head attention."""
        seq_len, d_model = X.shape

        # Project
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        # Split into heads
        Q = Q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        # Attention per head
        outputs = []
        for h in range(self.num_heads):
            head_output, _ = self_attention_forward(Q[h], K[h], V[h])
            outputs.append(head_output)

        # Concatenate
        concat = np.stack(outputs, axis=1).reshape(seq_len, d_model)

        # Output projection
        return concat @ self.W_o

print("✓ Helper functions loaded!\n")

# ==============================================================================
# SETUP
# ==============================================================================

print("=" * 70)
print("SETUP: Creating Input Data")
print("=" * 70)

sentence = ["The", "cat", "sat", "on", "the", "mat"]
seq_len = len(sentence)
d_model = 16  # Embedding dimension

print(f"Sentence: {' '.join(sentence)}")
print(f"Sequence length: {seq_len}")
print(f"Embedding dimension: {d_model}")

# Input embeddings
X = np.random.randn(seq_len, d_model) * 0.5
print(f"\nInput shape: {X.shape}\n")

# ==============================================================================
# TODO 1: Implement Layer Normalization
# ==============================================================================

print("=" * 70)
print("TODO 1: Implement Layer Normalization")
print("=" * 70)

print("""
Task: Implement layer normalization for stable training.

Layer norm normalizes across features for each position:
1. Compute mean across features (axis=-1)
2. Compute variance across features (axis=-1)
3. Normalize: (x - mean) / sqrt(variance + epsilon)
4. Scale and shift: gamma * normalized + beta

Formula:
    mean = average(x across features)
    var = variance(x across features)
    x_norm = (x - mean) / sqrt(var + epsilon)
    output = gamma * x_norm + beta

Where gamma and beta are learnable parameters (start with 1s and 0s).

C# Analogy: Like normalizing each row of a matrix:
    normalized = row.Select(val => (val - mean) / stddev)
""")

class LayerNorm:
    """Layer Normalization."""

    def __init__(self, d_model, epsilon=1e-6):
        """
        Initialize layer normalization.

        Args:
            d_model: Feature dimension
            epsilon: Small constant for numerical stability
        """
        # TODO: Initialize parameters
        # YOUR CODE HERE:
        pass

        # SOLUTION (uncomment to see/verify):
        # self.d_model = d_model
        # self.epsilon = epsilon
        # self.gamma = np.ones(d_model)   # Scale parameter
        # self.beta = np.zeros(d_model)   # Shift parameter
        # print(f"  ✓ LayerNorm created with d_model={d_model}")

    def forward(self, x):
        """
        Apply layer normalization.

        Args:
            x: Input tensor, shape (..., d_model)

        Returns:
            Normalized tensor, same shape as input
        """
        # TODO: Implement layer normalization
        # YOUR CODE HERE:
        pass

        # SOLUTION (uncomment to see/verify):
        # # Compute statistics across last dimension (features)
        # mean = np.mean(x, axis=-1, keepdims=True)
        # variance = np.var(x, axis=-1, keepdims=True)
        #
        # # Normalize
        # x_norm = (x - mean) / np.sqrt(variance + self.epsilon)
        #
        # # Scale and shift
        # return self.gamma * x_norm + self.beta

# Test layer normalization
try:
    layer_norm = LayerNorm(d_model)
    norm_output = layer_norm.forward(X)

    if norm_output is not None:
        print("\n✓ Layer normalization works!")
        print(f"  Input shape:  {X.shape}")
        print(f"  Output shape: {norm_output.shape}")

        # Check normalization properties
        mean_after = np.mean(norm_output, axis=-1)
        var_after = np.var(norm_output, axis=-1)

        print(f"\n  Mean after normalization (should be ~0): {mean_after[0]:.6f}")
        print(f"  Variance after normalization (should be ~1): {var_after[0]:.6f}")

        if np.allclose(mean_after, 0, atol=1e-5) and np.allclose(var_after, 1, atol=1e-1):
            print("\n✓ Normalization properties are correct!")
        else:
            print("\n⚠ Check your normalization - mean should be ~0, variance ~1")
    else:
        print("⚠ Layer normalization returned None. Complete TODO 1!")
except Exception as e:
    print(f"⚠ Error in layer normalization: {e}")
    print("Complete TODO 1 first!")

# ==============================================================================
# TODO 2: Implement Feed-Forward Network
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 2: Implement Feed-Forward Network")
print("=" * 70)

print("""
Task: Implement a 2-layer feed-forward network with ReLU activation.

Architecture:
    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

Where:
- W1: (d_model, d_ff) - first layer weights
- b1: (d_ff,) - first layer bias
- W2: (d_ff, d_model) - second layer weights
- b2: (d_model,) - second layer bias
- ReLU(x) = max(0, x) - activation function

Typical dimensions:
- d_ff = 4 × d_model (hidden layer is larger!)

Hints:
- Use np.random.randn for initialization
- Use np.maximum(0, x) for ReLU activation
- Apply to each position independently

C# Analogy:
    var hidden = input.Select(x => ReLU(x.MatMul(W1) + b1));
    var output = hidden.Select(h => h.MatMul(W2) + b2);
""")

class FeedForward:
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model, d_ff):
        """
        Initialize feed-forward network.

        Args:
            d_model: Input/output dimension
            d_ff: Hidden layer dimension (typically 4 × d_model)
        """
        # TODO: Initialize weights and biases
        # YOUR CODE HERE:
        pass

        # SOLUTION (uncomment to see/verify):
        # self.d_model = d_model
        # self.d_ff = d_ff
        #
        # # First layer
        # self.W1 = np.random.randn(d_model, d_ff) * 0.01
        # self.b1 = np.zeros(d_ff)
        #
        # # Second layer
        # self.W2 = np.random.randn(d_ff, d_model) * 0.01
        # self.b2 = np.zeros(d_model)
        #
        # print(f"  ✓ FeedForward: {d_model} -> {d_ff} -> {d_model}")

    def forward(self, x):
        """
        Apply feed-forward network.

        Args:
            x: Input tensor, shape (..., d_model)

        Returns:
            Output tensor, shape (..., d_model)
        """
        # TODO: Implement feed-forward pass
        # YOUR CODE HERE:
        pass

        # SOLUTION (uncomment to see/verify):
        # # First layer with ReLU
        # hidden = np.maximum(0, x @ self.W1 + self.b1)
        #
        # # Second layer (linear)
        # output = hidden @ self.W2 + self.b2
        #
        # return output

# Test feed-forward network
try:
    d_ff = d_model * 4  # Typical: 4x larger hidden layer
    ffn = FeedForward(d_model, d_ff)
    ffn_output = ffn.forward(X)

    if ffn_output is not None:
        print(f"\n✓ Feed-forward network works!")
        print(f"  Input shape:  {X.shape}")
        print(f"  Output shape: {ffn_output.shape}")
        print(f"  Hidden dimension: {d_ff}")

        if ffn_output.shape == X.shape:
            print("\n✓ Input and output shapes match!")
        else:
            print("\n✗ Output shape should match input shape")
    else:
        print("⚠ Feed-forward network returned None. Complete TODO 2!")
except Exception as e:
    print(f"⚠ Error in feed-forward network: {e}")
    print("Complete TODO 2 first!")

# ==============================================================================
# TODO 3: Build Complete Transformer Block
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 3: Build Complete Transformer Block")
print("=" * 70)

print("""
Task: Combine everything into a complete transformer block!

Architecture:
    # Sublayer 1: Multi-head attention
    attn_output = MultiHeadAttention(x)
    x = LayerNorm(x + attn_output)  ← Residual connection!

    # Sublayer 2: Feed-forward
    ffn_output = FeedForward(x)
    x = LayerNorm(x + ffn_output)   ← Residual connection!

Key concepts:
- Residual connection: Add input to output (x + sublayer(x))
- Layer norm: Normalize after adding residual
- Two sublayers: attention then feed-forward

This is the CORE of transformers!
""")

class TransformerBlock:
    """Complete Transformer Block."""

    def __init__(self, d_model, num_heads, d_ff):
        """
        Initialize transformer block.

        Args:
            d_model: Embedding dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
        """
        # TODO: Initialize all components
        # YOUR CODE HERE:
        pass

        # SOLUTION (uncomment to see/verify):
        # self.d_model = d_model
        #
        # print(f"\n  Creating Transformer Block:")
        # print(f"    d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}")
        #
        # # Sublayer 1: Multi-head attention
        # self.attention = MultiHeadAttention(d_model, num_heads)
        # self.norm1 = LayerNorm(d_model)
        #
        # # Sublayer 2: Feed-forward
        # self.ffn = FeedForward(d_model, d_ff)
        # self.norm2 = LayerNorm(d_model)
        #
        # print("  ✓ Transformer Block created!")

    def forward(self, x):
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor, shape (seq_len, d_model)

        Returns:
            Output tensor, same shape as input
        """
        # TODO: Implement transformer block forward pass
        # YOUR CODE HERE:
        pass

        # SOLUTION (uncomment to see/verify):
        # # Sublayer 1: Multi-head attention + residual + norm
        # attn_output = self.attention.forward(x)
        # x = self.norm1.forward(x + attn_output)  # Residual connection!
        #
        # # Sublayer 2: Feed-forward + residual + norm
        # ffn_output = self.ffn.forward(x)
        # x = self.norm2.forward(x + ffn_output)  # Residual connection!
        #
        # return x

# Test transformer block
try:
    num_heads = 4
    d_ff = d_model * 4

    transformer = TransformerBlock(d_model, num_heads, d_ff)
    transformer_output = transformer.forward(X)

    if transformer_output is not None:
        print(f"\n✓ Transformer block works!")
        print(f"  Input shape:  {X.shape}")
        print(f"  Output shape: {transformer_output.shape}")

        # Analyze transformation
        input_norm = np.linalg.norm(X)
        output_norm = np.linalg.norm(transformer_output)
        change = np.linalg.norm(transformer_output - X)

        print(f"\n  Input magnitude:  {input_norm:.4f}")
        print(f"  Output magnitude: {output_norm:.4f}")
        print(f"  Change magnitude: {change:.4f}")

        print("\n✓ The transformer block has processed your input!")
        print("  Each word now has a richer, context-aware representation!")
    else:
        print("⚠ Transformer block returned None. Complete TODO 3!")
except Exception as e:
    print(f"⚠ Error in transformer block: {e}")
    print("Complete TODO 3 first!")

# ==============================================================================
# TODO 4: Stack Multiple Transformer Blocks
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 4: Stack Multiple Transformer Blocks")
print("=" * 70)

print("""
Task: Create a function that stacks N transformer blocks.

Real transformers use many layers:
- GPT-2 Small: 12 layers
- GPT-2 Medium: 24 layers
- GPT-3: 96 layers!

Your function should:
1. Create a list of N transformer blocks
2. Pass input through each block sequentially
3. Return final output

C# Analogy:
    var blocks = Enumerable.Range(0, numLayers)
        .Select(_ => new TransformerBlock(...))
        .ToList();

    var output = blocks.Aggregate(input, (x, block) => block.Forward(x));
""")

def create_stacked_transformer(d_model, num_heads, d_ff, num_layers):
    """
    Create a stack of transformer blocks.

    Args:
        d_model: Embedding dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        num_layers: Number of transformer blocks to stack

    Returns:
        List of transformer blocks
    """
    # TODO: Create list of transformer blocks
    # YOUR CODE HERE:
    pass

    # SOLUTION (uncomment to see/verify):
    # blocks = []
    # print(f"\nCreating {num_layers} transformer blocks:")
    # for i in range(num_layers):
    #     print(f"  Layer {i+1}:")
    #     block = TransformerBlock(d_model, num_heads, d_ff)
    #     blocks.append(block)
    # return blocks


def forward_through_stack(blocks, x):
    """
    Pass input through all transformer blocks.

    Args:
        blocks: List of transformer blocks
        x: Input tensor

    Returns:
        Output tensor after all blocks
    """
    # TODO: Pass input through each block sequentially
    # YOUR CODE HERE:
    pass

    # SOLUTION (uncomment to see/verify):
    # for block in blocks:
    #     x = block.forward(x)
    # return x

# Test stacked transformers
try:
    num_layers = 3
    blocks = create_stacked_transformer(d_model, num_heads=4, d_ff=d_model*4, num_layers=num_layers)

    if blocks is not None and len(blocks) > 0:
        print(f"\n✓ Created stack of {len(blocks)} transformer blocks!")

        # Process through all layers
        output = forward_through_stack(blocks, X)

        if output is not None:
            print(f"\n✓ Forward pass through all {num_layers} layers successful!")
            print(f"  Input shape:  {X.shape}")
            print(f"  Output shape: {output.shape}")

            # Visualize change through layers
            print(f"\nAnalyzing transformations through layers:")

            current = X
            for i, block in enumerate(blocks):
                next_output = block.forward(current)
                change = np.linalg.norm(next_output - current)
                print(f"  Layer {i+1}: change magnitude = {change:.4f}")
                current = next_output

            print("\n✓ You've built a mini-transformer stack!")
        else:
            print("⚠ forward_through_stack returned None. Complete TODO 4!")
    else:
        print("⚠ create_stacked_transformer returned None or empty list. Complete TODO 4!")
except Exception as e:
    print(f"⚠ Error in stacked transformers: {e}")
    print("Complete TODO 4 first!")

# ==============================================================================
# BONUS TODO 5: Analyze Representation Changes
# ==============================================================================

print("\n" + "=" * 70)
print("BONUS TODO 5: Analyze How Representations Evolve")
print("=" * 70)

print("""
BONUS Task: Visualize how word representations change through layers.

Create a visualization that shows:
1. Original embedding for each word
2. Representation after each transformer layer
3. Final representation

This helps understand what each layer learns!

Hints:
- Store output after each layer
- Create heatmap showing embeddings × layers
- Use matplotlib/seaborn for visualization
""")

# TODO: Create visualization of representation evolution
# YOUR CODE HERE:

# SOLUTION (uncomment to see/verify):
# if blocks is not None and len(blocks) > 0:
#     # Collect representations at each layer
#     representations = [X]  # Start with input
#     current = X
#     for block in blocks:
#         current = block.forward(current)
#         representations.append(current)
#
#     # Visualize for each word
#     fig, axes = plt.subplots(2, 3, figsize=(15, 8))
#     axes = axes.flatten()
#
#     for word_idx in range(min(6, seq_len)):
#         ax = axes[word_idx]
#
#         # Get representation evolution for this word
#         word_evolution = np.array([rep[word_idx] for rep in representations])
#
#         # Create heatmap
#         im = ax.imshow(word_evolution.T, cmap='RdBu_r', aspect='auto')
#         ax.set_xlabel('Layer', fontsize=9)
#         ax.set_ylabel('Embedding Dimension', fontsize=9)
#         ax.set_title(f'Word: "{sentence[word_idx]}"', fontsize=10, fontweight='bold')
#         ax.set_xticks(range(len(representations)))
#         ax.set_xticklabels(['Input'] + [f'L{i+1}' for i in range(len(blocks))])
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#
#     plt.suptitle('How Word Representations Evolve Through Transformer Layers',
#                  fontsize=13, fontweight='bold')
#     plt.tight_layout()
#     plt.show()
#
#     print("\n✓ BONUS visualization created!")
#     print("  Notice how each word's representation changes through layers!")

# ==============================================================================
# CONGRATULATIONS!
# ==============================================================================

print("\n" + "=" * 70)
print("🎉 EXERCISE 03 COMPLETE! 🎉")
print("=" * 70)

print("""
OUTSTANDING WORK! You've built a complete transformer architecture!

What you implemented:
✓ Layer Normalization - stabilizes training
✓ Feed-Forward Network - adds non-linear transformations
✓ Transformer Block - combines attention + FFN + norms + residuals
✓ Stacked Transformers - multiple layers for deep learning
✓ BONUS: Visualization of representation evolution

This is the SAME architecture used in:
  - GPT (Generative Pre-trained Transformer)
  - BERT (Bidirectional Encoder Representations from Transformers)
  - T5, BART, and many other state-of-the-art models!

Key Insights:
1. Residual connections (x + sublayer(x)) help gradients flow
2. Layer normalization keeps values stable
3. Multiple layers build increasingly abstract representations
4. Each layer refines understanding of context

What You Can Do Now:
  ✓ Understand transformer papers (Attention Is All You Need, GPT, BERT)
  ✓ Read transformer code and know what it does
  ✓ Implement your own transformer variants
  ✓ Move to Module 5: Training and fine-tuning LLMs!

Module 4 Completion Status:
  ✓ Lesson 01: Attention mechanism
  ✓ Lesson 02: Self-attention
  ✓ Lesson 03: Multi-head attention
  ✓ Lesson 04: Positional encoding
  ✓ Lesson 05: Transformer block
  ✓ Lesson 06: Complete GPT architecture

  ✓ Example 01: Basic attention
  ✓ Example 02: Self-attention layer
  ✓ Example 03: Multi-head attention
  ✓ Example 04: Positional encoding
  ✓ Example 05: Transformer block
  ✓ Example 06: Mini-GPT

  ✓ Exercise 01: Implementing attention ← YOU DID IT!
  ✓ Exercise 02: Building self-attention ← YOU DID IT!
  ✓ Exercise 03: Complete transformer ← YOU JUST FINISHED!

YOU'VE MASTERED TRANSFORMERS! 🌟🚀

Next stop: Module 5 - Building and training complete LLMs!

Amazing progress! You went from .NET developer with no Python experience
to building transformers from scratch. That's incredible! 👏
""")

print("\n" + "=" * 70)
print("END OF EXERCISE 03")
print("=" * 70)
