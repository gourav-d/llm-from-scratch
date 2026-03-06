"""
Example 05: Complete Transformer Block

This example demonstrates a COMPLETE transformer block - the building block of
models like GPT, BERT, and other transformers. We combine everything we've
learned: multi-head attention, feed-forward networks, layer normalization,
and residual connections!

What you'll see:
1. Multi-head self-attention sublayer
2. Feed-forward network (FFN) sublayer
3. Layer normalization for stable training
4. Residual connections (skip connections)
5. How all components work together
6. Complete forward pass through a transformer block

C# Analogy:
Think of it like a processing pipeline with branching and merging:
    var attended = Attention(input);
    var normalized1 = Normalize(input + attended);  // Residual
    var processed = FeedForward(normalized1);
    var output = Normalize(normalized1 + processed);  // Residual

Each transformer layer does this twice: once for attention, once for FFN.
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

print("=" * 70)
print("COMPLETE TRANSFORMER BLOCK")
print("=" * 70)

# ==============================================================================
# PART 1: Components of a Transformer Block
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Transformer Block Architecture")
print("=" * 70)

print("""
A transformer block consists of two main sublayers:

┌─────────────────────────────────────────────┐
│          TRANSFORMER BLOCK                   │
│                                              │
│  Input                                       │
│    │                                         │
│    ├──────────────┐                         │
│    │              │                         │
│    │    Multi-Head Attention                │
│    │              │                         │
│    └──────(+)─────┘  ← Residual Connection │
│         │                                    │
│    Layer Norm                                │
│         │                                    │
│    ├──────────────┐                         │
│    │              │                         │
│    │  Feed-Forward Network                  │
│    │              │                         │
│    └──────(+)─────┘  ← Residual Connection │
│         │                                    │
│    Layer Norm                                │
│         │                                    │
│  Output                                      │
└─────────────────────────────────────────────┘

Components:
1. Multi-Head Self-Attention (we built this in example_03!)
2. Feed-Forward Network (position-wise MLP)
3. Layer Normalization (for stable training)
4. Residual Connections (to help gradients flow)

Let's implement each component!
""")

# ==============================================================================
# PART 2: Layer Normalization
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Layer Normalization")
print("=" * 70)

print("""
Layer Normalization normalizes across the feature dimension for each example.

For each position, compute:
    mean = average of all features
    variance = variance of all features
    normalized = (x - mean) / sqrt(variance + epsilon)
    output = gamma * normalized + beta

Where gamma and beta are learnable parameters.

C# Analogy: Like normalizing each row of a matrix:
    normalized = row.Select(x => (x - mean) / stddev)

This keeps values in a reasonable range and helps training stability.
""")

class LayerNorm:
    """
    Layer Normalization.

    Similar to C# class:
        public class LayerNorm {
            private Vector gamma, beta;
            public Matrix Normalize(Matrix input) { ... }
        }
    """

    def __init__(self, d_model, epsilon=1e-6):
        """
        Initialize layer normalization.

        Args:
            d_model: Feature dimension
            epsilon: Small constant for numerical stability
        """
        self.d_model = d_model
        self.epsilon = epsilon

        # Learnable parameters (initialized to 1 and 0)
        self.gamma = np.ones(d_model)   # Scale parameter
        self.beta = np.zeros(d_model)   # Shift parameter

        print(f"  ✓ Created LayerNorm with d_model={d_model}")

    def forward(self, x):
        """
        Apply layer normalization.

        Args:
            x: Input tensor, shape (..., d_model)

        Returns:
            Normalized tensor, same shape as input
        """
        # Compute mean and variance across last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / np.sqrt(variance + self.epsilon)

        # Scale and shift
        return self.gamma * x_norm + self.beta

# ==============================================================================
# PART 3: Feed-Forward Network
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Position-wise Feed-Forward Network")
print("=" * 70)

print("""
The FFN is a simple 2-layer neural network applied to each position
independently and identically.

FFN(x) = max(0, x @ W1 + b1) @ W2 + b2
         └─────────────────────┘
              ReLU activation

Typical dimensions:
  - Input:  d_model (e.g., 512)
  - Hidden: d_ff = 4 × d_model (e.g., 2048)
  - Output: d_model (e.g., 512)

C# Analogy: Like applying the same function to each item in a list:
    output = input.Select(x => FeedForward(x)).ToList()

This adds non-linearity and allows the model to learn complex patterns.
""")

class FeedForward:
    """
    Position-wise Feed-Forward Network.

    Similar to C# class:
        public class FeedForward {
            private Matrix W1, W2;
            private Vector b1, b2;
            public Matrix Forward(Matrix input) { ... }
        }
    """

    def __init__(self, d_model, d_ff):
        """
        Initialize feed-forward network.

        Args:
            d_model: Input/output dimension
            d_ff: Hidden layer dimension (typically 4 × d_model)
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Initialize weights
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

        print(f"  ✓ Created FeedForward: {d_model} -> {d_ff} -> {d_model}")

    def forward(self, x):
        """
        Apply feed-forward network.

        Args:
            x: Input tensor, shape (..., d_model)

        Returns:
            Output tensor, shape (..., d_model)
        """
        # First layer with ReLU activation
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU: max(0, x)

        # Second layer (linear)
        output = hidden @ self.W2 + self.b2

        return output

# ==============================================================================
# PART 4: Multi-Head Attention (Simplified)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Multi-Head Attention (Recap)")
print("=" * 70)

print("Reusing our multi-head attention from example_03...\n")

class MultiHeadAttention:
    """Multi-Head Attention (simplified version from example_03)."""

    def __init__(self, d_model, num_heads):
        """Initialize multi-head attention."""
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01

        print(f"  ✓ Created MultiHeadAttention: {num_heads} heads, d_k={self.d_k}")

    def forward(self, X):
        """Forward pass."""
        batch_size, seq_len, d_model = X.shape

        # Linear projections
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        # Split into heads
        Q = self._split_heads(Q, batch_size, seq_len)
        K = self._split_heads(K, batch_size, seq_len)
        V = self._split_heads(V, batch_size, seq_len)

        # Attention
        attn_output = self._attention(Q, K, V)

        # Combine heads
        concat = self._combine_heads(attn_output, batch_size, seq_len)

        # Final projection
        output = concat @ self.W_o

        return output

    def _split_heads(self, x, batch_size, seq_len):
        """Split into multiple heads."""
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x, batch_size, seq_len):
        """Combine heads back."""
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)

    def _attention(self, Q, K, V):
        """Scaled dot-product attention."""
        scores = Q @ K.transpose(-2, -1) / np.sqrt(self.d_k)
        weights = self._softmax(scores)
        return weights @ V

    @staticmethod
    def _softmax(x):
        """Softmax function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# ==============================================================================
# PART 5: Complete Transformer Block
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Assembling the Transformer Block")
print("=" * 70)

class TransformerBlock:
    """
    Complete Transformer Block.

    Combines:
    1. Multi-head self-attention + residual + layer norm
    2. Feed-forward network + residual + layer norm

    This is the core building block of GPT, BERT, and other transformers!
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize transformer block.

        Args:
            d_model: Embedding dimension (e.g., 512)
            num_heads: Number of attention heads (e.g., 8)
            d_ff: Feed-forward hidden dimension (e.g., 2048)
            dropout: Dropout rate (we'll skip actual dropout for simplicity)
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        print(f"\nCreating Transformer Block:")
        print(f"  d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}")

        # Sublayer 1: Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)

        # Sublayer 2: Feed-forward network
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

        print("✓ Transformer Block created!\n")

    def forward(self, x):
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor, same shape as input
        """
        # Sublayer 1: Multi-head attention with residual connection
        # Formula: LayerNorm(x + MultiHeadAttention(x))
        attn_output = self.attention.forward(x)
        x = self.norm1.forward(x + attn_output)  # Residual + norm

        # Sublayer 2: Feed-forward with residual connection
        # Formula: LayerNorm(x + FFN(x))
        ffn_output = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_output)  # Residual + norm

        return x

# ==============================================================================
# PART 6: Testing the Transformer Block
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Running a Transformer Block")
print("=" * 70)

# Input parameters
sentence = ["The", "cat", "sat", "on", "the", "mat"]
seq_len = len(sentence)
batch_size = 1
d_model = 64      # Embedding dimension
num_heads = 4     # Number of attention heads
d_ff = 256        # Feed-forward hidden dimension (4 × d_model)

print(f"Input sentence: {' '.join(sentence)}")
print(f"\nConfiguration:")
print(f"  Sequence length: {seq_len}")
print(f"  d_model: {d_model}")
print(f"  num_heads: {num_heads}")
print(f"  d_ff: {d_ff}")

# Create input embeddings
X = np.random.randn(batch_size, seq_len, d_model) * 0.5
print(f"\nInput shape: {X.shape}")

# Create transformer block
transformer_block = TransformerBlock(d_model, num_heads, d_ff)

# Forward pass
print("Running forward pass through transformer block...")
output = transformer_block.forward(X)

print(f"\nOutput shape: {output.shape}")
print("✓ Forward pass completed successfully!")

# ==============================================================================
# PART 7: Analyzing the Transformation
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: How Did the Representations Change?")
print("=" * 70)

print("\nComparing input vs output for each word:\n")
print(f"{'Word':<6} {'Input Norm':<12} {'Output Norm':<12} {'Change':<12}")
print("-" * 50)

for i, word in enumerate(sentence):
    input_norm = np.linalg.norm(X[0, i])
    output_norm = np.linalg.norm(output[0, i])
    change = np.linalg.norm(output[0, i] - X[0, i])

    print(f"{word:<6} {input_norm:<12.4f} {output_norm:<12.4f} {change:<12.4f}")

print("\nThe transformer block has transformed each word's representation")
print("to incorporate context from all other words!")

# Compute similarity between input and output
similarity = np.mean([np.dot(X[0, i], output[0, i]) /
                      (np.linalg.norm(X[0, i]) * np.linalg.norm(output[0, i]))
                      for i in range(seq_len)])
print(f"\nAverage cosine similarity (input vs output): {similarity:.4f}")
print("(Values close to 1 mean similar directions, close to 0 mean different)")

# ==============================================================================
# PART 8: Stacking Transformer Blocks
# ==============================================================================

print("\n" + "=" * 70)
print("PART 8: Stacking Multiple Transformer Blocks")
print("=" * 70)

print("""
Real transformers stack MANY of these blocks:
  - GPT-2 Small: 12 layers
  - GPT-2 Medium: 24 layers
  - GPT-2 Large: 36 layers
  - GPT-3: 96 layers!

Each layer refines the representations further, building increasingly
abstract and context-aware understanding.

Let's simulate a mini-transformer with 3 blocks:
""")

# Create a stack of 3 transformer blocks
num_layers = 3
print(f"\nCreating {num_layers} transformer blocks:")

blocks = []
for i in range(num_layers):
    print(f"\n--- Layer {i+1} ---")
    block = TransformerBlock(d_model, num_heads, d_ff)
    blocks.append(block)

# Process through all blocks
print("\n" + "=" * 70)
print(f"Processing through {num_layers} layers...")
print("=" * 70)

current_output = X
layer_outputs = [X[0]]  # Store output from each layer

for i, block in enumerate(blocks):
    print(f"\nLayer {i+1}:")
    current_output = block.forward(current_output)
    layer_outputs.append(current_output[0])
    print(f"  Output shape: {current_output.shape}")

    # Compute change from previous layer
    if i > 0:
        change = np.mean([np.linalg.norm(current_output[0, j] - layer_outputs[i][j])
                         for j in range(seq_len)])
        print(f"  Average change from previous layer: {change:.4f}")

print("\n✓ All layers completed!")

# ==============================================================================
# PART 9: Visualizing Layer-by-Layer Changes
# ==============================================================================

print("\n" + "=" * 70)
print("PART 9: Visualizing Representations Through Layers")
print("=" * 70)

# Visualize how representations change through layers
fig, axes = plt.subplots(1, num_layers + 1, figsize=(16, 4))

for i, ax in enumerate(axes):
    # Get representation at this layer
    layer_repr = layer_outputs[i]  # Shape: (seq_len, d_model)

    # Create heatmap
    im = ax.imshow(layer_repr.T, cmap='RdBu_r', aspect='auto')
    ax.set_yticks(range(0, d_model, 8))
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(sentence, rotation=45, ha='right')

    if i == 0:
        ax.set_title(f'Input\nLayer 0', fontsize=10, fontweight='bold')
        ax.set_ylabel('Embedding Dimension', fontsize=9)
    else:
        ax.set_title(f'After Block {i}\nLayer {i}', fontsize=10, fontweight='bold')

    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('How Word Representations Evolve Through Transformer Layers',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\nNotice how the patterns change as we go deeper!")
print("Each layer refines the representations to be more context-aware.")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - What We Learned")
print("=" * 70)

print("""
✓ Transformer Block = 2 sublayers (attention + FFN)
✓ Each sublayer has residual connection + layer normalization
✓ Multi-head attention captures relationships between positions
✓ Feed-forward network adds non-linearity and complexity
✓ Layer norm keeps values stable for training
✓ Residual connections help gradients flow through deep networks

The Architecture (per block):
    x = LayerNorm(x + MultiHeadAttention(x))
    x = LayerNorm(x + FeedForward(x))

Real transformers stack many of these blocks:
  - More layers = deeper understanding
  - Each layer refines representations
  - Bottom layers: simple patterns (syntax, local context)
  - Top layers: complex patterns (semantics, global context)

This is the CORE of GPT, BERT, and all modern transformers! 🚀

Next Step:
- example_06: Mini-GPT (complete model with embedding, positional encoding,
              stacked transformer blocks, and language modeling head!)
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 05")
print("=" * 70)
