"""
Example 03: Multi-Head Attention

This example demonstrates how transformers use MULTIPLE attention mechanisms in
parallel (called "heads"). Each head can learn to focus on different aspects
of the relationships between words!

What you'll see:
1. Why we need multiple attention heads
2. How to split embeddings across heads
3. Running attention in parallel for each head
4. Concatenating and projecting head outputs
5. Visualization showing different heads learning different patterns

C# Analogy:
Think of it like running multiple LINQ queries in parallel:
    var results = Enumerable.Range(0, numHeads).AsParallel()
        .Select(h => ComputeAttention(input, headWeights[h]))
        .ToArray();

Each head is like a different "perspective" on the data!
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed
np.random.seed(42)

print("=" * 70)
print("MULTI-HEAD ATTENTION")
print("=" * 70)

# ==============================================================================
# PART 1: Why Multiple Heads?
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: The Motivation - Why Multiple Attention Heads?")
print("=" * 70)

print("""
Single attention head might focus on ONE type of relationship:
  - Syntactic relationships (subject-verb agreement)
  - Semantic relationships (similar meanings)
  - Positional relationships (nearby words)

But language has MANY types of relationships simultaneously!

SOLUTION: Use multiple attention heads in parallel!
  - Head 1 might learn syntactic patterns
  - Head 2 might learn semantic similarity
  - Head 3 might learn positional proximity
  - Head 4 might learn domain-specific patterns
  - ... and so on!

Real transformers typically use 8-16 heads (GPT-3 uses 96 heads!).

C# Analogy: Like having multiple threads analyzing the same data
from different perspectives, then combining their findings.
""")

# ==============================================================================
# PART 2: Multi-Head Attention Architecture
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Multi-Head Attention Architecture")
print("=" * 70)

print("""
How it works:

1. SPLIT: Divide d_model into num_heads chunks
   - If d_model=512 and num_heads=8, each head gets 64 dimensions
   - d_k = d_model // num_heads

2. PARALLEL ATTENTION: Run attention separately for each head
   - Each head has its own W_q, W_k, W_v matrices
   - Each head processes its chunk of dimensions independently

3. CONCATENATE: Combine all head outputs back together
   - Stack results from all heads

4. PROJECT: Apply final linear transformation
   - W_o matrix to combine information from all heads

Formula:
    head_i = Attention(Q_i, K_i, V_i)
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
""")

# ==============================================================================
# PART 3: Implementing Multi-Head Attention
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Multi-Head Attention Implementation")
print("=" * 70)

class MultiHeadAttention:
    """
    Multi-Head Attention layer.

    Similar to C# class:
        public class MultiHeadAttention {
            private int numHeads;
            private int d_k;
            private Matrix[] headWeights;
            public Matrix Forward(Matrix input) { ... }
        }
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Total embedding dimension (e.g., 512)
            num_heads: Number of attention heads (e.g., 8)
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Weight matrices for each head
        # In practice, we use one big matrix and split it (more efficient)
        # But conceptually, each head has its own weights
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1

        # Output projection (combines all heads)
        self.W_o = np.random.randn(d_model, d_model) * 0.1

        print(f"✓ Created MultiHeadAttention:")
        print(f"    d_model={d_model}, num_heads={num_heads}, d_k={self.d_k}")

    def split_heads(self, x):
        """
        Split last dimension into (num_heads, d_k).

        Args:
            x: Shape (batch_size, seq_len, d_model)

        Returns:
            Shape (batch_size, num_heads, seq_len, d_k)

        C# Analogy: Like splitting an array into chunks:
            array.Chunk(d_k).ToArray()
        """
        batch_size, seq_len, d_model = x.shape

        # Reshape: (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose: (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        # This groups all data for each head together
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        """
        Inverse of split_heads.

        Args:
            x: Shape (batch_size, num_heads, seq_len, d_k)

        Returns:
            Shape (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.shape

        # Transpose back: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        x = x.transpose(0, 2, 1, 3)

        # Reshape: (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        return x.reshape(batch_size, seq_len, self.d_model)

    def attention(self, Q, K, V):
        """
        Scaled dot-product attention.

        Args:
            Q, K, V: Shape (..., seq_len, d_k)

        Returns:
            output: Attention output, shape (..., seq_len, d_k)
            weights: Attention weights, shape (..., seq_len, seq_len)
        """
        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / np.sqrt(self.d_k)

        # Apply softmax
        weights = self.softmax(scores)

        # Weighted sum of values
        output = weights @ V

        return output, weights

    def forward(self, X):
        """
        Forward pass of multi-head attention.

        Args:
            X: Input tensor, shape (batch_size, seq_len, d_model)

        Returns:
            output: Output tensor, shape (batch_size, seq_len, d_model)
            all_attention_weights: List of attention weights for each head
        """
        batch_size, seq_len, d_model = X.shape

        # 1. Linear projections
        Q = X @ self.W_q  # (batch, seq_len, d_model)
        K = X @ self.W_k
        V = X @ self.W_v

        # 2. Split into multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. Apply attention for each head (in parallel conceptually)
        attn_output, attn_weights = self.attention(Q, K, V)
        # attn_output: (batch, num_heads, seq_len, d_k)
        # attn_weights: (batch, num_heads, seq_len, seq_len)

        # 4. Concatenate heads
        concat_output = self.combine_heads(attn_output)  # (batch, seq_len, d_model)

        # 5. Final linear projection
        output = concat_output @ self.W_o  # (batch, seq_len, d_model)

        return output, attn_weights

    @staticmethod
    def softmax(x):
        """Apply softmax function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# ==============================================================================
# PART 4: Testing Multi-Head Attention
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Running Multi-Head Attention")
print("=" * 70)

# Create input sequence
sentence = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
seq_len = len(sentence)

print(f"Input sentence: {' '.join(sentence)}")
print(f"Sequence length: {seq_len}")

# Parameters
d_model = 16      # Total embedding dimension (small for demonstration)
num_heads = 4     # Number of attention heads

print(f"\nParameters:")
print(f"  d_model = {d_model}")
print(f"  num_heads = {num_heads}")
print(f"  d_k (per head) = {d_model // num_heads}")

# Create input embeddings
# Add batch dimension for realistic shape: (batch_size=1, seq_len, d_model)
X = np.random.randn(1, seq_len, d_model) * 0.5

print(f"\nInput shape: {X.shape}")

# Create multi-head attention layer
print("\nInitializing multi-head attention:")
mha = MultiHeadAttention(d_model, num_heads)

# Forward pass
print("\nRunning forward pass...")
output, attention_weights = mha.forward(X)

print(f"\nOutput shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
print(f"  (batch=1, num_heads={num_heads}, seq_len={seq_len}, seq_len={seq_len})")

# ==============================================================================
# PART 5: Analyzing Different Attention Heads
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: What Did Each Head Learn?")
print("=" * 70)

print("""
Each head has its own attention pattern!
Let's look at what each head is focusing on.

In a trained model, different heads often learn to focus on:
  - Head 1: Local context (nearby words)
  - Head 2: Syntactic relationships (noun-verb)
  - Head 3: Long-range dependencies
  - Head 4: Semantic similarity
  - etc.

(In our random untrained example, patterns will be random,
but in real transformers these patterns emerge during training!)
""")

# Extract attention weights for each head (remove batch dimension)
head_weights = attention_weights[0]  # Shape: (num_heads, seq_len, seq_len)

print(f"\nAttention statistics for each head:\n")
for h in range(num_heads):
    # Analyze attention pattern for this head
    head_attn = head_weights[h]

    # Compute metrics
    self_attention = np.mean(np.diag(head_attn))  # Attention to self
    neighbor_attention = np.mean([head_attn[i, i-1] + head_attn[i, i+1]
                                   for i in range(1, seq_len-1)]) / 2
    distant_attention = np.mean([head_attn[i, j]
                                 for i in range(seq_len)
                                 for j in range(seq_len)
                                 if abs(i-j) > 2])

    print(f"Head {h+1}:")
    print(f"  Self-attention (diagonal):    {self_attention:.3f}")
    print(f"  Neighbor attention (±1 pos):  {neighbor_attention:.3f}")
    print(f"  Distant attention (>2 apart): {distant_attention:.3f}")
    print()

# ==============================================================================
# PART 6: Visualizing Multi-Head Attention
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Visualizing All Attention Heads")
print("=" * 70)

# Create a grid of heatmaps for all heads
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for h in range(num_heads):
    ax = axes[h]

    # Get attention weights for this head
    head_attn = head_weights[h]

    # Create heatmap
    sns.heatmap(head_attn,
                annot=True,
                fmt='.2f',
                cmap='viridis',
                xticklabels=sentence,
                yticklabels=sentence,
                cbar_kws={'label': 'Attention Weight'},
                ax=ax,
                vmin=0,
                vmax=head_attn.max())

    ax.set_title(f'Head {h+1} Attention Pattern', fontsize=12, fontweight='bold')
    ax.set_xlabel('Keys (attending TO)', fontsize=9)
    ax.set_ylabel('Queries (attending FROM)', fontsize=9)

    # Rotate x labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.suptitle('Multi-Head Attention: Different Perspectives on the Same Sentence',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\nNotice how each head has a DIFFERENT attention pattern!")
print("In a trained transformer, these would represent different")
print("linguistic relationships learned during training.")

# ==============================================================================
# PART 7: Comparing Single-Head vs Multi-Head
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: Single-Head vs Multi-Head Attention")
print("=" * 70)

print("""
SINGLE-HEAD ATTENTION:
  ✓ Simple and straightforward
  ✗ Limited to ONE perspective on relationships
  ✗ Might miss important patterns

MULTI-HEAD ATTENTION:
  ✓ Multiple perspectives simultaneously
  ✓ Can capture different types of relationships
  ✓ More expressive and powerful
  ✗ More parameters to learn
  ✗ More computation (but parallelizable!)

Real-world configurations:
  - BERT base: 12 layers × 12 heads = 144 attention mechanisms!
  - GPT-2: 12 layers × 12 heads = 144 attention mechanisms!
  - GPT-3: 96 layers × 96 heads = 9,216 attention mechanisms!

Each head can specialize in a different aspect of language understanding.
""")

# Demonstrate the difference with a simple example
print("\nParameter count comparison:")
print(f"  Single-head: 3 × {d_model}×{d_model} = {3 * d_model * d_model} parameters")
print(f"  Multi-head ({num_heads} heads): 4 × {d_model}×{d_model} = {4 * d_model * d_model} parameters")
print(f"  (4 matrices: W_q, W_k, W_v, W_o)")

# ==============================================================================
# PART 8: Visualization - Head Specialization
# ==============================================================================

print("\n" + "=" * 70)
print("PART 8: Visualizing Head Specialization")
print("=" * 70)

# Create a visualization showing average attention distance for each head
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Average attention distance per head
distances = []
for h in range(num_heads):
    head_attn = head_weights[h]
    # Compute average attention distance
    avg_distance = sum(head_attn[i, j] * abs(i - j)
                       for i in range(seq_len)
                       for j in range(seq_len))
    distances.append(avg_distance)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
ax1.bar(range(1, num_heads + 1), distances, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Attention Head', fontsize=11)
ax1.set_ylabel('Average Attention Distance', fontsize=11)
ax1.set_title('Head Specialization: Attention Distance\n(Lower = focuses on nearby words)',
              fontsize=12, fontweight='bold')
ax1.set_xticks(range(1, num_heads + 1))
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(distances):
    ax1.text(i + 1, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

# Subplot 2: Attention concentration (entropy)
entropies = []
for h in range(num_heads):
    head_attn = head_weights[h]
    # Compute average entropy (lower = more focused, higher = more dispersed)
    entropy = -np.mean([np.sum(head_attn[i] * np.log(head_attn[i] + 1e-9))
                        for i in range(seq_len)])
    entropies.append(entropy)

ax2.bar(range(1, num_heads + 1), entropies, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Attention Head', fontsize=11)
ax2.set_ylabel('Attention Entropy', fontsize=11)
ax2.set_title('Head Specialization: Attention Focus\n(Lower = more focused, higher = more dispersed)',
              fontsize=12, fontweight='bold')
ax2.set_xticks(range(1, num_heads + 1))
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(entropies):
    ax2.text(i + 1, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

print("\nThese metrics show how different heads behave differently:")
print("  - Some heads focus on nearby words (low distance)")
print("  - Some heads spread attention widely (high entropy)")
print("  - This diversity gives multi-head attention its power!")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - What We Learned")
print("=" * 70)

print("""
✓ Multi-Head Attention runs multiple attention mechanisms in parallel
✓ Each head gets d_k = d_model / num_heads dimensions
✓ Heads can learn different types of relationships
✓ Outputs are concatenated and projected with W_o
✓ More expressive than single-head attention

The Architecture:
    1. Split d_model into num_heads chunks
    2. Apply attention separately for each head
    3. Concatenate all head outputs
    4. Project with W_o to get final output

Key Insight:
Different heads can specialize in different aspects:
  - Syntactic patterns (grammar)
  - Semantic relationships (meaning)
  - Positional proximity (nearby words)
  - Long-range dependencies (distant words)

This is why transformers are so powerful - they can understand
language from multiple perspectives simultaneously!

Next Steps:
- example_05: Transformer block (multi-head attention + FFN + norms)
- example_06: Mini-GPT (complete architecture!)
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 03")
print("=" * 70)
