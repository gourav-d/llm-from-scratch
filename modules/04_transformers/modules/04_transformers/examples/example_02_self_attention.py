"""
Example 02: Self-Attention Layer

This example demonstrates SELF-attention, where Query, Key, and Value all come
from the SAME input sequence. This is the key mechanism that allows transformers
to understand context and relationships between words.

What you'll see:
1. Learned weight matrices (W_q, W_k, W_v) that transform inputs
2. Self-attention: each word attending to all words in the same sequence
3. Context-aware representations (each word "knows" about other words)
4. Comparison: original embeddings vs. context-aware embeddings
5. Visualization showing which words attend to which

C# Analogy:
Think of it like LINQ joining a list with itself:
    from word1 in sentence
    from word2 in sentence
    where IsRelevant(word1, word2)
    select new { word1, word2, relevance }
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("SELF-ATTENTION LAYER")
print("=" * 70)

# ==============================================================================
# PART 1: Setting Up the Input Sequence
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Input Sequence and Embeddings")
print("=" * 70)

# A more interesting sentence to see attention patterns
sentence = ["The", "quick", "brown", "fox", "jumps"]
print(f"Input sentence: {' '.join(sentence)}")

# Embedding dimension (keeping it small for clarity)
d_model = 8  # Slightly larger than example_01 for more expressive power

print(f"\nEmbedding dimension: {d_model}")
print("(Real transformers use 512, 768, or even larger!)")

# Create word embeddings
# Shape: (sequence_length, d_model) = (5, 8)
num_words = len(sentence)
X = np.random.randn(num_words, d_model) * 0.5  # Input embeddings

print(f"\nInput embeddings X shape: {X.shape}")
print("These are the initial word representations (before attention)\n")

# ==============================================================================
# PART 2: Learned Weight Matrices
# ==============================================================================

print("=" * 70)
print("PART 2: Creating Learned Weight Matrices")
print("=" * 70)

print("""
Unlike example_01 where Q=K=V directly, in REAL transformers we learn
three separate weight matrices that transform the input:

W_q: Transforms input into Queries  (what to look for)
W_k: Transforms input into Keys     (what to offer)
W_v: Transforms input into Values   (what information to give)

These are LEARNED during training! (Like weights in a neural network)

C# Analogy: Similar to applying different LINQ Select transformations:
    Q = input.Select(x => x.Transform(W_q))
    K = input.Select(x => x.Transform(W_k))
    V = input.Select(x => x.Transform(W_v))
""")

# Initialize weight matrices (in training, these would be learned)
# Each matrix transforms d_model dimensions to d_model dimensions
# Shape: (d_model, d_model) = (8, 8)

W_q = np.random.randn(d_model, d_model) * 0.1  # Query weight matrix
W_k = np.random.randn(d_model, d_model) * 0.1  # Key weight matrix
W_v = np.random.randn(d_model, d_model) * 0.1  # Value weight matrix

print(f"W_q shape: {W_q.shape}")
print(f"W_k shape: {W_k.shape}")
print(f"W_v shape: {W_v.shape}")

# ==============================================================================
# PART 3: Computing Q, K, V through Projection
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Projecting Input to Q, K, V")
print("=" * 70)

print("Now we transform our input X using the weight matrices:\n")

# Project input to queries, keys, and values
# Matrix multiplication: X @ W_q
# Shape: (5, 8) @ (8, 8) = (5, 8)

Q = X @ W_q  # Queries: what each word is looking for
K = X @ W_k  # Keys: what each word offers
V = X @ W_v  # Values: information each word contains

print(f"Q (Queries) = X @ W_q")
print(f"  Shape: {X.shape} @ {W_q.shape} = {Q.shape}")

print(f"\nK (Keys) = X @ W_k")
print(f"  Shape: {X.shape} @ {W_k.shape} = {K.shape}")

print(f"\nV (Values) = X @ W_v")
print(f"  Shape: {X.shape} @ {W_v.shape} = {V.shape}")

print("""
Key insight: Q, K, V all came from the SAME input X!
This is why it's called SELF-attention.
""")

# ==============================================================================
# PART 4: Self-Attention Class Implementation
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Self-Attention Class")
print("=" * 70)

class SelfAttention:
    """
    Self-Attention layer implementation.

    Similar to a C# class with methods for forward pass:
        public class SelfAttention {
            private Matrix W_q, W_k, W_v;
            public Matrix Forward(Matrix input) { ... }
        }
    """

    def __init__(self, d_model):
        """
        Initialize self-attention layer.

        Args:
            d_model: Dimension of embeddings (like 512 in real transformers)
        """
        self.d_model = d_model

        # Initialize weight matrices
        # In practice, these are learned through backpropagation
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1

        print(f"✓ Created SelfAttention layer with d_model={d_model}")

    def forward(self, X):
        """
        Forward pass: compute self-attention.

        Args:
            X: Input embeddings, shape (seq_len, d_model)

        Returns:
            output: Context-aware embeddings, shape (seq_len, d_model)
            attention_weights: Attention matrix, shape (seq_len, seq_len)
        """
        # Step 1: Project to Q, K, V
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        # Step 2: Compute attention scores
        # Q @ K^T gives similarity between all pairs of positions
        d_k = self.d_model
        scores = Q @ K.T / np.sqrt(d_k)  # Scaled dot-product

        # Step 3: Apply softmax to get attention weights
        attention_weights = self.softmax(scores)

        # Step 4: Compute weighted sum of values
        output = attention_weights @ V

        return output, attention_weights

    @staticmethod
    def softmax(x):
        """Apply softmax function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Create and test the self-attention layer
print("\nCreating self-attention layer:")
attention_layer = SelfAttention(d_model)

# ==============================================================================
# PART 5: Running Self-Attention
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Computing Self-Attention")
print("=" * 70)

print("Running forward pass through self-attention layer...\n")

# Perform forward pass
output, attention_weights = attention_layer.forward(X)

print(f"Input shape:  {X.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")

print("\n✓ Each word now has a context-aware representation!")
print("  It 'knows' about the other words in the sentence.")

# ==============================================================================
# PART 6: Analyzing Attention Patterns
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Understanding Attention Patterns")
print("=" * 70)

print("\nAttention weights matrix:")
print("(Shows how much each word attends to every other word)\n")

# Print attention weights with word labels
print("      ", end="")
for word in sentence:
    print(f"{word:>8s}", end="")
print()

for i, word in enumerate(sentence):
    print(f"{word:6s}", end="")
    for j in range(len(sentence)):
        print(f"{attention_weights[i, j]:8.3f}", end="")
    print()

print("\nInterpretation:")
print("- Each row shows how one word distributes its attention")
print("- Each column shows how much attention one word receives")
print("- All rows sum to 1.0 (probability distribution)")

# ==============================================================================
# PART 7: Visualizing Self-Attention
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: Visualizing Self-Attention Patterns")
print("=" * 70)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Attention heatmap
sns.heatmap(attention_weights,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=sentence,
            yticklabels=sentence,
            cbar_kws={'label': 'Attention Weight'},
            ax=ax1)
ax1.set_title('Self-Attention Weights', fontsize=14, fontweight='bold')
ax1.set_xlabel('Attending TO (Keys)', fontsize=11)
ax1.set_ylabel('Attending FROM (Queries)', fontsize=11)

# Subplot 2: Attention pattern for one word
focus_idx = 3  # "fox"
focus_word = sentence[focus_idx]

ax2.bar(range(len(sentence)), attention_weights[focus_idx], color='skyblue', edgecolor='navy')
ax2.set_xticks(range(len(sentence)))
ax2.set_xticklabels(sentence, rotation=45)
ax2.set_ylabel('Attention Weight', fontsize=11)
ax2.set_title(f'How "{focus_word}" attends to all words', fontsize=14, fontweight='bold')
ax2.set_ylim(0, max(attention_weights[focus_idx]) * 1.1)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(attention_weights[focus_idx]):
    ax2.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ==============================================================================
# PART 8: Comparing Before and After Attention
# ==============================================================================

print("\n" + "=" * 70)
print("PART 8: Before vs. After Self-Attention")
print("=" * 70)

print(f"\nLet's see how '{sentence[2]}' (brown) changed:\n")

print("BEFORE attention (original embedding):")
print(f"  {X[2]}")

print("\nAFTER attention (context-aware):")
print(f"  {output[2]}")

print("\nWhat happened?")
print(f"  The word '{sentence[2]}' now incorporates information from:")
for i, word in enumerate(sentence):
    weight = attention_weights[2, i]
    print(f"    - {weight*100:5.1f}% from '{word}'")

print("\nThis new representation is CONTEXT-AWARE!")
print("It knows about the surrounding words, not just itself.")

# ==============================================================================
# PART 9: Key Differences from Basic Attention
# ==============================================================================

print("\n" + "=" * 70)
print("PART 9: Self-Attention vs. Basic Attention")
print("=" * 70)

print("""
BASIC ATTENTION (Example 01):
  - Q, K, V could come from different sources
  - Used in encoder-decoder architectures
  - Query from one sequence, Key/Value from another

SELF-ATTENTION (This Example):
  - Q, K, V all from the SAME input sequence
  - Each position attends to all positions (including itself)
  - Learns relationships WITHIN a sequence
  - Uses learned weight matrices (W_q, W_k, W_v)

Why Self-Attention is Powerful:
  ✓ Captures long-range dependencies (word 1 can attend to word 100)
  ✓ No sequential processing needed (all positions computed in parallel)
  ✓ Context-aware representations (each word knows about others)
  ✓ Position-independent (same computation regardless of position)

C# LINQ Analogy:
  Basic Attention:    from a in list1 join b in list2 ...
  Self-Attention:     from a in list join b in list ...
                      (joining with itself!)
""")

# ==============================================================================
# PART 10: Practical Example with Real Meaning
# ==============================================================================

print("\n" + "=" * 70)
print("PART 10: Why This Matters - Pronoun Resolution Example")
print("=" * 70)

print("""
Consider: "The cat chased the mouse. It was fast."

What does "It" refer to? The cat or the mouse?

Self-attention allows the model to:
1. Look at "It" and attend back to "cat" and "mouse"
2. Use context (chased, fast) to determine "It" likely means the cat
3. Build a representation of "It" that incorporates "cat" information

This is how transformers understand context and relationships!
""")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - What We Learned")
print("=" * 70)

print("""
✓ Self-Attention: Q, K, V all from same input sequence
✓ Learned Matrices: W_q, W_k, W_v transform inputs
✓ Context-Aware: Each word's representation includes info from all words
✓ Attention Weights: Show which words are most relevant to each other
✓ Parallel Processing: All positions computed simultaneously

The Formula:
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

Key Insight:
Self-attention is how transformers build understanding of relationships
between words, without any notion of word order (yet - that's next!).

Next Steps:
- example_03: Multi-head attention (multiple attention patterns)
- example_04: Positional encoding (adding sequence order)
- example_05: Complete transformer block (attention + feed-forward)
- example_06: Mini-GPT (putting it all together!)
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 02")
print("=" * 70)
