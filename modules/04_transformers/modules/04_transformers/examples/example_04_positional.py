"""
Example 04: Positional Encoding

This example demonstrates how transformers add position information to embeddings.
Self-attention has no built-in sense of word order - "cat sat mat" and "mat sat cat"
look identical! Positional encoding solves this problem.

What you'll see:
1. Why we need positional information (order matters!)
2. Sinusoidal positional encoding (the clever math trick)
3. How position encodings are added to word embeddings
4. Visualization of position patterns across dimensions
5. Different positions create unique "fingerprints"

C# Analogy:
Think of it like adding an index to a list:
    words.Select((word, index) => new { word, position = index })

But instead of just 0, 1, 2, ..., we use a clever pattern of sine and cosine
waves that helps the model understand relative positions!
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed
np.random.seed(42)

print("=" * 70)
print("POSITIONAL ENCODING")
print("=" * 70)

# ==============================================================================
# PART 1: The Problem - Self-Attention Has No Sense of Order
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: The Problem - Why We Need Positional Encoding")
print("=" * 70)

print("""
Self-attention is PERMUTATION INVARIANT - word order doesn't matter!

Example sentences:
  1. "The cat sat on the mat"
  2. "The mat sat on the cat"

To self-attention (without positional encoding), these look THE SAME!
Both have the same words, just in different order.

But meaning is completely different:
  Sentence 1: Cat is sitting (makes sense)
  Sentence 2: Mat is sitting (nonsense!)

SOLUTION: Add positional information to each word's embedding.
          Each position gets a unique "fingerprint" that encodes WHERE it is.
""")

sentence1 = ["The", "cat", "sat", "on", "the", "mat"]
sentence2 = ["The", "mat", "sat", "on", "the", "cat"]

print(f"Sentence 1: {' '.join(sentence1)}")
print(f"Sentence 2: {' '.join(sentence2)}")
print("\nWithout position info, self-attention can't tell these apart!")

# ==============================================================================
# PART 2: Sinusoidal Positional Encoding
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Sinusoidal Positional Encoding")
print("=" * 70)

print("""
The original transformer paper uses sine and cosine functions:

For position 'pos' and dimension 'i':
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Why sine and cosine?
  1. Values are bounded between -1 and +1 (won't dominate embeddings)
  2. Different frequencies for different dimensions
  3. Model can learn to attend by relative position
  4. Works for sequences of any length (even longer than training!)

C# Analogy: Like creating a hash code for each position, but with
mathematical properties that help the model learn relationships.
""")

def positional_encoding(max_seq_len, d_model):
    """
    Generate sinusoidal positional encodings.

    Args:
        max_seq_len: Maximum sequence length
        d_model: Embedding dimension (must be even)

    Returns:
        Positional encoding matrix of shape (max_seq_len, d_model)
    """
    # Create position indices: [0, 1, 2, ..., max_seq_len-1]
    # Shape: (max_seq_len, 1)
    position = np.arange(max_seq_len)[:, np.newaxis]

    # Create dimension indices: [0, 2, 4, ..., d_model-2]
    # We use every other dimension (even indices)
    # Shape: (d_model // 2,)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Initialize positional encoding matrix
    # Shape: (max_seq_len, d_model)
    PE = np.zeros((max_seq_len, d_model))

    # Even dimensions: use sine
    # PE[:, 0], PE[:, 2], PE[:, 4], ...
    PE[:, 0::2] = np.sin(position * div_term)

    # Odd dimensions: use cosine
    # PE[:, 1], PE[:, 3], PE[:, 5], ...
    PE[:, 1::2] = np.cos(position * div_term)

    return PE

# ==============================================================================
# PART 3: Creating Positional Encodings
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Generating Positional Encodings")
print("=" * 70)

# Parameters
max_seq_len = 50  # Support sequences up to 50 words
d_model = 64      # Embedding dimension (using 64 for better visualization)

print(f"Creating positional encodings for:")
print(f"  - Maximum sequence length: {max_seq_len}")
print(f"  - Embedding dimension: {d_model}")

# Generate positional encodings
PE = positional_encoding(max_seq_len, d_model)

print(f"\nPositional encoding shape: {PE.shape}")
print(f"  ({max_seq_len} positions × {d_model} dimensions)")

# Show encoding for first few positions
print("\nPositional encoding for first 5 positions:")
print("Position | First 8 dimensions")
print("-" * 50)
for pos in range(5):
    values = PE[pos, :8]
    values_str = ' '.join([f'{v:6.3f}' for v in values])
    print(f"   {pos}     | {values_str} ...")

print("\nNotice how each position has a UNIQUE pattern of values!")

# ==============================================================================
# PART 4: Visualizing Positional Encodings
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Visualizing Positional Encoding Patterns")
print("=" * 70)

# Create a comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Subplot 1: Full positional encoding heatmap
ax1 = fig.add_subplot(gs[0, :])
sns.heatmap(PE.T,  # Transpose to show dimensions as rows
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Encoding Value'},
            ax=ax1)
ax1.set_title('Positional Encoding Heatmap\n(Each column is a position, each row is a dimension)',
              fontsize=12, fontweight='bold')
ax1.set_xlabel('Position in Sequence', fontsize=10)
ax1.set_ylabel('Embedding Dimension', fontsize=10)

# Subplot 2: Encoding for specific dimensions
ax2 = fig.add_subplot(gs[1, 0])
dimensions_to_plot = [0, 1, 4, 5, 16, 17]
for dim in dimensions_to_plot:
    ax2.plot(PE[:, dim], label=f'Dim {dim}', linewidth=2)
ax2.set_xlabel('Position', fontsize=10)
ax2.set_ylabel('Encoding Value', fontsize=10)
ax2.set_title('Positional Encoding Curves\n(Different frequencies for different dimensions)',
              fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)

# Subplot 3: Encoding for specific positions
ax3 = fig.add_subplot(gs[1, 1])
positions_to_plot = [0, 5, 10, 20, 30, 40]
for pos in positions_to_plot:
    ax3.plot(PE[pos, :32], label=f'Pos {pos}', linewidth=2)  # Plot first 32 dims
ax3.set_xlabel('Embedding Dimension', fontsize=10)
ax3.set_ylabel('Encoding Value', fontsize=10)
ax3.set_title('Position "Fingerprints"\n(Each position has unique pattern across dimensions)',
              fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(True, alpha=0.3)

# Subplot 4: 2D visualization using first two dimensions
ax4 = fig.add_subplot(gs[2, 0])
scatter = ax4.scatter(PE[:, 0], PE[:, 1], c=np.arange(max_seq_len),
                     cmap='viridis', s=100, alpha=0.6, edgecolors='black')
ax4.set_xlabel('Dimension 0 (sin)', fontsize=10)
ax4.set_ylabel('Dimension 1 (cos)', fontsize=10)
ax4.set_title('2D Position Space (Dims 0 & 1)\n(Notice the spiral pattern!)',
              fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Position')

# Subplot 5: Frequency analysis
ax5 = fig.add_subplot(gs[2, 1])
# Show how frequency changes across dimensions
freqs = 1.0 / (10000.0 ** (np.arange(0, d_model, 2) / d_model))
ax5.plot(np.arange(0, d_model, 2), freqs, marker='o', linewidth=2, markersize=6)
ax5.set_xlabel('Dimension (even only)', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title('Frequency Spectrum Across Dimensions\n(Lower dimensions = higher frequency)',
              fontsize=11, fontweight='bold')
ax5.set_yscale('log')
ax5.grid(True, alpha=0.3, which='both')

plt.suptitle('Understanding Positional Encoding', fontsize=16, fontweight='bold', y=0.995)
plt.show()

print("\nKey observations from visualizations:")
print("  1. Each position has a unique 'fingerprint' across dimensions")
print("  2. Lower dimensions oscillate faster (high frequency)")
print("  3. Higher dimensions oscillate slower (low frequency)")
print("  4. This multi-scale pattern helps model learn both local and global position info")

# ==============================================================================
# PART 5: Adding Positional Encoding to Word Embeddings
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Combining Embeddings with Positional Encoding")
print("=" * 70)

print("""
In a transformer, we ADD positional encoding to word embeddings:

    final_embedding = word_embedding + positional_encoding

This gives each word two types of information:
  1. WHAT the word is (from word embedding)
  2. WHERE the word is (from positional encoding)
""")

# Example with our sentence
sentence = ["The", "cat", "sat", "on", "the", "mat"]
seq_len = len(sentence)
d_model_small = 8  # Smaller for this example

print(f"\nSentence: {' '.join(sentence)}")
print(f"Using d_model = {d_model_small} for this example\n")

# Create word embeddings (random for this example)
word_embeddings = np.random.randn(seq_len, d_model_small) * 0.5

# Get positional encodings for this sequence length
pos_encodings = positional_encoding(seq_len, d_model_small)

# Combine them (simple addition!)
final_embeddings = word_embeddings + pos_encodings

print("Word embeddings (what the word means):")
print(word_embeddings)

print("\nPositional encodings (where the word is):")
print(pos_encodings)

print("\nFinal embeddings (word meaning + position):")
print(final_embeddings)

print("\nNow each word has:")
print("  ✓ Its semantic meaning (from word embedding)")
print("  ✓ Its position in the sentence (from positional encoding)")

# ==============================================================================
# PART 6: Demonstrating Position Matters
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Position Makes a Difference!")
print("=" * 70)

# Create two embeddings for the word "cat" at different positions
word_embedding_cat = np.random.randn(d_model_small) * 0.5

# "cat" at position 1
cat_at_pos1 = word_embedding_cat + pos_encodings[1]

# "cat" at position 4
cat_at_pos4 = word_embedding_cat + pos_encodings[4]

print(f"Same word 'cat', different positions:\n")
print("'cat' at position 1:")
print(f"  {cat_at_pos1}")

print("\n'cat' at position 4:")
print(f"  {cat_at_pos4}")

# Compute similarity (cosine similarity would be better, but let's use simple distance)
difference = np.linalg.norm(cat_at_pos1 - cat_at_pos4)
print(f"\nDifference between them: {difference:.4f}")
print("Even though it's the same word, the position makes them different!")

# ==============================================================================
# PART 7: Why Sine and Cosine?
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: The Magic of Sine and Cosine")
print("=" * 70)

print("""
Why use sin/cos instead of simple numbers like [0, 1, 2, 3, ...]?

1. BOUNDED VALUES:
   - sin/cos stay between -1 and +1
   - Won't dominate the word embeddings (which are also small numbers)

2. PERIODIC PATTERNS:
   - Model can learn relative positions
   - PE(pos + k) can be expressed as function of PE(pos)

3. UNIQUE FINGERPRINTS:
   - Different positions have different patterns
   - Even for very long sequences!

4. SCALABILITY:
   - Works for sequences longer than seen during training
   - No maximum sequence length!

C# Analogy: Like using a hash function that:
  - Produces similar hashes for nearby items (pos 5 and pos 6)
  - Produces different hashes for distant items (pos 5 and pos 500)
  - Never runs out of unique hashes (infinite sequence support)
""")

# Demonstrate similarity between nearby positions
pos_5 = PE[5, :]
pos_6 = PE[6, :]
pos_25 = PE[25, :]

similarity_nearby = np.dot(pos_5, pos_6) / (np.linalg.norm(pos_5) * np.linalg.norm(pos_6))
similarity_far = np.dot(pos_5, pos_25) / (np.linalg.norm(pos_5) * np.linalg.norm(pos_25))

print(f"\nCosine similarity:")
print(f"  Position 5 vs Position 6 (nearby):  {similarity_nearby:.4f}")
print(f"  Position 5 vs Position 25 (far):    {similarity_far:.4f}")
print("\nNearby positions are more similar!")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - What We Learned")
print("=" * 70)

print("""
✓ Self-Attention is order-agnostic (word order doesn't matter without help)
✓ Positional Encoding adds position information to embeddings
✓ Sinusoidal encoding uses sin/cos with different frequencies
✓ Each position gets a unique "fingerprint" across dimensions
✓ We simply ADD positional encoding to word embeddings
✓ This gives each word both MEANING (what) and POSITION (where)

The Formula:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    final_embedding = word_embedding + PE

Key Insight:
Positional encoding is what gives transformers their sense of word order!
Without it, "cat sat mat" and "mat sat cat" would be identical.

Next Steps:
- example_03: Multi-head attention (multiple attention patterns)
- example_05: Transformer block (attention + FFN + norms)
- example_06: Mini-GPT (complete architecture with positional encoding!)
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 04")
print("=" * 70)
